import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_DEFAULT = 'Qwen/Qwen3-8B'
MINI_DEFAULT = 'sentence-transformers/all-MiniLM-L6-v2'
ALIASES = {
    'movie': ['movie','Movie_Name','movie_name','Movie'],
    'speaker': ['speaker','character_name','Character','character'],
    'source': ['source','english_dialogue','English_Dailouge','English_Dialogue','english'],
    'target': ['target','hindi_dialogue','Hindi_Dailouge','Hindi_Dialogue','hindi'],
    'emotion': ['Emotion','emotion','emotion_label'],
    'scene': ['Scene_ID','scene_id','scene'],
    'start': ['Start_Time','start_time','start'],
}

def normalize_columns(df):
    names = {}
    for standard, candidates in ALIASES.items():
        match = next((c for c in candidates if c in df.columns), None)
        if match: names[match] = standard
    return df.rename(columns=names)

def load_csvs(paths):
    parts = []
    for p in paths:
        df = normalize_columns(pd.read_csv(p))
        missing = set(['speaker','source','target']) - set(df.columns)
        if missing:
            raise ValueError(f'{p}: missing columns {sorted(missing)}; got {list(df.columns)}')
        df['movie'] = df['movie'] if 'movie' in df else Path(p).stem
        df['movie'] = df['movie'].fillna('Unknown').astype(str).str.strip()
        df['speaker'] = df['speaker'].fillna('Unknown').astype(str).str.strip()
        df['source'] = df['source'].fillna('').astype(str).str.strip()
        df['target'] = df['target'].fillna('').astype(str).str.strip()
        df['emotion'] = df['emotion'].fillna('unknown').astype(str).str.strip() if 'emotion' in df else 'unknown'
        df['scene'] = df['scene'].fillna('').astype(str).str.strip() if 'scene' in df else ''
        if 'start' in df:
            df['_seconds'] = df['start'].apply(parse_time)
        else:
            df['_seconds'] = np.nan
        df['_order'] = np.arange(len(df))
        # If timestamps are unusable, retain file order; else sort by time within each film.
        df = df.sort_values(['movie','_seconds','_order'], kind='stable', na_position='last')
        df = df[(df.source != '') & (df.target != '') & (df.movie != 'Unknown')].copy()
        df = df.reset_index(drop=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)

def parse_time(x):
    if pd.isna(x): return np.nan
    try:
        s = str(x).strip().replace(',', '.')
        if not s: return np.nan
        if ':' not in s: return float(s)
        result=0.0
        for a in s.split(':'): result=result*60+float(a)
        return result
    except (ValueError, TypeError): return np.nan

class StyleBridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size,1024),nn.GELU(),nn.Dropout(.1),
                                 nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,hidden_size),
                                 nn.LayerNorm(hidden_size))
    def forward(self, x): return self.net(x)

@torch.no_grad()
def sentence_vectors(df, tokenizer, model, device, batch_size=64):
    vectors = []
    for start in tqdm(range(0,len(df),batch_size),desc='MiniLM encodings'):
        lines = df.source.iloc[start:start+batch_size].tolist()
        tokens = tokenizer(lines,padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
        out = model(**tokens).last_hidden_state
        mask = tokens.attention_mask.unsqueeze(-1).to(out.dtype)
        vec = (out*mask).sum(1)/mask.sum(1).clamp(min=1)
        vec = nn.functional.normalize(vec.float(),p=2,dim=-1)
        vectors.append(vec.cpu())
    return torch.cat(vectors) if vectors else torch.empty(0,384)

def build_examples(df, line_vectors, history_limit=50, context_turns=3):
    # History uses only earlier dialogue *within the same film*; current text excluded.
    rows=[]
    for movie, film in df.groupby('movie',sort=False):
        memory = defaultdict(list)
        prior = []
        for ix, r in film.iterrows():
            who=str(r.speaker)
            known=memory[who][-history_limit:]
            if known:
                anchor=line_vectors[known].mean(0)
                anchor=nn.functional.normalize(anchor,dim=0)
            else:
                anchor=torch.zeros(line_vectors.shape[-1])
            scene_id=str(r.scene)
            # Scene boundaries usable only where actually populated.
            context=[p for p in prior if not scene_id or not p['scene'] or p['scene']==scene_id][-context_turns:]
            rows.append({'movie':movie,'speaker':who,'source':str(r.source),'target':str(r.target),
                         'emotion':str(r.emotion),'context':context,'anchor':anchor})
            memory[who].append(ix)
            prior.append({'speaker':who,'source':str(r.source),'scene':scene_id})
    return rows

class MovieDataset(Dataset):
    def __init__(self, examples, tokenizer, mode, max_tokens):
        self.items=[]
        self.mode=mode
        for r in examples:
            context='\n'.join(f"{p['speaker']}: {p['source']}" for p in r['context']) or '(start of scene)'
            instructions='Translate the current English movie dialogue into natural Hindi. Preserve its meaning. Return only the Hindi translation.'
            content=f"Current English dialogue: {r['source']}"
            if mode=='narrative':
                content=(f"Previous dialogue turns (context only):\n{context}\n"
                         f"Character: {r['speaker']}\n"
                         f"Current English dialogue: {r['source']}")
            messages=[{'role':'system','content':instructions},{'role':'user','content':content}]
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            p=tokenizer(prompt,add_special_tokens=False).input_ids
            t=tokenizer(r['target'],add_special_tokens=False).input_ids+[tokenizer.eos_token_id]
            # Keep complete target, trim source prompt from the left if needed.
            if len(t)>max_tokens//2: t=t[:max_tokens//2-1]+[tokenizer.eos_token_id]
            p=p[-(max_tokens-len(t)):]
            self.items.append({'ids':p+t,'labels':[-100]*len(p)+t,'anchor':r['anchor']})
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]

def collate(items, pad_id):
    n=max(len(x['ids']) for x in items)
    ids=torch.full((len(items),n),pad_id,dtype=torch.long)
    labels=torch.full_like(ids,-100)
    mask=torch.zeros_like(ids)
    for i,item in enumerate(items):
        k=len(item['ids']); ids[i,:k]=torch.tensor(item['ids']);labels[i,:k]=torch.tensor(item['labels']);mask[i,:k]=1
    return {'input_ids':ids,'labels':labels,'attention_mask':mask,
            'anchor':torch.stack([x['anchor'] for x in items])}

def model_forward(model, bridge, batch):
    embedding=model.get_input_embeddings()
    device=embedding.weight.device
    ids=batch['input_ids'].to(device)
    mask=batch['attention_mask'].to(device)
    labels=batch['labels'].to(device)
    if bridge is None:
        return model(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False)
    token=bridge(batch['anchor'].to(device,dtype=torch.float32)).unsqueeze(1)
    token=token.to(dtype=embedding.weight.dtype)
    embeddings=torch.cat([token,embedding(ids)],dim=1)
    labels=torch.cat([torch.full((len(ids),1),-100,device=device,dtype=torch.long),labels],dim=1)
    mask=torch.cat([torch.ones((len(ids),1),device=device,dtype=mask.dtype),mask],dim=1)
    return model(inputs_embeds=embeddings,attention_mask=mask,labels=labels,use_cache=False)


def make_prompt(tokenizer, r, mode):
    context='\n'.join(f"{p['speaker']}: {p['source']}" for p in r['context']) or '(start of scene)'
    instructions='Translate the current English movie dialogue into natural Hindi. Preserve its meaning. Return only the Hindi translation.'
    content=f"Current English dialogue: {r['source']}"
    if mode=='narrative':
        content=(f"Previous dialogue turns (context only):\n{context}\n"
                 f"Character: {r['speaker']}\n"
                 f"Current English dialogue: {r['source']}")
    messages=[{'role':'system','content':instructions},{'role':'user','content':content}]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)


def train_one(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for this training configuration')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=load_csvs(args.train);valid=load_csvs(args.valid)
    if not args.allow_same_movie_validation:
        assert not(set(train.movie)&set(valid.movie)), 'Movie overlap: validation must use held-out films'
    else:
        # Ensure validation examples are disjoint from train examples within each movie.
        train_keys=set(zip(train.movie,train.source,train._seconds.fillna(-1)))
        valid_keys=set(zip(valid.movie,valid.source,valid._seconds.fillna(-1)))
        if train_keys & valid_keys:
            raise ValueError('Training and validation rows overlap')
    print('Train',len(train),sorted(train.movie.unique()),'Validation',len(valid),sorted(valid.movie.unique()))
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    # Precompute embeddings with a frozen MiniLM before loading Qwen to save VRAM.
    if args.mode!='baseline':
        mini_device='cuda'
        mtok=AutoTokenizer.from_pretrained(args.mini)
        mini=AutoModel.from_pretrained(args.mini).to(mini_device).eval()
        tr_vec=sentence_vectors(train,mtok,mini,mini_device)
        va_vec=sentence_vectors(valid,mtok,mini,mini_device)
        del mini;torch.cuda.empty_cache()
    else:
        tr_vec=torch.zeros((len(train),384));va_vec=torch.zeros((len(valid),384))
    training=MovieDataset(build_examples(train,tr_vec),tok,args.mode,args.max_tokens)
    validation=MovieDataset(build_examples(valid,va_vec),tok,args.mode,args.max_tokens)
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=True,collate_fn=lambda x:collate(x,tok.pad_token_id))
    vloader=DataLoader(validation,batch_size=args.batch_size,shuffle=False,collate_fn=lambda x:collate(x,tok.pad_token_id))
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'})
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
                                target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
    bridge=None
    if args.mode!='baseline':
        bridge=StyleBridge(tr_vec.shape[-1],model.config.hidden_size).to('cuda',dtype=torch.float32)
    params=[p for p in model.parameters() if p.requires_grad]
    if bridge is not None:params+=list(bridge.parameters())
    optim=AdamW(params,lr=args.lr)
    history=[]
    for ep in range(args.epochs):
        model.train()
        if bridge is not None:bridge.train()
        optim.zero_grad(set_to_none=True)
        total=0.0
        for step,b in enumerate(tqdm(loader,desc=f'Epoch {ep+1}')):
            loss=model_forward(model,bridge,b).loss
            (loss/args.grad_accum).backward()
            total+=float(loss.detach())
            if (step+1)%args.grad_accum==0 or step+1==len(loader):
                torch.nn.utils.clip_grad_norm_(params,1.0)
                optim.step();optim.zero_grad(set_to_none=True)
        model.eval()
        if bridge is not None:bridge.eval()
        val_loss=0.0
        with torch.no_grad():
            for b in tqdm(vloader,desc='Validation'):
                val_loss+=float(model_forward(model,bridge,b).loss)
        metrics={'epoch':ep+1,'train_loss':total/max(len(loader),1),'val_loss':val_loss/max(len(vloader),1)}
        history.append(metrics);print(metrics)
        (out/'training_history.json').write_text(json.dumps(history,indent=2))
        model.save_pretrained(out/'lora')
        tok.save_pretrained(out/'lora')
        if bridge is not None:torch.save(bridge.state_dict(),out/'stylebridge.pt')
    config={'model':args.model,'mini':args.mini,'mode':args.mode,'max_tokens':args.max_tokens,
            'pooling':'masked-mean, normalized per sentence; mean over preceding same-film character lines',
            'history_limit':50,'context_turns':3,'train_movies':sorted(train.movie.unique().tolist()),
            'valid_movies':sorted(valid.movie.unique().tolist()),'seed':args.seed}
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2))
    print('Saved:',out)
    del model, bridge, optim, loader, vloader

def evaluate_one(a):
    if not torch.cuda.is_available(): raise ValueError('CUDA GPU required')
    run=Path(a.run)
    cfg=json.loads((run/'experiment_config.json').read_text())
    if cfg['mode']!=a.mode:raise ValueError('Mode differs from trained checkpoint')
    df=load_csvs([a.test]);
    overlap=set(df.movie.unique()) & set(cfg['train_movies'])
    if overlap:raise ValueError(f'Test film was present in training: {overlap}')
    model_name=cfg.get('model',MODEL_DEFAULT)
    tok=AutoTokenizer.from_pretrained(run/'lora')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if a.mode!='baseline':
        mtok=AutoTokenizer.from_pretrained(cfg.get('mini',MINI_DEFAULT))
        mini=AutoModel.from_pretrained(cfg.get('mini',MINI_DEFAULT)).to('cuda').eval()
        vec=sentence_vectors(df,mtok,mini,'cuda')
        del mini;torch.cuda.empty_cache()
    else: vec=torch.zeros((len(df),384))
    examples=build_examples(df,vec,history_limit=cfg.get('history_limit',50),context_turns=cfg.get('context_turns',3))
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map={'':'cuda:0'})
    model=PeftModel.from_pretrained(model,run/'lora').eval()
    embed=model.get_input_embeddings();device=embed.weight.device
    bridge=None
    if a.mode!='baseline':
        bridge=StyleBridge(vec.shape[-1],model.config.hidden_size).to(device,dtype=torch.float32)
        bridge.load_state_dict(torch.load(run/'stylebridge.pt',map_location=device,weights_only=True));bridge.eval()
    output=[]
    for r in tqdm(examples[:a.limit or None],desc='Held-out movie test'):
        prompt=make_prompt(tok,r,a.mode)
        ids=tok(prompt,add_special_tokens=False,return_tensors='pt').input_ids.to(device)
        mask=torch.ones_like(ids)
        with torch.inference_mode():
            if bridge is not None:
                anchor=bridge(r['anchor'].unsqueeze(0).to(device,dtype=torch.float32)).unsqueeze(1).to(embed.weight.dtype)
                inp=torch.cat([anchor,embed(ids)],1)
                attention=torch.cat([torch.ones((1,1),dtype=mask.dtype,device=device),mask],1)
                generated=model.generate(inputs_embeds=inp,attention_mask=attention,max_new_tokens=a.max_new_tokens,
                                         do_sample=False,pad_token_id=tok.eos_token_id)
            else:
                generated=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=a.max_new_tokens,
                                         do_sample=False,pad_token_id=tok.eos_token_id)
        seq=generated[0]
        # When generating from input_ids the output includes the prompt; when using inputs_embeds
        # many transformers versions return only the completion, so trim only in baseline mode.
        if bridge is None:seq=seq[ids.shape[1]:]
        pred=tok.decode(seq,skip_special_tokens=True).strip()
        output.append({'movie':r['movie'],'character':r['speaker'],'source':r['source'],
                       'reference_provisional':r['target'],'emotion':r['emotion'],
                       'prediction':pred})
    run.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_csv(run/'predictions.csv',index=False)
    metrics={'examples':len(output),'test_movies':sorted(df.movie.unique().tolist()),
             'note':'Automatic scores use unverified/provisional Hindi reference translations.'}
    if output:
        try:
            import sacrebleu
            preds=[r['prediction'] for r in output]
            refs=[r['reference_provisional'] for r in output]
            metrics['BLEU_provisional']=sacrebleu.corpus_bleu(preds,[refs]).score
            metrics['chrFpp_provisional']=sacrebleu.corpus_chrf(preds,[refs],word_order=2).score
        except ImportError:
            metrics['scores']='Install sacrebleu to calculate BLEU/chrF++'
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(metrics)
    del model, bridge


def main():
    parser = argparse.ArgumentParser(description="English-Hindi 10-film Qwen3 StyleBridge: 8 train / 1 validation / 1 test")
    parser.add_argument('--data',required=True,help='CSV with movie, speaker, source and target columns')
    parser.add_argument('--test-movie',default='Get_Out')
    parser.add_argument('--valid-movie',default='Arrival')
    parser.add_argument('--output',default='runs/hindi_stylebridge')
    parser.add_argument('--model',default=MODEL_DEFAULT)
    parser.add_argument('--mini',default=MINI_DEFAULT)
    parser.add_argument('--modes',nargs='+',choices=['baseline','style','narrative'],default=['baseline','style','narrative'])
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--grad-accum',type=int,default=8)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--skip-eval',action='store_true')
    args=parser.parse_args()
    if args.valid_movie==args.test_movie:
        parser.error('Validation and test movie must be different')
    df=load_csvs([args.data])
    movies=set(df.movie.unique())
    if args.valid_movie not in movies or args.test_movie not in movies:
        parser.error(f'Specified movie absent. Available: {sorted(movies)}')
    if len(movies)<3: parser.error('Need at least three distinct movies')
    out=Path(args.output);split=out/'splits';split.mkdir(parents=True,exist_ok=True)
    train=df[~df.movie.isin([args.test_movie,args.valid_movie])].copy()
    valid=df[df.movie==args.valid_movie].copy()
    test=df[df.movie==args.test_movie].copy()
    paths={}
    for name,part in [('train',train),('valid',valid),('test',test)]:
        path=split/f'{name}.csv';part.to_csv(path,index=False);paths[name]=str(path)
        print(f'{name.upper():5s}: {len(part):5d} lines from {sorted(part.movie.unique())}',flush=True)
    manifest={'split':'movie-disjoint 8/1/1','counts':{x:len(y) for x,y in [('train',train),('valid',valid),('test',test)]},
              'train_movies':sorted(train.movie.unique().tolist()), 'valid_movie':args.valid_movie,'test_movie':args.test_movie,
              'columns':['movie','speaker','source','target'],'target_language':'Hindi',
              'references':'Provided Hindi translations; not independently verified.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.dry_run:
        print('DRY RUN OK — no GPU models loaded.');return
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. Dataset splits were saved; run training on your GPU machine.')
    for mode in args.modes:
        run=out/mode
        print(f'\n=== {mode.upper()} ===',flush=True)
        tr_args=argparse.Namespace(train=[paths['train']],valid=[paths['valid']],mode=mode,output=str(run),
             model=args.model,mini=args.mini,epochs=args.epochs,batch_size=args.batch_size,
             grad_accum=args.grad_accum,lr=args.lr,max_tokens=args.max_tokens,seed=args.seed,
             allow_same_movie_validation=False)
        train_one(tr_args)
        gc.collect();torch.cuda.empty_cache()
        if not args.skip_eval:
            ev_args=argparse.Namespace(run=str(run),test=paths['test'],mode=mode,
                                       max_new_tokens=args.max_new_tokens,limit=args.limit)
            evaluate_one(ev_args)
            gc.collect();torch.cuda.empty_cache()
    print('Saved results to',out)

if __name__=='__main__':
    main()
