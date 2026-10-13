import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel

MINI = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_TEXT = Path('runs/hindi_stylebridge/narrative')
DEFAULT_MANIFEST = Path('mm_manifest')
DEFAULT_FEATURES = Path('av_features_v2')

class StyleBridge(nn.Module):
    """EXACT architecture of the user's previously trained StyleBridge MLP."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024), nn.GELU(), nn.Dropout(.1),
            nn.Linear(1024, 2048), nn.GELU(),
            nn.Linear(2048, hidden_size), nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class AVEmotionAdapter(nn.Module):
    """Conditioned audio, video and predicted-emotion soft tokens.

    Gold labels are NOT an input. They supervise ``logits`` through a separate
    classifier head, while emotion probabilities produce the soft prefix.
    Every modality ablation masks the unavailable input *before* the classifier.
    """
    def __init__(self, hidden: int, n_emotions: int):
        super().__init__()
        self.n_emotions = n_emotions
        self.audio_projection = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 1024), nn.GELU(),
            nn.Linear(1024, hidden), nn.LayerNorm(hidden))
        self.video_projection = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, hidden), nn.LayerNorm(hidden))
        self.emotion_trunk = nn.Sequential(
            nn.LayerNorm(768 + 512 + 384),
            nn.Linear(768 + 512 + 384, 512), nn.GELU(), nn.Dropout(.1))
        self.emotion_classifier = nn.Linear(512, n_emotions)
        self.emotion_embeddings = nn.Embedding(n_emotions, hidden)
        self.emotion_gate = nn.Linear(384 + n_emotions, 1)
        self.audio_gate = nn.Linear(768 + 384 + n_emotions, 1)
        self.video_gate = nn.Linear(512 + 384 + n_emotions, 1)
        for gate in (self.audio_gate, self.video_gate, self.emotion_gate):
            nn.init.constant_(gate.bias, -2.0)
        nn.init.normal_(self.emotion_embeddings.weight, std=0.02)

    def forward(self, audio, video, anchor, modalities='both'):
        if modalities not in ('audio', 'video', 'both'):
            raise ValueError(modalities)
        a = audio if modalities in ('audio', 'both') else torch.zeros_like(audio)
        v = video if modalities in ('video', 'both') else torch.zeros_like(video)
        joint = torch.cat([a, v, anchor], dim=-1)
        logits = self.emotion_classifier(self.emotion_trunk(joint))
        probabilities = F.softmax(logits.float(), dim=-1)
        ag = torch.sigmoid(self.audio_gate(torch.cat([a, anchor, probabilities], -1)))
        vg = torch.sigmoid(self.video_gate(torch.cat([v, anchor, probabilities], -1)))
        eg = torch.sigmoid(self.emotion_gate(torch.cat([anchor, probabilities], -1)))
        z_audio = ag * self.audio_projection(a)
        z_video = vg * self.video_projection(v)
        z_emotion = eg * (probabilities @ self.emotion_embeddings.weight)
        prefixes = torch.stack([z_audio, z_video, z_emotion], dim=1)
        # Training uses a fixed 4-prefix sequence: style, audio, video, emotion.
        # Disabled modality prefixes remain masked even though MLP biases exist.
        return prefixes, logits, probabilities


def normalize_emotion(value):
    key = str(value or '').strip().casefold()
    if key in ('', 'nan', 'none', 'unknown', 'n/a', 'na', 'null', 'unlabeled'):
        return ''
    return key  # Do not relabel rare/emotion nuances as neutral.


def make_emotion_vocab(train_rows, min_count):
    counts = Counter(normalize_emotion(r.get('emotion_metadata', '')) for r in train_rows)
    counts.pop('', None)
    labels = sorted(label for label, n in counts.items() if n >= min_count)
    if len(labels) < 2:
        raise ValueError(f'Not enough emotion classes with >= {min_count} train rows: {counts}')
    weights = np.array([counts[x] for x in labels], dtype=np.float64)
    weights = np.sqrt(weights.sum() / (len(labels) * weights))
    weights = np.clip(weights, 0.25, 5.0)
    print('Emotion labels (TRAIN ONLY):', {x: counts[x] for x in labels}, flush=True)
    print('Class weights:', dict(zip(labels, weights.round(3).tolist())), flush=True)
    return labels, torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def attach_text_profiles(rows, text_master, mini_id, device, batch_size=64):
    """Rebuild original text-only anchors and 3 previous turns from ALL text lines.

    This follows the supplied original ``sentence_vectors``/``build_examples``:
    first normalize each sentence vector, mean last 50 earlier SAME speaker
    vectors within one film, then normalize the resulting character anchor.
    No target translations or gold emotions are used as features.
    """
    tok = AutoTokenizer.from_pretrained(mini_id)
    mini = AutoModel.from_pretrained(mini_id).to(device).eval()
    original = text_master.reset_index(drop=True).fillna('')
    texts = original.source.astype(str).tolist()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc='MiniLM history vectors'):
        batch = tok(texts[start:start+batch_size], padding=True,
                    truncation=True, max_length=256, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        hidden = mini(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        avg = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        chunks.append(F.normalize(avg.float(), dim=-1).cpu())
    del mini
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    vectors = torch.cat(chunks) if chunks else torch.empty((0,384))
    if vectors.shape[-1] != 384:
        raise RuntimeError(f'MiniLM width mismatch: {vectors.shape}')
    profiles = {}
    histories = defaultdict(list)
    preceding = defaultdict(list)
    # Text input is already in original source order, same as training loader.
    for index, r in original.iterrows():
        movie = str(r.movie).strip()
        speaker = str(r.speaker).strip() or 'Unknown'
        previous = histories[(movie, speaker)][-50:]
        anchor = F.normalize(vectors[previous].mean(0), dim=0) if previous else torch.zeros(384)
        context = preceding[movie][-3:]
        profile_context = ('\n'.join(f"{x['speaker']}: {x['source']}" for x in context)
                           or '(start of scene)')
        profiles[index] = (anchor, profile_context, speaker)
        histories[(movie, speaker)].append(index)
        preceding[movie].append({'speaker':speaker,'source':str(r.source)})
    for row in rows:
        idx = int(row['master_index'])
        if idx not in profiles:
            raise RuntimeError(f'Unmatched source index {idx}; rerun 01_prepare_manifest.py')
        src_row = original.iloc[idx]
        if str(src_row.movie).strip() != str(row['movie']).strip() or str(src_row.source).strip() != str(row['source']).strip():
            raise RuntimeError(f'Mismatched dialogue in {row["uid"]}; rerun prepare script')
        row['anchor'], row['context'], row['speaker'] = profiles[idx]
    return rows


def make_prompt(tokenizer, row):
    instruction = ('Translate the current English movie dialogue into natural Hindi. '
                   'Preserve its meaning. Return only the Hindi translation.')
    content = (f"Previous dialogue turns (context only):\n{row['context']}\n"
               f"Character: {row['speaker']}\n"
               f"Current English dialogue: {row['source']}")
    return tokenizer.apply_chat_template([
        {'role':'system', 'content':instruction},
        {'role':'user', 'content':content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


class EmotionMovieDataset(Dataset):
    def __init__(self, rows, tokenizer, feature_dir, label_index, max_tokens=512):
        self.items = []
        for row in rows:
            pp = tokenizer(make_prompt(tokenizer, row), add_special_tokens=False).input_ids
            yy = tokenizer(str(row['target']), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            if len(yy) > max_tokens // 2:
                yy = yy[:max_tokens//2-1] + [tokenizer.eos_token_id]
            pp = pp[-max(1, max_tokens-len(yy)):]
            with np.load(feature_dir / f"{row['uid']}.npz") as npz:
                audio = torch.tensor(npz['audio'].astype('float32'))
                video = torch.tensor(npz['video'].astype('float32'))
            if audio.shape != (768,) or video.shape != (512,):
                raise RuntimeError(f'Wrong cached feature shapes: {row["uid"]} {audio.shape} {video.shape}')
            label = label_index.get(normalize_emotion(row.get('emotion_metadata', '')), -100)
            self.items.append({'ids': pp + yy,
                               'labels': [-100]*len(pp) + yy,
                               'anchor': row['anchor'],
                               'audio': audio, 'video':video,
                               'emotion':label})

    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def pack(items, pad_id):
    seq = max(len(r['ids']) for r in items)
    ids = torch.full((len(items), seq), pad_id, dtype=torch.long)
    labels = torch.full((len(items), seq), -100, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(items):
        length=len(r['ids'])
        ids[i,:length] = torch.tensor(r['ids'])
        labels[i,:length] = torch.tensor(r['labels'])
        mask[i,:length] = 1
    return dict(input_ids=ids, labels=labels, attention_mask=mask,
                anchor=torch.stack([r['anchor'] for r in items]),
                audio=torch.stack([r['audio'] for r in items]),
                video=torch.stack([r['video'] for r in items]),
                emotion=torch.tensor([r['emotion'] for r in items],dtype=torch.long))


def supervised_forward(model, bridge, adapter, batch, modalities, emotion_weight, weights):
    """Loss with gradients through *frozen* LLM to train the new AV prefix.

    Gold emotion IDs are used ONLY for the classifier CE; prefixes use softmax
    predictions even at training time. Translation labels mask prompt/prefix.
    """
    embedding = model.get_input_embeddings()
    dev = embedding.weight.device
    ids = batch['input_ids'].to(dev)
    labels = batch['labels'].to(dev)
    mask = batch['attention_mask'].to(dev)
    anchor = batch['anchor'].to(dev, dtype=torch.float32)
    audio = batch['audio'].to(dev, dtype=torch.float32)
    video = batch['video'].to(dev, dtype=torch.float32)
    emotion_id = batch['emotion'].to(dev)
    with torch.no_grad():
        style = bridge(anchor).unsqueeze(1).to(embedding.weight.dtype)
    av, emo_logits, _ = adapter(audio, video, anchor, modalities)
    av = av.to(embedding.weight.dtype)
    # Order: [style] [audio] [video] [predicted-emotion] [narrative prompt]
    embeds = torch.cat([style, av, embedding(ids)], dim=1)
    prefix_mask = torch.tensor([1, int(modalities in ('audio','both')),
                                 int(modalities in ('video','both')), 1],
                                device=dev, dtype=mask.dtype)[None,:].expand(len(ids),-1)
    attention = torch.cat([prefix_mask, mask], dim=1)
    padded_labels = torch.cat([torch.full((len(ids),4),-100,device=dev,dtype=torch.long), labels],dim=1)
    mt_loss = model(inputs_embeds=embeds, attention_mask=attention,
                    labels=padded_labels, use_cache=False).loss
    labeled = emotion_id.ne(-100)
    # CE ignores unknown/rare labels instead of implicitly calling them Neutral.
    emotion_loss = (F.cross_entropy(emo_logits[labeled], emotion_id[labeled],
                                    weight=weights.to(dev)) if labeled.any()
                    else emo_logits.sum()*0.0)
    loss = mt_loss + emotion_weight * emotion_loss
    return loss, mt_loss.detach(), emotion_loss.detach(), emo_logits.detach(), emotion_id.detach()


def load_rows(manifest_csv, feature_dir, limit):
    rows = pd.read_csv(manifest_csv,keep_default_na=False).to_dict('records')
    chosen = []
    for r in rows:
        p = feature_dir/f"{r['uid']}.npz"
        if (p.is_file() and p.stat().st_size > 100 and
            str(r['source']).strip() and str(r['target']).strip()):
            chosen.append(r)
    if limit: chosen = chosen[:limit]
    print(f'{manifest_csv.name}: {len(chosen)}/{len(rows)} with cached audio+video',flush=True)
    return chosen


def load_text_model(text_ckpt, device):
    cfg_path = text_ckpt/'experiment_config.json'
    if not cfg_path.is_file() or not (text_ckpt/'stylebridge.pt').is_file() or not (text_ckpt/'lora').is_dir():
        raise SystemExit(f'Missing complete narrative StyleBridge checkpoint: {text_ckpt}')
    cfg = json.loads(cfg_path.read_text())
    if cfg.get('mode') != 'narrative':
        raise SystemExit('This multimodal extension requires your trained NARRATIVE StyleBridge checkpoint')
    tok = AutoTokenizer.from_pretrained(text_ckpt/'lora')
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.get('model','Qwen/Qwen3-8B'),
                                                torch_dtype=torch.bfloat16,
                                                device_map={'':'cuda:0'})
    model = PeftModel.from_pretrained(base, text_ckpt/'lora', is_trainable=False).eval()
    model.config.use_cache = False
    for parameter in model.parameters(): parameter.requires_grad_(False)
    bridge = StyleBridge(384, model.config.hidden_size).to(device)
    bridge.load_state_dict(torch.load(text_ckpt/'stylebridge.pt', map_location='cpu',weights_only=True))
    bridge.eval()
    for parameter in bridge.parameters():parameter.requires_grad_(False)
    return cfg, tok, model, bridge


@torch.inference_mode()
def evaluate(model, bridge, adapter, tokenizer, rows, feature_dir, out_dir,
             modalities, max_new_tokens, labels, shuffle=False, seed=42,
             baseline_csv=None, save_diagnostics=True):
    """Generate Hindi only. Save optional emotion predictions for analysis.

    Shuffling moves audio/video pairs among test rows without leaking gold labels.
    Scores use provisional Hindi references; baseline paired on original text index.
    """
    import sacrebleu
    adapter.eval(); model.eval();bridge.eval()
    dev = model.get_input_embeddings().weight.device
    embedding=model.get_input_embeddings()
    out_dir.mkdir(parents=True,exist_ok=True)
    shuffled=list(range(len(rows)))
    if shuffle:
        rng=random.Random(seed);rng.shuffle(shuffled)
        if len(shuffled)>1 and all(i==j for i,j in enumerate(shuffled)):
            shuffled=shuffled[1:]+shuffled[:1]
    data=[]
    for i,r in enumerate(tqdm(rows,desc='Held-out Hindi generation')):
        from_row=rows[shuffled[i]]
        with np.load(feature_dir/f"{from_row['uid']}.npz") as z:
            a=torch.tensor(z['audio'].astype('float32'), device=dev).unsqueeze(0)
            v=torch.tensor(z['video'].astype('float32'), device=dev).unsqueeze(0)
        anchor=r['anchor'].unsqueeze(0).to(dev,dtype=torch.float32)
        ids=tokenizer(make_prompt(tokenizer,r),add_special_tokens=False,return_tensors='pt').input_ids.to(dev)
        style=bridge(anchor).unsqueeze(1).to(embedding.weight.dtype)
        av,emo_logits,probs=adapter(a,v,anchor,modalities)
        embs=torch.cat([style,av.to(embedding.weight.dtype),embedding(ids)],dim=1)
        att=torch.cat([
            torch.tensor([[1,int(modalities in ('audio','both')),
                           int(modalities in ('video','both')),1]],device=dev,dtype=torch.long),
            torch.ones_like(ids)],dim=1)
        generated=model.generate(inputs_embeds=embs,attention_mask=att,
                                 max_new_tokens=max_new_tokens,do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        completion=generated[0]
        # HF versions differ on whether prompt ids are returned for inputs_embeds.
        if len(completion)>=len(ids[0]) and torch.equal(completion[:len(ids[0])],ids[0]):
            completion=completion[len(ids[0]):]
        prediction=tokenizer.decode(completion,skip_special_tokens=True).strip()
        entry={'uid':r['uid'],'movie':r['movie'],'text_movie_index':int(r['text_movie_index']),
               'speaker':r['speaker'],'source':r['source'],'reference_provisional':r['target'],
               'emotion_gold_unverified':r.get('emotion_metadata',''),
               'prediction':prediction}
        if save_diagnostics:
            predicted=int(emo_logits.argmax(-1).item())
            entry['emotion_predicted_diagnostic']=labels[predicted]
            entry['emotion_probability_diagnostic']=round(float(probs[0,predicted].item()),5)
        data.append(entry)
    if not data:
        raise RuntimeError('No test rows with cached audio/video features')
    output=out_dir/'predictions.csv'
    pd.DataFrame(data).to_csv(output,index=False)
    pred=[r['prediction'] for r in data]
    ref=[r['reference_provisional'] for r in data]
    metrics={'examples':len(data),'modalities':modalities,'shuffled_features':shuffle,
             'BLEU_provisional':sacrebleu.corpus_bleu(pred,[ref]).score,
             'chrFpp_provisional':sacrebleu.corpus_chrf(pred,[ref],word_order=2).score,
             'note':'Translations and emotion labels are provisional; alignments unverified.'}
    if save_diagnostics:
        gold=[normalize_emotion(r.get('emotion_gold_unverified','')) for r in data]
        pred_em=[r['emotion_predicted_diagnostic'] for r in data]
        present=[i for i,x in enumerate(gold) if x in labels]
        if present:
            try:
                from sklearn.metrics import f1_score
                metrics['emotion_macro_F1_diagnostic']=float(f1_score(
                    [gold[i] for i in present],[pred_em[i] for i in present],
                    labels=labels,average='macro',zero_division=0))
                metrics['emotion_labeled_examples']=len(present)
            except ImportError:
                pass
    if baseline_csv and baseline_csv.is_file():
        b=pd.read_csv(baseline_csv,keep_default_na=False)
        if {'movie','source','prediction','reference_provisional'}<=set(b):
            b['text_movie_index']=b.groupby('movie',sort=False).cumcount()
            pair=pd.DataFrame(data).merge(
                b[['movie','text_movie_index','source','reference_provisional','prediction']],
                on=['movie','text_movie_index'],validate='one_to_one',suffixes=('_mm','_baseline'))
            good=(pair.source_mm.astype(str).str.strip()==pair.source_baseline.astype(str).str.strip()) & (
                pair.reference_provisional_mm.astype(str).str.strip()==pair.reference_provisional_baseline.astype(str).str.strip())
            pair=pair[good]
            if len(pair)==len(data):
                metrics['matched_text_comparison']={
                    'examples':len(pair),
                    'text_BLEU':sacrebleu.corpus_bleu(pair.prediction_baseline.to_list(),
                                                      [pair.reference_provisional_mm.to_list()]).score,
                    'mm_BLEU':metrics['BLEU_provisional'],
                    'text_chrFpp':sacrebleu.corpus_chrf(pair.prediction_baseline.to_list(),
                                                        [pair.reference_provisional_mm.to_list()],word_order=2).score,
                    'mm_chrFpp':metrics['chrFpp_provisional']}
            else:
                metrics['matched_text_comparison_warning']=(
                    f'Only {len(pair)} of {len(data)} rows safely matched; no paired scores reported')
    (out_dir/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
    print(json.dumps(metrics,indent=2,ensure_ascii=False),flush=True)
    return metrics



@torch.inference_mode()
def translate_one(model, bridge, adapter, tokenizer, feature_path, source,
                  speaker, previous_turns, speaker_history, modalities,
                  max_new_tokens, mini_id, device):
    """Single-clip inference: generates ONLY Hindi, never an emotion label.

    previous_turns: prior chronological text strings, e.g. 'Alex: Stay here'.
    speaker_history: earlier dialogue utterances of the SAME character,
      strictly prior to the current source line (for causal character anchor).
    """
    with np.load(feature_path) as features:
        a=torch.tensor(features['audio'].astype('float32'),device=device).unsqueeze(0)
        v=torch.tensor(features['video'].astype('float32'),device=device).unsqueeze(0)
    mtok=AutoTokenizer.from_pretrained(mini_id)
    mini=AutoModel.from_pretrained(mini_id).to(device).eval()
    if speaker_history:
        lines=speaker_history[-50:]
        encoded=mtok(lines,padding=True,truncation=True,max_length=256,return_tensors='pt')
        encoded={k:v.to(device) for k,v in encoded.items()}
        hidden=mini(**encoded).last_hidden_state
        mask=encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        per_line=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1),dim=-1)
        anchor=F.normalize(per_line.mean(0),dim=-1).unsqueeze(0)
    else:
        anchor=torch.zeros((1,384),device=device)
    del mini
    row={'source':source,'speaker':speaker,
         'context':'\n'.join(previous_turns[-3:]) or '(start of scene)'}
    prompt=make_prompt(tokenizer,row)
    emb=model.get_input_embeddings()
    ids=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').input_ids.to(device)
    style=bridge(anchor).unsqueeze(1).to(emb.weight.dtype)
    av,_,_=adapter(a,v,anchor,modalities)
    prefix=torch.cat([style,av.to(emb.weight.dtype),emb(ids)],dim=1)
    mask=torch.cat([torch.tensor([[1,int(modalities in ('audio','both')),
                                   int(modalities in ('video','both')),1]],device=device),
                    torch.ones_like(ids)],dim=1)
    tokens=model.generate(inputs_embeds=prefix,attention_mask=mask,do_sample=False,
                          max_new_tokens=max_new_tokens,pad_token_id=tokenizer.eos_token_id)[0]
    if len(tokens)>=len(ids[0]) and torch.equal(tokens[:len(ids[0])],ids[0]):
        tokens=tokens[len(ids[0]):]
    return tokenizer.decode(tokens,skip_special_tokens=True).strip()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['train','eval','predict'],default='train')
    p.add_argument('--manifest-dir',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--feature-dir',type=Path,default=DEFAULT_FEATURES)
    p.add_argument('--text-checkpoint',type=Path,default=DEFAULT_TEXT)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--modalities',choices=['audio','video','both'],required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--emotion-weight',type=float,default=0.2)
    p.add_argument('--min-emotion-count',type=int,default=10)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=8)
    p.add_argument('--max-tokens',type=int,default=512)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--limit-train',type=int,default=0)
    p.add_argument('--limit-valid',type=int,default=0)
    p.add_argument('--limit-test',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--shuffle-test-features',action='store_true',help='Evaluation ablation; --stage eval only')
    p.add_argument('--text-predictions',type=Path,default=None)
    p.add_argument('--skip-test',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume training from last complete epoch')
    p.add_argument('--feature-file',type=Path,help='For --stage predict: cached audio/video .npz')
    p.add_argument('--source',help='For --stage predict: current English dialogue')
    p.add_argument('--speaker',default='Unknown')
    p.add_argument('--previous-turn',action='append',default=[],help='e.g. "Chris: What happened?"; can repeat')
    p.add_argument('--speaker-history',action='append',default=[],help='Prior English utterance of the current speaker; repeat')
    args=p.parse_args()
    if not torch.cuda.is_available():p.error('CUDA is required; select GPU via CUDA_VISIBLE_DEVICES=1')
    if args.stage!='eval' and args.shuffle_test_features:p.error('--shuffle-test-features only supported with --stage eval')
    if args.resume and args.stage!='train':p.error('--resume requires --stage train')
    if args.emotion_weight<0:p.error('--emotion-weight must be nonnegative')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    dev=torch.device('cuda:0')
    args.manifest_dir=args.manifest_dir.expanduser()
    args.feature_dir=args.feature_dir.expanduser()
    args.text_checkpoint=args.text_checkpoint.expanduser()
    args.output_dir=args.output_dir.expanduser()
    # Standalone inference does not require a manifest or any emotion labels.
    if args.stage=='predict':
        if not args.feature_file or not args.feature_file.is_file() or not args.source:
            p.error('--stage predict requires --feature-file EXISTING.npz and --source TEXT')
        config_file=args.output_dir/'mm_emotion_config.json'
        label_file=args.output_dir/'emotion_classes.json'
        weights=args.output_dir/'best_mm_emotion_adapter.pt'
        if not all(x.is_file() for x in (config_file,label_file,weights)):
            p.error('Missing trained adapter, config or emotion labels in --output-dir')
        run_cfg=json.loads(config_file.read_text())
        if run_cfg['modalities']!=args.modalities:
            p.error('Mode mismatch: choose the trained audio/video/both adapter')
        classes=json.loads(label_file.read_text())['labels']
        cfg,tok,model,bridge=load_text_model(args.text_checkpoint,dev)
        adapter=AVEmotionAdapter(model.config.hidden_size,len(classes)).to(dev)
        adapter.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        adapter.eval()
        output=translate_one(model,bridge,adapter,tok,args.feature_file,args.source,
                             args.speaker,args.previous_turn,args.speaker_history,
                             args.modalities,args.max_new_tokens,cfg.get('mini',MINI),dev)
        print('\n'+output,flush=True)
        return
    parts={s:load_rows(args.manifest_dir/f'{s}.csv',args.feature_dir,getattr(args,f'limit_{s}'))
           for s in ('train','valid','test')}
    if not parts['train'] or not parts['valid']:
        p.error('Need train and valid cached features: run 01_prepare_manifest.py + 02_cache_av_features.py first')
    sets={s:{r['movie'] for r in x} for s,x in parts.items()}
    for s in sets:
        for t in sets:
            if s!=t and (sets[s]&sets[t]):p.error(f'Movie overlap/leakage {s}/{t}: {sets[s]&sets[t]}')
    cfg=json.loads((args.text_checkpoint/'experiment_config.json').read_text())
    original_train=set(cfg.get('train_movies',[]))
    if original_train & (sets['valid']|sets['test']):
        p.error('Original narrative checkpoint trained on a held-out A/V film')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config_file=args.output_dir/'mm_emotion_config.json'
    ckpt_file=args.output_dir/'best_mm_emotion_adapter.pt'
    labels_file=args.output_dir/'emotion_classes.json'
    if args.stage=='train' and args.resume:
        if not labels_file.is_file() or not config_file.is_file():
            p.error('Cannot resume: saved config/emotion_classes.json missing')
        saved_cfg=json.loads(config_file.read_text())
        if saved_cfg['modalities']!=args.modalities or saved_cfg['emotion_weight']!=args.emotion_weight:
            p.error('Cannot resume with changed modality or emotion weight')
        saved_classes=json.loads(labels_file.read_text())
        classes=saved_classes['labels'];weights=torch.tensor(saved_classes['weights'],dtype=torch.float32)
    elif args.stage=='train':
        classes,weights=make_emotion_vocab(parts['train'],args.min_emotion_count)
        classes_payload={'labels':classes,'weights':weights.tolist(),
                         'min_count':args.min_emotion_count,'training_only_vocabulary':True,
                         'missing_or_rare_emotions':'ignored_in_auxiliary_loss'}
        labels_file.write_text(json.dumps(classes_payload,indent=2))
    else:
        if not ckpt_file.exists() or not config_file.exists() or not labels_file.exists():
            p.error('Missing trained multimodal checkpoint/config/emotion vocabulary')
        saved_cfg=json.loads(config_file.read_text())
        if saved_cfg['modalities']!=args.modalities:
            p.error('The requested modality differs from the trained adapter; use the correct run folder')
        saved_classes=json.loads(labels_file.read_text())
        classes=saved_classes['labels']
        weights=torch.tensor(saved_classes['weights'],dtype=torch.float32)
    label_map={x:i for i,x in enumerate(classes)}
    cfg,tok,model,bridge=load_text_model(args.text_checkpoint,dev)
    master=pd.read_csv(args.manifest_dir/'text_master_sources.csv',keep_default_na=False)
    all_rows=[r for section in ('train','valid','test') for r in parts[section]]
    attach_text_profiles(all_rows,master,cfg.get('mini',MINI),dev)
    if args.stage=='train':
        train_ds=EmotionMovieDataset(parts['train'],tok,args.feature_dir,label_map,args.max_tokens)
        valid_ds=EmotionMovieDataset(parts['valid'],tok,args.feature_dir,label_map,args.max_tokens)
        generator=torch.Generator().manual_seed(args.seed)
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,generator=generator,
                                collate_fn=lambda rows:pack(rows,tok.pad_token_id))
        valid_loader=DataLoader(valid_ds,batch_size=args.batch_size,shuffle=False,
                                collate_fn=lambda rows:pack(rows,tok.pad_token_id))
    adapter=AVEmotionAdapter(model.config.hidden_size,len(classes)).to(dev,dtype=torch.float32)
    if args.stage=='eval':
        adapter.load_state_dict(torch.load(ckpt_file,map_location='cpu',weights_only=True))
    else:
        optimizer=AdamW(adapter.parameters(),lr=args.lr,weight_decay=0.01)
        best=float('inf')
        history=[]
        start_epoch=0
        last_state=args.output_dir/'latest_training_state.pt'
        if args.resume:
            if not last_state.is_file():p.error('No latest_training_state.pt to resume')
            state=torch.load(last_state,map_location='cpu',weights_only=False)
            adapter.load_state_dict(state['adapter'])
            optimizer.load_state_dict(state['optimizer'])
            # Restoring CUDA tensors for AdamW after loading CPU optimizer state.
            for optimizer_state in optimizer.state.values():
                for k,v in optimizer_state.items():
                    if torch.is_tensor(v):optimizer_state[k]=v.to(dev)
            best=state['best_val_translation']
            history=state['history']
            start_epoch=state['epoch']
            print('Resuming after epoch:',start_epoch,flush=True)
        run_cfg={'modalities':args.modalities,'emotion_weight':args.emotion_weight,
                 'min_emotion_count':args.min_emotion_count,'text_checkpoint':str(args.text_checkpoint.resolve()),
                 'train_movies':sorted(sets['train']),'valid_movies':sorted(sets['valid']),
                 'test_movies':sorted(sets['test']),'seed':args.seed,
                 'trainable_components':'audio/video projections, emotion head/embeddings and gates',
                 'emotion_conditioning':'predicted probabilities only (never gold labels)',
                 'provisional_references':True}
        config_file.write_text(json.dumps(run_cfg,indent=2))
        for epoch in range(start_epoch,args.epochs):
            model.eval();bridge.eval();adapter.train();optimizer.zero_grad(set_to_none=True)
            total=mt_total=e_total=0.0
            for step,batch in enumerate(tqdm(train_loader,desc=f'{args.modalities} epoch {epoch+1}')):
                loss,mt,emo,_,_=supervised_forward(model,bridge,adapter,batch,
                              args.modalities,args.emotion_weight,weights)
                (loss/args.grad_accum).backward()
                total+=loss.item();mt_total+=mt.item();e_total+=emo.item()
                if (step+1)%args.grad_accum==0 or step+1==len(train_loader):
                    nn.utils.clip_grad_norm_(adapter.parameters(),1.)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            adapter.eval();v_mt=v_em=v_total=0.0
            hits=classified=0
            with torch.no_grad():
                for batch in tqdm(valid_loader,desc='Validation'):
                    loss,mt,em,logits,truth=supervised_forward(model,bridge,adapter,batch,
                               args.modalities,args.emotion_weight,weights)
                    v_total+=float(loss);v_mt+=float(mt);v_em+=float(em)
                    valid=truth.ne(-100)
                    classified+=int(valid.sum())
                    hits+=int((logits.argmax(-1)[valid]==truth[valid]).sum())
            record={'epoch':epoch+1,
                    'train_total_loss':total/len(train_loader),
                    'train_translation_loss':mt_total/len(train_loader),
                    'train_emotion_loss':e_total/len(train_loader),
                    'val_total_loss':v_total/len(valid_loader),
                    'val_translation_loss':v_mt/len(valid_loader),
                    'val_emotion_loss':v_em/len(valid_loader),
                    'val_emotion_accuracy_labeled_only':(hits/classified if classified else None),
                    'val_labeled_emotion_examples':classified}
            history.append(record);print(record,flush=True)
            (args.output_dir/'training_history.json').write_text(json.dumps(history,indent=2))
            # Best checkpoint by translation validation loss, not emotion loss;
            # translation is the sole end-user output.
            if record['val_translation_loss']<best:
                best=record['val_translation_loss']
                torch.save(adapter.state_dict(),ckpt_file)
                print('Saved best adapter:',ckpt_file,flush=True)
            torch.save({'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),
                        'best_val_translation':best,'history':history,'epoch':epoch+1},last_state)
        adapter.load_state_dict(torch.load(ckpt_file,map_location='cpu',weights_only=True))
    if not args.skip_test and parts['test']:
        dest=args.output_dir/'shuffled_test' if args.shuffle_test_features else args.output_dir
        evaluate(model,bridge,adapter,tok,parts['test'],args.feature_dir,dest,
                 args.modalities,args.max_new_tokens,classes,args.shuffle_test_features,
                 args.seed,args.text_predictions)
    else:
        print('Training/evaluation finished. No test generation was requested.',flush=True)


if __name__=='__main__':
    main()
