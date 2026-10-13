from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random

import pandas as pd

BASE_PROMPT = "Translate the current English movie dialogue into natural Hindi. Preserve its meaning. Return only the Hindi translation."


def prepare_manifest(path: Path, expected_movie: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = {"uid", "movie", "source", "target", "text_movie_index"}
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    frame = frame[frame["movie"] == expected_movie].copy()
    if frame.empty:
        raise ValueError(f"No examples for movie={expected_movie!r} found in {path}")
    if frame.uid.eq("").any() or frame.uid.duplicated().any():
        raise ValueError("Manifest UID is missing or duplicated")
    if frame.source.eq("").any() or frame.target.eq("").any():
        raise ValueError("Manifest contains empty source or reference")
    frame["text_movie_index"] = pd.to_numeric(frame.text_movie_index, errors="raise").astype(int)
    if frame["text_movie_index"].duplicated().any():
        raise ValueError("Manifest contains duplicated text_movie_index")
    return frame.reset_index(drop=True)


def prompt_text(tokenizer, english: str) -> str:
    # Match make_prompt(...,mode='baseline') in stylebridge_qwen3_all_in_one.py.
    messages = [
        {"role": "system", "content": BASE_PROMPT},
        {"role": "user", "content": f"Current English dialogue: {english}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def save_predictions(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows, columns=["uid", "movie", "text_movie_index", "character", "source",
                                    "reference_provisional", "prediction"]).to_csv(temp, index=False)
    os.replace(temp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=Path('mm_manifest/test.csv'))
    ap.add_argument('--output-dir', type=Path, default=Path('runs/hindi_stylebridge/zero_shot'))
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--movie', default='Get_Out')
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--limit', type=int, default=0,
                    help='For a smoke test only; run with 0 for the complete matched test')
    ap.add_argument('--overwrite', action='store_true', help='Discard existing output and start again')
    ap.add_argument('--dry-run', action='store_true', help='Validate inputs; do not load models')
    args = ap.parse_args()
    if args.max_new_tokens < 1 or args.save_every < 1 or args.limit < 0:
        ap.error('Numeric parameters must be positive (limit may be zero)')
    df = prepare_manifest(args.manifest, args.movie)
    if args.limit:
        df = df.iloc[:args.limit].copy().reset_index(drop=True)
    out = args.output_dir / 'predictions.csv'
    print(f"Zero-shot Get Out subset: {len(df)} unique rows from {args.manifest}", flush=True)
    if args.dry_run:
        print(df[['uid', 'movie', 'text_movie_index', 'source']].head(3).to_string(index=False))
        print('DRY RUN: no GPU model loaded or predictions generated.')
        return
    if not df['movie'].eq('Get_Out').all():
        raise ValueError('Zero-shot benchmark is configured for Get_Out only')
    if not args.limit and len(df) != 1173:
        print(f'NOTICE: manifest contains {len(df)} rows instead of the previous 1,173; '
              'the matched evaluator will validate the common intersection.', flush=True)
    existing = []
    if out.exists() and not args.overwrite:
        old = pd.read_csv(out, dtype=str, keep_default_na=False)
        for k in ('uid','source','reference_provisional','prediction','text_movie_index'):
            if k not in old.columns: raise ValueError(f'Existing predictions missing {k}; use --overwrite after inspection')
        if old.uid.duplicated().any(): raise ValueError('Duplicate UID in existing predictions')
        ref = df.set_index('uid')
        if not set(old.uid).issubset(set(df.uid)):
            raise ValueError('Existing results contain different UIDs; use a separate output directory')
        for _,r in old.iterrows():
            m=ref.loc[r['uid']]
            if (r['source'] != m['source'] or r['reference_provisional'] != m['target'] or
                    int(r['text_movie_index']) != int(m['text_movie_index'])):
                raise ValueError(f'Existing zero-shot result conflicts with the manifest at {r["uid"]}')
        existing = old.to_dict(orient='records')
        print(f'Resuming: {len(existing)} completed rows already saved.', flush=True)
    done = {row['uid']:row for row in existing}
    remaining = [r for r in df.to_dict('records') if r['uid'] not in done]
    if not remaining:
        print('All selected zero-shot predictions are already present. No model load necessary.')
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm.auto import tqdm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; run on your GPU server, not on the local laptop')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}).eval()
    device = model.get_input_embeddings().weight.device
    rows = dict(done)
    for i, item in enumerate(tqdm(remaining, desc='Zero-shot Get Out generation'), 1):
        text = prompt_text(tokenizer, item['source'])
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=args.max_new_tokens, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        # For input_ids generation, returned sequence includes the prompt tokens.
        completion = generated[0, ids.shape[1]:]
        prediction = tokenizer.decode(completion, skip_special_tokens=True).strip()
        rows[item['uid']] = dict(uid=item['uid'], movie=item['movie'],
                                  text_movie_index=item['text_movie_index'],
                                  character=item.get('speaker', ''), source=item['source'],
                                  reference_provisional=item['target'], prediction=prediction)
        if i % args.save_every == 0:
            save_predictions([rows[r['uid']] for r in df.to_dict('records') if r['uid'] in rows], out)
    save_predictions([rows[r['uid']] for r in df.to_dict('records') if r['uid'] in rows], out)
    (args.output_dir / 'zero_shot_config.json').write_text(json.dumps({
        'model':args.model, 'mode':'zero-shot', 'fine_tuned':False,
        'prompts':'same as Qwen3 baseline; current English line only',
        'max_new_tokens':args.max_new_tokens, 'generation':'greedy',
        'movie':args.movie, 'samples':len(df),
        'note':'Provisional Hindi references; no training/test data used for model adaptation.'
    },indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} zero-shot predictions: {out}', flush=True)


if __name__ == '__main__':
    main()
