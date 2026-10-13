import argparse
import io
import subprocess
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, CLIPModel, CLIPProcessor

AUDIO_ID = 'microsoft/wavlm-base-plus'
IMAGE_ID = 'openai/clip-vit-base-patch32'


def read_wav(path, max_sec):
    # Existing extraction script produces PCM16, mono, 16 kHz WAV.
    # ffmpeg fallback covers differing WAV formats in the first three movies.
    with wave.open(str(path), 'rb') as w:
        if w.getsampwidth() != 2 or w.getframerate() != 16000:
            raise ValueError(f'Expected PCM16 16-kHz audio, got {w.getparams()}')
        nch = w.getnchannels()
        samples = w.readframes(min(w.getnframes(), int(max_sec*16000)))
    x = np.frombuffer(samples, dtype=np.int16).reshape(-1, nch).astype(np.float32) / 32768
    x = x.mean(axis=1)  # Always produce a 1D waveform
    if x.size < 1600: raise ValueError('Audio too short')
    return x


def fallback_audio(path, max_sec):
    result = subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-t',str(max_sec),
                             '-f','s16le','-ac','1','-ar','16000','pipe:1'],
                            capture_output=True)
    if result.returncode or len(result.stdout)<3200:
        raise RuntimeError(f'Audio decode failed: {result.stderr.decode(errors="replace")[-400:]}')
    return np.frombuffer(result.stdout,dtype='<i2').astype('float32') / 32768.


def frame_from_clip(video, relative_second):
    cmd = ['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-i',str(video),
           '-ss', f'{relative_second:.3f}','-frames:v','1', '-f','image2pipe',
           '-vcodec','mjpeg','pipe:1']
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode or not r.stdout:
        raise RuntimeError(f'Frame decode failed: {r.stderr.decode(errors="replace")[-350:]}')
    return Image.open(io.BytesIO(r.stdout)).convert('RGB')


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--manifest-dir',required=True,type=Path)
    pa.add_argument('--feature-dir',required=True,type=Path)
    pa.add_argument('--device',default='cuda:0')
    pa.add_argument('--max-audio-sec',type=float,default=12)
    pa.add_argument('--frames',type=int,default=2)
    pa.add_argument('--limit',type=int,default=0)
    pa.add_argument('--limit-per-split',type=int,default=0,help='Cache first N train, valid and test clips for a quick smoke test')
    args = pa.parse_args()
    if args.frames < 1: raise SystemExit('--frames must be >=1')
    chunks=[]
    for split in ('train','valid','test'):
        piece=pd.read_csv(args.manifest_dir/f'{split}.csv',keep_default_na=False)
        if args.limit_per_split:piece=piece.head(args.limit_per_split)
        chunks.append(piece)
    manifest=pd.concat(chunks,ignore_index=True)
    if args.limit:manifest=manifest.head(args.limit)
    args.feature_dir.mkdir(parents=True,exist_ok=True)
    device = torch.device(args.device)
    print(f'Loading frozen CLIP image encoder: {IMAGE_ID}',flush=True)
    cproc=CLIPProcessor.from_pretrained(IMAGE_ID)
    clip=CLIPModel.from_pretrained(IMAGE_ID).to(device).eval()
    print(f'Loading frozen WavLM speech encoder: {AUDIO_ID}',flush=True)
    aproc=AutoFeatureExtractor.from_pretrained(AUDIO_ID)
    wavlm=WavLMModel.from_pretrained(AUDIO_ID).to(device).eval()
    done=already=failed=0
    for row in tqdm(manifest.to_dict('records'),desc='Audio+video features'):
        path=args.feature_dir/f'{row["uid"]}.npz'
        if path.exists() and path.stat().st_size>0:
            already+=1
            continue
        try:
            audio_file=Path(row['audio_path']); video_file=Path(row['video_path'])
            try: audio = read_wav(audio_file,args.max_audio_sec)
            except Exception: audio = fallback_audio(audio_file,args.max_audio_sec)
            proc=aproc(audio,sampling_rate=16000,return_tensors='pt')
            args_a = {k:v.to(device) for k,v in proc.items() if k in ('input_values','attention_mask')}
            with torch.inference_mode():
                h=wavlm(**args_a).last_hidden_state
                # Each clip is fed independently without pad; mean pooling is valid.
                a=h.mean(dim=1).float()
                a=torch.nn.functional.normalize(a,dim=-1)

            # Audio WAV duration closely tracks video segment duration.
            duration=max(len(audio)/16000,0.1)
            positions=[duration*(i+1)/(args.frames+1) for i in range(args.frames)]
            frames=[frame_from_clip(video_file,t) for t in positions]
            pixels=cproc(images=frames,return_tensors='pt')['pixel_values'].to(device)
            with torch.inference_mode():
                result=clip.get_image_features(pixel_values=pixels)
                v=result.pooler_output if hasattr(result,'pooler_output') else result
                v=v.float().mean(0,keepdim=True)
                v=torch.nn.functional.normalize(v,dim=-1)
            if a.shape[-1] != 768 or v.shape[-1] != 512:
                raise ValueError(f'Wrong dimensions: audio={a.shape}, video={v.shape}')
            tmp=path.with_suffix('.tmp.npz')
            np.savez(tmp,audio=a[0].cpu().numpy().astype('float16'),
                     video=v[0].cpu().numpy().astype('float16'))
            tmp.replace(path)
            done+=1
        except Exception as exc:
            failed+=1
            print(f'FAILED {row["movie"]} {row["uid"]}: {exc}',flush=True)
    print(f'Features complete: new={done}, reused={already}, failed={failed}, total={len(manifest)}')
    print('Only successfully cached rows will be used for adapter training/evaluation.')

if __name__=='__main__':main()
