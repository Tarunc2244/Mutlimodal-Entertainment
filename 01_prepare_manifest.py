import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import pandas as pd

REQUIRED = {'movie','source_segment_id','start_time','end_time','character','english_dialogue',
            'hindi_dialogue','audio_path','video_path','status'}
EXCLUDED = {'cityofgod'}

def norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).casefold())

def clean(s):
    return str(s).strip().casefold()

def seconds(s):
    try:
        parts=[float(p) for p in str(s).replace(',','.').split(':')]
        t=0.
        for p in parts: t=t*60+p
        return t
    except (ValueError,TypeError): return float('inf')

def media_path(value, base, sub):
    value=str(value).strip()
    if not value or value.lower() in {'nan','none'}:return None
    p=Path(value).expanduser()
    for q in (p,base/p,base/sub/p.name):
        if q.is_file() and q.stat().st_size>0:return str(q.resolve())
    return None

def make_manifest(df, master, check_files=True, limit_per_movie=0):
    """Pure Pandas/FS part, separated so preprocessing is CPU-testable."""
    missing=REQUIRED-set(df)
    if missing:raise ValueError(f'combined CSV missing columns: {sorted(missing)}')
    if not {'movie','speaker','source','target'} <= set(master):
        raise ValueError('Original text CSV must have movie,speaker,source,target')

    # Chronological alignment via matching blocks avoids ambiguities when short
    # lines like "Yes" repeat and some earlier movie clips have been filtered.
    master=master.fillna('').copy().reset_index(drop=True)
    original=len(df)
    ordered=df.copy().fillna('')
    ordered['_time']=ordered['start_time'].map(seconds)
    ordered['_original_order']=range(len(ordered))
    ordered=ordered.sort_values(['movie','_time','_original_order'],kind='stable')
    movie_text_pos={}
    master_indices_by_movie=defaultdict(list)
    for i,r in master.iterrows():
        movie=norm(r.movie)
        movie_text_pos[i]=len(master_indices_by_movie[movie])
        master_indices_by_movie[movie].append(i)
    index_mapping={}
    for movie, clipped in ordered.groupby('movie',sort=False):
        whole_indices=master_indices_by_movie[norm(movie)]
        whole=master.iloc[whole_indices]
        clip_keys=[(clean(e),clean(h)) for e,h in zip(clipped.english_dialogue,clipped.hindi_dialogue)]
        whole_keys=[(clean(e),clean(h)) for e,h in zip(whole.source,whole.target)]
        matcher=SequenceMatcher(None,clip_keys,whole_keys,autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                orig_idx=int(clipped.iloc[block.a+offset]['_original_order'])
                text_idx=whole_indices[block.b+offset]
                index_mapping[orig_idx]=(text_idx,movie_text_pos[text_idx])

    kept=[]
    rejects=Counter()
    by_movie=Counter()
    for r in ordered.to_dict('records'):
        movie=str(r['movie']).strip()
        if norm(movie) in EXCLUDED:
            rejects['excluded_non_English_movie']+=1;continue
        if limit_per_movie and by_movie[movie]>=limit_per_movie: continue
        source=str(r['english_dialogue']).strip()
        target=str(r['hindi_dialogue']).strip()
        if not source or not target:
            rejects['missing_source_or_hindi']+=1;continue
        # Legacy Arrival metadata uses status='unknown' despite existing clips.
        # Only explicit failures are rejected; real media checks decide eligibility.
        if str(r['status']).strip().lower() in {'failed','error','skipped','missing'}:
            rejects['unsuccessful_extraction']+=1;continue
        if check_files:
            folder=Path(str(r['audio_path'])).parent.parent
            audio=media_path(r['audio_path'],folder,'audio')
            video=media_path(r['video_path'],folder,'clips')
            if not audio or not video:
                rejects['media_file_missing_or_zero_bytes']+=1;continue
        else:
            audio=str(r['audio_path']).strip();video=str(r['video_path']).strip()
            if not audio or not video:
                rejects['empty_media_paths']+=1;continue

        row_pos=int(r['_original_order'])
        if row_pos not in index_mapping:
            rejects['not_found_in_original_text_training_file']+=1;continue
        master_idx,movie_idx=index_mapping[row_pos]
        # Avoid leaks: Never use the master Hindi target as an input feature.
        uid=str(r.get('source_segment_id','')).strip()
        if not uid:
            uid=hashlib.sha1(f'{movie}\0{r["start_time"]}\0{source}'.encode()).hexdigest()[:24]
        kept.append({'uid':uid,'movie':movie,'speaker':str(r['character']).strip() or 'Unknown',
          'source':source,'target':target,'emotion_metadata':str(r.get('emotion','')).strip(),
          'start_time':str(r['start_time']),'end_time':str(r['end_time']),
          'audio_path':audio,'video_path':video,
          'master_index':master_idx,'text_movie_index':movie_idx,
          'alignment_verified':str(r.get('alignment_verified','')).lower()=='true',
          'reference_verified':str(r.get('reference_verified','')).lower()=='true'})
        by_movie[movie]+=1

    if not kept:raise ValueError(f'No usable rows, rejects={dict(rejects)}')
    out=pd.DataFrame(kept)
    if out.uid.duplicated().any():
        bad=out[out.uid.duplicated(False)].uid.head().tolist()
        raise ValueError(f'Duplicate source_segment_id: {bad}')
    if out.master_index.duplicated().any():
        raise ValueError('A text source entry was mapped to multiple media rows')
    return out, dict(rejects), original

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--csv',required=True,type=Path,help='Combined metadata CSV with media paths')
    pa.add_argument('--text-data',required=True,type=Path,help='Original 4-column StyleBridge text CSV')
    pa.add_argument('--output-dir',required=True,type=Path)
    pa.add_argument('--valid-movie',default='Arrival')
    pa.add_argument('--test-movie',default='Get_Out')
    pa.add_argument('--limit-per-movie',type=int,default=0)
    pa.add_argument('--no-check-files',action='store_true',help='Only for local manifest tests; NOT for GPU training')
    args=pa.parse_args()
    combined=pd.read_csv(args.csv,keep_default_na=False,low_memory=False)
    master=pd.read_csv(args.text_data,keep_default_na=False,low_memory=False)
    data,rejects,total=make_manifest(combined,master,not args.no_check_files,args.limit_per_movie)
    if norm(args.valid_movie)==norm(args.test_movie):raise SystemExit('Validation and test cannot be same movie')
    data['split']='train'
    data.loc[data.movie.map(norm)==norm(args.valid_movie),'split']='valid'
    data.loc[data.movie.map(norm)==norm(args.test_movie),'split']='test'
    if not {'train','valid','test'}<=set(data.split):
        raise SystemExit(f'All train/valid/test splits needed. Available: {data.split.value_counts().to_dict()}')
    dest=args.output_dir.expanduser();dest.mkdir(parents=True,exist_ok=True)
    # Master includes sources/characters only: no test Hindi reference leakage into features.
    master[['movie','speaker','source']].fillna('').to_csv(dest/'text_master_sources.csv',index=False)
    for part in ('train','valid','test'):
        d=data[data.split==part].sort_values(['movie','master_index'],kind='stable')
        d.to_csv(dest/f'{part}.csv',index=False)
        print(f'{part}: {len(d)} clips from {sorted(d.movie.unique())}')
    data.to_csv(dest/'all_manifest.csv',index=False)
    info={'original_metadata_rows':total,'kept':len(data),'by_movie':data.movie.value_counts().to_dict(),
          'by_split':data.split.value_counts().to_dict(),'excluded_rows':rejects,
          'master_text_rows':len(master),'alignment_verified_true':int(data.alignment_verified.sum()),
          'reference_verified_true':int(data.reference_verified.sum()),
          'warning':'Extracted media exists; no automatic validation of actual sync or Hindi translation quality.'}
    (dest/'manifest_summary.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('Wrote manifest:',dest,'\nExcluded:',rejects)

if __name__=='__main__':main()
