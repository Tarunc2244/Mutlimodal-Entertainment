from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

VARIANTS = ("audio", "video", "both")
REQUIRED_MANIFEST = ("train.csv", "valid.csv", "test.csv", "text_master_sources.csv")


def realpath(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def run(cmd: list[str], env: dict[str, str], dry: bool) -> None:
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry:
        return
    # Inherit stdout/stderr, including tqdm progress bars; a failed child stops all
    # downstream experiments instead of silently moving to the next ablation.
    subprocess.run(cmd, env=env, check=True)


def last_epoch(history: Path) -> int:
    try:
        data = json.loads(history.read_text())
        return max((int(row["epoch"]) for row in data), default=0)
    except (ValueError, OSError, TypeError, KeyError):
        return 0


def is_complete(out: Path, epochs: int) -> bool:
    return (last_epoch(out / "training_history.json") >= epochs
            and (out / "best_mm_emotion_adapter.pt").is_file()
            and (out / "predictions.csv").is_file()
            and (out / "metrics.json").is_file())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-dir", default=".", help="Folder containing original training/data files")
    p.add_argument("--script-dir", default=str(Path(__file__).resolve().parent), help="Directory containing the four Python scripts")
    p.add_argument("--csv", default="/mnt/storage/paritosh/datasetcp/combined_metadata.csv")
    p.add_argument("--text-data", default="stylebridge_english_hindi.csv")
    p.add_argument("--manifest-dir", default="mm_manifest")
    p.add_argument("--feature-dir", default="/mnt/storage/paritosh/av_features_v2")
    p.add_argument("--text-checkpoint", default="runs/hindi_stylebridge/narrative")
    p.add_argument("--output-base", default="runs/hindi_stylebridge")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--emotion-weight", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--gpu", default=None, help="GPU index to expose, e.g. 1; if absent use CUDA_VISIBLE_DEVICES or 1")
    p.add_argument("--setup", action="store_true", help="Create/refesh manifest and cache WavLM+CLIP features first")
    p.add_argument("--smoke", action="store_true", help="Run 1 epoch on 16/8/4 examples into separate *_smoke folders")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without executing")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        p.error("epochs, batch-size and grad-accum must be positive")
    if args.emotion_weight < 0:
        p.error("emotion-weight must be nonnegative")
    root = Path(args.project_dir).expanduser().resolve()
    scripts = Path(args.script_dir).expanduser().resolve()
    csv_path = realpath(args.csv, root)
    master = realpath(args.text_data, root)
    manifest = realpath(args.manifest_dir, root)
    features = realpath(args.feature_dir, root)
    narrative = realpath(args.text_checkpoint, root)
    outputs = realpath(args.output_base, root)
    trainer = scripts / "stylebridge_qwen3_mm_emotion.py"
    prepare = scripts / "01_prepare_manifest.py"
    cache = scripts / "02_cache_av_features.py"
    for path in (trainer, prepare, cache):
        if not path.is_file():
            p.error(f"Script not found: {path}. Extract/copy all files into one folder.")
    if not args.dry_run:
        if not (narrative / "stylebridge.pt").is_file() or not (narrative / "lora").is_dir():
            p.error(f"Trained narrative StyleBridge checkpoint missing under: {narrative}")
        if not master.is_file():
            p.error(f"Original text CSV missing: {master}")
        if args.setup and not csv_path.is_file():
            p.error(f"Combined metadata CSV missing: {csv_path}")

    gpu = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    print("""
===============================================
  Emotion-Supervised Multimodal StyleBridge
  Sequence: AUDIO -> VIDEO -> AUDIO + VIDEO
===============================================""")
    print(f"Project: {root}\nGPU: {gpu}\nNarrative checkpoint: {narrative}\n"
          f"Manifest: {manifest}\nFeatures: {features}\nOutputs: {outputs}", flush=True)

    if args.setup:
        if args.smoke:
            print("SMOKE: Preparing full split metadata, caching 16/8/4 samples per split (cache stage caps at 16/split).")
        run([sys.executable, str(prepare), "--csv", str(csv_path),
             "--text-data", str(master), "--output-dir", str(manifest)], env, args.dry_run)
        cache_cmd = [sys.executable, str(cache), "--manifest-dir", str(manifest),
                     "--feature-dir", str(features), "--frames", str(args.frames)]
        if args.smoke:
            cache_cmd.extend(["--limit-per-split", "16"])
        run(cache_cmd, env, args.dry_run)
    elif not args.dry_run:
        missing = [name for name in REQUIRED_MANIFEST if not (manifest / name).is_file()]
        if missing:
            p.error(f"Manifest missing {missing}. Run with --setup to prepare and cache features.")
        if not features.is_dir() or not any(features.glob("*.npz")):
            p.error(f"No cached AV features in {features}. Run with --setup first.")

    for index, mode in enumerate(VARIANTS, 1):
        suffix = f"mm_emotion_{mode}{'_smoke' if args.smoke else ''}"
        out = outputs / suffix
        print(f"\n========== EXPERIMENT {index}/3: {mode.upper()} ==========", flush=True)
        if not args.dry_run and is_complete(out, 1 if args.smoke else args.epochs):
            print(f"ALREADY COMPLETE; skipping: {out}", flush=True)
            continue
        cmd = [sys.executable, str(trainer),
               "--stage", "train", "--modalities", mode,
               "--manifest-dir", str(manifest),
               "--feature-dir", str(features),
               "--text-checkpoint", str(narrative),
               "--output-dir", str(out),
               "--epochs", "1" if args.smoke else str(args.epochs),
               "--batch-size", str(args.batch_size),
               "--grad-accum", str(args.grad_accum),
               "--lr", str(args.lr),
               "--emotion-weight", str(args.emotion_weight)]
        text_preds = narrative / "predictions.csv"
        if text_preds.is_file():
            cmd.extend(["--text-predictions", str(text_preds)])
        if args.smoke:
            cmd.extend(["--limit-train", "16", "--limit-valid", "8", "--limit-test", "4"])
        resume_file = out / "latest_training_state.pt"
        if resume_file.is_file():
            cmd.append("--resume")
            print(f"RESUME from last completed epoch: {out}", flush=True)
        elif (out / "mm_emotion_config.json").exists():
            p.error(f"Partial output at {out}, but no optimizer state to resume. "
                    "Back up/rename this folder before starting a fresh run.")
        print(f"Start: {datetime.now(timezone.utc).isoformat()}", flush=True)
        try:
            run(cmd, env, args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"\nFAILED on {mode} (exit={exc.returncode}). Stopped; subsequent experiments NOT launched."
                  f"\nAfter resolving the error, rerun this launcher to resume.", file=sys.stderr, flush=True)
            raise SystemExit(exc.returncode)
        print(f"{'PLANNED' if args.dry_run else 'FINISHED'} {mode}: {out}", flush=True)
    print("\nALL THREE VARIANTS PLANNED (dry run)." if args.dry_run
          else "\nALL THREE VARIANTS FINISHED (or were already complete).", flush=True)
    for mode in VARIANTS:
        out = outputs / f"mm_emotion_{mode}{'_smoke' if args.smoke else ''}"
        print(f"{mode:>5s}: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
