#!/usr/bin/env python3
"""Batch-generate music with ACE-Step 1.5 from a YAML job list.

Run through the ACE-Step environment, e.g.:
    uv run --project vendor/ACE-Step-1.5 python scripts/generate.py \
        --config config/prompts.yaml --out out

Every finished take gets a .json sidecar (model, commit, prompt, seed, date) so
you can later prove how a track was made.  Takes whose sidecar already exists
are skipped, so the script is safe to re-run after a Colab disconnect.

--dry-run lists the planned work without importing ACE-Step (no GPU needed).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_jobs(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {}) or {}
    jobs = []
    for raw in cfg.get("jobs", []) or []:
        job = {**defaults, **raw}
        if not job.get("enabled", True):
            print(f"[skip] {job.get('name')}: enabled=false")
            continue
        lyrics = job.get("lyrics", "") or ""
        if job.get("lyrics_file"):
            lf = Path(job["lyrics_file"])
            if not lf.is_absolute():
                lf = ROOT / lf
            lyrics = "\n".join(
                ln for ln in lf.read_text(encoding="utf-8").splitlines()
                if not ln.lstrip().startswith("#")
            ).strip()
        if job.get("instrumental"):
            lyrics = "[Instrumental]"
        elif not lyrics:
            print(f"[skip] {job.get('name')}: no lyrics (fill lyrics/lyrics_file or set instrumental: true)")
            continue
        job["_lyrics"] = lyrics
        jobs.append(job)
    return jobs


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def plan(jobs, out_dir: Path):
    todo = []
    for job in jobs:
        for i in range(int(job["takes"])):
            seed = int(job["seed"]) + i
            stem = f"{job['name']}_s{seed}"
            sidecar = out_dir / job["name"] / f"{stem}.json"
            todo.append((job, seed, stem, sidecar, sidecar.exists()))
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "prompts.yaml"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--ace-root", default=str(ROOT / "vendor" / "ACE-Step-1.5"))
    ap.add_argument("--only", nargs="*", help="run only these job names")
    ap.add_argument("--lm-model", default=None,
                    help="e.g. acestep-5Hz-lm-0.6B. Default: no LM (DiT-only, safest on low VRAM)")
    ap.add_argument("--lm-backend", default="pt", choices=["pt", "vllm"])
    ap.add_argument("--offload", choices=["auto", "on", "off"], default="auto",
                    help="CPU offload of weights. auto = on when VRAM < 20GB")
    ap.add_argument("--int8", choices=["auto", "on", "off"], default="auto",
                    help="INT8 weight-only quantization. auto = on when VRAM < 8GB")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    jobs = load_jobs(Path(args.config))
    if args.only:
        jobs = [j for j in jobs if j["name"] in args.only]
    todo = plan(jobs, out_dir)

    print(f"{len(todo)} takes planned, {sum(1 for t in todo if t[4])} already done")
    if args.dry_run:
        for job, seed, stem, _, done in todo:
            print(f"  {'done' if done else 'todo'}  {stem}  ({job['duration']}s, lang={job['vocal_language']})")
        return
    todo = [t for t in todo if not t[4]]
    if not todo:
        print("Nothing to do.")
        return

    ace_root = Path(args.ace_root).resolve()
    if not ace_root.exists():
        sys.exit(f"ACE-Step not found at {ace_root}. Run scripts/setup.sh first.")
    sys.path.insert(0, str(ace_root))

    import torch  # noqa: E402
    from acestep.handler import AceStepHandler  # noqa: E402
    from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_gb = 0.0
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}  {vram_gb:.1f} GB")
    else:
        print("WARNING: no CUDA GPU, running on CPU (very slow)")

    offload = args.offload == "on" or (args.offload == "auto" and device == "cuda" and vram_gb < 20)
    int8 = args.int8 == "on" or (args.int8 == "auto" and 0 < vram_gb < 8)

    models = sorted({j["model"] for j in todo_jobs(todo)})
    commit = git_commit(ace_root)

    llm = None
    if args.lm_model:
        from acestep.llm_inference import LLMHandler  # noqa: E402
        llm = LLMHandler()
        llm.initialize(
            checkpoint_dir=os.environ.get("ACESTEP_CHECKPOINTS_DIR", str(ace_root / "checkpoints")),
            lm_model_path=args.lm_model,
            backend=args.lm_backend,
            device=device,
        )

    handlers = {}
    for model in models:
        h = AceStepHandler()
        kw = dict(project_root=str(ace_root), config_path=model, device=device)
        # offload / quantization options are documented for initialize_service;
        # fall back gracefully if this ACE-Step version names them differently.
        try:
            h.initialize_service(**kw, offload_to_cpu=offload,
                                 quantization="int8_weight_only" if int8 else None)
        except TypeError:
            print("note: initialize_service() rejected offload/quantization kwargs; using defaults")
            h.initialize_service(**kw)
        handlers[model] = h

    for n, (job, seed, stem, sidecar, _) in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {stem}")
        job_dir = out_dir / job["name"]
        job_dir.mkdir(parents=True, exist_ok=True)

        params = GenerationParams(
            task_type="text2music",
            caption=" ".join(str(job["caption"]).split()),
            lyrics=job["_lyrics"],
            instrumental=bool(job.get("instrumental", False)),
            vocal_language=job["vocal_language"],
            bpm=job.get("bpm"),
            keyscale=job.get("keyscale", ""),
            duration=float(job["duration"]),
            inference_steps=int(job["steps"]),
            shift=float(job["shift"]),
            seed=seed,
            thinking=llm is not None,
        )
        config = GenerationConfig(
            batch_size=1, use_random_seed=False, seeds=[seed], audio_format=job["format"]
        )
        t0 = time.time()
        result = generate_music(handlers[job["model"]], llm, params, config, save_dir=str(job_dir))
        if not result.success or not result.audios:
            print(f"  FAILED: {result.error}")
            continue

        src = Path(result.audios[0]["path"])
        dst = job_dir / f"{stem}{src.suffix}"
        if src.resolve() != dst.resolve():
            src.replace(dst)
        meta = {
            "file": dst.name,
            "job": job["name"],
            "seed": seed,
            "caption": params.caption,
            "lyrics": job["_lyrics"],
            "vocal_language": job["vocal_language"],
            "duration_requested": job["duration"],
            "model": job["model"],
            "lm_model": args.lm_model,
            "steps": job["steps"],
            "shift": job["shift"],
            "acestep_commit": commit,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "seconds_to_generate": round(time.time() - t0, 1),
            "license_note": "ACE-Step 1.5 code/weights: MIT (verify at time of use)",
        }
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ok -> {dst}  ({meta['seconds_to_generate']}s)")


def todo_jobs(todo):
    return [t[0] for t in todo]


if __name__ == "__main__":
    main()
