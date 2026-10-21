# Buddhist meditation music generator

Local-weights, licence-friendly pipeline for a meditation / chanting YouTube channel:

1. **Generate** raw tracks with ACE-Step 1.5 (MIT) on a free Colab T4. Notebook: `notebooks/colab_generate.ipynb`.
2. **Pick** the best takes (listen!).
3. **Loop** them into a 1–3 hour video on your own PC with `scripts/make_loop.py`. Only ffmpeg is needed, no GPU.

```
config/prompts.yaml     what to generate (edit this)
texts/da_bei_zhou.txt   paste your chant text here
scripts/setup.sh        clones + installs ACE-Step (Colab/Linux GPU)
scripts/generate.py     batch generator, writes a .json provenance file per take
scripts/make_loop.py    crossfade + seamless loop + loudness + video
notebooks/              Colab notebook
```

## Quick start

1. Put this folder on GitHub; edit `REPO_URL` in the notebook's first cell.
2. Open the notebook in Colab, choose a T4 GPU, run the cells in order. Weights and outputs live on Google Drive, so a disconnect loses nothing.
3. Listen to `out/<job>/*.flac` and keep what sounds right.

Build the long video (PC, CPU only):

```bash
# bed + looping footage, 3 hours
python scripts/make_loop.py --music out/bowls_drone/*.flac --duration 3h \
    --bg footage/forest.mp4 --out final/bowls_3h.mp4

# bed + chant layer over a still image
python scripts/make_loop.py --music out/deep_focus_pad/*_s1000.flac \
    --voice out/om_vocal_drone/*_s1000.flac --voice-db -9 \
    --duration 2h --bg footage/lake.jpg --out final/om_2h.mp4
```

Free footage: Pexels / Pixabay (check each clip's licence; avoid recognisable people and logos).

## Your own GTX 960 (4 GB)

Not recommended for generation: it is Maxwell, which current PyTorch builds no longer support. PyTorch 2.14 with CUDA 12.6 is the last that does. Use Colab for generating, and the PC for `make_loop.py`.

## Chanting: what to expect

- Instrumental beds are the strongest use. The model leans toward vocals, so listen to confirm "instrumental" really is.
- Chinese/Sanskrit pronunciation can be wrong. ACE-Step takes lyrics plus `vocal_language`; it does not document IPA input. If a line is wrong, try pinyin with tone numbers for that line, or re-roll the seed.
- Check every chant take with someone who knows the text before publishing.
- Do not use reference audio / cover mode on other people's recordings, and do not clone a real singer's voice without their consent.

## Licence & provenance

- ACE-Step 1.5 is MIT (model card says trained on licensed, royalty-free and synthetic data). Re-read `vendor/ACE-Step-1.5/LICENSE` and the Hugging Face model card when you start and keep a copy.
- `generate.py` writes a `.json` next to each take (model, git commit, prompt, seed, UTC time). Keep them with your uploads, in case of a Content ID dispute.
- Check YouTube's current policy on AI-generated and repetitive content before monetising.
- This is not legal advice.

## What was tested

- `make_loop.py`: run for real (video, still image and audio-only modes; the loop point has no audible/measurable click).
- `generate.py`: job parsing, resume-skip and provenance files tested against a stub ACE-Step.
- Not tested: the real ACE-Step install and generation (no GPU / model download in the authoring environment).

## Known unknowns

- `initialize_service(offload_to_cpu=..., quantization=...)` follows ACE-Step's INFERENCE.md; if your version renames them, `generate.py` falls back to defaults.
- Where weights land follows `ACESTEP_CHECKPOINTS_DIR` per INSTALL.md; confirm on first run.
- Instrumental quality and Chinese chant accuracy need your ears.
