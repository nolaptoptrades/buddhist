#!/usr/bin/env bash
# One-shot setup for Colab / Linux GPU machines.
#   bash scripts/setup.sh
# Clones ACE-Step 1.5 into vendor/ and installs it with uv (its own venv, so it
# will not disturb Colab's preinstalled PyTorch).
set -euo pipefail
cd "$(dirname "$0")/.."

ACE_REF="${ACE_REF:-main}"          # pin a commit/tag here once you have a working version
VENDOR=vendor/ACE-Step-1.5

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

mkdir -p vendor
if [ ! -d "$VENDOR/.git" ]; then
  git clone https://github.com/ACE-Step/ACE-Step-1.5.git "$VENDOR"
fi
git -C "$VENDOR" fetch --quiet origin
git -C "$VENDOR" checkout --quiet "$ACE_REF"

(cd "$VENDOR" && uv sync)

# Pre-download weights (to $ACESTEP_CHECKPOINTS_DIR if set, e.g. on Google Drive)
(cd "$VENDOR" && uv run acestep-download) || echo "acestep-download failed or unavailable; models will download on first run"

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No NVIDIA GPU visible"
echo "ACE-Step commit: $(git -C "$VENDOR" rev-parse HEAD)"
echo "Setup done."
