#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! python3 -c 'import torch' 2>/dev/null; then
  echo "ERROR: PyTorch is required before setup-asd.sh" >&2
  exit 1
fi

if [[ ! -x .venv-asd/bin/python ]]; then
  if ! python3 -m venv --system-site-packages .venv-asd; then
    python3 -m venv --without-pip --system-site-packages .venv-asd
  fi
fi

python3 -m pip --python .venv-asd install -r requirements/asd.txt
./lab fetch-models
./lab asd-status
