#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  if ! python3 -m venv .venv; then
    # Ubuntu can omit ensurepip when python3-venv is not installed. A venv
    # without pip is still isolated; the host pip can install into it.
    python3 -m venv --without-pip .venv
  fi
fi
python3 -m pip --python .venv install -r requirements/runtime.txt
./lab fetch-models
