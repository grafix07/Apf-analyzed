#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/studio${PYTHONPATH:+:$PYTHONPATH}"
python3 "$ROOT/studio/bbr_vector_studio.py" "$@"
