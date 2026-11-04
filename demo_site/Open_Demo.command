#!/bin/zsh
set -eu
demo_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
printf 'Open http://127.0.0.1:5812/\nKeep this window open during your demo. Control-C stops the server.\n'
cd "$demo_dir"
exec /usr/bin/env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 "$demo_dir/../.venv/bin/python" server.py
