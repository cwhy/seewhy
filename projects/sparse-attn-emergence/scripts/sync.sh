#!/usr/bin/env bash
# Safe sync with the GPU box. Run from the repo root.
#
#   scripts/sync.sh push    code up   (never results.jsonl)
#   scripts/sync.sh pull    results down
#
# results.jsonl is appended to ON THE BOX. Pushing a stale local copy silently
# overwrites rows a running experiment has written — that is how 10 exp2 cells were
# lost. Hence: pushes exclude it, and it only ever travels remote -> local.
set -euo pipefail

HOST=195.133.135.186
REMOTE="Projects/seewhy/projects/sparse-attn-emergence/"
LOCAL="projects/sparse-attn-emergence/"

[[ -d "$LOCAL" ]] || { echo "run from the repo root (no $LOCAL here)" >&2; exit 2; }

case "${1:-}" in
  push)
    rsync -az --exclude 'logs/' --exclude '__pycache__/' --exclude '*.pkl' \
              --exclude 'results.jsonl' "$LOCAL" "$HOST:$REMOTE"
    echo "pushed code (results.jsonl excluded)"
    ;;
  pull)
    rsync -az "$HOST:${REMOTE}results.jsonl" "$LOCAL"
    echo "pulled results.jsonl ($(wc -l < "${LOCAL}results.jsonl") rows)"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
