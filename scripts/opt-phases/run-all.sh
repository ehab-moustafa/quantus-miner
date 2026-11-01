#!/usr/bin/env bash
# Run measurement gates in order. Stop on the first phase that has no files yet.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
for p in 1 2 3 4 5 6 7; do
  echo "######## phase $p ########"
  "$DIR/run-phase.sh" "$p"
done
