#!/usr/bin/env bash
# Run one optimization phase's benches and tests. Agents are launched separately.
# Usage: ./scripts/opt-phases/run-phase.sh 1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 <phase-number>"
  exit 2
fi
mkdir -p "$ROOT/scripts/opt-phases/results"
cd "$ROOT/quantus-miner"

case "$PHASE" in
  1)
    echo "=== phase 1: library parity tests ==="
    shopt -s nullglob
    tests=("$ROOT"/quantus-miner/crates/pow-core/tests/p1_*.rs)
    if [[ ${#tests[@]} -eq 0 ]]; then
      echo "no p1_*.rs tests yet"
      exit 1
    fi
    for t in "${tests[@]}"; do
      name="$(basename "$t" .rs)"
      echo "--- $name ---"
      cargo test -p pow-core --test "$name" -- --nocapture
    done
    ;;
  2|3|4|5|6)
    echo "=== phase $PHASE: crate tests ==="
    cargo test -p pow-core
    cargo test -p engine-cuda
    ;;
  7)
    echo "=== phase 7: collect result rows ==="
    out="$ROOT/scripts/opt-phases/RESULTS.md"
    {
      echo "# Measured changes"
      echo
      echo "| Item | Old | New | Ratio | Hash-rate note |"
      echo "|------|-----|-----|-------|----------------|"
      for f in "$ROOT"/scripts/opt-phases/results/*.md; do
        [[ -f "$f" ]] || continue
        grep -E '^\| ' "$f" | grep -v '^| Item' | grep -v '^|---' || true
      done
    } > "$out"
    echo "wrote $out"
    ;;
  *)
    echo "unknown phase $PHASE"
    exit 2
    ;;
esac
