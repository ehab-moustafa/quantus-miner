#!/usr/bin/env bash
# Local gate for default mining_main vs pow_core samples and MINER_CUDA_NONCE_LINE stubs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MINER_CUDA_NONCE_LINE=1

echo "=== engine-cuda: nonce-line env + mining_main / pow_core tests ==="
cargo test -p engine-cuda -- --nocapture \
  miner_cuda_nonce_line \
  k_segment_plan_nonce_line \
  cuda_default_mining_main \
  cuda_mining_main_search_multisegment \
  cuda_nonce_line \
  cuda_matches_nonce_hash_golden \
  cuda_hash_batches_match_cpu

echo
echo "OK: verify-nonce-line-local (GPU tests skip gracefully when CUDA is absent)"
