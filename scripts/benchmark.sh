#!/usr/bin/env bash
# Usage: scripts/benchmark.sh <image>
# Correctness check, GPU-vs-CPU timing, and the DP parameter sweep used in the README.
set -euo pipefail
IMG=${1:?usage: scripts/benchmark.sh <image>}
BIN=./seam_cuda

echo "== correctness + CPU comparison (50% each way) =="
$BIN "$IMG" 539 960 --K 48 --verify --cpu --reps 3

echo "== DP v1 (block/shared) vs v2 (warp/registers) =="
$BIN "$IMG" 539 960 --dp 1 --K 32 --bw 128 --reps 3 | grep -E "DP|compute|us/seam"
for C in 2 4 8; do
  for K in 16 32 48; do
    if [ "$K" -lt $((16 * C)) ]; then
      $BIN "$IMG" 539 960 --cpt "$C" --K "$K" --reps 3 | grep -E "DP|compute|us/seam"
    fi
  done
done

echo "== CUDA graph on/off =="
$BIN "$IMG" 539 960 --K 48 --no-graph --reps 3 | grep -E "DP|compute|us/seam"
