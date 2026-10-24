# GPU-Accelerated Seam Carving (CUDA)

Exact content-aware image resizing on the GPU. Seam carving removes the lowest-energy connected
path of pixels, one seam at a time, so the image shrinks while important content stays intact.

This is a redesign of my CPU implementation (`reference/seam_carving_original.cpp`, C++/OpenCV,
recursive memoized DP) for GPU execution. It includes a CPU version of the same algorithm as the
baseline and correctness reference. GPU output is checked **bit for bit** against that CPU version.

## Results (Tesla T4, Google Colab, 1920×1079 photo)

| Reduction | GPU | CPU (same algorithm) | Speedup | Verified |
|---|---|---|---|---|
| 100 rows + 400 cols | 232 ms | 6.86 s | 29.5× | exact match |
| 539 rows + 960 cols (to 960×540) | 479–528 ms | 13.9–16.4 s | 26–32× | exact match |
| 900 rows + 1700 cols (to 220×179) | 669 ms | 14.2 s | 21.2× | exact match |
| 1500 cols only | 549 ms | 13.1 s | 23.9× | exact match |

*GPU time is the median over repeated runs, excluding image file I/O. The CPU baseline is
single-threaded for the DP, and Colab's CPU timings vary between sessions, so the conservative
figure is ~25×.*

## Algorithm changes vs. the original

| | Original (`reference/`) | This version |
|---|---|---|
| Energy | √(Δx² + Δy²) with wrap-around borders | Integer L1 gradient Σ\|Δx\|+\|Δy\| over RGB, clamped borders |
| Energy update per seam | Full recompute, O(HW) | Only the 2 pixels per row next to the seam, O(H) |
| DP | Recursive top-down memoization | Bottom-up, banded and tiled for the GPU |
| Memory | Arrays reallocated every seam | Allocated once, compacted in place |

**Why a seam-local update is exact.** Removing seam `s` shifts row `i` left by one at column `s_i`.
Seams move at most one column per row. With clamped borders, only the energies at `s_i − 1` and
`s_i` can change. Wrap-around borders would also change columns 0 and W−1, which is why the
borders are clamped. Integer energy makes ties deterministic, so CPU and GPU can be compared exactly.

## GPU design

Per seam (captured once as a **CUDA Graph** and replayed for every seam):

1. **Banded DP.** Rows are split into bands of K rows, one launch per band. Each tile also loads a
   K-wide halo on both sides and recomputes a shrinking trapezoid, so tiles never synchronize
   with each other inside a band. The kernel also records where each column's path entered the band.
   - `--dp 2` (default) **warp version**: one warp per tile, 2/4/8 columns per thread held in
     registers. Neighbours come from `__shfl_up_sync`/`__shfl_down_sync`, with no `__syncthreads`
     and no bounds checks per row: invalid halo cells are computed but can never reach an output column.
   - `--dp 1` **block version**: one thread per column, shared-memory ping-pong, one barrier per row.
2. **Argmin**: warp-shuffle reduction on a packed `(cost << 32) | column` key.
3. **Two-level backtrack**: one thread hops band entry points (H/K steps), then all bands trace
   their K rows in parallel, instead of H dependent reads.
4. **Seam removal**: one block per row, chunked in-place left shift of the pixels and energies.
5. **Energy update**: recomputes the 2 changed pixels per row and updates the width on the device.

Horizontal seams use a tiled shared-memory transpose (with +1 padding against bank conflicts),
the same kernels, and a transpose back. The width lives in device memory, so the per-seam graph
is identical for every seam.

## Profiling findings

- **Nsight Systems** (300 column seams, kernel time share):

  | Kernel | Share |
  |---|---|
  | Banded DP | 80.1% |
  | Seam removal | 10.8% |
  | Backtrack | 6.1% |
  | Energy update | 2.2% |
  | Argmin | 0.6% |

- **Nsight Compute** (block DP): 30 blocks for 40 SMs on a 1920-wide image, **12.5% achieved
  occupancy** (one warp per scheduler), and **6.4 cycles per issued instruction**. Every
  instruction's latency is exposed.
- **The DP is bound by latency along its rows.** Row r+1 depends on row r, so each row costs one
  dependent chain: neighbour exchange, then compare/select, then add. Moving from shared memory
  plus a barrier (v1) to shuffles with 4 independent columns per thread (v2) raised throughput but
  barely shortened that chain (~380 → ~360 µs per seam at the best K).
- **Tuning:**
  - Larger bands help up to K≈32–48 (fewer launches and dependency gaps). Past that, the halo
    recomputation cancels the gain.
  - 4 columns per thread is best. With 8, too few warps cover the 40 SMs.
  - CUDA Graphs saved only ~2%, because CPU launch overhead isn't the bottleneck.

## Build and run

Requires an NVIDIA GPU and the CUDA toolkit (tested on a T4, sm_75).

```bash
make                      # fetches stb headers, builds ./seam_cuda
make test                 # CPU emulation tests of the kernel index math (no GPU needed)

./seam_cuda photo.jpg <rows_to_remove> <cols_to_remove> [options]
./seam_cuda photo.jpg 539 960 --K 48 --verify --cpu
scripts/benchmark.sh photo.jpg
```

| Option | Default | Meaning |
|---|---|---|
| `--out f` | `output.jpg` | output path (`.png` for lossless) |
| `--dp 1\|2` | `2` | DP kernel: block/shared (1) or warp/registers (2) |
| `--cpt 2\|4\|8` | `4` | columns per thread (v2) |
| `--K n` | `32` | rows per DP band (best measured: 48 with `--cpt 4`) |
| `--bw 128\|256\|512` | `128` | threads per block (v1 only) |
| `--reps n` | `5` | timed repetitions (median reported, after 1 warm-up) |
| `--no-graph` | off | launch kernels individually instead of a CUDA Graph |
| `--cpu` | off | also time the CPU implementation and print the speedup |
| `--verify` | off | compare against a CPU run that recomputes all energies every seam |

On Colab: select a T4 runtime, upload the repo files, then `%cd` into the folder and run `!make`.

## Verification

- `--verify`: compares the GPU output pixel for pixel against a CPU run that recomputes the full
  energy map after every seam. This independently checks the seam-local update. It matched on
  every run above, including 1,700 consecutive seams.
- `tests/dp_emulation_test.cpp`: replays both DP kernels' tile, halo and shuffle-edge logic on the
  CPU over thousands of random sizes and tie-heavy energies. It checks against a plain DP, and
  also that no output ever depends on an invalid halo cell.

## Possible next steps

- Keep a register halo in each thread and exchange edges every R rows instead of every row. This
  takes the shuffle latency off most row steps.
- Lock or report the SM clock (Colab T4s are power-capped), so cycle-level numbers can be compared.
