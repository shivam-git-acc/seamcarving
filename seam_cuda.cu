// seam_cuda.cu - exact content-aware image resizing (seam carving) on the GPU.
// Target: NVIDIA T4 (sm_75). Also contains a CPU implementation of the same
// algorithm, used as the baseline and as the correctness reference.
//
// Algorithm (identical on CPU and GPU, so outputs must match bit for bit):
//   energy  e(i,j) = sum over RGB of |P(i,j+1)-P(i,j-1)| + |P(i+1,j)-P(i-1,j)|, clamped borders
//   DP      M(i,j) = e(i,j) + min(M(i-1,j), M(i-1,j-1), M(i-1,j+1)); ties: up, then left, then right
//   seam    argmin of last row (ties: smallest column), then backtrack
//   after a seam is removed, only e(i, s_i-1) and e(i, s_i) can change -> O(H) energy update
//
// GPU per seam (captured once as a CUDA graph, replayed per seam):
//   DP x ceil((H-1)/K) launches: banded DP, K rows per launch, column tiles with a K-wide halo
//                                (trapezoid recompute) so blocks never sync with each other inside a band;
//                                also tracks, for every column, where its path enters the band
//     --dp 2 (default) k_dp_band_warp: one warp per tile, C columns per thread held in registers
//                      (independent chains -> ILP), neighbours via warp shuffles, no __syncthreads,
//                      no per-row validity math (invalid halo cells are computed but never reach output)
//     --dp 1           k_dp_band: one thread per column, shared-memory ping-pong + __syncthreads per row
//   k_argmin                   : warp-shuffle argmin over the last row
//   k_backtrack                : one thread hops band entries (H/K steps), then all bands trace K rows in parallel
//   k_remove                   : one block per row, in-place left shift of pixels + energy right of the seam
//   k_energy_update            : recompute the 2 changed energies per row, commit new width
// Horizontal seams: tiled shared-memory transpose, same kernels, transpose back.
//
// Build (Colab/Kaggle, T4):
//   wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
//   wget -q https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
//   nvcc -O3 -arch=native -Xcompiler -fopenmp -lgomp -diag-suppress 550 -o seam_cuda seam_cuda.cu
// Run:
//   ./seam_cuda <image> [rows_to_remove cols_to_remove] [--out output.jpg] [--dp 2] [--cpt 4] [--K 32]
//               [--bw 128 (dp 1 only)] [--reps 5] [--no-graph] [--cpu] [--verify]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------- shared helpers
__host__ __device__ inline int ad(int a, int b) { return a > b ? a - b : b - a; }

__host__ __device__ inline uint16_t pixEnergy(const uchar4* img, int P, int H, int W, int i, int j)
{
    int jl = j > 0 ? j - 1 : j, jr = j < W - 1 ? j + 1 : j;
    int iu = i > 0 ? i - 1 : i, id = i < H - 1 ? i + 1 : i;
    uchar4 a = img[(size_t)i * P + jl], b = img[(size_t)i * P + jr];
    uchar4 c = img[(size_t)iu * P + j], d = img[(size_t)id * P + j];
    return (uint16_t)(ad(a.x, b.x) + ad(a.y, b.y) + ad(a.z, b.z) +
                      ad(c.x, d.x) + ad(c.y, d.y) + ad(c.z, d.z));
}

// ---------------------------------------------------------------- GPU kernels
__global__ void k_energy_full(const uchar4* __restrict__ img, uint16_t* __restrict__ E,
                              int P, int H, const int* __restrict__ dW)
{
    int W = *dW;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < H && j < W) E[(size_t)i * P + j] = pixEnergy(img, P, H, W, i, j);
}

// One band = rows [r0, r0+K) with r0 = 1 + b*K. Block owns output columns [c0, c0+T), T = BW - 2K,
// loads [c0-K, c0+T+K) of the previous row and recomputes a shrinking trapezoid in shared memory.
template <int BW>
__global__ void __launch_bounds__(BW)
k_dp_band(const uint16_t* __restrict__ E, int P, int H, const int* __restrict__ dW, int K, int b,
          uint32_t* __restrict__ Mb, int* __restrict__ entry, int8_t* __restrict__ dir)
{
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t* sM = (uint32_t*)smem;          // [2][BW] cumulative cost, ping-pong
    int*      sO = (int*)(sM + 2 * BW);      // [2][BW] column where the path entered the band
    uint16_t* sE = (uint16_t*)(sO + 2 * BW); // [K][BW] energy tile

    const int W = *dW;
    const int T = BW - 2 * K;
    const int c0 = blockIdx.x * T;
    if (c0 >= W) return;                     // uniform per block

    const int t = threadIdx.x;
    const int gL = c0 - K;
    const int g = gL + t;
    const bool in = (g >= 0) && (g < W);
    const int r0 = 1 + b * K;
    const int rows = min(K, H - r0);

    // energy tile: loads issued in batches of 8 into registers before any shared-memory store,
    // so 8 global loads are in flight per thread instead of one load-latency per row
    int k0 = 0;
    for (; k0 + 8 <= rows; k0 += 8) {
        uint16_t v[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = in ? E[(size_t)(r0 + k0 + u) * P + g] : (uint16_t)0;
        #pragma unroll
        for (int u = 0; u < 8; ++u) sE[(k0 + u) * BW + t] = v[u];
    }
    for (; k0 < rows; ++k0)
        sE[k0 * BW + t] = in ? E[(size_t)(r0 + k0) * P + g] : (uint16_t)0;

    uint32_t pm = 0;
    if (in) pm = (b == 0) ? (uint32_t)E[g] : Mb[(size_t)(b - 1) * P + g];
    sM[t] = pm;
    sO[t] = g;
    __syncthreads();

    const bool leftReal  = gL <= 0;        // tile touches the true left border: no shrink there
    const bool rightReal = gL + BW >= W;   // tile touches the true right border
    const bool isOut = in && t >= K && t < K + T;

    int cur = 0;
    for (int k = 0; k < rows; ++k) {
        const int nxt = cur ^ 1;
        const int lo = leftReal ? 0 : k + 1;
        const int hi = rightReal ? BW - 1 : BW - 2 - k;
        if (in && t >= lo && t <= hi) {
            const uint32_t* pr = sM + cur * BW;
            uint32_t best = pr[t];
            int bd = 0;
            if (g > 0     && pr[t - 1] < best) { best = pr[t - 1]; bd = -1; }
            if (g < W - 1 && pr[t + 1] < best) { best = pr[t + 1]; bd = 1; }
            sM[nxt * BW + t] = best + sE[k * BW + t];
            sO[nxt * BW + t] = sO[cur * BW + t + bd];
            if (isOut) dir[(size_t)(r0 + k) * P + g] = (int8_t)bd;
        }
        __syncthreads();
        cur = nxt;
    }
    if (isOut) {
        Mb[(size_t)b * P + g]    = sM[cur * BW + t];
        entry[(size_t)b * P + g] = sO[cur * BW + t];
    }
}

// v2: one warp per tile of 32*C columns (incl. K-wide halos); lane l owns columns gL + l*C + [0, C).
// Every cell is recomputed every row; cells outside the shrinking valid trapezoid hold garbage, but
// (as in v1) a valid cell only ever reads valid parents and every output cell stays valid for K rows.
// Image borders are handled by hasL/hasR, tile edges by the trapezoid argument, so no per-row bounds.
template <int C>
__global__ void __launch_bounds__(32)
k_dp_band_warp(const uint16_t* __restrict__ E, int P, int H, const int* __restrict__ dW, int K, int b,
               uint32_t* __restrict__ Mb, int* __restrict__ entry, int8_t* __restrict__ dir)
{
    constexpr int TW = 32 * C;
    constexpr unsigned FULL = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];
    uint16_t* sE = (uint16_t*)smem;          // [K][TW]; each lane only touches its own C slots -> no barrier

    const int W = *dW;
    const int T = TW - 2 * K;
    const int c0 = blockIdx.x * T;
    if (c0 >= W) return;                     // uniform for the warp

    const int lane = threadIdx.x;
    const int gL = c0 - K;
    const int g0 = gL + lane * C;
    const int r0 = 1 + b * K;
    const int rows = min(K, H - r0);

    bool in[C], hasL[C], hasR[C], isOut[C];
    #pragma unroll
    for (int c = 0; c < C; ++c) {
        const int g = g0 + c, u = lane * C + c;
        in[c] = g >= 0 && g < W;
        hasL[c] = g > 0;
        hasR[c] = g < W - 1;
        isOut[c] = in[c] && u >= K && u < K + T;
    }

    // energy tile for this lane's columns (C independent loads per row, 4 rows unrolled -> 4C in flight)
    uint16_t* myE = sE + lane * C;
    #pragma unroll 4
    for (int k = 0; k < rows; ++k) {
        const long long base = (long long)(r0 + k) * P + g0;
        #pragma unroll
        for (int c = 0; c < C; ++c) myE[k * TW + c] = in[c] ? E[base + c] : (uint16_t)0;
    }

    uint32_t m[C];
    int o[C];
    #pragma unroll
    for (int c = 0; c < C; ++c) {
        const int g = g0 + c;
        m[c] = in[c] ? ((b == 0) ? (uint32_t)E[g] : Mb[(size_t)(b - 1) * P + g]) : 0u;
        o[c] = g;
    }

    long long drow = (long long)r0 * P + g0;
    for (int k = 0; k < rows; ++k, drow += P) {
        // neighbours across lane boundaries (lane 0 / lane 31 get their own value: halo garbage, never used validly)
        const uint32_t mL = __shfl_up_sync(FULL, m[C - 1], 1);
        const uint32_t mR = __shfl_down_sync(FULL, m[0], 1);
        const int oL = __shfl_up_sync(FULL, o[C - 1], 1);
        const int oR = __shfl_down_sync(FULL, o[0], 1);

        uint16_t e[C];
        const uint16_t* er = myE + k * TW;
        if constexpr (C % 4 == 0) {
            #pragma unroll
            for (int q = 0; q < C / 4; ++q) {
                const uint2 v = *(const uint2*)(er + 4 * q);
                e[4*q+0] = (uint16_t)(v.x & 0xffff); e[4*q+1] = (uint16_t)(v.x >> 16);
                e[4*q+2] = (uint16_t)(v.y & 0xffff); e[4*q+3] = (uint16_t)(v.y >> 16);
            }
        } else if constexpr (C == 2) {
            const uint32_t v = *(const uint32_t*)er;
            e[0] = (uint16_t)(v & 0xffff); e[1] = (uint16_t)(v >> 16);
        } else {
            #pragma unroll
            for (int c = 0; c < C; ++c) e[c] = er[c];
        }

        uint32_t nm[C];
        int no[C];
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            const uint32_t L  = (c == 0) ? mL : m[c - 1];
            const uint32_t R  = (c == C - 1) ? mR : m[c + 1];
            const int      oLc = (c == 0) ? oL : o[c - 1];
            const int      oRc = (c == C - 1) ? oR : o[c + 1];
            uint32_t best = m[c];
            int bo = o[c], bd = 0;
            if (hasL[c] && L < best) { best = L; bo = oLc; bd = -1; }
            if (hasR[c] && R < best) { best = R; bo = oRc; bd = 1; }
            nm[c] = best + e[c];
            no[c] = bo;
            if (isOut[c]) dir[drow + c] = (int8_t)bd;
        }
        #pragma unroll
        for (int c = 0; c < C; ++c) { m[c] = nm[c]; o[c] = no[c]; }
    }

    #pragma unroll
    for (int c = 0; c < C; ++c)
        if (isOut[c]) {
            const int g = g0 + c;
            Mb[(size_t)b * P + g]    = m[c];
            entry[(size_t)b * P + g] = o[c];
        }
}

__global__ void k_argmin(const uint32_t* __restrict__ M, const int* __restrict__ dW, int* __restrict__ dBest)
{
    __shared__ unsigned long long sw[32];
    const int W = *dW;
    unsigned long long best = ~0ull;
    for (int j = threadIdx.x; j < W; j += blockDim.x) {
        unsigned long long key = ((unsigned long long)M[j] << 32) | (unsigned)j;
        best = key < best ? key : best;
    }
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long v = __shfl_down_sync(0xffffffffu, best, o);
        best = v < best ? v : best;
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sw[w] = best;
    __syncthreads();
    if (w == 0) {
        best = lane < (int)(blockDim.x >> 5) ? sw[lane] : ~0ull;
        for (int o = 16; o > 0; o >>= 1) {
            unsigned long long v = __shfl_down_sync(0xffffffffu, best, o);
            best = v < best ? v : best;
        }
        if (lane == 0) *dBest = (int)(best & 0xffffffffull);
    }
}

__global__ void k_backtrack(const int* __restrict__ entry, const int8_t* __restrict__ dir, int P, int H,
                            int K, int nb, const int* __restrict__ dBest, int* __restrict__ seam)
{
    extern __shared__ int sEnd[];          // column of the seam at the last row of each band
    if (threadIdx.x == 0) {
        int c = *dBest;
        for (int b = nb - 1; b >= 0; --b) { sEnd[b] = c; c = entry[(size_t)b * P + c]; }
        seam[0] = c;
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const int r0 = 1 + b * K, rl = min(r0 + K, H) - 1;
        int c = sEnd[b];
        for (int r = rl; r >= r0; --r) { seam[r] = c; c += dir[(size_t)r * P + c]; }
    }
}

// in-place left shift of the row tail, chunk by chunk (read chunk -> barrier -> write chunk)
__global__ void k_remove(uchar4* __restrict__ img, uint16_t* __restrict__ E, int P,
                         const int* __restrict__ seam, const int* __restrict__ dW)
{
    const int i = blockIdx.x, W = *dW, s = seam[i];
    uchar4* r = img + (size_t)i * P;
    uint16_t* e = E + (size_t)i * P;
    for (int base = s; base < W - 1; base += blockDim.x) {
        const int j = base + threadIdx.x;
        const bool ok = j < W - 1;
        uchar4 p = make_uchar4(0, 0, 0, 0);
        uint16_t v = 0;
        if (ok) { p = r[j + 1]; v = e[j + 1]; }
        __syncthreads();
        if (ok) { r[j] = p; e[j] = v; }
    }
}

__global__ void k_energy_update(const uchar4* __restrict__ img, uint16_t* __restrict__ E, int P, int H,
                                const int* __restrict__ seam, int* __restrict__ dW)
{
    const int Wn = *dW - 1;
    for (int idx = threadIdx.x; idx < 2 * H; idx += blockDim.x) {
        const int i = idx >> 1, j = seam[i] - 1 + (idx & 1);
        if (j >= 0 && j < Wn) E[(size_t)i * P + j] = pixEnergy(img, P, H, Wn, i, j);
    }
    __syncthreads();
    if (threadIdx.x == 0) *dW = Wn;
}

// out[x][y] = in[y][x], 32x32 tiles, +1 padding against bank conflicts
__global__ void k_transpose(const uchar4* __restrict__ in, int pin, int H, int W,
                            uchar4* __restrict__ out, int pout)
{
    __shared__ uchar4 tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < W && y + k < H) tile[threadIdx.y + k][threadIdx.x] = in[(size_t)(y + k) * pin + x];
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < H && y + k < W) out[(size_t)(y + k) * pout + x] = tile[threadIdx.x][threadIdx.y + k];
}

static void transposeGPU(const uchar4* in, int pin, int H, int W, uchar4* out, int pout, cudaStream_t st)
{
    dim3 blk(32, 8), grd((W + 31) / 32, (H + 31) / 32);
    k_transpose<<<grd, blk, 0, st>>>(in, pin, H, W, out, pout);
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------- GPU driver (vertical seams)
typedef void (*DPKernel)(const uint16_t*, int, int, const int*, int, int, uint32_t*, int*, int8_t*);

struct GpuCarver {
    int H = 0, P = 0, K = 32, BW = 128, T = 0, nb = 0, nbx = 0, dpVer = 2, C = 4, dpThreads = 32;
    size_t smemDP = 0;
    bool useGraph = true;
    DPKernel dpk = nullptr;
    uchar4* img = nullptr; uint16_t* E = nullptr; int8_t* dir = nullptr;
    uint32_t* Mb = nullptr; int* entry = nullptr; int* seam = nullptr; int* dW = nullptr; int* dBest = nullptr;
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;

    void init(int H_, int Wmax, int K_, int BW_, int dpVer_, int C_, bool graph_, cudaStream_t st)
    {
        H = H_; P = Wmax; K = K_; BW = BW_; dpVer = dpVer_; C = C_; useGraph = graph_;
        nb = H > 1 ? (H - 1 + K - 1) / K : 0;
        if (dpVer == 1) {
            T = BW - 2 * K;
            dpThreads = BW;
            smemDP = (size_t)4 * BW * sizeof(uint32_t) + (size_t)K * BW * sizeof(uint16_t);
            if (BW == 128) dpk = k_dp_band<128>;
            else if (BW == 256) dpk = k_dp_band<256>;
            else dpk = k_dp_band<512>;
        } else {
            T = 32 * C - 2 * K;
            dpThreads = 32;
            smemDP = (size_t)K * 32 * C * sizeof(uint16_t);
            if (C == 2) dpk = k_dp_band_warp<2>;
            else if (C == 4) dpk = k_dp_band_warp<4>;
            else dpk = k_dp_band_warp<8>;
        }
        nbx = (Wmax + T - 1) / T;
        int dev; CK(cudaGetDevice(&dev));
        int maxOptin; CK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        if (smemDP > (size_t)maxOptin) {
            fprintf(stderr, "this K needs %zu B shared memory (max %d). Lower K.\n", smemDP, maxOptin);
            exit(1);
        }
        CK(cudaFuncSetAttribute((const void*)dpk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemDP));

        size_t HP = (size_t)H * P, BP = (size_t)max(nb, 1) * P;
        CK(cudaMalloc(&img, HP * sizeof(uchar4)));
        CK(cudaMalloc(&E, HP * sizeof(uint16_t)));
        CK(cudaMalloc(&dir, HP * sizeof(int8_t)));
        CK(cudaMalloc(&Mb, BP * sizeof(uint32_t)));
        CK(cudaMalloc(&entry, BP * sizeof(int)));
        CK(cudaMalloc(&seam, (size_t)H * sizeof(int)));
        CK(cudaMalloc(&dW, sizeof(int)));
        CK(cudaMalloc(&dBest, sizeof(int)));

        if (useGraph && nb > 0) {   // every seam launches the same sequence; width lives in device memory
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
            enqueueSeam(st);
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        }
    }

    void enqueueSeam(cudaStream_t st)
    {
        for (int b = 0; b < nb; ++b)
            dpk<<<nbx, dpThreads, smemDP, st>>>(E, P, H, dW, K, b, Mb, entry, dir);
        k_argmin<<<1, 1024, 0, st>>>(Mb + (size_t)(nb - 1) * P, dW, dBest);
        int bt = min(1024, ((nb + 31) / 32) * 32);
        k_backtrack<<<1, bt, nb * sizeof(int), st>>>(entry, dir, P, H, K, nb, dBest, seam);
        k_remove<<<H, 256, 0, st>>>(img, E, P, seam, dW);
        k_energy_update<<<1, 1024, 0, st>>>(img, E, P, H, seam, dW);
    }

    // remove n vertical seams from img (H x W, pitch P), already on the device
    void run(int W, int n, cudaStream_t st)
    {
        CK(cudaMemcpyAsync(dW, &W, sizeof(int), cudaMemcpyHostToDevice, st));
        dim3 blk(32, 8), grd((W + 31) / 32, (H + 7) / 8);
        k_energy_full<<<grd, blk, 0, st>>>(img, E, P, H, dW);
        if (n == 0) return;
        if (useGraph) for (int s = 0; s < n; ++s) CK(cudaGraphLaunch(exec, st));
        else          for (int s = 0; s < n; ++s) enqueueSeam(st);
        CK(cudaGetLastError());
    }

    void destroy()
    {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        cudaFree(img); cudaFree(E); cudaFree(dir); cudaFree(Mb); cudaFree(entry);
        cudaFree(seam); cudaFree(dW); cudaFree(dBest);
    }
};

// ---------------------------------------------------------------- CPU implementation
// Removes n vertical seams in place (pitch P). fullEnergy=true recomputes all energies every seam
// (independent check of the seam-local update).
static void cpuCarve(vector<uchar4>& img, int P, int H, int W, int n, bool fullEnergy)
{
    vector<uint16_t> E((size_t)H * P);
    vector<int8_t> dir((size_t)H * P);
    vector<uint32_t> prev(W), cur(W);
    vector<int> seam(H);
    auto energyAll = [&](int Wc) {
        #pragma omp parallel for
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < Wc; ++j) E[(size_t)i * P + j] = pixEnergy(img.data(), P, H, Wc, i, j);
    };
    energyAll(W);
    for (int s = 0; s < n; ++s) {
        for (int j = 0; j < W; ++j) prev[j] = E[j];
        for (int i = 1; i < H; ++i) {
            const uint16_t* e = &E[(size_t)i * P];
            int8_t* d = &dir[(size_t)i * P];
            for (int j = 0; j < W; ++j) {
                uint32_t best = prev[j]; int bd = 0;
                if (j > 0     && prev[j - 1] < best) { best = prev[j - 1]; bd = -1; }
                if (j < W - 1 && prev[j + 1] < best) { best = prev[j + 1]; bd = 1; }
                cur[j] = best + e[j];
                d[j] = (int8_t)bd;
            }
            swap(prev, cur);
        }
        int col = 0;
        for (int j = 1; j < W; ++j) if (prev[j] < prev[col]) col = j;
        seam[H - 1] = col;
        for (int i = H - 1; i > 0; --i) seam[i - 1] = seam[i] + dir[(size_t)i * P + seam[i]];

        #pragma omp parallel for
        for (int i = 0; i < H; ++i) {
            int c = seam[i];
            memmove(&img[(size_t)i * P + c], &img[(size_t)i * P + c + 1], (size_t)(W - 1 - c) * sizeof(uchar4));
            memmove(&E[(size_t)i * P + c], &E[(size_t)i * P + c + 1], (size_t)(W - 1 - c) * sizeof(uint16_t));
        }
        --W;
        if (fullEnergy) energyAll(W);
        else
            for (int i = 0; i < H; ++i)
                for (int j = seam[i] - 1; j <= seam[i]; ++j)
                    if (j >= 0 && j < W) E[(size_t)i * P + j] = pixEnergy(img.data(), P, H, W, i, j);
    }
}

static void transposeCPU(const vector<uchar4>& in, int pin, int H, int W, vector<uchar4>& out, int pout)
{
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) out[(size_t)j * pout + i] = in[(size_t)i * pin + j];
}

// full pipeline on the CPU: cols first, then rows (same order as the original program)
static vector<uchar4> cpuPipeline(vector<uchar4> img, int H0, int W0, int nr, int nc, bool full)
{
    cpuCarve(img, W0, H0, W0, nc, full);
    int W1 = W0 - nc, H1 = H0 - nr;
    vector<uchar4> t((size_t)W1 * H0);
    transposeCPU(img, W0, H0, W1, t, H0);
    cpuCarve(t, H0, W1, H0, nr, full);
    vector<uchar4> out((size_t)H1 * W1);
    transposeCPU(t, H0, W1, H1, out, W1);
    return out;
}

// ---------------------------------------------------------------- main
static double median(vector<double> v) { sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <image> [rows cols] [--out f] [--dp 2] [--cpt 4] [--K 32] [--bw 128] "
                        "[--reps 5] [--no-graph] [--cpu] [--verify]\n", argv[0]);
        return 1;
    }
    string inPath = argv[1], outPath = "output.jpg";
    int nr = -1, nc = -1, K = 32, BW = 128, reps = 5, dpVer = 2, C = 4;
    bool graph = true, doCpu = false, verify = false;
    vector<string> pos;
    for (int a = 2; a < argc; ++a) {
        string s = argv[a];
        if (s == "--out") outPath = argv[++a];
        else if (s == "--K") K = atoi(argv[++a]);
        else if (s == "--bw") BW = atoi(argv[++a]);
        else if (s == "--dp") dpVer = atoi(argv[++a]);
        else if (s == "--cpt") C = atoi(argv[++a]);
        else if (s == "--reps") reps = max(1, atoi(argv[++a]));
        else if (s == "--no-graph") graph = false;
        else if (s == "--cpu") doCpu = true;
        else if (s == "--verify") verify = true;
        else pos.push_back(s);
    }
    if (pos.size() >= 2) { nr = atoi(pos[0].c_str()); nc = atoi(pos[1].c_str()); }

    int W0, H0, ch;
    unsigned char* raw = stbi_load(inPath.c_str(), &W0, &H0, &ch, 3);
    if (!raw) { cout << "Could not read the image" << endl; return 1; }
    cout << "Image read succesfully, dimensions: " << H0 << "x" << W0 << endl;
    vector<uchar4> host((size_t)H0 * W0);
    for (size_t p = 0; p < host.size(); ++p) host[p] = make_uchar4(raw[3 * p], raw[3 * p + 1], raw[3 * p + 2], 0);
    stbi_image_free(raw);

    if (nr < 0) {
        cout << "Enter how much rows you want to remove? " << endl; cin >> nr;
        cout << "Enter how much cols you want to remove? " << endl; cin >> nc;
    }
    if (nr < 0 || nc < 0 || nr >= H0 || nc >= W0) { cout << "compression not possible!!" << endl; return 0; }
    if ((nc > 0 && H0 < 2) || (nr > 0 && W0 - nc < 2)) {
        cout << "need at least 2 rows/cols in the direction being carved" << endl; return 0;
    }
    if (dpVer != 1 && dpVer != 2) { cout << "--dp must be 1 or 2" << endl; return 1; }
    if (dpVer == 1) {
        if (BW != 128 && BW != 256 && BW != 512) { cout << "--bw must be 128, 256 or 512" << endl; return 1; }
        if (K < 1 || BW - 2 * K < 1) { cout << "need 1 <= K < bw/2" << endl; return 1; }
    } else {
        if (C != 2 && C != 4 && C != 8) { cout << "--cpt must be 2, 4 or 8" << endl; return 1; }
        if (K < 1 || 32 * C - 2 * K < 1) { cout << "need 1 <= K < 16*cpt" << endl; return 1; }
    }
    const int H1 = H0 - nr, W1 = W0 - nc;

    int ndev = 0;
    bool haveGpu = cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
    vector<uchar4> gpuOut;
    double gpuMs = -1;

    if (haveGpu) {
        cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
        if (dpVer == 1)
            printf("GPU: %s (sm_%d%d), DP v1 (block/shared), K=%d, BW=%d (tile T=%d), graph=%s\n", prop.name,
                   prop.major, prop.minor, K, BW, BW - 2 * K, graph ? "on" : "off");
        else
            printf("GPU: %s (sm_%d%d), DP v2 (warp/registers), K=%d, cols/thread=%d (tile T=%d), graph=%s\n",
                   prop.name, prop.major, prop.minor, K, C, 32 * C - 2 * K, graph ? "on" : "off");
        cudaStream_t st; CK(cudaStreamCreate(&st));
        GpuCarver c1, c2;
        c1.init(H0, W0, K, BW, dpVer, C, graph, st);   // vertical seams on the original image
        c2.init(W1, H0, K, BW, dpVer, C, graph, st);   // vertical seams on the transposed image = horizontal seams
        uchar4* dOut; CK(cudaMalloc(&dOut, (size_t)H1 * W1 * sizeof(uchar4)));
        cudaEvent_t e0, e1, e2, e3, e4;
        for (cudaEvent_t* e : {&e0, &e1, &e2, &e3, &e4}) CK(cudaEventCreate(e));

        vector<double> tAll, tCompute, tCols, tRows;
        for (int r = 0; r <= reps; ++r) {    // r == 0 is warm-up
            CK(cudaEventRecord(e0, st));
            CK(cudaMemcpyAsync(c1.img, host.data(), host.size() * sizeof(uchar4), cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(e1, st));
            c1.run(W0, nc, st);
            CK(cudaEventRecord(e2, st));
            transposeGPU(c1.img, W0, H0, W1, c2.img, H0, st);
            c2.run(H0, nr, st);
            transposeGPU(c2.img, H0, W1, H1, dOut, W1, st);
            CK(cudaEventRecord(e3, st));
            gpuOut.resize((size_t)H1 * W1);
            CK(cudaMemcpyAsync(gpuOut.data(), dOut, gpuOut.size() * sizeof(uchar4), cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(e4, st));
            CK(cudaEventSynchronize(e4));
            if (r == 0) continue;
            float a, b, c, d;
            CK(cudaEventElapsedTime(&a, e0, e4)); CK(cudaEventElapsedTime(&b, e1, e3));
            CK(cudaEventElapsedTime(&c, e1, e2)); CK(cudaEventElapsedTime(&d, e2, e3));
            tAll.push_back(a); tCompute.push_back(b); tCols.push_back(c); tRows.push_back(d);
        }
        double mc = median(tCompute);
        gpuMs = mc;
        printf("GPU median of %d runs: compute %.3f ms, end-to-end (incl. copies) %.3f ms\n",
               reps, mc, median(tAll));
        if (nc) printf("  col seams: %.3f ms total, %.1f us/seam\n", median(tCols), 1000.0 * median(tCols) / nc);
        if (nr) printf("  row seams: %.3f ms total (incl. transposes), %.1f us/seam\n",
                       median(tRows), 1000.0 * median(tRows) / nr);
        c1.destroy(); c2.destroy(); cudaFree(dOut); cudaStreamDestroy(st);
    } else {
        printf("No CUDA device found: running the CPU path only.\n");
    }

    vector<uchar4> cpuOut;
    if (doCpu || !haveGpu) {
        auto t0 = chrono::high_resolution_clock::now();
        cpuOut = cpuPipeline(host, H0, W0, nr, nc, false);
        double ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count();
        printf("CPU (same algorithm, seam-local energy update): %.3f ms\n", ms);
        if (haveGpu) printf("  speedup (CPU / GPU compute): %.1fx\n", ms / gpuMs);
    }

    if (verify) {
        vector<uchar4> ref = cpuPipeline(host, H0, W0, nr, nc, true);   // full energy recompute each seam
        auto cmp = [&](const vector<uchar4>& x, const char* name) {
            size_t bad = 0;
            for (size_t p = 0; p < ref.size(); ++p)
                if (x[p].x != ref[p].x || x[p].y != ref[p].y || x[p].z != ref[p].z) ++bad;
            printf("verify %s vs CPU reference: %s (%zu mismatching pixels)\n", name, bad ? "MISMATCH" : "MATCH", bad);
        };
        if (haveGpu) cmp(gpuOut, "GPU");
        if (!cpuOut.empty()) cmp(cpuOut, "CPU incremental");
    }

    const vector<uchar4>& res = haveGpu ? gpuOut : cpuOut;
    vector<unsigned char> o((size_t)H1 * W1 * 3);
    for (size_t p = 0; p < res.size(); ++p) { o[3 * p] = res[p].x; o[3 * p + 1] = res[p].y; o[3 * p + 2] = res[p].z; }
    string ext = outPath.size() > 4 ? outPath.substr(outPath.size() - 4) : "";
    bool ok = (ext == ".png") ? stbi_write_png(outPath.c_str(), W1, H1, 3, o.data(), W1 * 3)
                              : stbi_write_jpg(outPath.c_str(), W1, H1, 3, o.data(), 95);
    cout << (ok ? "Image saved successfully: " + outPath : string("Error: Could not save the image!")) << endl;
    return 0;
}
