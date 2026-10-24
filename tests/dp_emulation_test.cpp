// CPU emulation of the GPU kernels' index math, compared against a plain bottom-up DP:
//   - k_dp_band (v1, block/shared memory) + k_argmin + two-level k_backtrack
//   - k_dp_band_warp (v2, warp/registers), with exact __shfl_up/__shfl_down edge semantics
//   - k_remove's chunked in-place row shift
// Thread loops stand in for parallel execution; a barrier = finishing all threads of a step.
// Random widths/heights/K/tile sizes, including tie-heavy energies.
// Build & run:  g++ -O2 -std=c++17 -o dp_emulation_test dp_emulation_test.cpp && ./dp_emulation_test
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

static vector<int> plainSeam(const vector<uint16_t>& E, int P, int H, int W) {
    vector<uint32_t> prev(W), cur(W); vector<int8_t> dir((size_t)H * P);
    for (int j = 0; j < W; ++j) prev[j] = E[j];
    for (int i = 1; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            uint32_t best = prev[j]; int bd = 0;
            if (j > 0 && prev[j-1] < best) { best = prev[j-1]; bd = -1; }
            if (j < W-1 && prev[j+1] < best) { best = prev[j+1]; bd = 1; }
            cur[j] = best + E[(size_t)i*P+j]; dir[(size_t)i*P+j] = bd;
        }
        swap(prev, cur);
    }
    int col = 0; for (int j = 1; j < W; ++j) if (prev[j] < prev[col]) col = j;
    vector<int> s(H); s[H-1] = col;
    for (int i = H-1; i > 0; --i) s[i-1] = s[i] + dir[(size_t)i*P+s[i]];
    return s;
}

static vector<int> bandSeam(const vector<uint16_t>& E, int P, int H, int W, int K, int BW) {
    int T = BW - 2*K, nb = (H - 1 + K - 1) / K, nbx = (P + T - 1) / T;   // grid sized for max width P
    vector<uint32_t> Mb((size_t)nb * P, 0xDEADBEEF); vector<int> entry((size_t)nb * P, -999999);
    vector<int8_t> dir((size_t)H * P, 99);
    for (int b = 0; b < nb; ++b)
        for (int bx = 0; bx < nbx; ++bx) {
            int c0 = bx * T; if (c0 >= W) continue;
            vector<uint32_t> sM(2*BW, 0xABCDEF); vector<int> sO(2*BW, -777); vector<uint16_t> sE((size_t)K*BW);
            int gL = c0 - K, r0 = 1 + b*K, rows = min(K, H - r0);
            for (int t = 0; t < BW; ++t) {
                int g = gL + t; bool in = g >= 0 && g < W;
                for (int k = 0; k < rows; ++k) sE[k*BW+t] = in ? E[(size_t)(r0+k)*P+g] : 0;
                uint32_t pm = 0; if (in) pm = b == 0 ? E[g] : Mb[(size_t)(b-1)*P+g];
                sM[t] = pm; sO[t] = g;
            }
            bool leftReal = gL <= 0, rightReal = gL + BW >= W;
            int cur = 0;
            for (int k = 0; k < rows; ++k) {
                int nxt = cur ^ 1, lo = leftReal ? 0 : k+1, hi = rightReal ? BW-1 : BW-2-k;
                for (int t = 0; t < BW; ++t) {
                    int g = gL + t; bool in = g >= 0 && g < W;
                    bool isOut = in && t >= K && t < K+T;
                    if (in && t >= lo && t <= hi) {
                        const uint32_t* pr = &sM[cur*BW];
                        if (t-1 < 0 && g > 0) { printf("OOB left\n"); exit(1); }
                        if (t+1 >= BW && g < W-1) { printf("OOB right\n"); exit(1); }
                        uint32_t best = pr[t]; int bd = 0;
                        if (g > 0 && pr[t-1] < best) { best = pr[t-1]; bd = -1; }
                        if (g < W-1 && pr[t+1] < best) { best = pr[t+1]; bd = 1; }
                        sM[nxt*BW+t] = best + sE[k*BW+t];
                        sO[nxt*BW+t] = sO[cur*BW+t+bd];
                        if (isOut) dir[(size_t)(r0+k)*P+g] = bd;
                    }
                }
                cur = nxt;
            }
            for (int t = 0; t < BW; ++t) {
                int g = gL + t; bool in = g >= 0 && g < W;
                if (in && t >= K && t < K+T) { Mb[(size_t)b*P+g] = sM[cur*BW+t]; entry[(size_t)b*P+g] = sO[cur*BW+t]; }
            }
        }
    // argmin
    unsigned long long best = ~0ull;
    for (int j = 0; j < W; ++j) best = min(best, ((unsigned long long)Mb[(size_t)(nb-1)*P+j] << 32) | (unsigned)j);
    // backtrack
    vector<int> s(H), sEnd(nb);
    int c = (int)(best & 0xffffffffull);
    for (int b = nb-1; b >= 0; --b) { sEnd[b] = c; c = entry[(size_t)b*P+c]; }
    s[0] = c;
    for (int b = 0; b < nb; ++b) {
        int r0 = 1 + b*K, rl = min(r0+K, H) - 1, cc = sEnd[b];
        for (int r = rl; r >= r0; --r) { s[r] = cc; cc += dir[(size_t)r*P+cc]; }
        if (cc != (b ? sEnd[b-1] : s[0])) { printf("band chain mismatch\n"); exit(1); }
    }
    return s;
}


// v2 emulation: one warp, lane l owns u = l*C + c. Registers m/o for all lanes are a TW array,
// updated from a snapshot each row (lockstep). __shfl_up at lane 0 / __shfl_down at lane 31 return own value.
static vector<int> bandSeamV2(const vector<uint16_t>& E, int P, int H, int W, int K, int C) {
    int TW = 32 * C, T = TW - 2*K, nb = (H - 1 + K - 1) / K, nbx = (P + T - 1) / T;
    vector<uint32_t> Mb((size_t)nb * P, 0xDEADBEEF); vector<int> entry((size_t)nb * P, -999999);
    vector<int8_t> dir((size_t)H * P, 99);
    for (int b = 0; b < nb; ++b)
        for (int bx = 0; bx < nbx; ++bx) {
            int c0 = bx * T; if (c0 >= W) continue;
            int gL = c0 - K, r0 = 1 + b*K, rows = min(K, H - r0);
            vector<uint32_t> m(TW); vector<int> o(TW);
            auto inImg = [&](int g) { return g >= 0 && g < W; };
            for (int u = 0; u < TW; ++u) {
                int g = gL + u;
                m[u] = inImg(g) ? (b == 0 ? E[g] : Mb[(size_t)(b-1)*P+g]) : 0u; o[u] = g;
            }
            for (int k = 0; k < rows; ++k) {
                vector<uint32_t> nm(TW); vector<int> no(TW);
                for (int lane = 0; lane < 32; ++lane) {
                    uint32_t mL = lane > 0 ? m[(lane-1)*C + C-1] : m[lane*C + C-1];
                    uint32_t mR = lane < 31 ? m[(lane+1)*C] : m[lane*C];
                    int oL = lane > 0 ? o[(lane-1)*C + C-1] : o[lane*C + C-1];
                    int oR = lane < 31 ? o[(lane+1)*C] : o[lane*C];
                    for (int c = 0; c < C; ++c) {
                        int u = lane*C + c, g = gL + u;
                        bool in = inImg(g), hasL = g > 0, hasR = g < W-1, isOut = in && u >= K && u < K+T;
                        uint16_t e = in ? E[(size_t)(r0+k)*P+g] : 0;
                        uint32_t L = c == 0 ? mL : m[u-1], R = c == C-1 ? mR : m[u+1];
                        int oLc = c == 0 ? oL : o[u-1], oRc = c == C-1 ? oR : o[u+1];
                        uint32_t best = m[u]; int bo = o[u], bd = 0;
                        if (hasL && L < best) { best = L; bo = oLc; bd = -1; }
                        if (hasR && R < best) { best = R; bo = oRc; bd = 1; }
                        nm[u] = best + e; no[u] = bo;
                        if (isOut) dir[(size_t)(r0+k)*P+g] = bd;
                    }
                }
                m = nm; o = no;
            }
            for (int u = 0; u < TW; ++u) {
                int g = gL + u;
                if (inImg(g) && u >= K && u < K+T) { Mb[(size_t)b*P+g] = m[u]; entry[(size_t)b*P+g] = o[u]; }
            }
        }
    unsigned long long best = ~0ull;
    for (int j = 0; j < W; ++j) best = min(best, ((unsigned long long)Mb[(size_t)(nb-1)*P+j] << 32) | (unsigned)j);
    vector<int> s(H), sEnd(nb);
    int c = (int)(best & 0xffffffffull);
    for (int b = nb-1; b >= 0; --b) { sEnd[b] = c; if (entry[(size_t)b*P+c] < 0) { printf("bad entry\n"); exit(1);} c = entry[(size_t)b*P+c]; }
    s[0] = c;
    for (int b = 0; b < nb; ++b) {
        int r0 = 1 + b*K, rl = min(r0+K, H) - 1, cc = sEnd[b];
        for (int r = rl; r >= r0; --r) { s[r] = cc; if (dir[(size_t)r*P+cc] == 99) { printf("unwritten dir\n"); exit(1);} cc += dir[(size_t)r*P+cc]; }
    }
    return s;
}

static bool removeEmu(int W, int s, int B, mt19937& rng) {    // chunked in-place shift
    vector<int> row(W); for (int j = 0; j < W; ++j) row[j] = rng();
    vector<int> want = row; want.erase(want.begin() + s);
    for (int base = s; base < W-1; base += B) {
        vector<int> tmp(B); vector<bool> ok(B);
        for (int t = 0; t < B; ++t) { int j = base + t; ok[t] = j < W-1; if (ok[t]) tmp[t] = row[j+1]; }
        for (int t = 0; t < B; ++t) if (ok[t]) row[base + t] = tmp[t];
    }
    for (int j = 0; j < W-1; ++j) if (row[j] != want[j]) return false;
    return true;
}

int main() {
    mt19937 rng(42);
    int tests = 0;
    for (int it = 0; it < 3000; ++it) {
        int BWs[3] = {128, 256, 512};
        int BW = BWs[rng() % 3];
        int K = 1 + rng() % (BW/2 - 1);
        int P = 1 + rng() % 1500;          // max width (grid size)
        int W = 1 + rng() % P;             // current width (shrunk)
        int H = 2 + rng() % 300;
        int maxE = (rng() % 3 == 0) ? 3 : 1530;   // small range -> many ties
        vector<uint16_t> E((size_t)H * P);
        for (auto& e : E) e = rng() % (maxE + 1);
        if (plainSeam(E, P, H, W) != bandSeam(E, P, H, W, K, BW)) {
            printf("SEAM MISMATCH H=%d W=%d P=%d K=%d BW=%d\n", H, W, P, K, BW); return 1;
        }
        ++tests;
    }
    int tests2 = 0;
    for (int it = 0; it < 3000; ++it) {
        int Cs[3] = {2, 4, 8};
        int C = Cs[rng() % 3];
        int K = 1 + rng() % (16*C - 1);
        int P = 1 + rng() % 1500, W = 1 + rng() % P, H = 2 + rng() % 300;
        int maxE = (rng() % 3 == 0) ? 3 : 1530;
        vector<uint16_t> E((size_t)H * P);
        for (auto& e : E) e = rng() % (maxE + 1);
        if (plainSeam(E, P, H, W) != bandSeamV2(E, P, H, W, K, C)) {
            printf("V2 SEAM MISMATCH H=%d W=%d P=%d K=%d C=%d\n", H, W, P, K, C); return 1;
        }
        ++tests2;
    }
    printf("v2 OK: %d configs\n", tests2);
    for (int it = 0; it < 2000; ++it) {
        int W = 2 + rng() % 3000, s = rng() % W, B = 256;
        if (!removeEmu(W, s, B, rng)) { printf("REMOVE MISMATCH W=%d s=%d\n", W, s); return 1; }
    }
    printf("OK: %d banded-DP configs + 2000 removals match\n", tests);
}
