// Copyright (c) 2026 David Dalcu. MIT. Adapted from mlx-serve; see LICENSE and NOTICE.
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint n = threadgroup_position_in_grid.y;
uint row = threadgroup_position_in_grid.z;
threadgroup float part[8];
const int K = HC * H;
const device T* xn = xn_in + (size_t)row * (size_t)K;
const device float* ipart = ipart_in + (size_t)row * (size_t)(HC * HC);
device T* act = act_out + (size_t)row * (size_t)R;
device T* inj = inj_out + (size_t)row * (size_t)HC;
const int VPW = 32 / BITS;
const int K_by_p = K / VPW;
const int K_by_gs = K / GS;
const int SLICE = K_by_p / 8;
const int ITERS = SLICE / 32;
uint mask = (1u << BITS) - 1u;
if (n < uint(R)) {
  size_t wbase = (size_t)n * (size_t)K_by_p;
  size_t gbase = (size_t)n * (size_t)K_by_gs;
  int p0 = int(sg) * SLICE + int(lane);
  uint32_t pw[ITERS];
  for (int i = 0; i < ITERS; ++i) pw[i] = dw_q[wbase + (size_t)(p0 + 32 * i)];
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int i = 0; i < ITERS; ++i) {
    int k_base = (p0 + 32 * i) * VPW;
    int gi = k_base / GS;
    float sj = float(dw_s[gbase + (size_t)gi]);
    float bj = float(dw_b[gbase + (size_t)gi]);
    for (int ki = 0; ki < VPW; ki += 4) {
      int k = k_base + ki;
      uint32_t q = pw[i] >> (ki * BITS);
      a0 += float(xn[k + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
      a1 += float(xn[k + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
      a2 += float(xn[k + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
      a3 += float(xn[k + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
    }
  }
  float acc = simd_sum((a0 + a1) + (a2 + a3));
  if (lane == 0) part[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float t = 0.0f;
    for (int g = 0; g < 8; ++g) t += part[g];
    T v = T(float(T(t)) / float(HC));
    T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    act[n] = v * sig;
  }
} else if (tid == 0) {
  int c = int(n) - R;
  float t = 0.0f;
  for (int hh = 0; hh < HC; ++hh) t += ipart[hh * HC + c];
  T v = T(float(T(t)) / float(HC));
  T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj[c] = sig * T(2.0f);
}
