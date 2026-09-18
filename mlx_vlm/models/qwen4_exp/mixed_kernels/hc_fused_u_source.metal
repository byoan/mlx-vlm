// Copyright (c) 2026 David Dalcu. MIT. Adapted from mlx-serve; see donor-LICENSE.
uint lane = thread_index_in_simdgroup;
uint j = thread_position_in_grid.y;
uint row = thread_position_in_grid.z;
const device T* xn = xn_in + (size_t)row * (size_t)(HC * H);
const device T* act = act_in + (size_t)row * (size_t)R;
device T* mixed = mixed_out + (size_t)row * (size_t)H;
const int VPW = 32 / BITS;
const int R_by_p = R / VPW;
const int R_by_gs = R / GS;
const int RIT = (R_by_p + 31) / 32;
uint mask = (1u << BITS) - 1u;
float sum = 0.0f;
for (int h = 0; h < HC; ++h) {
  size_t row = (size_t)h * (size_t)H + (size_t)j;
  size_t wbase = row * (size_t)R_by_p;
  size_t gbase = row * (size_t)R_by_gs;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int i = 0; i < RIT; ++i) {
    int pack = int(lane) + 32 * i;
    if (pack < R_by_p) {
      uint32_t pw = uw_q[wbase + (size_t)pack];
      int k_base = pack * VPW;
      int gi = k_base / GS;
      float sj = float(uw_s[gbase + (size_t)gi]);
      float bj = float(uw_b[gbase + (size_t)gi]);
      for (int ki = 0; ki < VPW; ki += 4) {
        int k = k_base + ki;
        uint32_t q = pw >> (ki * BITS);
        a0 += float(act[k + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
        a1 += float(act[k + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
        a2 += float(act[k + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
        a3 += float(act[k + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
      }
    }
  }
  float acc = simd_sum((a0 + a1) + (a2 + a3));
  T u = T(acc);
  T sg = T(1.0f / (1.0f + metal::exp(-float(u))));
  sum += float(T(float(sg) * float(xn[row])));
}
if (lane == 0) mixed[j] = T(float(T(sum)) * float(T(1.0f / float(HC))));
