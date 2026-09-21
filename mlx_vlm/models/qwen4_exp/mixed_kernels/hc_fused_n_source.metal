// Copyright (c) 2026 David Dalcu. MIT. Adapted from mlx-serve; see LICENSE and NOTICE.
uint tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint h = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;
threadgroup float tgs[8];
threadgroup float tgi[8 * HC];
const int base = int(h) * H;
const int PER = H / 256;
// Rows (batch*seq) are independent: every per-row buffer is offset here once.
const device T* x = x_in + (size_t)row * (size_t)(HC * H);
device T* xn = xn_out + (size_t)row * (size_t)(HC * H);
device T* xs = xs_out + (WR ? (size_t)row * (size_t)(HC * H) : 0);
device float* ipart = ipart_out + (size_t)row * (size_t)(HC * HC);
const device T* wo = wo_in + (size_t)row * (size_t)H;
float xv[PER];
if (WR) {
  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
  // (`wi_in` can be < 8 elements and land in `constant`: no pointer rebind.)
  float g = float(wi_in[(size_t)row * (size_t)HC + h]);
  for (int i = 0; i < PER; ++i) {
    int k = base + int(tid) + 256 * i;
    T v = T(float(x[k]) + float(T(float(wo[k - base]) * g)));
    xs[k] = v;
    xv[i] = float(v);
  }
} else {
  for (int i = 0; i < PER; ++i) xv[i] = float(x[base + int(tid) + 256 * i]);
}
float a = 0.0f;
for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
a = simd_sum(a);
if (lane == 0) tgs[sg] = a;
threadgroup_barrier(mem_flags::mem_threadgroup);
float t = 0.0f;
for (int g = 0; g < 8; ++g) t += tgs[g];
float rsh = rsqrt(t / float(H) + eps[0]);
float ip[HC];
for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
for (int i = 0; i < PER; ++i) {
  int k = base + int(tid) + 256 * i;
  T v = T((xv[i] * rsh) * (1.0f + float(nw[k])));
  xn[k] = v;
  if (INJ) { for (int c = 0; c < HC; ++c) ip[c] += float(v) * float(iw[(size_t)k * (size_t)HC + (size_t)c]); }
}
if (INJ) {
  for (int c = 0; c < HC; ++c) { float pa = simd_sum(ip[c]); if (lane == 0) tgi[sg * HC + c] = pa; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < uint(HC)) {
    float tt = 0.0f;
    for (int g = 0; g < 8; ++g) tt += tgi[g * HC + tid];
    ipart[h * HC + tid] = tt;
  }
}
