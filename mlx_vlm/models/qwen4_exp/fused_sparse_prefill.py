"""Direct selected-key QSA prefill, adapted from mlx-serve.

Source: ddalcu/mlx-serve src/transformer.zig, commit
9dd536a1ef860b08c9677c5f1d739ed04b33515e (gatherQsa256).
The upstream design credits oMLX PR #3244; its kernel body derives from
mlx-serve's msv_attn_p256. Adapted here to Python MLX with shared Boolean
masks and zero output for fully masked query rows.

MIT License

Copyright (c) 2026 David Dalcu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
#include <metal_simdgroup_matrix>

// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
// holding one row differ in lane bits 0 and 3 (see msv_coord).
inline short2 msv_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}

inline void msv_mma(thread float2 &d, float2 a, float2 b) {
  metal::simdgroup_float8x8 D, A, B, C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = d.x;
  C.thread_elements()[1] = d.y;
  simdgroup_multiply_accumulate(D, A, B, C);
  d.x = D.thread_elements()[0];
  d.y = D.thread_elements()[1];
}

inline float msv_row_max(float2 v) {
  float t = metal::max(v.x, v.y);
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
  return t;
}

inline float msv_row_sum(float2 v) {
  float t = v.x + v.y;
  t += metal::simd_shuffle_xor(t, ushort(1));
  t += metal::simd_shuffle_xor(t, ushort(8));
  return t;
}

inline int msv_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
  const int b = vi / ratio;
  return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
}

"""
_SOURCE = r"""
constexpr int BD = 256;
constexpr int LDK = BK + 8;
constexpr int LDV = BD + 8;
constexpr int NT = 32 * NSG;
constexpr int KT = BK / 8;

const int qL = q_shape[2];
const int kL = k_shape[2];
const int Hq = q_shape[1];
const int Hk = k_shape[1];
const int gqa = Hq / Hk;
const int KB = blocks_shape[2];
const int valid_stride = valid_shape[1];

const int s = int(threadgroup_position_in_grid.x);
const int hk = int(threadgroup_position_in_grid.y);
const int bb = int(threadgroup_position_in_grid.z);
const ushort lane = ushort(thread_index_in_simdgroup);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const int tix = int(thread_index_in_threadgroup);

const float scale_log2e = scl[0] * 1.44269504088896340736f;
// Query s sits at absolute key position p (bottom-right aligned). Its
// visible keys: `count` complete blocks, then the tail [tail_start, p].
const int p = (kL - qL) + s;
const int complete = (p + 1) / RATIO;
const int count = metal::min(complete, KB);
const int sel_len = count * RATIO;
const int tail_start = complete * RATIO;
const int L = sel_len + (p + 1 - tail_start);

const device int* blk = blocks + (long)bb * blocks_strides[0] + (long)s * blocks_strides[1];
const device T* Kp = k + bb * k_strides[0] + hk * k_strides[1];
const device T* Vp = v + bb * v_strides[0] + hk * v_strides[1];

threadgroup T KVs[LDK * BD];
threadgroup T* Ks = KVs;
threadgroup T* Vs = KVs;

const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const short tm = 8 * short(warp);
const int Ks_off = sm * LDK + sn;
const int Vs_off = sm * LDV + sn;
const int row = tm + sm;
const bool row_ok = row < gqa;

float2 Qfrag[BD / 8];
if (row_ok) {
  const device T* Qrow = q + bb * q_strides[0] + (long)(hk * gqa + row) * q_strides[1] + (long)s * q_strides[2];
  for (int dd = 0; dd < BD / 8; ++dd) {
    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    Qfrag[dd] = float2(float(pr.x), float(pr.y));
  }
} else {
  for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
}
float2 Ofrag[BD / 8];
for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
float max_score = -3.0e38f;
float sum_score = 0.0f;

for (int t0 = 0; t0 < L; t0 += BK) {
  const int rows_k = metal::min(BK, L - t0);

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    uint4 w = uint4(0);
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      w = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + c8);
    }
    thread T* e = (thread T*)&w;
    const int cb = c8 * 8;
    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  float2 Sfrag[KT];
  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
  for (int dd = 0; dd < BD / 8; ++dd) {
    const float2 qf = Qfrag[dd];
    const int kbase = Ks_off + dd * 8 * LDK;
    for (int kt = 0; kt < KT; ++kt) {
      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
      msv_mma(Sfrag[kt], qf, kf);
    }
  }
  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
  {
    for (int kt = 0; kt < KT; ++kt) {
      if (kt * 8 + sn >= rows_k || !valid[s * valid_stride + t0 + kt * 8 + sn]) Sfrag[kt].x = -INFINITY;
      if (kt * 8 + sn + 1 >= rows_k || !valid[s * valid_stride + t0 + kt * 8 + sn + 1]) Sfrag[kt].y = -INFINITY;
    }
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BD / 8); i += NT) {
    const int r = i >> 5;
    const int c8 = i & 31;
    uint4 w = uint4(0);
    if (r < rows_k) {
      const int pos = msv_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
      w = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + c8);
    }
    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
  }

  float new_max = max_score;
  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, msv_row_max(Sfrag[kt]));
  float rowsum = 0.0f;
  for (int kt = 0; kt < KT; ++kt) {
    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    rowsum += msv_row_sum(Sfrag[kt]);
  }
  const float factor = metal::exp2(max_score - new_max);
  max_score = new_max;
  sum_score = sum_score * factor + rowsum;
  for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int id = 0; id < BD / 8; ++id) {
    const int vbase = Vs_off + id * 8;
    for (int kt = 0; kt < KT; ++kt) {
      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
      msv_mma(Ofrag[id], Sfrag[kt], vf);
    }
  }
}

if (row_ok) {
  const float inv = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;
  device T* Optr = out + (((long)bb * Hq + (hk * gqa + row)) * (long)qL + (long)s) * BD + sn;
  for (int id = 0; id < BD / 8; ++id) {
    Optr[id * 8] = T(Ofrag[id].x * inv);
    Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
  }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="qwen4_fused_sparse_prefill",
        input_names=["q", "k", "v", "scl", "blocks", "valid"],
        output_names=["out"],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


def attention(queries, keys, values, blocks, valid, scale):
    return _kernel()(
        inputs=[
            queries,
            keys,
            values,
            mx.array([scale], dtype=mx.float32),
            blocks[None].astype(mx.int32),
            mx.contiguous(valid),
        ],
        template=[("T", queries.dtype), ("NSG", 2), ("BK", 32), ("RATIO", 4)],
        grid=(queries.shape[2] * 32, keys.shape[1] * 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
