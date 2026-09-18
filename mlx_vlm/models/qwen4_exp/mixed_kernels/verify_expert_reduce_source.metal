// Copyright (c) 2026 David Dalcu. MIT. Adapted from mlx-serve; see donor-LICENSE.
uint p = thread_position_in_grid.x;
uint row = p / 2560;
uint col = p % 2560;
T partial[8];
#pragma unroll
for (uint i = 0; i < 8; ++i) partial[i] = T(0);
#pragma unroll
for (uint e = 0; e < 10; ++e) {
  uint at = row * 10 + e;
  T product = down[size_t(inverse[at]) * 2560 + col] * scores[at];
  partial[e % 8] = product + partial[e % 8];
}
T total = partial[0];
#pragma unroll
for (uint i = 1; i < 8; ++i) total = partial[i] + total;
y[p] = total;
