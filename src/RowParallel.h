#pragma once

#include <cstddef>
#include <cstdint>

namespace whistle {

// Installs a two-core row-parallel hook for the CQ GEMV: the caller keeps the
// first half of the output rows and a worker task pinned to the other core runs
// the second half, with the caller blocking until both halves finish. Returns
// false when the task/semaphores cannot be created, in which case the GEMV stays
// single-core. Safe to call more than once; call it from setup() before use.
bool enableDualCoreRows();

// Two-core parallel over independent rows (the Hadamard-MLP time frames).
// fn(ctx, r0, r1, out) computes rows [r0, r1), each `rowStride` contiguous
// floats, into `out` (which must hold (r1-r0)*rowStride floats). The caller runs
// [0, r0) directly into `out`; a pinned worker runs [r0, nrows) into an internal
// staging buffer with the same layout, copied to out + r0*rowStride after the
// join. `ro`/`roBytes` is a shared read-only input written back before the
// worker starts. Returns false (and runs nothing) when the second core or the
// staging buffer is unavailable, in which case the caller must compute
// [0, nrows) itself.
using RowChunkFn = void (*)(void* ctx, uint32_t r0, uint32_t r1, float* out);
bool runRowParallel(RowChunkFn fn, void* ctx, uint32_t nrows, size_t rowStride,
                    float* out, const void* ro, size_t roBytes);

}  // namespace whistle
