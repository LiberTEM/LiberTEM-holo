"""GPU (cupy) implementation of the weighted problem of the IRLS phase unwrapping.

PROTOTYPE. The counterpart of :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`
for cupy input. At the image sizes of holography (~500² to 2048²), the work per CG
iteration is a few passes over one image, so kernel launches and host
synchronizations cost more than the arithmetic. Hence:

* The preconditioner (the unweighted Neumann Poisson solve) is done as a periodic
  Poisson solve on the mirror-extended grid: the even extension turns the DCT-II into
  a DFT, and the symbol is the same. That is one mirror kernel, a real FFT pair with
  pre-built cuFFT plans, and a scaling kernel, instead of a DCT pair.
* The CG scalars (alpha, beta, the residuals, the stop flag) stay on the device. Each
  kernel returns at once when the stop flag is set, so a fixed sequence of the
  preamble and :code:`cg_max_iterations` iterations does what the CPU loop does.
  That sequence is captured once as a CUDA graph and replayed every round: one host
  synchronization per round's CG, instead of several per iteration.
* Dot products are reduced in float64 and in a fixed order (per-block partial sums,
  then the last block to finish adds them up), so results do not vary from run to
  run.

The kernels are written in CUDA C++ (cupy :class:`~cupy.RawModule`): a numba
:code:`@cuda.jit` launch with cupy arrays costs several times more.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import numpy as np

try:
    import cupy as cp
    from cupy.cuda import cufft
    from cupyx.scipy.fft import get_fft_plan
except ImportError:
    # Importable without cupy (for example for the doctests); only cupy input
    # gets here.
    cp = None

from .irls import _is_smooth

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .irls import BranchesPrior, CGRound, IRLSOptions, PhasePrior, _Candidate

_BLOCK = 256
# CG iterations per captured graph: the default `cg_max_iterations` fits in one.
_GRAPH_ITERATIONS = 16  # even: see `_cg_iterations`
# Grid of the grid-stride kernels: at most 8 blocks of 256 threads per
# multiprocessor, which is full occupancy (4 left the streaming kernels at 45 %
# of DRAM bandwidth, measured with Nsight Compute; 8 made a CG round 12 % faster
# at 2048^2, more made the reductions slower), and at least 2 elements per
# thread, which keeps the reduction tails of small images short.
_THREADS_PER_SM = 2048  # the most an SM holds at once
_MIN_PER_THREAD = 2
# PROTOTYPE switches: which CG dot products are added up from their partial sums
# by the kernels that need them, instead of by the last block of the kernel that
# computes them. That takes the last block's serial tail (~0.5 us) off the
# critical path, which gains ~4 % at ~500^2 for all three, but every block of the
# consuming kernels then adds up all partial sums, which costs ~4 % at 2048^2
# (measured on the GH200). So only up to `_DEFER_MAX_GRID` blocks.
_DEFER_RZ = True  # r . z, in apply_operator (beta) and cg_update (alpha)
_DEFER_ALPHA = True  # p . Ap, in cg_update
_DEFER_RR = True  # r . r, in makhoul_spectrum (the convergence check)
_DEFER_MAX_GRID = 512
# EXPERIMENT switch: run the hot CG kernels as numba CUDA kernels (`_irls_numba_cuda`).
_KERNELS: Literal["cuda", "numba"] = "numba"

# Indices into the double scalars `s` on the device:
(_S_NORM, _S_RZ, _S_ALPHA, _S_BETA, _S_MEAN, _S_TURNS, _S_COST, _S_LARGEST, _S_INVALID,
 _S_NAN) = range(10)
# Indices into the int flags `f` on the device:
_F_DONE, _F_ITER, _F_ITERATED, _F_ZERO, _F_PENDING = range(5)

_SOURCE = r"""
#ifndef BLOCK  // threads per block, set at compile time (see `_kernels`)
#define BLOCK 256
#endif

// Indices, as in the Python module:
#define S_NORM 0
#define S_RZ 1
#define S_ALPHA 2
#define S_BETA 3
#define S_MEAN 4
#define S_TURNS 5
#define S_COST 6
#define S_LARGEST 7
#define S_INVALID 8
#define S_NAN 9
#define F_DONE 0
#define F_ITER 1
#define F_ITERATED 2
#define F_ZERO 3
#define F_PENDING 4  // an iteration's residual was checked; `crop` counts it

// Block reductions: within each warp by shuffles, then across the warps in the
// first one. A fixed order, so the same result on every run. All threads of the
// block must call them; `sh` holds at least BLOCK / 32 + 1 values.
#define WARPS (BLOCK / 32)

__device__ double block_sum(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double r = lane < WARPS ? sh[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
        if (lane == 0) sh[WARPS] = r;
    }
    __syncthreads();
    double r = sh[WARPS];
    __syncthreads();  // `sh` may be reused at once
    return r;
}

// Write this block's sums of `v0` (and `v1`) to `partials`. In the last block to
// finish, return true and the totals over all blocks, added up in block order.
__device__ bool last_block(
    double v0, double v1, int two, double* partials, unsigned int* counter,
    double* total0, double* total1
) {
    __shared__ double sh[BLOCK];
    __shared__ bool is_last;
    double b0 = block_sum(v0, sh);
    double b1 = two ? block_sum(v1, sh) : 0.0;
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = b0;
        if (two) partials[gridDim.x + blockIdx.x] = b1;
        __threadfence();
        unsigned int ticket = atomicAdd(counter, 1u);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last) return false;
    __threadfence();
    volatile double* vp = partials;
    double a0 = 0.0, a1 = 0.0;
    for (int k = threadIdx.x; k < gridDim.x; k += BLOCK) {
        a0 += vp[k];
        if (two) a1 += vp[gridDim.x + k];
    }
    *total0 = block_sum(a0, sh);
    *total1 = two ? block_sum(a1, sh) : 0.0;
    if (threadIdx.x == 0) *counter = 0u;
    return true;
}

// 32-bit indices: 64-bit division (for the row and column) is slow on a GPU.
// The Python side checks that every array has fewer than 2^31 elements.
// Write this block's sum of `v` to partials[blockIdx.x] (no final sum).
__device__ void block_partial(double v, double* partials) {
    __shared__ double sh[WARPS + 1];
    double b = block_sum(v, sh);
    if (threadIdx.x == 0) partials[blockIdx.x] = b;
}

// The sum of `count` partial sums, in the same order as the last block of
// `last_block` adds them up, so with the same result; in every calling block.
__device__ double sum_partials(const double* partials, int count) {
    __shared__ double sh[WARPS + 1];
    double a = 0.0;
    for (int k = threadIdx.x; k < count; k += BLOCK) a += partials[k];
    return block_sum(a, sh);
}

#define GRID_STRIDE(k, n) \
    for (int k = blockIdx.x * BLOCK + threadIdx.x; k < (int)(n); \
         k += gridDim.x * BLOCK)

extern "C" __global__ void set_params(double* p, double a, double b) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { p[0] = a; p[1] = b; }
}

extern "C" __global__ void reset_flags(int* f) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        f[F_DONE] = 0; f[F_ITER] = 0; f[F_ITERATED] = 0; f[F_ZERO] = 0;
        f[F_PENDING] = 0;
    }
}

// -div(w grad p) into `out`, as `apply_operator` in _irls_numba.py. mode 0: only
// that; mode 1: also p . out, and alpha = rz / (p . out) from it. mode 2: as 1,
// for the next CG direction p' = beta p + z, computed on the fly at each point of
// the stencil (as `next_direction` would, so with the same result), and written
// to `p_new` (another buffer: neighbours still read the old p).
// With `rz_count` > 0, r . z is not in s[S_RZ] but the sum of the partial sums
// in `rz_new` (the last `crop`), and the one before in `rz_old`; every block
// adds them up itself (see `sum_partials`). With `pap_out`, the blocks only write
// their partial sums of p . Ap there, and `cg_update` adds them up for alpha.
template<typename T>
__global__ void apply_operator(
    const T* p, const T* z, T* p_new, const T* wx, const T* wy, T* out,
    int rows, int cols,
    double* partials, unsigned int* counter, double* s, int* f, int mode, double tiny,
    const double* rz_new, const double* rz_old, int rz_count, double* pap_out
) {
    if (f[F_DONE]) return;
    long long n = (long long)rows * cols;
    const bool update = mode == 2;
    double rz;
    T beta = (T)0;
    if (rz_count > 0) {
        rz = sum_partials(rz_new, rz_count);
        if (update) beta = (T)(rz / sum_partials(rz_old, rz_count));
    } else {
        rz = s[S_RZ];
        if (update) beta = (T)s[S_BETA];
    }
#define AT(x) (update ? p[x] * beta + z[x] : p[x])
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        T v = 0;
        T c = AT(k);
        if (j > 0) v += wx[i * (cols - 1) + j - 1] * (c - AT(k - 1));
        if (j < cols - 1) v -= wx[i * (cols - 1) + j] * (AT(k + 1) - c);
        if (i > 0) v += wy[(i - 1) * cols + j] * (c - AT(k - cols));
        if (i < rows - 1) v -= wy[i * cols + j] * (AT(k + cols) - c);
        out[k] = v;
        if (update) p_new[k] = c;
        acc += (double)c * (double)v;
    }
#undef AT
    if (mode == 0) return;
    if (pap_out) {
        block_partial(acc, pap_out);
        return;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        if (fabs(total) <= tiny) {
            f[F_DONE] = 1;  // zero curvature: converged
        } else {
            s[S_ALPHA] = (double)(T)(rz / total);
        }
    }
}

// b = -div(w g), and its mean into s[S_MEAN].
template<typename T>
__global__ void right_hand_side(
    const T* wx, const T* tx, const T* wy, const T* ty, T* b, int rows, int cols,
    double* partials, unsigned int* counter, double* s
) {
    long long n = (long long)rows * cols;
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        T v = 0;
        if (j > 0) v += wx[i * (cols - 1) + j - 1] * tx[i * (cols - 1) + j - 1];
        if (j < cols - 1) v -= wx[i * (cols - 1) + j] * tx[i * (cols - 1) + j];
        if (i > 0) v += wy[(i - 1) * cols + j] * ty[(i - 1) * cols + j];
        if (i < rows - 1) v -= wy[i * cols + j] * ty[i * cols + j];
        b[k] = v;
        acc += (double)v;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        s[S_MEAN] = total / (double)n;
    }
}

// b -= mean; norm = |b|. A zero norm stops the solve: nothing to do.
template<typename T>
__global__ void center_and_norm(
    T* b, long long n, double* partials, unsigned int* counter, double* s, int* f,
    double* history
) {
    T mean = (T)s[S_MEAN];
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        T v = b[k] - mean;
        b[k] = v;
        acc += (double)v * (double)v;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        double norm = sqrt(total);
        s[S_NORM] = norm;
        if (!(norm > 0)) {
            f[F_DONE] = 1; f[F_ZERO] = 1; history[0] = 0.0;
        }
    }
}

// r = b - q, and the initial relative residual.
template<typename T>
__global__ void initial_residual(
    const T* b, const T* q, T* r, long long n, double* partials,
    unsigned int* counter, double* s, int* f, double* history, double tol
) {
    if (f[F_DONE]) return;
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        T v = b[k] - q[k];
        r[k] = v;
        acc += (double)v * (double)v;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        double rel = sqrt(total) / s[S_NORM];
        history[0] = rel;
        if (rel <= tol) f[F_DONE] = 1; else f[F_ITERATED] = 1;
    }
}

// Where the Makhoul reordering puts index n of a length-N axis: the even indices
// first, ascending, then the odd ones, descending.
__device__ __forceinline__ int reordered(int n, int length) {
    return (n % 2 == 0) ? n / 2 : length - (n + 1) / 2;
}

// phi += alpha d; r -= alpha q; then the relative residual of the new r. Stops
// with F_DONE = 1 (converged) at the tolerance, with F_DONE = 2 at the limit.
// With `v`, also writes the new r to its place in the Makhoul-reordered (P1, P2)
// array v, the input of the preconditioner's FFT (whose padding stays zero).
// With `pap`, alpha = rz / (p . Ap) comes from the partial sums of p . Ap (and of
// r . z in `rz_cur`, else s[S_RZ]), which every block adds up itself; zero
// curvature stops CG in every block alike. With `rr_out`, the blocks only write
// their partial sums of r . r there, and `makhoul_spectrum` checks convergence.
template<typename T>
__global__ void cg_update(
    T* phi, const T* d, T* r, const T* q, long long n, double* partials,
    unsigned int* counter, double* s, int* f, double* history, double tol, int limit,
    T* v, int cols, int p1, int p2,
    const double* pap, const double* rz_cur, int count, double tiny, double* rr_out
) {
    if (f[F_DONE]) return;
    T alpha;
    if (pap) {
        double curvature = sum_partials(pap, count);
        if (fabs(curvature) <= tiny) {  // the direction is in the nullspace
            if (blockIdx.x == 0 && threadIdx.x == 0) f[F_DONE] = 1;
            return;
        }
        double rz = rz_cur ? sum_partials(rz_cur, count) : s[S_RZ];
        alpha = (T)(rz / curvature);
    } else {
        alpha = (T)s[S_ALPHA];
    }
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        phi[k] += alpha * d[k];
        T value = r[k] - alpha * q[k];
        r[k] = value;
        if (v) {
            int i = k / cols, j = k % cols;
            v[reordered(i, p1) * p2 + reordered(j, p2)] = value;
        }
        acc += (double)value * (double)value;
    }
    if (rr_out) {
        block_partial(acc, rr_out);
        return;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        double rel = sqrt(total) / s[S_NORM];
        int it = f[F_ITER] + 1;
        f[F_ITER] = it;
        history[it] = rel;
        if (rel <= tol) f[F_DONE] = 1;
        else if (it >= limit) f[F_DONE] = 2;
    }
}

// The even extension of r, zero-padded to (P1, P2), into the (2 P1, 2 P2) grid e.
template<typename T>
__global__ void mirror(
    const T* r, T* e, int rows, int cols, int p1, int p2, const int* f
) {
    if (f[F_DONE]) return;
    long long w = 2LL * p2;
    long long n = 2LL * p1 * w;
    GRID_STRIDE(k, n) {
        int i = k / w, j = k % w;
        int si = i < p1 ? i : 2 * p1 - 1 - i;
        int sj = j < p2 ? j : 2 * p2 - 1 - j;
        e[k] = (si < rows && sj < cols) ? r[(long long)si * cols + sj] : (T)0;
    }
}

template<typename C, typename T>
__global__ void scale_spectrum(
    C* spectrum, const T* inverse, long long n, const int* f
) {
    if (f[F_DONE]) return;
    GRID_STRIDE(k, n) {
        T a = inverse[k];
        spectrum[k].x *= a;
        spectrum[k].y *= a;
    }
}

// v = x reordered on both axes, where x is r zero-padded to (P1, P2). The DCT-II
// of x is then a twiddled FFT of v (J. Makhoul, 1980).
template<typename T>
__global__ void reorder(
    const T* r, T* v, int rows, int cols, int p1, int p2, const int* f
) {
    if (f[F_DONE]) return;
    long long n = (long long)p1 * p2;
    GRID_STRIDE(k, n) {
        int m1 = k / p2, m2 = k % p2;
        int i = m1 < (p1 + 1) / 2 ? 2 * m1 : 2 * (p1 - m1) - 1;
        int j = m2 < (p2 + 1) / 2 ? 2 * m2 : 2 * (p2 - m2) - 1;
        v[k] = (i < rows && j < cols) ? r[(long long)i * cols + j] : (T)0;
    }
}

// Re[w1 (w2 v0 + conj(w2) v1)]: the unnormalized DCT-II at (a, b) from the FFT V
// of the reordered array, with v0 = V[a, b], v1 = V[a, -b], w = exp(-i pi k / 2N).
template<typename C, typename T>
__device__ __forceinline__ T dct_at(C w1, C w2, C v0, C v1) {
    T ux = w2.x * v0.x - w2.y * v0.y + w2.x * v1.x + w2.y * v1.y;
    T uy = w2.x * v0.y + w2.y * v0.x + w2.x * v1.y - w2.y * v1.x;
    return w1.x * ux - w1.y * uy;
}

template<typename C>
__device__ __forceinline__ C conj(C c) { c.y = -c.y; return c; }

// The preconditioner in the spectral domain: from the half spectrum V of the
// reordered residual to the half spectrum V' of the reordered result. With
// Y = (inverse symbol) DCT-II, -k meaning N - k, and Y = 0 at index N:
// V'[k1, k2] = conj(w1(k1) w2(k2))
//     (Y[k1, k2] - Y[-k1, -k2] - i (Y[-k1, k2] + Y[k1, -k2])).
// Rows k1 and -k1 need the same four Y, and for a real input, V[-k1, -k2] =
// conj(V[k1, k2]), so these four come from two values of the half spectrum.
// One thread per row pair (k1 <= P1 / 2) and column k2 <= P2 / 2.
// With `rr`, first the convergence check of the CG iteration whose residual this
// is, from the partial sums of r . r (see `cg_update`): every block computes the
// relative residual and whether CG stops; block 0 records it, and marks the
// iteration for `crop` to count (F_ITER must not change while blocks read it).
template<typename C, typename T>
__global__ void makhoul_spectrum(
    const C* half, C* out, const T* inverse, const C* tw1, const C* tw2,
    int p1, int p2, int* f,
    const double* rr, int count, const double* s, double* history, double tol, int limit
) {
    if (f[F_DONE]) return;
    if (rr) {
        double rel = sqrt(sum_partials(rr, count)) / s[S_NORM];
        int it = f[F_ITER] + 1;
        int done = rel <= tol ? 1 : (it >= limit ? 2 : 0);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            history[it] = rel;
            f[F_PENDING] = 1;
            if (done) f[F_DONE] = done;
        }
        if (done) return;
    }
    int h = p2 / 2 + 1;
    int n = (p1 / 2 + 1) * h;
    GRID_STRIDE(k, n) {
        int k1 = k / h, k2 = k % h;
        int m1 = (p1 - k1) % p1;  // the row of -k1
        C va = half[k1 * h + k2];  // V[k1, k2]
        C vc = half[m1 * h + k2];  // V[-k1, k2]
        // V[k1, -k2] = conj(vc), V[-k1, -k2] = conj(va)
        C w1a = tw1[k1], w1c = tw1[p1 - k1], w2a = tw2[k2], w2b = tw2[p2 - k2];
        T y_aa = dct_at<C, T>(w1a, w2a, va, conj(vc)) * inverse[k1 * p2 + k2];
        T y_ab = k2 == 0 ? (T)0
            : dct_at<C, T>(w1a, w2b, conj(vc), va) * inverse[k1 * p2 + (p2 - k2)];
        T y_ca = k1 == 0 ? (T)0
            : dct_at<C, T>(w1c, w2a, vc, conj(va)) * inverse[m1 * p2 + k2];
        T y_cb = (k1 == 0 || k2 == 0) ? (T)0
            : dct_at<C, T>(w1c, w2b, conj(va), vc) * inverse[m1 * p2 + (p2 - k2)];
        // Row k1: Y[k1, k2] - Y[-k1, -k2] - i (Y[-k1, k2] + Y[k1, -k2])
        // Row -k1: Y[-k1, k2] - Y[k1, -k2] - i (Y[k1, k2] + Y[-k1, -k2])
        C w2c = conj(w2a);
        for (int row = 0; row < 2; ++row) {
            if (row == 1 && (k1 == 0 || m1 == k1)) break;
            int r1 = row == 0 ? k1 : m1;
            T re = row == 0 ? y_aa - y_cb : y_ca - y_ab;
            T im = row == 0 ? -(y_ca + y_ab) : -(y_aa + y_cb);
            C w1 = conj(tw1[r1]);
            T cx = w1.x * w2c.x - w1.y * w2c.y;
            T cy = w1.x * w2c.y + w1.y * w2c.x;
            C o;
            o.x = cx * re - cy * im;
            o.y = cx * im + cy * re;
            out[r1 * h + k2] = o;
        }
    }
}

// z = e[:rows, :cols] (`unreorder` 0), or z = x[:rows, :cols] for x the
// Makhoul-reordered array e (`unreorder` 1); `width` is the row length of e.
// mode 0: rz = r . z, and d = z (the start of CG).
// mode 1: beta = (r . z) / rz, rz = r . z. mode 2: only z.
// mode 3: only the blocks' partial sums of r . z, into `partials`, for the next
// kernels to add up (see `apply_operator`). mode 4: as 3, and d = z.
// With `count_iteration`, block 0 first counts an iteration that
// `makhoul_spectrum` checked, also the last one, after which CG stops.
template<typename T>
__global__ void crop(
    const T* e, const T* r, T* z, T* d, int rows, int cols, int p1, int p2,
    int width, int unreorder,
    double* partials, unsigned int* counter, double* s, int* f, int mode,
    int count_iteration
) {
    if (count_iteration && blockIdx.x == 0 && threadIdx.x == 0 && f[F_PENDING]) {
        f[F_ITER] += 1;
        f[F_PENDING] = 0;
    }
    if (f[F_DONE]) return;
    long long n = (long long)rows * cols;
    double acc = 0.0;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        if (unreorder) {
            i = reordered(i, p1);
            j = reordered(j, p2);
        }
        T v = e[(long long)i * width + j];
        z[k] = v;
        if (mode == 0 || mode == 4) d[k] = v;
        acc += (double)r[k] * (double)v;
    }
    if (mode == 2) return;
    if (mode >= 3) {
        block_partial(acc, partials);
        return;
    }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        if (mode == 1) s[S_BETA] = (double)(T)(total / s[S_RZ]);
        s[S_RZ] = total;
    }
}

// The mean of x into s[S_MEAN], once CG has stopped.
template<typename T>
__global__ void mean(
    const T* x, long long n, double* partials, unsigned int* counter, double* s,
    const int* f
) {
    if (!f[F_DONE]) return;
    double acc = 0.0;
    GRID_STRIDE(k, n) { acc += (double)x[k]; }
    double total, unused;
    if (last_block(acc, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        s[S_MEAN] = total / (double)n;
    }
}

// x -= mean, once CG has stopped, if it iterated (the CPU code: `if iterated:
// phi -= phi.mean()`).
template<typename T>
__global__ void subtract_mean_if_iterated(
    T* x, long long n, const double* s, const int* f
) {
    if (!f[F_DONE] || !f[F_ITERATED]) return;
    T m = (T)s[S_MEAN];
    GRID_STRIDE(k, n) { x[k] -= m; }
}

// w = C eps^2 / sqrt((grad phi - g)^2 + eps^2), as `charbonnier_weights`, with
// eps^2 = params[0]. With params[1] != 0 ("unless settled"), nothing happens if
// the last CG ran no iteration: the IRLS loop's fixed point, decided on the
// device. The parameters are in device memory, so that a captured graph can
// take new ones each round.
template<typename T>
__global__ void charbonnier_weights(
    const T* phi, const T* tx, const T* ty, const T* cx, const T* cy,
    const double* params, T* wx, T* wy, int rows, int cols, const int* f
) {
    if (params[1] != 0.0 && f[F_ITER] == 0) return;
    T squared = (T)params[0];
    long long n = (long long)rows * cols;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        if (j < cols - 1) {
            long long e = (long long)i * (cols - 1) + j;
            T r = (phi[k + 1] - phi[k]) - tx[e];
            wx[e] = (squared / sqrt(r * r + squared)) * cx[e];
        }
        if (i < rows - 1) {
            T r = (phi[k + cols] - phi[k]) - ty[k];
            wy[k] = (squared / sqrt(r * r + squared)) * cy[k];
        }
    }
}

template<typename T>
__global__ void wrapped_difference(
    const T* phi, const T* wrapped, T tau, T* out, long long n
) {
    GRID_STRIDE(k, n) {
        T v = phi[k] - wrapped[k];
        out[k] = v - rint(v / tau) * tau;
    }
}

template<typename T>
__global__ void branches_and_phase(
    const T* phi, const T* wrapped, const T* offset, T tau, int* branches,
    double* phase, long long n
) {
    T off = offset[0];
    GRID_STRIDE(k, n) {
        int b = (int)rint((phi[k] - off - wrapped[k]) / tau);
        branches[k] = b;
        phase[k] = (double)wrapped[k] + 2.0 * 3.141592653589793 * (double)b;
    }
}

// Cut turns |jumps + diff(branches)| and their cost, weighted by C, summed over all
// edges into s[S_TURNS] and s[S_COST].
template<typename T>
__global__ void cuts(
    const int* br, const int* jx, const int* jy, const T* cx, const T* cy,
    int rows, int cols, double* partials, unsigned int* counter, double* s
) {
    long long n = (long long)rows * cols;
    double turns = 0.0, cost = 0.0;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        if (j < cols - 1) {
            long long e = (long long)i * (cols - 1) + j;
            int t = abs(jx[e] + (br[k + 1] - br[k]));
            turns += t;
            cost += t * (double)cx[e];
        }
        if (i < rows - 1) {
            int t = abs(jy[k] + (br[k + cols] - br[k]));
            turns += t;
            cost += t * (double)cy[k];
        }
    }
    double t0, t1;
    if (last_block(turns, cost, 1, partials, counter, &t0, &t1)
            && threadIdx.x == 0) {
        s[S_TURNS] = t0;
        s[S_COST] = t1;
    }
}

// ---- Setup ----------------------------------------------------------------

// Copy the wrapped phase (`in` may be `wrapped`), and compute the wrapped target
// gradients and their whole-turn jumps, as `_WeightedProblem`:
// g = d - rint(d / tau) tau, jumps = rint((d - g) / tau), in T. With
// `unit_edges`, also C = w = 1 on every edge.
template<typename T>
__global__ void prepare_targets(
    const T* in, T* wrapped, T* tx, T* ty, int* jx, int* jy,
    T* cx, T* cy, T* wx, T* wy, int rows, int cols, int unit_edges
) {
    const T tau = (T)6.283185307179586;
    int n = rows * cols;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        T c = in[k];
        wrapped[k] = c;
        if (j < cols - 1) {
            int e = i * (cols - 1) + j;
            T d = in[k + 1] - c;
            T g = d - rint(d / tau) * tau;
            tx[e] = g;
            jx[e] = (int)rint((d - g) / tau);
            if (unit_edges) { cx[e] = (T)1; wx[e] = (T)1; }
        }
        if (i < rows - 1) {
            T d = in[k + cols] - c;
            T g = d - rint(d / tau) * tau;
            ty[k] = g;
            jy[k] = (int)rint((d - g) / tau);
            if (unit_edges) { cy[k] = (T)1; wy[k] = (T)1; }
        }
    }
}

// The phase and the amplitude of a complex wave, as cupy's angle and abs, and the
// number of NaN amplitudes into s[S_NAN]: numpy's percentile is NaN then.
template<typename C, typename T>
__global__ void wave_phase_amplitude(
    const C* wave, T* phase, T* amplitude, int n,
    double* partials, unsigned int* counter, double* s
) {
    double nans = 0.0;
    GRID_STRIDE(k, n) {
        C z = wave[k];
        phase[k] = atan2(z.y, z.x);
        T a = hypot(z.x, z.y);
        amplitude[k] = a;
        nans += isnan(a);
    }
    double total, unused;
    if (last_block(nans, 0.0, 0, partials, counter, &total, &unused)
            && threadIdx.x == 0) {
        s[S_NAN] = total;
    }
}

// amplitude_edge_weights in double: clip(a / ref, floor, 1), or 1 without signal
// (ref not > 0); NaN propagates, as in numpy's clip and minimum.
__device__ __forceinline__ double relative_amplitude(
    double a, double ref, double floor
) {
    if (!(ref > 0)) return 1.0;
    double r = a / ref;
    if (isnan(r)) return r;
    return r < floor ? floor : (r > 1.0 ? 1.0 : r);
}
__device__ __forceinline__ double nan_minimum(double a, double b) {
    return (isnan(a) || isnan(b)) ? (a + b) : (a < b ? a : b);
}

__device__ double block_max(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        double r = lane < WARPS ? sh[lane] : sh[0];
        for (int o = 16; o > 0; o >>= 1) {
            r = fmax(r, __shfl_down_sync(0xffffffffu, r, o));
        }
        if (lane == 0) sh[WARPS] = r;
    }
    __syncthreads();
    double r = sh[WARPS];
    __syncthreads();
    return r;
}

// The edge weights from the amplitude, with ref = params[0] (the percentile).
// mode 0: their maximum into s[S_LARGEST] (1 without edges), and the number of
// edges whose weight is not finite and greater than zero into s[S_INVALID].
// mode 1: C = w = (T)(weight / s[S_LARGEST]), as `_check_edge_weights`.
template<typename T>
__global__ void amplitude_edges(
    const T* amp, const double* params, double floor, int rows, int cols, int mode,
    double* partials, unsigned int* counter, double* s,
    T* cx, T* cy, T* wx, T* wy
) {
    __shared__ double sh[BLOCK];
    __shared__ bool is_last;
    // As numpy: a NaN amplitude makes the percentile NaN, so all weights one.
    double ref = s[S_NAN] > 0 ? __longlong_as_double(0x7ff8000000000000LL) : params[0];
    double largest = mode == 1 ? s[S_LARGEST] : 0.0;
    double top = -__longlong_as_double(0x7ff0000000000000LL), invalid = 0.0;
    int n = rows * cols;
    GRID_STRIDE(k, n) {
        int i = k / cols, j = k % cols;
        double a = relative_amplitude((double)amp[k], ref, floor);
        if (j < cols - 1) {
            double w = nan_minimum(
                relative_amplitude((double)amp[k + 1], ref, floor), a);
            if (mode == 0) {
                top = fmax(top, w);
                invalid += !(isfinite(w) && w > 0);
            } else {
                int e = i * (cols - 1) + j;
                cx[e] = (T)(w / largest);
                wx[e] = cx[e];
            }
        }
        if (i < rows - 1) {
            double w = nan_minimum(
                relative_amplitude((double)amp[k + cols], ref, floor), a);
            if (mode == 0) {
                top = fmax(top, w);
                invalid += !(isfinite(w) && w > 0);
            } else {
                cy[k] = (T)(w / largest);
                wy[k] = cy[k];
            }
        }
    }
    if (mode == 1) return;
    double bmax = block_max(top, sh);
    double bsum = block_sum(invalid, sh);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = bmax;
        partials[gridDim.x + blockIdx.x] = bsum;
        __threadfence();
        is_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    volatile double* vp = partials;
    double m = -__longlong_as_double(0x7ff0000000000000LL), c = 0.0;
    for (int b = threadIdx.x; b < gridDim.x; b += BLOCK) {
        m = fmax(m, vp[b]);
        c += vp[gridDim.x + b];
    }
    m = block_max(m, sh);
    c = block_sum(c, sh);
    if (threadIdx.x == 0) {
        s[S_LARGEST] = m == -__longlong_as_double(0x7ff0000000000000LL) ? 1.0 : m;
        s[S_INVALID] = c;
        *counter = 0u;
    }
}

// The starting phase of a solve from a prior, as `irls._starting_phase`: mode 0
// (a phase prior) the branch of the wrapped value nearest to the prior, in
// double; mode 1 (a branch prior) wrapped + 2 pi branches, in T.
template<typename T>
__global__ void starting_phase(
    const T* wrapped, const double* prior_phase, const int* prior_branches, int mode,
    T* phi, long long n
) {
    const double tau = 6.283185307179586;
    GRID_STRIDE(k, n) {
        if (mode == 0) {
            double raw = (double)wrapped[k];
            double b = rint((prior_phase[k] - raw) / tau);
            phi[k] = (T)(raw + tau * b);
        } else {
            phi[k] = wrapped[k] + (T)tau * (T)prior_branches[k];
        }
    }
}

// ---- Exact selection of order statistics (radix select) ------------------
//
// The k-th smallest value, by its bit pattern: keys that sort like the floats
// (flip all bits of negative values, set the sign bit of positive ones), and one
// pass per 11-bit digit, most significant first. Each pass takes the histogram of
// the digit among the values that match the digits found so far; the last block
// to finish finds the bucket that holds rank k, and narrows the prefix and the
// rank to it. Two ranks at once, k0 <= k1: the two middle values of a median, or
// the two neighbours of a percentile. No host synchronization.

#define DIGIT_BITS 11
#define BUCKETS (1 << DIGIT_BITS)

__device__ __forceinline__ unsigned long long to_key(float v) {
    unsigned int u = __float_as_uint(v);
    return (u & 0x80000000u) ? (unsigned long long)(~u)
                             : (unsigned long long)(u | 0x80000000u);
}
__device__ __forceinline__ unsigned long long to_key(double v) {
    unsigned long long u = (unsigned long long)__double_as_longlong(v);
    return (u >> 63) ? ~u : (u | (1ULL << 63));
}
__device__ __forceinline__ float from_key(unsigned long long key, float) {
    unsigned int k = (unsigned int)key;
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
__device__ __forceinline__ double from_key(unsigned long long key, double) {
    return __longlong_as_double(
        (long long)((key >> 63) ? (key & ~(1ULL << 63)) : ~key));
}

template<typename T> struct KeyBits { static const int value = 8 * sizeof(T); };

// Find the bucket of rank k in hist (all threads of the block take part).
// Returns the bucket in *bucket and the rank within it in *rank (thread 0).
__device__ void find_bucket(
    const unsigned int* hist, unsigned long long k, int* bucket,
    unsigned long long* rank, unsigned long long* sh
) {
    const int per = BUCKETS / BLOCK;
    unsigned long long own = 0;
    for (int b = 0; b < per; ++b) own += hist[threadIdx.x * per + b];
    sh[threadIdx.x] = own;
    __syncthreads();
    // inclusive scan (Hillis-Steele)
    for (int off = 1; off < BLOCK; off <<= 1) {
        unsigned long long v = threadIdx.x >= off ? sh[threadIdx.x - off] : 0;
        __syncthreads();
        sh[threadIdx.x] += v;
        __syncthreads();
    }
    unsigned long long before = sh[threadIdx.x] - own;
    if (k >= before && k < sh[threadIdx.x]) {
        unsigned long long r = k - before;
        for (int b = 0; b < per; ++b) {
            unsigned long long c = hist[threadIdx.x * per + b];
            if (r < c) { *bucket = threadIdx.x * per + b; *rank = r; break; }
            r -= c;
        }
    }
    __syncthreads();
}

// One pass of the selection. `st` holds prefix0, prefix1, rank0, rank1 (set from
// k0, k1 in pass 0). `hist` is 2 * BUCKETS zeros, and left so. After the last pass,
// mode 0 writes the median (v0 + v1) / 2 in T to out_t[0] (v0 == v1 for odd n, so
// that is v0), mode 1 the percentile by linear interpolation with weight t, as
// numpy's `_lerp`, in double to out_d[0]; both modes write v0 and v1 to out_d[1:3].
template<typename T>
__global__ void select_pass(
    const T* x, int n, int pass, unsigned long long k0, unsigned long long k1,
    unsigned long long* st, unsigned int* hist, unsigned int* counter,
    int mode, double t, T* out_t, double* out_d
) {
    __shared__ unsigned int h0[BUCKETS];
    __shared__ unsigned int h1[BUCKETS];
    __shared__ bool is_last;
    __shared__ unsigned long long sh[BLOCK];
    __shared__ int found_bucket[2];
    __shared__ unsigned long long found_rank[2];
    const int bits = KeyBits<T>::value;
    int fixed = DIGIT_BITS * pass;  // high bits fixed by earlier passes
    int width = min(DIGIT_BITS, bits - fixed);
    int shift = bits - fixed - width;
    unsigned long long digit_mask = (1ULL << width) - 1;
    unsigned long long high_mask = fixed == 0 ? 0ULL : (~0ULL << (bits - fixed));
    if (bits == 32) high_mask &= 0xffffffffULL;
    unsigned long long p0 = pass == 0 ? 0ULL : st[0];
    unsigned long long p1 = pass == 0 ? 0ULL : st[1];
    for (int b = threadIdx.x; b < BUCKETS; b += BLOCK) { h0[b] = 0; h1[b] = 0; }
    __syncthreads();
    GRID_STRIDE(k, n) {
        unsigned long long key = to_key(x[k]);
        unsigned long long high = key & high_mask;
        int d = (int)((key >> shift) & digit_mask);
        if (high == p0) atomicAdd(&h0[d], 1u);
        if (high == p1) atomicAdd(&h1[d], 1u);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < BUCKETS; b += BLOCK) {
        if (h0[b]) atomicAdd(&hist[b], h0[b]);
        if (h1[b]) atomicAdd(&hist[BUCKETS + b], h1[b]);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(counter, 1u);
        is_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    unsigned long long r0 = pass == 0 ? k0 : st[2];
    unsigned long long r1 = pass == 0 ? k1 : st[3];
    find_bucket(hist, r0, &found_bucket[0], &found_rank[0], sh);
    find_bucket(hist + BUCKETS, r1, &found_bucket[1], &found_rank[1], sh);
    for (int b = threadIdx.x; b < 2 * BUCKETS; b += BLOCK) hist[b] = 0;
    if (threadIdx.x == 0) {
        unsigned long long np0 = p0 | ((unsigned long long)found_bucket[0] << shift);
        unsigned long long np1 = p1 | ((unsigned long long)found_bucket[1] << shift);
        st[0] = np0; st[1] = np1; st[2] = found_rank[0]; st[3] = found_rank[1];
        *counter = 0u;
        if (shift == 0) {  // the last pass: the keys are complete
            T v0 = from_key(np0, (T)0), v1 = from_key(np1, (T)0);
            out_d[1] = (double)v0;
            out_d[2] = (double)v1;
            if (mode == 0) {
                out_t[0] = (v0 + v1) / (T)2;
            } else {
                double a = (double)v0, b = (double)v1, diff = b - a;
                out_d[0] = t >= 0.5 ? b - diff * (1.0 - t) : a + diff * t;
            }
        }
    }
}
"""

_TEMPLATES = (
    "apply_operator", "right_hand_side", "center_and_norm", "initial_residual",
    "cg_update", "mirror", "reorder", "crop", "mean",
    "subtract_mean_if_iterated", "charbonnier_weights", "wrapped_difference",
    "branches_and_phase", "cuts", "select_pass", "prepare_targets",
    "amplitude_edges", "starting_phase",
)
_CTYPES = {np.dtype(np.float32): ("float", "float2"),
           np.dtype(np.float64): ("double", "double2")}


def _kernel_names(dtype: np.dtype) -> dict[str, str]:
    real, cplx = _CTYPES[np.dtype(dtype)]
    names = {name: f"{name}<{real}>" for name in _TEMPLATES}
    names["scale_spectrum"] = f"scale_spectrum<{cplx}, {real}>"
    names["makhoul_spectrum"] = f"makhoul_spectrum<{cplx}, {real}>"
    names["wave_phase_amplitude"] = f"wave_phase_amplitude<{cplx}, {real}>"
    return names


_modules: dict[tuple[np.dtype, int], tuple[cp.RawModule, dict[str, str]]] = {}


def _fft_plans(extended: cp.ndarray, spectrum: cp.ndarray) -> tuple:
    """Return new cuFFT plans for the real FFT pair on the extended grid.

    Each workspace has its own: a plan has a work area for its intermediate
    results, which every execution of the plan uses, and its stream is set at each
    call. Shared between workspaces, and so between threads, concurrent solves of
    the same shape would overwrite each other's intermediate results. A plan pair
    costs ~10 ms to create, once per workspace.
    """
    return (
        get_fft_plan(extended, value_type="R2C"),
        get_fft_plan(spectrum, shape=extended.shape, value_type="C2R"),
    )


def _kernels(dtype: np.dtype, block: int) -> dict[str, cp.RawKernel]:
    """Compile the kernels for `dtype` and `block` threads per block, once."""
    key = (np.dtype(dtype), block)
    if key not in _modules:
        names = _kernel_names(key[0])
        module = cp.RawModule(
            code=_SOURCE,
            options=("--std=c++17", "--fmad=false", f"-DBLOCK={block}"),
            name_expressions=[*names.values(), "reset_flags", "set_params"],
        )
        _modules[key] = (module, names)
    module, names = _modules[key]
    kernels = {k: module.get_function(v) for k, v in names.items()}
    kernels["reset_flags"] = module.get_function("reset_flags")
    kernels["set_params"] = module.get_function("set_params")
    return kernels


# PROTOTYPE switches, for comparisons:
PRECONDITIONER: Literal["makhoul", "mirror"] = "makhoul"
PADDED_LENGTH: Callable[[int], int] | None = None  # None: irls._padded_length

_local = threading.local()
_CACHE_SIZE = 2


def problem_for(
    wrapped: cp.ndarray,
    edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    *,
    deferred: bool = False,
) -> CupyWeightedProblem:
    """Return a problem for `wrapped`, reusing a workspace of the same shape.

    Buffers, FFT plans and the captured CUDA graph only depend on the shape and the
    dtype, so a series of same-sized inputs reuses them. Workspaces are kept per
    thread (two concurrent solves must not share one), for the last
    :code:`_CACHE_SIZE` shapes. With `deferred`, the input is only copied into the
    workspace, and :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start` sets the problem up, which must come
    next.
    """
    problem = _workspace(wrapped.shape, wrapped.dtype)
    if deferred:
        problem.stage_phase(wrapped, edge_weights)
    else:
        problem.fill(wrapped, edge_weights)
    return problem


def problem_for_wave(
    wave: cp.ndarray, floor: float, percentile: float, *, deferred: bool = False,
) -> CupyWeightedProblem:
    """Return a problem for the phase of `wave`, with edge weights from its amplitude.

    See :func:`problem_for` and :meth:`CupyWeightedProblem.fill_from_wave`. With
    `deferred`, the wave is only copied into the workspace, and the problem is set
    up by :meth:`CupyWeightedProblem.cold_start` or
    :meth:`~CupyWeightedProblem.warm_start`, which must come next.
    """
    dtype = np.float32 if wave.dtype == np.complex64 else np.float64
    wave = cp.ascontiguousarray(wave, dtype=np.result_type(dtype, np.complex64))
    problem = _workspace(wave.shape, np.dtype(dtype))
    if deferred:
        problem.stage_wave(wave, floor, percentile)
    else:
        problem.fill_from_wave(wave, floor, percentile)
    return problem


def _workspace(shape: tuple[int, int], dtype: np.dtype) -> CupyWeightedProblem:
    """Return the cached workspace for `shape` and `dtype` (see :func:`problem_for`)."""
    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = OrderedDict()
    key = (
        shape, np.dtype(dtype).str, cp.cuda.Device().id,
        PRECONDITIONER, PADDED_LENGTH, _BLOCK, _MIN_PER_THREAD,
        _DEFER_RZ, _DEFER_ALPHA, _DEFER_RR, _DEFER_MAX_GRID, _KERNELS,
    )
    problem = cache.pop(key, None)
    if problem is None:
        problem = CupyWeightedProblem(
            shape, dtype, preconditioner=PRECONDITIONER, padded_length=PADDED_LENGTH,
        )
    cache[key] = problem
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)
    return problem


def _gpu_padded_length(length: int) -> int:
    """Return the smallest even 3-smooth number that is at least `length`.

    cuFFT is fastest on these lengths. Lengths with factors of 5 or 7, which the
    CPU version allows, are much slower with cuFFT for some sizes (500 against
    512, 1008 against 1024), and measured slower overall, padding included.
    """
    candidate = max(length, 2)
    while candidate % 2 != 0 or not _is_smooth(candidate, limit=3):
        candidate += 1
    return candidate


def _pinned(size: int, dtype: type) -> np.ndarray:
    """Return a numpy array of `size` elements in pinned (page-locked) memory."""
    dtype = np.dtype(dtype)
    memory = cp.cuda.alloc_pinned_memory(max(size, 1) * dtype.itemsize)
    return np.frombuffer(memory, dtype=dtype, count=size)


_multiprocessors: dict[int, int] = {}


def _multiprocessor_count() -> int:
    device = cp.cuda.Device()
    if device.id not in _multiprocessors:
        # `Device.attributes` queries all attributes, which takes milliseconds:
        _multiprocessors[device.id] = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMultiProcessorCount, device.id,
        )
    return _multiprocessors[device.id]


class CupyWeightedProblem:
    """The weighted problem of one IRLS round, on the GPU.

    Same interface as :class:`~libertem_holo.base.unwrap.irls._WeightedProblem`.
    All arrays are cupy arrays. Use :func:`problem_for` to get one.

    Everything runs on a private non-blocking stream (stream capture needs one),
    ordered with the caller's stream on entry and exit (:meth:`_on_stream`).
    """

    xp = cp

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype,
        preconditioner: Literal["makhoul", "mirror"] = "makhoul",
        padded_length: Callable[[int], int] | None = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.shape = rows, cols = shape
        self.size = rows * cols
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.block = _BLOCK
        self.k = _kernels(self.dtype, self.block)
        self.grid = max(1, min(
            -(-self.size // (self.block * _MIN_PER_THREAD)),
            min(_THREADS_PER_SM // self.block, 32) * _multiprocessor_count(),
        ))
        self._numba = None
        if _KERNELS == "numba":
            self._init_numba(preconditioner)
        self._set_deferrals(preconditioner)
        image = lambda shape=shape: cp.empty(shape, dtype=self.dtype)  # noqa: E731
        with self.stream:
            edges_x, edges_y = (rows, cols - 1), (rows - 1, cols)
            self.target_x, self.target_y = image(edges_x), image(edges_y)
            self.jumps_x = cp.empty((rows, cols - 1), dtype=cp.int32)
            self.jumps_y = cp.empty((rows - 1, cols), dtype=cp.int32)
            self.edge_x, self.edge_y = image(edges_x), image(edges_y)
            self.weights_x, self.weights_y = image(edges_x), image(edges_y)
            self.unit_x = cp.ones((rows, cols - 1), dtype=self.dtype)
            self.unit_y = cp.ones((rows - 1, cols), dtype=self.dtype)

            self._allocate_scalars()

            self._setup_preconditioner(
                preconditioner, padded_length or _gpu_padded_length,
            )

            # CG work arrays; the graph runs on `phi`, see `solve`:
            self.phi, self.b, self.q, self.r, self.z = (image() for _ in range(5))
            # The CG direction, in two buffers: an iteration computes the next one
            # from the current one (see `apply_operator`, mode 2).
            self.directions = (image(), image())
            self._direction = 0
            self.difference = image()
            # Inputs at fixed addresses, for the captured graphs:
            self.wrapped = image()
            self.wave = None  # a staged wave (see `stage_wave`)
            self.prior_phase = None  # a staged prior (see `warm_start`)
            self.prior_branches = None
            self.start_params = cp.zeros(2, dtype=cp.float64)  # eps^2 of a prior
            self.l2 = image()  # the L2 solution of the cold start
            self.round_params = cp.zeros(2, dtype=cp.float64)  # eps^2, unless settled
            self.direct_params = cp.zeros(2, dtype=cp.float64)  # the same, ungraphed
            # Two slots for a round's candidate result (phase, branches, weights):
            # a round writes into the slot that does not hold the best result so
            # far, so that keeping a candidate needs no copy (see `keep`).
            self.slots = tuple(
                (cp.empty(self.shape, dtype=cp.float64),
                 cp.empty(self.shape, dtype=cp.int32),
                 image(edges_x), image(edges_y))
                for _ in range(2)
            )
            self._kept: int | None = None
            # Reused for the stream ordering of `_on_stream`:
            self._event_caller = cp.cuda.Event(disable_timing=True)
            self._event_own = cp.cuda.Event(disable_timing=True)
            # The radix selection (see `select_pass`):
            self.select_state = cp.zeros(4, dtype=cp.uint64)
            self.select_hist = cp.zeros(2 * 2048, dtype=cp.uint32)
            self.select_counter = cp.zeros(1, dtype=cp.uint32)
            self.select_out = cp.zeros(3, dtype=cp.float64)
            self.offset = cp.zeros(1, dtype=self.dtype)
        if self.extended.size >= 2**31:
            msg = f"image too large for the GPU solver: {shape}"
            raise ValueError(msg)
        self._init_host_state()

    def _set_deferrals(self, preconditioner: str) -> None:
        """Which dot products to defer, for this grid (see `_DEFER_RZ`)."""
        small = self.grid <= _DEFER_MAX_GRID
        self.defer_rz = _DEFER_RZ and small
        self.defer_alpha = _DEFER_ALPHA and small
        self.defer_rr = _DEFER_RR and small and preconditioner == "makhoul"

    def _init_host_state(self) -> None:
        """Pinned host buffers, captured graphs, and the state of a solve."""
        self._host_s = _pinned(self.s.size, np.float64)
        self._host_params = _pinned(2, np.float64)
        self._history_length = 0
        self._graph = None
        self._graph_key = None
        self._start_graphs: dict[tuple, cp.cuda.graph.Graph] = {}
        self._host_start_params = _pinned(2, np.float64)
        self._check_weights = False
        self._staged = None
        self._iterations = 0

    def _setup_preconditioner(
        self, kind: str, padded_length: Callable[[int], int],
    ) -> None:
        """Set up the Neumann Poisson solve on the zero-padded grid (P1, P2).

        `"mirror"`: a periodic solve on the even extension, (2 P1, 2 P2), whose DFT
        is the DCT of the padded image; the symbol is the same.
        `"makhoul"`: DCTs by FFTs of size (P1, P2) of the reordered image, with
        twiddles. The preconditioner is C3 D C2 for the orthonormal DCTs C2, C3 =
        C2^-1 and the diagonal D; the normalizations of the DCTs cancel around D,
        so it is B D F for the unnormalized forward transform F and its inverse B.
        Both are a reordering, an FFT and a twiddle.
        """
        rows, cols = self.shape
        self.preconditioner = kind
        self.padded_shape = p1, p2 = padded_length(rows), padded_length(cols)
        cdtype = cp.complex64 if self.dtype == np.float32 else cp.complex128
        if kind == "mirror":
            grid = (2 * p1, 2 * p2)
            u = cp.arange(grid[0], dtype=cp.float64)[:, None]
            v = cp.arange(p2 + 1, dtype=cp.float64)[None, :]
            # cuFFT does not normalize: fold 1/N of the inverse transform in.
            scale = grid[0] * grid[1]
        elif kind == "makhoul":
            grid = (p1, p2)
            u = cp.arange(p1, dtype=cp.float64)[:, None]
            v = cp.arange(p2, dtype=cp.float64)[None, :]
            # B F = 2, and 1/N of the inverse FFT:
            scale = 2 * p1 * p2
            self.twiddles = [
                cp.exp(-1j * np.pi * cp.arange(n + 1) / (2 * n)).astype(cdtype)
                for n in (p1, p2)
            ]
            self.spectrum_out = cp.zeros((p1, p2 // 2 + 1), dtype=cdtype)
            # The FFT's input, which `cg_update` writes into directly. Separate
            # from `extended`, which the inverse FFT overwrites, so that its
            # padding stays zero.
            self.fft_input = cp.zeros(grid, dtype=self.dtype)
        else:
            msg = f"unknown preconditioner {kind!r}"
            raise ValueError(msg)
        symbol = 4 - 2 * cp.cos(np.pi * u / p1) - 2 * cp.cos(np.pi * v / p2)
        inverse = 1.0 / (cp.where(symbol == 0, 1.0, symbol) * scale)
        inverse[symbol == 0] = 0
        self.inverse_symbol = inverse.astype(self.dtype)
        self.extended = cp.zeros(grid, dtype=self.dtype)
        if kind == "mirror":
            self.fft_input = self.extended
        self.spectrum = cp.zeros((grid[0], grid[1] // 2 + 1), dtype=cdtype)
        self.plan_forward, self.plan_inverse = _fft_plans(self.extended, self.spectrum)

    def fill(
        self,
        wrapped: cp.ndarray,
        edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    ) -> None:
        """Set up the problem for `wrapped`, in this workspace."""
        with self._on_stream():
            if edge_weights is not None:
                self.edge_x[...] = edge_weights[0]
                self.edge_y[...] = edge_weights[1]
                self._memcpy(self.weights_x, self.edge_x)
                self._memcpy(self.weights_y, self.edge_y)
            self._prepare_targets(
                cp.ascontiguousarray(wrapped, dtype=self.dtype),
                unit_edges=edge_weights is None,
            )
        self._iterations = 0
        self._kept = None
        self._staged = None

    def fill_from_wave(self, wave: cp.ndarray, floor: float, percentile: float) -> None:
        """Set up the problem for the phase of `wave`, with amplitude edge weights.

        The same as `fill(angle(wave), amplitude_edge_weights(abs(wave), floor,
        percentile))`, in a few kernels instead of a few dozen cupy calls. Whether
        the weights are valid is checked at the next :meth:`cuts`, which
        synchronizes anyway.
        """
        with self._on_stream():
            self._enqueue_fill_from_wave(wave, floor, percentile)
        self._check_weights = True
        self._iterations = 0
        self._kept = None
        self._staged = None

    def _enqueue_fill_from_wave(
        self, wave: cp.ndarray, floor: float, percentile: float,
    ) -> None:
        rows, cols = self.shape
        n = self.size
        amplitude = self.difference  # free until the first `congruent`
        virtual = (n - 1) * (percentile / 100)  # numpy's "linear" method
        low = math.floor(virtual)
        self._launch(
            "wave_phase_amplitude",
            (wave, self.wrapped, amplitude, np.int32(n), self.partials,
             self.counter, self.s),
        )
        self._select(amplitude, (low, min(low + 1, n - 1)), mode=1, t=virtual - low)
        for mode in (0, 1):
            self._launch(
                "amplitude_edges",
                (amplitude, self.select_out, np.float64(floor), np.int32(rows),
                 np.int32(cols), np.int32(mode), self.partials, self.counter,
                 self.s, self.edge_x, self.edge_y, self.weights_x, self.weights_y),
            )
        self._prepare_targets(self.wrapped, unit_edges=False)

    def stage_wave(self, wave: cp.ndarray, floor: float, percentile: float) -> None:
        """Copy `wave` into the workspace; the start graph sets up from it."""
        with self._on_stream():
            if self.wave is None or self.wave.dtype != wave.dtype:
                self.wave = cp.empty(self.shape, dtype=wave.dtype)
            self._memcpy(self.wave, wave)
        self._stage(("wave", floor, percentile))

    def stage_phase(
        self,
        wrapped: cp.ndarray,
        edge_weights: tuple[cp.ndarray, cp.ndarray] | None,
    ) -> None:
        """Copy the wrapped phase (and edge weights) into the workspace."""
        with self._on_stream():
            self._memcpy(self.wrapped, cp.ascontiguousarray(wrapped, dtype=self.dtype))
            if edge_weights is not None:
                self.edge_x[...] = edge_weights[0]
                self.edge_y[...] = edge_weights[1]
        self._stage(("phase", edge_weights is not None))

    def _stage(self, staged: tuple) -> None:
        self._staged = staged
        self._check_weights = False
        self._iterations = 0
        self._kept = None

    def cold_start(
        self, wrapped: cp.ndarray, options: IRLSOptions,
    ) -> tuple[cp.ndarray, _Candidate, float]:
        """Start a solve without a prior; see `irls._cold_start`.

        After staging, as one CUDA graph: the setup, the rounded L2 solution as
        the first candidate (into slot 0), and the starting point. One
        synchronization.
        """
        from .irls import _cold_start  # noqa: PLC0415

        if self._staged is None:
            return _cold_start(self, wrapped, options)
        best = self._run_start(options.cold_start, options, weights_epsilon=math.inf)
        if options.cold_start == "zero" and best.cut_turns > 0:
            return self.phi, best, options.epsilon_start
        return self.phi, best, math.inf

    def warm_start(
        self,
        wrapped: cp.ndarray,
        prior: PhasePrior | BranchesPrior,
        options: IRLSOptions,
    ) -> tuple[cp.ndarray, _Candidate, float]:
        """Start a solve from a prior; see `irls._warm_start`.

        After staging, as one CUDA graph: the setup, the starting phase from the
        prior, its weights, and it rounded as the first candidate. The epsilon of
        the weights comes from pinned memory, so one graph serves every prior.
        """
        from .irls import PhasePrior, _warm_start  # noqa: PLC0415

        if self._staged is None:
            return _warm_start(self, wrapped, prior, options)
        epsilon = min(max(prior.epsilon, options.epsilon_min), options.epsilon_start)
        phase_prior = isinstance(prior, PhasePrior)
        with self._on_stream():
            if self.prior_phase is None:
                self.prior_phase = cp.empty(self.shape, dtype=cp.float64)
                self.prior_branches = cp.empty(self.shape, dtype=cp.int32)
            if phase_prior:
                target, source = self.prior_phase, prior.phase
            else:
                target, source = self.prior_branches, prior.branches
            source = cp.ascontiguousarray(cp.asarray(source), dtype=target.dtype)
            self._memcpy(target, source)
        self._host_start_params[0] = epsilon * epsilon
        self._host_start_params[1] = 0.0
        best = self._run_start(
            "phase_prior" if phase_prior else "branches_prior", options,
            weights_epsilon=epsilon,
        )
        return self.phi, best, epsilon

    def _run_start(
        self, kind: str, options: IRLSOptions, *, weights_epsilon: float,
    ) -> _Candidate:
        """Launch the start graph of `kind` for the staged input; its candidate."""
        from .irls import _Candidate  # noqa: PLC0415

        staged, self._staged = self._staged, None
        key = (staged, kind, options.epsilon_start if kind == "zero" else None)
        graph = self._start_graphs.get(key)
        if graph is None:
            graph = self._capture(lambda: self._enqueue_start(staged, kind, options))
            self._start_graphs[key] = graph
        with self._on_stream():
            graph.launch(self.stream)
            self.stream.synchronize()
        s = self._host_s
        if staged[0] == "wave" and s[_S_INVALID] > 0:
            msg = "edge_weights must be finite and greater than zero"
            raise ValueError(msg)
        phase, branches, weights_x, weights_y = self.slots[0]
        self._kept = 0
        return _Candidate(
            phase=phase, branches=branches,
            cut_turns=int(s[_S_TURNS]), cut_cost=float(s[_S_COST]),
            weights_x=weights_x, weights_y=weights_y,
            weights_epsilon=weights_epsilon, converged=True, round_index=-1,
        )

    def _enqueue_start(self, staged: tuple, kind: str, options: IRLSOptions) -> None:
        """Enqueue a start: the setup, the starting point, and the first candidate.

        `staged` is ("wave", floor, percentile) or ("phase", edge weights given);
        `kind` is "zero" or "least_squares" (a cold start), "phase_prior" or
        "branches_prior". Mirrors `irls._cold_start` and `irls._warm_start`.
        """
        rows, cols = self.shape
        dims = (np.int32(rows), np.int32(cols))
        if staged[0] == "wave":
            self._enqueue_fill_from_wave(self.wave, staged[1], staged[2])
        else:
            if staged[1]:
                self._memcpy(self.weights_x, self.edge_x)
                self._memcpy(self.weights_y, self.edge_y)
            self._prepare_targets(self.wrapped, unit_edges=not staged[1])
        phase, branches, weights_x, weights_y = self.slots[0]

        def candidate(phi: cp.ndarray) -> None:
            self._congruent_into(phi, self.wrapped, phase, branches)
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 *dims, self.partials, self.counter, self.s),
            )
            self._memcpy(weights_x, self.weights_x)
            self._memcpy(weights_y, self.weights_y)

        def reweight(params: cp.ndarray) -> None:
            self._launch(
                "charbonnier_weights",
                (self.phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
                 params, self.weights_x, self.weights_y, *dims, self.f),
            )

        if kind in ("zero", "least_squares"):
            # The rounded L2 solution is the first candidate:
            self._launch(
                "right_hand_side",
                (self.unit_x, self.target_x, self.unit_y, self.target_y, self.b,
                 *dims, self.partials, self.counter, self.s),
            )
            self._precondition(self.b, self.l2, self.f_free, mode=2)
            candidate(self.l2)
            if kind == "zero":
                # From zero, with its weights (used only if there are cuts):
                cp.cuda.runtime.memsetAsync(
                    self.phi.data.ptr, 0, self.phi.nbytes, self.stream.ptr,
                )
                eps = options.epsilon_start
                self._launch(
                    "set_params",
                    (self.direct_params, np.float64(eps * eps), np.float64(0)),
                    grid=1, block=1,
                )
                reweight(self.direct_params)
            else:
                self._memcpy(self.phi, self.l2)
        else:
            self._memcpy(self.start_params, self._host_start_params)
            self._launch(
                "starting_phase",
                (self.wrapped, self.prior_phase, self.prior_branches,
                 np.int32(kind == "branches_prior"), self.phi, np.int64(self.size)),
            )
            reweight(self.start_params)
            candidate(self.phi)
        self._memcpy(self._host_s, self.s)

    def _prepare_targets(self, wrapped: cp.ndarray, *, unit_edges: bool) -> None:
        rows, cols = self.shape
        self._launch(
            "prepare_targets",
            (wrapped, self.wrapped, self.target_x, self.target_y, self.jumps_x,
             self.jumps_y, self.edge_x, self.edge_y, self.weights_x, self.weights_y,
             np.int32(rows), np.int32(cols), np.int32(unit_edges)),
        )

    # -- launches ------------------------------------------------------------

    @contextlib.contextmanager
    def _on_stream(self) -> Iterator[None]:
        """Run on our stream, ordered after and before the caller's stream.

        Our stream first waits for the caller's, whose arrays we are about to use.
        At the end, the caller's stream waits for ours: that makes our results
        safe to use there, and it keeps the memory pool from handing memory that
        the caller frees after the call to a new array while our stream is still
        reading it. Neither costs a host synchronization.
        """
        caller = cp.cuda.get_current_stream()
        if caller == self.stream:
            yield
            return
        # A wait takes the event's state when it is enqueued, so the events can
        # be recorded again at the next call.
        self._event_caller.record(caller)
        self.stream.wait_event(self._event_caller)
        try:
            with self.stream:
                yield
        finally:
            self._event_own.record(self.stream)
            caller.wait_event(self._event_own)

    def _launch(
        self, name: str, args: tuple, grid: int | None = None, block: int | None = None,
    ) -> None:
        if self._numba is not None:
            with warnings.catch_warnings():
                # Small grids are deliberate (flags, small images); numba warns of
                # low occupancy:
                warnings.filterwarnings("ignore", message="Grid size", category=Warning)
                self._numba[name][
                    grid or self.grid, block or self.block, self._numba_stream,
                ](*self._numba_args(name, args))
            return
        self.k[name]((grid or self.grid,), (block or self.block,), args)

    def _allocate_scalars(self) -> None:
        """Allocate the device scalars, flags, and partial sums of reductions."""
        self.s = cp.zeros(10, dtype=cp.float64)
        self.f = cp.zeros(5, dtype=cp.int32)
        # Flags that are never set, for unconditional launches:
        self.f_free = cp.zeros(5, dtype=cp.int32)
        self.counter = cp.zeros(1, dtype=cp.uint32)
        self.partials = cp.zeros(2 * self.grid, dtype=cp.float64)
        # Partial sums of the deferred dot products (see `_DEFER_RZ`); r . z in
        # two buffers, which alternate: the latest and the one before.
        self.partials_rz = tuple(
            cp.zeros(self.grid, dtype=cp.float64) for _ in range(2)
        )
        self._rz_index = 0
        self.partials_pap = cp.zeros(self.grid, dtype=cp.float64)
        self.partials_rr = cp.zeros(self.grid, dtype=cp.float64)

    def _init_numba(self, preconditioner: str) -> None:
        """Use the numba kernels (see `_KERNELS`) instead of the CUDA C++ ones."""
        from numba import cuda as numba_cuda  # noqa: PLC0415

        from . import _irls_numba_cuda  # noqa: PLC0415
        if self.block != _irls_numba_cuda.BLOCK:
            msg = "the numba kernels are compiled for their own block size"
            raise ValueError(msg)
        if preconditioner != "makhoul":
            msg = "the numba kernels implement the Makhoul preconditioner only"
            raise ValueError(msg)
        self._numba = _irls_numba_cuda.kernels(self.dtype)
        self._numba_optional = _irls_numba_cuda.OPTIONAL
        self._numba_stream = numba_cuda.external_stream(self.stream.ptr)
        self._numba_arrays: dict[tuple, object] = {}
        self._numba_cuda = numba_cuda

    def _numba_args(self, name: str, args: tuple) -> tuple:
        """Return the arguments for a numba kernel.

        Arrays are wrapped once (without the stream synchronization of the array
        interface, which graph capture forbids), and absent arrays (NULL) are
        empty ones.
        """
        optional = self._numba_optional.get(name, {})
        out = []
        for index, arg in enumerate(args):
            if index in optional and isinstance(arg, np.uint64) and arg == 0:
                kind = optional[index]
                dtype = self.dtype if kind == "T" else np.dtype(kind)
                key = ("empty", dtype.str)
                if key not in self._numba_arrays:
                    self._numba_arrays[key] = self._numba_cuda.as_cuda_array(
                        cp.empty(0, dtype=dtype), sync=False,
                    )
                out.append(self._numba_arrays[key])
            elif isinstance(arg, cp.ndarray):
                key = (arg.data.ptr, arg.shape, arg.dtype.str)
                if key not in self._numba_arrays:
                    self._numba_arrays[key] = self._numba_cuda.as_cuda_array(
                        arg.ravel(), sync=False,
                    )
                out.append(self._numba_arrays[key])
            else:
                out.append(arg)
        return tuple(out)

    def _precondition(
        self,
        residual: cp.ndarray,
        out: cp.ndarray,
        flags: cp.ndarray,
        mode: int,
        *,
        reordered: bool = False,
        direction: cp.ndarray | None = None,
        rz_out: cp.ndarray | None = None,
        check: IRLSOptions | None = None,
    ) -> None:
        """Enqueue `out` = M `residual`; `mode` as in the `crop` kernel.

        With `reordered`, `cg_update` has already written the reordered residual
        into the FFT's input (Makhoul only). Modes 0 and 4 set `direction` to
        `out`; modes 3 and 4 write the partial sums of r . z to `rz_out`. With
        `check`, the spectral step first checks the convergence of the CG
        iteration from the partial sums of r . r (see `_DEFER_RR`).
        """
        rows, cols = self.shape
        p1, p2 = self.padded_shape
        size = self.extended.size
        grid = max(1, min(-(-size // self.block), 8 * self.grid))
        dims = (np.int32(rows), np.int32(cols), np.int32(p1), np.int32(p2))
        makhoul = self.preconditioner == "makhoul"
        if makhoul and not reordered:
            self._launch("reorder", (residual, self.fft_input, *dims, flags), grid=grid)
        elif not makhoul:
            self._launch("mirror", (residual, self.extended, *dims, flags), grid=grid)
        self.plan_forward.fft(self.fft_input, self.spectrum, cufft.CUFFT_FORWARD)
        if makhoul:
            history = self.history if check is not None else self.s
            self._launch(
                "makhoul_spectrum",
                (self.spectrum, self.spectrum_out, self.inverse_symbol,
                 *self.twiddles, np.int32(p1), np.int32(p2), flags,
                 self.partials_rr if check is not None else np.uint64(0),
                 np.int32(self.grid), self.s, history,
                 np.float64(check.cg_relative_tolerance if check else 0),
                 np.int32(check.cg_max_iterations if check else 0)),
                grid=grid,
            )
            self.plan_inverse.fft(self.spectrum_out, self.extended, cufft.CUFFT_INVERSE)
        else:
            self._launch(
                "scale_spectrum",
                (self.spectrum, self.inverse_symbol, np.int64(self.spectrum.size),
                 flags),
            )
            self.plan_inverse.fft(self.spectrum, self.extended, cufft.CUFFT_INVERSE)
        self._launch(
            "crop",
            (self.extended, residual, out,
             direction if direction is not None else out, *dims,
             np.int32(self.extended.shape[1]), np.int32(makhoul),
             rz_out if mode in (3, 4) else self.partials, self.counter, self.s,
             flags, np.int32(mode), np.int32(check is not None)),
        )

    def _select(
        self,
        x: cp.ndarray,
        ranks: tuple[int, int],
        *,
        mode: int,
        t: float = 0.0,
        out_t: cp.ndarray | None = None,
    ) -> None:
        """Enqueue the radix selection of the values of ranks `ranks` in `x`.

        mode 0: their mean in the dtype of `x` to `out_t` (the median, for the two
        middle ranks); mode 1: the linear interpolation with weight `t` between
        them, in float64, to `self.select_out[0]`.
        """
        n = x.size
        bits = 8 * x.dtype.itemsize
        grid = max(1, min(-(-n // (self.block * 8)), 2 * _multiprocessor_count()))
        for p in range(-(-bits // 11)):
            self._launch(
                "select_pass",
                (x, np.int32(n), np.int32(p), np.uint64(ranks[0]), np.uint64(ranks[1]),
                 self.select_state, self.select_hist, self.select_counter,
                 np.int32(mode), np.float64(t),
                 out_t if out_t is not None else self.offset, self.select_out),
                grid=grid,
            )

    def _apply(
        self,
        p: cp.ndarray,
        out: cp.ndarray,
        mode: int,
        *,
        z: cp.ndarray | None = None,
        p_new: cp.ndarray | None = None,
    ) -> None:
        """Enqueue `out` = A p; see the `apply_operator` kernel for the modes.

        In the CG iterations (modes 1 and 2), with the deferred dot products, r . z
        comes from the latest partial sums (`self._rz_index`) and the ones before.
        """
        rows, cols = self.shape
        deferred_rz = mode != 0 and self.defer_rz
        latest = self.partials_rz[self._rz_index]
        before = self.partials_rz[1 - self._rz_index]
        self._launch(
            "apply_operator",
            (p, z if z is not None else p, p_new if p_new is not None else p,
             self.weights_x, self.weights_y, out, np.int32(rows), np.int32(cols),
             self.partials, self.counter, self.s, self.f, np.int32(mode),
             np.float64(np.finfo(self.dtype).tiny), latest, before,
             np.int32(self.grid if deferred_rz else 0),
             self.partials_pap if mode != 0 and self.defer_alpha else np.uint64(0)),
        )

    def _cg_preamble(self, options: IRLSOptions) -> None:
        """Enqueue the start of a round's CG on `self.phi`."""
        rows, cols = self.shape
        n = np.int64(self.size)
        tol = np.float64(options.cg_relative_tolerance)
        phi = self.phi
        self._launch("reset_flags", (self.f,), grid=1, block=1)
        self._launch(
            "right_hand_side",
            (self.weights_x, self.target_x, self.weights_y, self.target_y, self.b,
             np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
        )
        self._launch(
            "center_and_norm",
            (self.b, n, self.partials, self.counter, self.s, self.f, self.history),
        )
        self._apply(phi, self.q, mode=0)
        self._launch(
            "initial_residual",
            (self.b, self.q, self.r, n, self.partials, self.counter, self.s, self.f,
             self.history, tol),
        )
        # z = M r; d = z; rz (as partial sums, into the first buffer, if deferred)
        self._direction = 0
        self._rz_index = 0
        self._precondition(
            self.r, self.z, self.f, mode=4 if self.defer_rz else 0,
            direction=self.directions[0], rz_out=self.partials_rz[0],
        )

    def _cg_iterations(
        self, count: int, options: IRLSOptions, *, first: bool,
    ) -> None:
        """Enqueue `count` CG iterations, then the mean removal once CG stopped.

        Iterations after CG has stopped (converged, or at the limit) do nothing,
        except for the FFTs, which cannot be skipped from the device. Each
        iteration but the `first` after the preamble computes its direction from
        the last one, in the other direction buffer, and the deferred r . z
        alternates between its two buffers likewise: `count` must be even for a
        graph that is replayed after itself.
        """
        _, cols = self.shape
        p1, p2 = self.padded_shape
        n = np.int64(self.size)
        tol = np.float64(options.cg_relative_tolerance)
        limit = np.int32(options.cg_max_iterations)
        makhoul = self.preconditioner == "makhoul"
        null = np.uint64(0)  # a NULL pointer argument
        reorder_into = self.fft_input if makhoul else null
        phi = self.phi
        for i in range(count):
            current = self.directions[self._direction]
            if first and i == 0:  # d = z, from the preamble
                self._apply(current, self.q, mode=1)  # q = A d; alpha
            else:  # d = beta d + z; q = A d; alpha
                self._direction = 1 - self._direction
                self._apply(
                    current, self.q, mode=2,
                    z=self.z, p_new=self.directions[self._direction],
                )
            d = self.directions[self._direction]
            latest_rz = self.partials_rz[self._rz_index]
            self._launch(
                "cg_update",
                (phi, d, self.r, self.q, n, self.partials, self.counter, self.s,
                 self.f, self.history, tol, limit, reorder_into, np.int32(cols),
                 np.int32(p1), np.int32(p2),
                 self.partials_pap if self.defer_alpha else null,
                 latest_rz if self.defer_rz else null, np.int32(self.grid),
                 np.float64(np.finfo(self.dtype).tiny),
                 self.partials_rr if self.defer_rr else null),
            )
            # z = M r; r . z: beta now, or the partial sums, into the other buffer
            self._precondition(
                self.r, self.z, self.f, mode=3 if self.defer_rz else 1,
                reordered=makhoul, rz_out=self.partials_rz[1 - self._rz_index],
                check=options if self.defer_rr else None,
            )
            if self.defer_rz:
                self._rz_index = 1 - self._rz_index
        self._launch("mean", (phi, n, self.partials, self.counter, self.s, self.f))
        self._launch("subtract_mean_if_iterated", (phi, n, self.s, self.f))

    def _capture(self, enqueue: Callable[[], None]) -> cp.cuda.graph.Graph:
        with self.stream:
            self.stream.begin_capture()
            try:
                enqueue()
            finally:
                graph = self.stream.end_capture()
        return graph

    def _memcpy(
        self, dst: cp.ndarray | np.ndarray, src: cp.ndarray | np.ndarray,
    ) -> None:
        """Enqueue a copy between device and/or pinned host arrays (capturable)."""
        if src.nbytes == 0:
            return
        kinds = {(True, True): 3, (True, False): 2, (False, True): 1}  # cudaMemcpyKind
        on_device = (isinstance(src, cp.ndarray), isinstance(dst, cp.ndarray))
        pointer = lambda a: a.data.ptr if isinstance(a, cp.ndarray) else a.ctypes.data  # noqa: E731
        cp.cuda.runtime.memcpyAsync(
            pointer(dst), pointer(src), src.nbytes, kinds[on_device], self.stream.ptr,
        )

    def _round_tail(self, slot: int) -> None:
        """Enqueue the end of a round, after its CG.

        Reweight (unless settled), round onto the branches, count the cuts, keep
        the weights with the candidate (all into `slot`), and copy the report to
        the host.
        """
        rows, cols = self.shape
        phase, branches, weights_x, weights_y = self.slots[slot]
        self._launch(
            "charbonnier_weights",
            (self.phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
             self.round_params, self.weights_x, self.weights_y,
             np.int32(rows), np.int32(cols), self.f),
        )
        self._congruent_into(self.phi, self.wrapped, phase, branches)
        self._launch(
            "cuts",
            (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
             np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
        )
        self._memcpy(weights_x, self.weights_x)
        self._memcpy(weights_y, self.weights_y)
        self._memcpy(self._host_flags, self.f)
        self._memcpy(self._host_history, self.history)
        self._memcpy(self._host_s, self.s)

    def _prepare_graph(self, options: IRLSOptions) -> None:
        """Capture a whole round as CUDA graphs.

        With up to `_GRAPH_ITERATIONS` CG iterations, one graph: the round's
        parameters from pinned memory, the CG, and the round's tail. With more, the
        first graph stops after that many iterations, `_graph_more` runs as many
        more, and the tail graph the tail. The graphs with the tail exist once per
        candidate slot.
        """
        key = (options.cg_max_iterations, options.cg_relative_tolerance)
        if self._graph is not None and self._graph_key == key:
            return
        if self._history_length < options.cg_max_iterations + 1:
            self._history_length = options.cg_max_iterations + 1
            with self.stream:
                self.history = cp.zeros(self._history_length, dtype=cp.float64)
            self._host_flags = _pinned(self.f.size, np.int32)
            self._host_history = _pinned(self.history.size, np.float64)
        chunk = min(options.cg_max_iterations, _GRAPH_ITERATIONS)
        single = options.cg_max_iterations <= chunk
        # A replayed chunk must leave the direction in the buffer it found it in:
        assert single or chunk % 2 == 0  # noqa: S101

        def first(slot: int) -> None:
            self._memcpy(self.round_params, self._host_params)
            self._cg_preamble(options)
            self._cg_iterations(chunk, options, first=True)
            if single:
                self._round_tail(slot)

        # Both first graphs end with the direction in the same buffer, where the
        # more graph starts.
        self._graphs = tuple(self._capture(lambda s=s: first(s)) for s in (0, 1))
        self._graph_more = (
            None if single
            else self._capture(lambda: self._cg_iterations(chunk, options, first=False))
        )
        self._graph_tails = (
            None if single
            else tuple(self._capture(lambda s=s: self._round_tail(s)) for s in (0, 1))
        )
        self._graph = self._graphs[0]
        self._graph_key = key

    # -- the interface of _WeightedProblem -----------------------------------

    def run_round(
        self,
        phi: cp.ndarray,
        wrapped: cp.ndarray,  # noqa: ARG002 (filled in by `fill`)
        *,
        epsilon: float,
        at_end: bool,
        options: IRLSOptions,
        index: int,
    ) -> tuple[CGRound, _Candidate]:
        """Run one IRLS round on `phi` (in place); see `_WeightedProblem.run_round`.

        One graph launch and one synchronization. The candidate's arrays are a
        workspace slot, valid until the next round unless it is kept (:meth:`keep`),
        and until the end of the solve (:meth:`release`) if it is.
        """
        from .irls import CGRound, _Candidate  # noqa: PLC0415

        self._prepare_graph(options)
        # Read by the graph when it runs; the last round has synchronized.
        self._host_params[0] = epsilon * epsilon
        self._host_params[1] = float(at_end)
        slot = 1 if self._kept == 0 else 0
        with self._on_stream():
            own = phi.data.ptr == self.phi.data.ptr
            if not own:
                self._memcpy(self.phi, phi)
            self._graphs[slot].launch(self.stream)
            if self._graph_more is not None:
                while True:  # more iterations than one graph holds
                    self._memcpy(self._host_flags, self.f)
                    self.stream.synchronize()
                    if self._host_flags[_F_DONE]:
                        break
                    self._graph_more.launch(self.stream)
                self._graph_tails[slot].launch(self.stream)
            if not own:
                self._memcpy(phi, self.phi)
            self.stream.synchronize()
        flags = self._host_flags
        iterations = int(flags[_F_ITER])
        self._iterations = iterations
        if flags[_F_ZERO]:
            report = CGRound(iterations=0, residuals=[0.0], converged=True)
        else:
            report = CGRound(
                iterations=iterations,
                residuals=[float(x) for x in self._host_history[: iterations + 1]],
                converged=bool(flags[_F_DONE] == 1),
            )
        phase, branches, weights_x, weights_y = self.slots[slot]
        candidate = _Candidate(
            phase=phase,
            branches=branches,
            cut_turns=int(self._host_s[_S_TURNS]),
            cut_cost=float(self._host_s[_S_COST]),
            weights_x=weights_x,
            weights_y=weights_y,
            weights_epsilon=math.nan,  # set by the caller
            converged=False,  # set by the caller
            round_index=index,
        )
        return report, candidate

    def _slot_of(self, candidate: _Candidate) -> int | None:
        for index, slot in enumerate(self.slots):
            if candidate.phase is slot[0]:
                return index
        return None

    def keep(self, candidate: _Candidate) -> _Candidate:
        """Keep a candidate past the next round: protect its slot from it."""
        slot = self._slot_of(candidate)
        if slot is not None:
            self._kept = slot
        return candidate

    def release(self, candidate: _Candidate) -> _Candidate:
        """Copy the result out of the workspace, which later solves reuse."""
        if self._slot_of(candidate) is None:
            return candidate
        self._kept = None
        with self._on_stream():
            arrays = {}
            for name in ("phase", "branches", "weights_x", "weights_y"):
                source = getattr(candidate, name)
                arrays[name] = cp.empty_like(source)
                self._memcpy(arrays[name], source)
            return dataclasses.replace(candidate, **arrays)

    def cuts(self, branches: cp.ndarray) -> tuple[int, float]:
        """Count the cut turns of `branches`, and their cost. Synchronizes."""
        rows, cols = self.shape
        with self._on_stream():
            self._launch(
                "cuts",
                (branches, self.jumps_x, self.jumps_y, self.edge_x, self.edge_y,
                 np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
            )
            s = self.s.get(stream=self.stream)
        if self._check_weights:
            self._check_weights = False
            if s[_S_INVALID] > 0:
                msg = "edge_weights must be finite and greater than zero"
                raise ValueError(msg)
        return int(s[_S_TURNS]), float(s[_S_COST])

    def reweight(
        self, phi: cp.ndarray, epsilon: float, *, unless_settled: bool = False,
    ) -> None:
        """Set the weights from `phi` at `epsilon`.

        With `unless_settled`, only if the last :meth:`solve` ran an iteration;
        decided on the device.
        """
        rows, cols = self.shape
        with self._on_stream():
            self._launch(
                "set_params",
                (self.direct_params, np.float64(epsilon * epsilon),
                 np.float64(unless_settled)),
                grid=1, block=1,
            )
            self._launch(
                "charbonnier_weights",
                (phi, self.target_x, self.target_y, self.edge_x, self.edge_y,
                 self.direct_params, self.weights_x, self.weights_y,
                 np.int32(rows), np.int32(cols), self.f),
            )

    def l2_solution(self) -> cp.ndarray:
        """Solve the unweighted problem, approximately; see `_WeightedProblem`."""
        rows, cols = self.shape
        with self._on_stream():
            out = cp.empty(self.shape, dtype=self.dtype)
            self._launch(
                "right_hand_side",
                (self.unit_x, self.target_x, self.unit_y, self.target_y, self.b,
                 np.int32(rows), np.int32(cols), self.partials, self.counter, self.s),
            )
            self._precondition(self.b, out, self.f_free, mode=2)
        return out

    def congruent(
        self, phi: cp.ndarray, wrapped: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Snap `phi` onto the branches of `wrapped`, see `irls._congruent`."""
        with self._on_stream():
            branches = cp.empty(self.shape, dtype=cp.int32)
            phase = cp.empty(self.shape, dtype=cp.float64)
            self._congruent_into(phi, wrapped, phase, branches)
        return phase, branches

    def _congruent_into(
        self,
        phi: cp.ndarray,
        wrapped: cp.ndarray,
        phase: cp.ndarray,
        branches: cp.ndarray,
    ) -> None:
        tau = self.dtype.type(2 * np.pi)
        n = np.int64(self.size)
        self._launch("wrapped_difference", (phi, wrapped, tau, self.difference, n))
        # The median, as np.median: the mean of the two middle values.
        self._select(
            self.difference, ((self.size - 1) // 2, self.size // 2),
            mode=0, out_t=self.offset,
        )
        self._launch(
            "branches_and_phase",
            (phi, wrapped, self.offset, tau, branches, phase, n),
        )
