"""The kernels of the GPU solvers (IRLS and ADMM), in numba CUDA.

Launched by `CupyWeightedProblem` in `_irls_cupy.py`; an absent (optional) array
argument is an empty array. The IRLS kernels give the CPU solver's results bit for
bit; for ADMM, see `_admm_numba`.

What keeps numba's code fast, found with Nsight Compute profiles and the SASS:

* **32-bit index arithmetic, truncated after every operation.** numba computes
  integer arithmetic at the machine width: `uint32 + uint32` is a 64-bit addition,
  and a `//` of 64-bit values takes a slow path. Every integer result that feeds
  an index is therefore wrapped in `u32(...)`: truncated, so that LLVM narrows the
  arithmetic back to 32 bits. Unsigned, because with signed integers numba also
  gives every array access Python's handling of negative indices, and `//` and `%`
  Python's rounding. Plain Python integer constants are int64 and would widen the
  arithmetic again: hence the typed constants `U0`, `U1`, `U2`, `UBLOCK`. Without
  this, the per-pixel kernels executed 2.4-2.6x the instructions and ran up to 2x
  as long. Narrowing cannot repair a comparison, `//` or `%` of a widened value,
  so loop bounds such as `u32(u32(rows) * cols)` are truncated too (a 64-bit loop
  test and a 64-bit `%` per element cost ~10 % in `apply_operator` and
  `makhoul_spectrum`).
* **Vector loads and stores of complex values** (`_cload`, `_cstore`): numba
  aligns a complex element as its components and moves the real and imaginary
  parts separately, which made `makhoul_spectrum` touch ~40 % more L2 sectors.
* **No fused multiply-adds.** NVVM contracts `a * b + c` into an FMA, and
  numba-cuda has no switch for that. Products that are added or subtracted are
  computed with round-to-nearest intrinsics (`mul`, `_dmul`: libdevice's
  `fmul_rn`, `dmul_rn`), which are never fused: so the results match the CPU
  solver's bit for bit. Allowing FMAs gained at most ~1 %.
* **One loop per mode.** numba does not hoist a loop-invariant test out of a loop;
  `apply_operator` has a loop for each of its modes.
* **`ld.global.cg` for the last block's sum of partial sums** (`cuda.ldcg`): L1 is
  not coherent between blocks, L2 is. numba has no `volatile` loads, and atomic
  reads cost ~1 us per reduction.
* **Short reductions without loops.** A deferred sum (partial sums that a later
  kernel adds up, see `_irls_cupy`) has at most two partial sums per thread, added
  in straight-line code (`_sum_partials`): LLVM's unrolled loop cost branches,
  and together with a barrier after `_ticket_last`'s flag, which is not needed,
  ~0.3 us per kernel, a few percent at ~500^2.
* **No launch bounds.** They do not help any kernel, and `makhoul_spectrum` spills
  registers under them.

Launching a numba kernel from Python costs ~40 us of host time (cupy: ~3 us), in
numba's argument handling, even for a compiled kernel launched directly. So these
kernels run only inside captured CUDA graphs, including the setup of a solve
(`CupyWeightedProblem.cold_start`, `warm_start`): launched from the host, they
made warm starts ~70 % slower. Capturing still pays the launch cost, once per
problem: ~25 ms for the first solve of a shape.

Compiling all kernels for one dtype takes ~7 s, so they are cached on disk
(`cache=True`); a fresh process then loads them in ~0.2 s, mostly numba's own
initialization. (The cache key hashes the bytecode, which differs between a module
compiled from source and one loaded from its .pyc: after an edit, the first two
processes compile.)
"""
# Kernels take their arrays as positional arguments, and T is a type. Modes are
# plain numbers, and plain comparisons keep numba's code simple (no tuple
# membership tests, no min/max with their NaN handling):
# ruff: noqa: PLR0917, N806, C901, PLR0912, PLR0915, ANN001, ANN202, PLR2004, PLR1714, FURB136
from __future__ import annotations

import functools
import math

import numba
import numpy as np
from llvmlite import ir
from numba import cuda, types
from numba.cuda import libdevice
from numba.cuda.extending import intrinsic

BLOCK = 256
WARPS = BLOCK // 32
SHARED_SUM = WARPS + 1  # shared array sizes must be plain constants
BUCKETS = 2048  # of the radix selection
DIGIT_BITS = 11
FULL_MASK = 0xFFFFFFFF
# Indices into the int flags `f` and the double scalars `s` on the device, as in
# `_irls_cupy`. F_PENDING: `makhoul_spectrum` checked an iteration's residual, and
# `crop` has yet to count that iteration.
F_DONE, F_ITER, F_ITERATED, F_ZERO, F_PENDING = range(5)
# S_INVALID: the number of edges whose weight is not finite and positive;
# S_INVALID_PIXELS: the number of invalid pixels (input not finite); S_MASK_SET: 1
# if the mask of invalid pixels has any set (see `prepare_targets`).
(S_NORM, S_RZ, S_ALPHA, S_BETA, S_MEAN, S_TURNS, S_COST, S_LARGEST, S_INVALID,
 S_INVALID_PIXELS, S_MASK_SET) = range(11)

u32 = numba.uint32
u64 = numba.uint64
U0 = u32(0)
U1 = u32(1)
U2 = u32(2)
UBLOCK = u32(BLOCK)

_dmul = libdevice.dmul_rn


def _element_pointer(context, builder, aryty, ary, idx):
    data = context.make_array(aryty)(context, builder, ary).data
    if idx.type.width < 64:
        idx = builder.zext(idx, ir.IntType(64))
    return builder.gep(data, [idx])


@intrinsic
def _cload(_typingctx, arr, idx):
    """`arr[idx]` of a complex array, aligned as float2/double2: one vector load.

    numba aligns a complex access as its components, so it loads and stores the
    real and imaginary parts separately, which costs L2 sectors.
    """
    def codegen(context, builder, sig, args):
        aryty = sig.args[0]
        ptr = _element_pointer(context, builder, aryty, args[0], args[1])
        size = context.get_abi_sizeof(context.get_data_type(aryty.dtype))
        return builder.load(ptr, align=size)
    return arr.dtype(arr, idx), codegen


@intrinsic
def _cstore(_typingctx, arr, idx, value):
    """`arr[idx] = value` for a complex array: one vector store (see `_cload`)."""
    def codegen(context, builder, sig, args):
        aryty, _, valty = sig.args
        ptr = _element_pointer(context, builder, aryty, args[0], args[1])
        size = context.get_abi_sizeof(context.get_data_type(aryty.dtype))
        value = context.cast(builder, args[2], valty, aryty.dtype)
        builder.store(value, ptr, align=size)
        return context.get_dummy_value()
    return types.none(arr, idx, value), codegen


@cuda.jit(device=True, inline=True)
def _block_sum(v):
    sh = cuda.shared.array(SHARED_SUM, numba.float64)
    o = 16
    while o > 0:
        v += cuda.shfl_down_sync(FULL_MASK, v, o)
        o >>= 1
    lane = u32(u32(cuda.threadIdx.x) & u32(31))
    warp = u32(u32(cuda.threadIdx.x) >> u32(5))
    if lane == 0:
        sh[warp] = v
    cuda.syncthreads()
    if warp == 0:
        r = sh[lane] if lane < u32(WARPS) else 0.0
        o = 16
        while o > 0:
            r += cuda.shfl_down_sync(FULL_MASK, r, o)
            o >>= 1
        if lane == 0:
            sh[WARPS] = r
    cuda.syncthreads()
    r = sh[WARPS]
    cuda.syncthreads()
    return r


@cuda.jit(device=True, inline=True)
def _block_max(v):
    sh = cuda.shared.array(SHARED_SUM, numba.float64)
    o = 16
    while o > 0:
        v = libdevice.fmax(v, cuda.shfl_down_sync(FULL_MASK, v, o))
        o >>= 1
    lane = u32(u32(cuda.threadIdx.x) & u32(31))
    warp = u32(u32(cuda.threadIdx.x) >> u32(5))
    if lane == 0:
        sh[warp] = v
    cuda.syncthreads()
    if warp == 0:
        r = sh[lane] if lane < u32(WARPS) else sh[0]
        o = 16
        while o > 0:
            r = libdevice.fmax(r, cuda.shfl_down_sync(FULL_MASK, r, o))
            o >>= 1
        if lane == 0:
            sh[WARPS] = r
    cuda.syncthreads()
    r = sh[WARPS]
    cuda.syncthreads()
    return r


@cuda.jit(device=True, inline=True)
def _block_partial(v, partials):
    b = _block_sum(v)
    if cuda.threadIdx.x == 0:
        partials[u32(cuda.blockIdx.x)] = b


@cuda.jit(device=True, inline=True)
def _sum_partials(partials, count):
    """Return the sum of `count` partial sums, in the order of `_last_block`.

    The first two per thread are added without a loop, since a deferred sum has
    at most 2 * BLOCK partial sums: LLVM's own unrolling of the loop costs a few
    branches.
    """
    a = 0.0
    k = u32(cuda.threadIdx.x)
    n = u32(count)
    if k < n:
        a += partials[k]
    k = u32(k + UBLOCK)
    if k < n:
        a += partials[k]
    k = u32(k + UBLOCK)
    while k < n:
        a += partials[k]
        k = u32(k + UBLOCK)
    return _block_sum(a)


@cuda.jit(device=True, inline=True)
def _ticket_last(counter):
    """Take a ticket: return whether this block is the last one to finish.

    Thread 0 takes the ticket from `counter`; all threads get the answer.
    """
    flag = cuda.shared.array(1, numba.int32)
    if cuda.threadIdx.x == 0:
        cuda.threadfence()
        ticket = cuda.atomic.add(counter, 0, U1)
        flag[0] = 1 if ticket == u32(u32(cuda.gridDim.x) - U1) else 0
    cuda.syncthreads()
    return flag[0] != 0  # written once per launch: no barrier needed after


@cuda.jit(device=True, inline=True)
def _coherent_sum(partials, offset):
    """In the last block: the sum of all blocks' partial sums, from partials[offset:].

    Read with `ld.global.cg`, which caches in L2 only: L1 is not coherent between
    blocks, and L2 holds what the other blocks wrote before their ticket.
    """
    a = 0.0
    k = u32(cuda.threadIdx.x)
    n = u32(cuda.gridDim.x)
    while k < n:
        a += cuda.ldcg(partials, u32(offset + k))
        k = u32(k + UBLOCK)
    return _block_sum(a)


@cuda.jit(device=True, inline=True)
def _last_block(v, partials, counter):
    """Sum v over the grid: return (whether this is the last block, the total).

    Each block writes its partial sum into `partials`; the last block to finish
    adds them up (the total is 0.0 in the other blocks) and resets `counter`.
    """
    b = _block_sum(v)
    if cuda.threadIdx.x == 0:
        partials[u32(cuda.blockIdx.x)] = b
    if not _ticket_last(counter):
        return False, 0.0
    cuda.threadfence()
    total = _coherent_sum(partials, U0)
    if cuda.threadIdx.x == 0:
        counter[0] = U0
    return True, total


@cuda.jit(device=True, inline=True)
def _last_block2(v0, v1, partials, counter):
    """`_last_block` for two values: (last block, total0, total1)."""
    b0 = _block_sum(v0)
    b1 = _block_sum(v1)
    if cuda.threadIdx.x == 0:
        partials[u32(cuda.blockIdx.x)] = b0
        partials[u32(u32(cuda.gridDim.x) + u32(cuda.blockIdx.x))] = b1
    if not _ticket_last(counter):
        return False, 0.0, 0.0
    cuda.threadfence()
    t0 = _coherent_sum(partials, U0)
    t1 = _coherent_sum(partials, u32(cuda.gridDim.x))
    if cuda.threadIdx.x == 0:
        counter[0] = U0
    return True, t0, t1


@cuda.jit(device=True, inline=True)
def _reordered(n, length):
    """Where the Makhoul reordering puts index n of an axis of `length` (unsigned)."""
    return u32(n // U2) if u32(n % U2) == U0 else u32(length - u32(u32(n + U1) // U2))


@cuda.jit(device=True, inline=True)
def _source(m, length):
    """Return the index that the Makhoul reordering puts at m (unsigned)."""
    if m < u32(u32(length + U1) // U2):
        return u32(U2 * m)
    return u32(U2 * u32(length - m) - U1)


@cuda.jit(device=True, inline=True)
def _relative_amplitude(a, ref, floor):
    if not ref > 0:
        return 1.0
    r = a / ref
    if math.isnan(r):
        return r
    return floor if r < floor else (1.0 if r > 1.0 else r)


@cuda.jit(device=True, inline=True)
def _nan_minimum(a, b):
    if math.isnan(a) or math.isnan(b):
        return a + b
    return a if a < b else b


@cuda.jit(device=True, inline=True)
def _key32(v):
    u = u32(libdevice.float_as_int(v))
    return u64(~u) if u & u32(0x80000000) else u64(u | u32(0x80000000))


@cuda.jit(device=True, inline=True)
def _value32(key):
    k = u32(key & u64(0xFFFFFFFF))
    bits = (k & u32(0x7FFFFFFF)) if k & u32(0x80000000) else ~k
    return libdevice.int_as_float(numba.int32(bits))


@cuda.jit(device=True, inline=True)
def _key64(v):
    u = u64(libdevice.double_as_longlong(v))
    top = u64(1) << u64(63)
    return ~u if u & top else u | top


@cuda.jit(device=True, inline=True)
def _value64(key):
    top = u64(1) << u64(63)
    bits = (key & ~top) if key & top else ~key
    return libdevice.longlong_as_double(numba.int64(bits))


@cuda.jit(device=True, inline=True)
def _find_bucket(hist, offset, k, found, found_rank, which, sh):
    """In the last block: the bucket of rank k in hist[offset:offset + BUCKETS].

    Writes the bucket to found[which], and k's rank within it to found_rank[which].
    Plain loads suffice here: the blocks wrote the histogram only with atomics,
    which act in L2, and this block has not read it before.
    """
    per = u32(BUCKETS // BLOCK)
    tid = u32(cuda.threadIdx.x)
    base = u32(u32(offset) + tid * per)
    own = u64(0)
    for b in range(per):
        own += u64(hist[u32(base + u32(b))])
    sh[tid] = own
    cuda.syncthreads()
    off = U1
    while off < UBLOCK:
        v = sh[u32(tid - off)] if tid >= off else u64(0)
        cuda.syncthreads()
        sh[tid] += v
        cuda.syncthreads()
        off = u32(off << U1)
    before = sh[tid] - own
    if before <= k < sh[tid]:
        r = k - before
        for b in range(per):
            c = u64(hist[u32(base + u32(b))])
            if r < c:
                found[which] = u32(tid * per + u32(b))
                found_rank[which] = r
                break
            r -= c
    cuda.syncthreads()


def kernels(dtype: np.dtype) -> dict:
    """Return the kernels for `dtype` (float32 or float64), by name."""
    return _kernels(np.dtype(dtype))[0]


def signatures(dtype: np.dtype) -> dict[str, tuple]:
    """Return the argument types of the kernels for `dtype`, by name.

    What `CupyWeightedProblem` launches them with, so the types they compile for:
    one signature per kernel and dtype.
    """
    return _kernels(np.dtype(dtype))[1]


@functools.cache
def _kernels(dtype: np.dtype) -> dict:
    single = dtype == np.float32
    T = numba.float32 if single else numba.float64
    BITS = 32 if single else 64
    # Products are rounded before they are added (no FMAs), which keeps the
    # results bit-identical with the CPU:
    mul = libdevice.fmul_rn if single else libdevice.dmul_rn
    _dmul = libdevice.dmul_rn
    rint = libdevice.rintf if single else libdevice.rint
    key = _key32 if single else _key64
    value = _value32 if single else _value64
    all_bits = u64(0xFFFFFFFF) if single else u64(0xFFFFFFFFFFFFFFFF)
    compiled = {}
    signatures = {}
    # Argument types: arrays (C-contiguous, 1-d) of T, of the complex type of T,
    # and of other dtypes; scalars.
    A, CA = T[::1], (numba.complex64 if single else numba.complex128)[::1]
    F64A, I32A, U32A, U64A = (t[::1] for t in (numba.float64, numba.int32, u32, u64))
    U8A = numba.uint8[::1]
    i32, i64, f64 = numba.int32, numba.int64, numba.float64

    def kernel(*arg_types: types.Type):
        """Declare a kernel with its argument types (see `signatures`)."""
        def declare(func):
            # Compiled lazily, and cached on disk: the index key covers the source
            # file and the closure (the dtype), but not options of
            # `cuda.jit`, which must not vary.
            compiled[func.__name__] = cuda.jit(cache=True)(func)
            signatures[func.__name__] = arg_types
            return compiled[func.__name__]
        return declare

    def start():
        return u32(u32(cuda.blockIdx.x) * UBLOCK + u32(cuda.threadIdx.x))

    start = cuda.jit(device=True, inline=True)(start)

    @cuda.jit(device=True, inline=True)
    def stride():
        return u32(u32(cuda.gridDim.x) * UBLOCK)

    @cuda.jit(device=True, inline=True)
    def stencil_plain(p, wx, wy, out, rows, cols):
        """`apply_operator`'s loop for p as it is; returns this thread's p . Ap."""
        acc = 0.0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            v = T(0)
            c = p[k]
            if j > U0:
                v += mul(wx[u32(i * cm1 + j - U1)], c - p[u32(k - U1)])
            if j < cm1:
                v -= mul(wx[u32(i * cm1 + j)], p[u32(k + U1)] - c)
            if i > U0:
                v += mul(wy[u32((i - U1) * cols_u + j)], c - p[u32(k - cols_u)])
            if i < rm1:
                v -= mul(wy[u32(i * cols_u + j)], p[u32(k + cols_u)] - c)
            out[k] = v
            acc += _dmul(numba.float64(c), numba.float64(v))
            k = u32(k + step)
        return acc

    @cuda.jit(device=True, inline=True)
    def next_direction(p, z, beta, k):
        return mul(p[k], beta) + z[k]

    @cuda.jit(device=True, inline=True)
    def stencil_update(p, z, beta, p_new, wx, wy, out, rows, cols):
        """`stencil_plain` for the next direction p' = beta p + z, also into p_new."""
        acc = 0.0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            v = T(0)
            c = next_direction(p, z, beta, k)
            if j > U0:
                left = next_direction(p, z, beta, u32(k - U1))
                v += mul(wx[u32(i * cm1 + j - U1)], c - left)
            if j < cm1:
                right = next_direction(p, z, beta, u32(k + U1))
                v -= mul(wx[u32(i * cm1 + j)], right - c)
            if i > U0:
                up = next_direction(p, z, beta, u32(k - cols_u))
                v += mul(wy[u32((i - U1) * cols_u + j)], c - up)
            if i < rm1:
                down = next_direction(p, z, beta, u32(k + cols_u))
                v -= mul(wy[u32(i * cols_u + j)], down - c)
            out[k] = v
            p_new[k] = c
            acc += _dmul(numba.float64(c), numba.float64(v))
            k = u32(k + step)
        return acc

    @kernel(I32A)
    def reset_flags(f):
        """Clear the flags of a CG solve."""
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            f[F_DONE] = 0
            f[F_ITER] = 0
            f[F_ITERATED] = 0
            f[F_ZERO] = 0
            f[F_PENDING] = 0

    @kernel(F64A, f64, f64)
    def set_params(p, a, b):
        """p[0], p[1] = a, b (as a node of a captured graph)."""
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            p[0] = a
            p[1] = b

    @kernel(
        A, A, A, A, A, A, i32, i32, F64A, U32A, F64A, I32A, i32, f64, F64A, F64A, i32,
        F64A,
    )
    def apply_operator(p, z, p_new, wx, wy, out, rows, cols, partials, counter, s, f,
                       mode, tiny, rz_new, rz_old, rz_count, pap_out):
        """Compute out = -div(w grad p), the operator of `irls._WeightedProblem`.

        mode 0: only that. mode 1: also alpha = rz / (p . out) into s[S_ALPHA];
        p . out <= `tiny` stops CG. mode 2: as 1, for the next CG direction
        p' = beta p + z instead of p, computed at each point of the stencil (with
        `next_direction`, so with the same result) and written to `p_new` (a
        separate buffer: neighbours still read the old p). With `rz_count` > 0,
        rz is not s[S_RZ] but the sum of the `rz_count` partial sums in `rz_new`
        (from the last `crop`), and beta = rz / (the sum of those in `rz_old`);
        every block adds them up itself (see `_sum_partials`). With `pap_out`, the
        blocks only write their partial sums of p . out there, and `cg_update`
        computes alpha.
        """
        if f[F_DONE]:
            return
        update = mode == 2
        beta = T(0)
        if rz_count > 0:
            rz = _sum_partials(rz_new, rz_count)
            if update:
                beta = T(rz / _sum_partials(rz_old, rz_count))
        else:
            rz = s[S_RZ]
            if update:
                beta = T(s[S_BETA])
        # One loop per mode, decided once: numba does not hoist the test out of
        # the loop by itself, which costs ~30 % more instructions.
        if update:
            acc = stencil_update(p, z, beta, p_new, wx, wy, out, rows, cols)
        else:
            acc = stencil_plain(p, wx, wy, out, rows, cols)
        if mode == 0:
            return
        if pap_out.size > 0:
            _block_partial(acc, pap_out)
            return
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            if abs(total) <= tiny:
                f[F_DONE] = 1
            else:
                s[S_ALPHA] = numba.float64(T(rz / total))

    @kernel(A, A, A, A, A, i32, i32, F64A, U32A, F64A)
    def right_hand_side(wx, tx, wy, ty, b, rows, cols, partials, counter, s):
        """Compute b = -div(w g), and its mean into s[S_MEAN]."""
        acc = 0.0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            v = T(0)
            if j > U0:
                e = u32(i * cm1 + j - U1)
                v += mul(wx[e], tx[e])
            if j < cm1:
                e = u32(i * cm1 + j)
                v -= mul(wx[e], tx[e])
            if i > U0:
                e = u32((i - U1) * cols_u + j)
                v += mul(wy[e], ty[e])
            if i < rm1:
                v -= mul(wy[k], ty[k])
            b[k] = v
            acc += numba.float64(v)
            k = u32(k + step)
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_MEAN] = total / numba.float64(n)

    @kernel(A, i64, F64A, U32A, F64A, I32A, F64A)
    def center_and_norm(b, n, partials, counter, s, f, history):
        """Center b, and its norm into s[S_NORM]. A zero norm stops the solve."""
        mean = T(s[S_MEAN])
        acc = 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            v = b[k] - mean
            b[k] = v
            acc += _dmul(numba.float64(v), numba.float64(v))
            k = u32(k + step)
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            norm = math.sqrt(total)
            s[S_NORM] = norm
            if not norm > 0:
                f[F_DONE] = 1
                f[F_ZERO] = 1
                history[0] = 0.0

    @kernel(A, A, A, i64, F64A, U32A, F64A, I32A, F64A, f64)
    def initial_residual(b, q, r, n, partials, counter, s, f, history, tol):
        """Compute r = b - q, and the initial relative residual."""
        if f[F_DONE]:
            return
        acc = 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            v = b[k] - q[k]
            r[k] = v
            acc += _dmul(numba.float64(v), numba.float64(v))
            k = u32(k + step)
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            rel = math.sqrt(total) / s[S_NORM]
            history[0] = rel
            if rel <= tol:
                f[F_DONE] = 1
            else:
                f[F_ITERATED] = 1

    @kernel(
        A, A, A, A, i64, F64A, U32A, F64A, I32A, F64A, f64, i32, A, i32, i32, i32,
        F64A, F64A, i32, f64, F64A,
    )
    def cg_update(phi, d, r, q, n, partials, counter, s, f, history, tol, limit,
                  v, cols, p1, p2, pap, rz_cur, count, tiny, rr_out):
        """Update phi += alpha d, r -= alpha q; then the relative residual of r.

        Stops with F_DONE = 1 (converged) at the tolerance, with F_DONE = 2 at
        `limit` iterations. Also writes the new r to its place in the
        Makhoul-reordered (P1, P2) array v, the input of the preconditioner's FFT
        (whose padding stays zero). With `pap`, alpha = rz / (p . Ap) comes from
        the partial sums of p . Ap (and of r . z in `rz_cur`, else s[S_RZ]), which
        every block adds up itself; p . Ap <= `tiny` stops CG in every block
        alike. With `rr_out`, the blocks only write their partial sums of r . r
        there, and `makhoul_spectrum` checks convergence.
        """
        if f[F_DONE]:
            return
        if pap.size > 0:
            curvature = _sum_partials(pap, count)
            if abs(curvature) <= tiny:
                if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
                    f[F_DONE] = 1
                return
            rz = _sum_partials(rz_cur, count) if rz_cur.size > 0 else s[S_RZ]
            alpha = T(rz / curvature)
        else:
            alpha = T(s[S_ALPHA])
        acc = 0.0
        cols_u, p1_u, p2_u, n_u = u32(cols), u32(p1), u32(p2), u32(n)
        k = start()
        step = stride()
        while k < n_u:
            phi[k] += mul(alpha, d[k])
            value = r[k] - mul(alpha, q[k])
            r[k] = value
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            v[u32(_reordered(i, p1_u) * p2_u + _reordered(j, p2_u))] = value
            acc += _dmul(numba.float64(value), numba.float64(value))
            k = u32(k + step)
        if rr_out.size > 0:
            _block_partial(acc, rr_out)
            return
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            rel = math.sqrt(total) / s[S_NORM]
            it = f[F_ITER] + 1
            f[F_ITER] = it
            history[it] = rel
            if rel <= tol:
                f[F_DONE] = 1
            elif it >= limit:
                f[F_DONE] = 2

    @kernel(A, A, i32, i32, i32, i32, I32A)
    def reorder(r, v, rows, cols, p1, p2, f):
        """Reorder x on both axes into v, for x = r zero-padded to (P1, P2).

        The DCT-II of x is then a twiddled FFT of v (J. Makhoul, 1980).
        """
        if f[F_DONE]:
            return
        rows_u, cols_u, p1_u, p2_u = u32(rows), u32(cols), u32(p1), u32(p2)
        n = u32(p1_u * p2_u)
        k = start()
        step = stride()
        while k < n:
            m1 = u32(k // p2_u)
            m2 = u32(k - m1 * p2_u)
            i = _source(m1, p1_u)
            j = _source(m2, p2_u)
            v[k] = r[u32(i * cols_u + j)] if (i < rows_u and j < cols_u) else T(0)
            k = u32(k + step)

    @cuda.jit(device=True, inline=True)
    def dct_at(w1, w2, v0, v1):
        ux = (mul(w2.real, v0.real) - mul(w2.imag, v0.imag) + mul(w2.real, v1.real)
              + mul(w2.imag, v1.imag))
        uy = (mul(w2.real, v0.imag) + mul(w2.imag, v0.real) + mul(w2.real, v1.imag)
              - mul(w2.imag, v1.real))
        return mul(w1.real, ux) - mul(w1.imag, uy)

    @kernel(CA, CA, A, CA, CA, i32, i32, I32A, F64A, i32, F64A, F64A, f64, i32)
    def makhoul_spectrum(half, out, inverse, tw1, tw2, p1, p2, f,
                         rr, count, s, history, tol, limit):
        """Apply the preconditioner in the spectral domain, on half spectra.

        From the half spectrum V of the reordered residual to the half spectrum V'
        of the reordered result. With Y = (inverse symbol) DCT-II, -k meaning
        N - k, and Y = 0 at index N:
        V'[k1, k2] = conj(w1(k1) w2(k2)) (Y[k1, k2] - Y[-k1, -k2]
        - i (Y[-k1, k2] + Y[k1, -k2])). Rows k1 and -k1 need the same four Y, and
        for a real input, V[-k1, -k2] = conj(V[k1, k2]), so these four come from
        two values of the half spectrum. One thread per row pair (k1 <= P1 / 2)
        and column k2 <= P2 / 2. With `rr`, first the convergence check of the CG
        iteration whose residual this is, from the partial sums of r . r (see
        `cg_update`): every block computes the relative residual and whether CG
        stops; block 0 records it, and marks the iteration for `crop` to count
        (F_ITER must not change while blocks read it).
        """
        if f[F_DONE]:
            return
        if rr.size > 0:
            rel = math.sqrt(_sum_partials(rr, count)) / s[S_NORM]
            it = f[F_ITER] + 1
            done = 1 if rel <= tol else (2 if it >= limit else 0)
            if cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0:
                history[it] = rel
                f[F_PENDING] = 1
                if done:
                    f[F_DONE] = done
            if done:
                return
        p1_u, p2_u = u32(p1), u32(p2)
        h = u32(u32(p2_u // U2) + U1)
        n = u32(u32(u32(p1_u // U2) + U1) * h)
        k = start()
        step = stride()
        while k < n:
            k1 = u32(k // h)
            k2 = u32(k - k1 * h)
            m1 = u32(u32(p1_u - k1) % p1_u)
            va = _cload(half, u32(k1 * h + k2))
            vc = _cload(half, u32(m1 * h + k2))
            w1a = _cload(tw1, k1)
            w1c = _cload(tw1, u32(p1_u - k1))
            w2a = _cload(tw2, k2)
            w2b = _cload(tw2, u32(p2_u - k2))
            row_a = u32(k1 * p2_u)
            row_c = u32(m1 * p2_u)
            k2b = u32(p2_u - k2)
            y_aa = mul(dct_at(w1a, w2a, va, vc.conjugate()), inverse[u32(row_a + k2)])
            y_ab = T(0) if k2 == U0 else mul(
                dct_at(w1a, w2b, vc.conjugate(), va), inverse[u32(row_a + k2b)])
            y_ca = T(0) if k1 == U0 else mul(
                dct_at(w1c, w2a, vc, va.conjugate()), inverse[u32(row_c + k2)])
            y_cb = T(0) if (k1 == U0 or k2 == U0) else mul(
                dct_at(w1c, w2b, va.conjugate(), vc), inverse[u32(row_c + k2b)])
            w2c = w2a.conjugate()
            for row in range(2):
                if row == 1 and (k1 == U0 or m1 == k1):
                    break
                r1 = k1 if row == 0 else m1
                re = (y_aa - y_cb) if row == 0 else (y_ca - y_ab)
                im = -(y_ca + y_ab) if row == 0 else -(y_aa + y_cb)
                w1 = _cload(tw1, r1).conjugate()
                cx = mul(w1.real, w2c.real) - mul(w1.imag, w2c.imag)
                cy = mul(w1.real, w2c.imag) + mul(w1.imag, w2c.real)
                _cstore(out, u32(r1 * h + k2), complex(mul(cx, re) - mul(cy, im),
                                                       mul(cx, im) + mul(cy, re)))
            k = u32(k + step)

    @kernel(A, A, A, A, i32, i32, i32, i32, F64A, U32A, F64A, I32A, i32, i32)
    def crop(e, r, z, d, rows, cols, p1, p2, partials, counter, s, f, mode,
             count_iteration):
        """Crop z = x[:rows, :cols] for x the Makhoul-reordered (P1, P2) array e.

        mode 0: rz = r . z, and d = z (the start of CG). mode 1: beta =
        (r . z) / rz, rz = r . z. mode 2: only z. mode 3: only the blocks' partial
        sums of r . z, into `partials`, for the next kernels to add up (see
        `apply_operator`). mode 4: as 3, and d = z. With `count_iteration`, block
        0 first counts the iteration that `makhoul_spectrum` checked (F_PENDING),
        also if CG stopped after it.
        """
        if (count_iteration and cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0
                and f[F_PENDING]):
            f[F_ITER] += 1
            f[F_PENDING] = 0
        if f[F_DONE]:
            return
        cols_u, p1_u, p2_u = u32(cols), u32(p1), u32(p2)
        n = u32(u32(rows) * cols_u)
        acc = 0.0
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            value = e[u32(_reordered(i, p1_u) * p2_u + _reordered(j, p2_u))]
            z[k] = value
            if mode == 0 or mode == 4:
                d[k] = value
            acc += _dmul(numba.float64(r[k]), numba.float64(value))
            k = u32(k + step)
        if mode == 2:
            return
        if mode == 3 or mode == 4:
            _block_partial(acc, partials)
            return
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            if mode == 1:
                s[S_BETA] = numba.float64(T(total / s[S_RZ]))
            s[S_RZ] = total

    @cuda.jit(device=True, inline=True)
    def admm_u(src, from_fft, k, i, j, p1_u, p2_u):
        """Return u at pixel k, from `phi` or (`from_fft`) the reordered FFT output."""
        if from_fft:
            return src[u32(_reordered(i, p1_u) * p2_u + _reordered(j, p2_u))]
        return src[k]

    @cuda.jit(device=True, inline=True)
    def admm_edge(grad, g, c, lam, rho, inv_rho, delta):
        """One edge: z = prox of (c / rho) huber_delta at t; returns (lam', v).

        t = grad u - g + lam / rho; lam' = lam + rho (grad u - g - z); v = g + z -
        lam' / rho, the edge's target of the next u-solve.
        """
        tau = c * inv_rho
        t = grad - g + mul(lam, inv_rho)
        if abs(t) <= delta + tau:
            z = mul(t, delta) / (delta + tau)
        elif t > T(0):
            z = t - tau
        else:
            z = t + tau
        lam_new = lam + mul(rho, grad - g - z)
        return lam_new, g + z - mul(lam_new, inv_rho)

    @kernel(
        A, i32, A, A, A, A, A, A, A, A, A, i32, i32, i32, i32, F64A, F64A, i32,
    )
    def admm_update(src, from_fft, gx, gy, cx, cy, lx, ly, lx_new, ly_new, v,
                    rows, cols, p1, p2, params, rho_, per):
        """One ADMM step for min sum C huber_delta(grad u - g).

        From u (`src`: `phi`, or the inverse FFT's output if `from_fft`), the
        z- and dual updates on the edges of each pixel (each thread writes the
        duals of its right and lower edges, into the other buffer), and the
        right-hand side G^T v of the next u-solve, Makhoul-reordered into
        `v` (the FFT's input). params[0]: epsilon^2 (delta = epsilon, the Huber
        scale); rho_[0]: rho.

        Each thread takes one column of a band of `per` rows (see
        `_irls_cupy._admm_shape`). Each edge is evaluated once: a thread keeps
        the value v of the edge below a pixel for the pixel below, and takes the
        value of the edge to the left from the lane to its left (`shfl`). Only
        a band's first row and lane 0 evaluate their upper and left edges
        themselves. A pixel adds up its values in the order of the CPU's
        `_admm_numba.admm_update`: right, left, lower, upper.
        """
        cols_u, rows_u, p1_u, p2_u = u32(cols), u32(rows), u32(p1), u32(p2)
        cm1 = u32(cols_u - U1)
        rm1 = u32(rows_u - U1)
        col_blocks = u32(u32(cols_u + UBLOCK - U1) // UBLOCK)
        block = u32(cuda.blockIdx.x)
        band = u32(block // col_blocks)
        j = u32(u32(block - band * col_blocks) * UBLOCK + u32(cuda.threadIdx.x))
        lane = u32(cuda.threadIdx.x) & u32(31)
        i = u32(band * u32(per))
        end = u32(min(i + u32(per), rows_u))
        inside = j < cols_u
        delta = T(math.sqrt(params[0]))
        rho = T(rho_[0])
        inv_rho = T(1.0 / rho_[0])
        u = T(0)
        upper = T(0)  # v of the edge above the current pixel
        if inside and i < end:
            k = u32(i * cols_u + j)
            u = admm_u(src, from_fft, k, i, j, p1_u, p2_u)
            if i > U0:
                e = u32(k - cols_u)
                grad = u - admm_u(src, from_fft, e, u32(i - U1), j, p1_u, p2_u)
                _, upper = admm_edge(grad, gy[e], cy[e], ly[e], rho, inv_rho, delta)
        while i < end:  # the same rows for the whole block
            k = u32(i * cols_u + j)
            # right edge (own): k -> k + 1, contributes -v
            right_u = cuda.shfl_down_sync(FULL_MASK, u, 1)
            right = T(0)
            if inside and j < cm1:
                if lane == u32(31):
                    right_u = admm_u(src, from_fft, u32(k + U1), i, u32(j + U1),
                                     p1_u, p2_u)
                e = u32(i * cm1 + j)
                lam, right = admm_edge(right_u - u, gx[e], cx[e], lx[e], rho, inv_rho,
                                       delta)
                lx_new[e] = lam
            # left edge: k - 1 -> k, the right edge of the lane to the left
            left = cuda.shfl_up_sync(FULL_MASK, right, 1)
            if inside and j > U0 and lane == U0:
                e = u32(i * cm1 + j - U1)
                left_u = admm_u(src, from_fft, u32(k - U1), i, u32(j - U1), p1_u, p2_u)
                grad = u - left_u
                _, left = admm_edge(grad, gx[e], cx[e], lx[e], rho, inv_rho, delta)
            # lower edge (own)
            below_u = T(0)
            lower = T(0)
            if inside and i < rm1:
                below_u = admm_u(src, from_fft, u32(k + cols_u), u32(i + U1), j,
                                 p1_u, p2_u)
                lam, lower = admm_edge(below_u - u, gy[k], cy[k], ly[k], rho, inv_rho,
                                       delta)
                ly_new[k] = lam
            if inside:
                rhs = T(0)
                if j < cm1:
                    rhs -= right
                if j > U0:
                    rhs += left
                if i < rm1:
                    rhs -= lower
                if i > U0:
                    rhs += upper
                v[u32(_reordered(i, p1_u) * p2_u + _reordered(j, p2_u))] = rhs
            upper = lower
            u = below_u
            i = u32(i + U1)

    @kernel(
        A, i32, A, A, A, A, A, A, A, A, A, i32, i32, i32, i32, F64A, F64A,
    )
    def admm_update_pixels(src, from_fft, gx, gy, cx, cy, lx, ly, lx_new, ly_new, v,
                           rows, cols, p1, p2, params, rho_):
        """`admm_update` with one pixel per thread, and a grid-stride loop.

        Each edge is evaluated by both of its pixels. Faster than `admm_update`
        where that would take one row per thread (see `_irls_cupy._admm_shape`):
        all lanes do work, without shuffles.
        """
        cols_u, p1_u, p2_u = u32(cols), u32(p1), u32(p2)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        delta = T(math.sqrt(params[0]))
        rho = T(rho_[0])
        inv_rho = T(1.0 / rho_[0])
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            u = admm_u(src, from_fft, k, i, j, p1_u, p2_u)
            rhs = T(0)
            if j < cm1:  # right edge (own): k -> k + 1, contributes -v
                e = u32(i * cm1 + j)
                right = admm_u(src, from_fft, u32(k + U1), i, u32(j + U1), p1_u, p2_u)
                grad = right - u
                lam, val = admm_edge(grad, gx[e], cx[e], lx[e], rho, inv_rho, delta)
                lx_new[e] = lam
                rhs -= val
            if j > U0:  # left edge: k - 1 -> k, contributes +v
                e = u32(i * cm1 + j - U1)
                left = admm_u(src, from_fft, u32(k - U1), i, u32(j - U1), p1_u, p2_u)
                grad = u - left
                _, val = admm_edge(grad, gx[e], cx[e], lx[e], rho, inv_rho, delta)
                rhs += val
            if i < rm1:  # lower edge (own)
                below = admm_u(src, from_fft, u32(k + cols_u), u32(i + U1), j,
                               p1_u, p2_u)
                grad = below - u
                lam, val = admm_edge(grad, gy[k], cy[k], ly[k], rho, inv_rho, delta)
                ly_new[k] = lam
                rhs -= val
            if i > U0:  # upper edge
                e = u32(k - cols_u)
                grad = u - admm_u(src, from_fft, e, u32(i - U1), j, p1_u, p2_u)
                _, val = admm_edge(grad, gy[e], cy[e], ly[e], rho, inv_rho, delta)
                rhs += val
            v[u32(_reordered(i, p1_u) * p2_u + _reordered(j, p2_u))] = rhs
            k = u32(k + step)

    @kernel(A, A, A, T, I32A, F64A, I32A, F64A, F64A, i64)
    def branches_and_phase_slot(phi, wrapped, offset, tau, br0, ph0, br1, ph1, rs, n):
        """`branches_and_phase` into the candidate slot rs[16] (0 or 1).

        Does nothing once the rounds have stopped (rs[4]; see `cuts_decide`).
        """
        if rs[4] != 0.0:
            return
        current = rs[16] != 0.0
        branches = br1 if current else br0
        phase = ph1 if current else ph0
        off = offset[0]
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            b = numba.int32(rint((phi[k] - off - wrapped[k]) / tau))
            branches[k] = b
            turns = _dmul(2.0 * math.pi, numba.float64(b))
            phase[k] = numba.float64(wrapped[k]) + turns
            k = u32(k + step)

    @kernel(
        I32A, I32A, I32A, I32A, A, A, U8A, i32, i32, F64A, U32A, F64A, F64A, F64A,
        F64A,
    )
    def cuts_decide(br0, br1, jx, jy, cx, cy, invalid, rows, cols, partials, counter,
                    s, rs, hist, params):
        """ADMM: `cuts` of the candidate in slot rs[16], then the stop rule.

        In the last block, the stop rule of `irls._irls_rounds` on the device,
        with the candidate slots chosen on the device: a candidate that is kept
        becomes the best (rs[17]), and the next round writes into the other slot.
        rs: 0 best cost, 1 best turns, 2 best round (-1: the start), 3 rounds
        without improvement, 4 done, 5 epsilon, 6 epsilon of the best, 7 round
        index, 8 maximum rounds, 9 stop patience, 10 stop tolerance,
        11 epsilon_min, 12 decay, 13 minimum improvement, 16 candidate slot,
        17 best's slot. Records the round's turns and cost into hist[2 index] and
        hist[2 index + 1], and the next epsilon^2 into params[0].
        """
        if rs[4] != 0.0:
            return
        br = br1 if rs[16] != 0.0 else br0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        if s[S_INVALID_PIXELS] > 0.0:
            turns, cost = cut_sums_masked(br, jx, jy, cx, cy, invalid, cols_u, cm1,
                                          rm1, n, k, step)
        else:
            turns, cost = cut_sums(br, jx, jy, cx, cy, cols_u, cm1, rm1, n, k, step)
        last, turns, cost = _last_block2(turns, cost, partials, counter)
        if not (last and cuda.threadIdx.x == 0):
            return
        s[S_TURNS] = turns
        s[S_COST] = cost
        index = numba.int64(rs[7])
        hist[2 * index] = turns
        hist[2 * index + 1] = cost
        best = rs[0]
        threshold = rs[13] if rs[13] > rs[10] * best else rs[10] * best
        stale = 0.0 if cost < best - threshold else rs[3] + 1.0
        rs[3] = stale
        if cost <= best:
            rs[0] = cost
            rs[1] = turns
            rs[2] = numba.float64(index)
            rs[6] = rs[5]
            rs[17] = rs[16]
            rs[16] = 1.0 - rs[16]
        eps = rs[5] * rs[12]
        rs[5] = eps if eps > rs[11] else rs[11]
        params[0] = rs[5] * rs[5]
        rs[7] = numba.float64(index + 1)
        if stale >= rs[9] or rs[1] == 0.0 or index + 1 >= numba.int64(rs[8]):
            rs[4] = 1.0

    @kernel(F64A, F64A, F64A, A, A, A, A, A, A, A, A, i32, i32)
    def admm_weights(rs, ph0, ph1, tx, ty, cx, cy, wx0, wy0, wx1, wy1, rows, cols):
        """Once the rounds have stopped (rs[4]), the result's Charbonnier weights.

        From the best candidate's phase (slot rs[17]) at its epsilon (rs[6]), into
        that slot's weights, as `_admm_numba.weights_from_phase`. Launched at the
        end of every chunk of rounds; the host launches no chunk after the stop,
        so this runs once.
        """
        if rs[4] == 0.0:
            return
        best = rs[17] != 0.0
        phase = ph1 if best else ph0
        wx = wx1 if best else wx0
        wy = wy1 if best else wy0
        squared = rs[6] * rs[6]
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            if j < cm1:
                e = u32(i * cm1 + j)
                r = (phase[u32(k + U1)] - phase[k]) - numba.float64(tx[e])
                wx[e] = T(squared / math.sqrt(_dmul(r, r) + squared)) * cx[e]
            if i < rm1:
                r = (phase[u32(k + cols_u)] - phase[k]) - numba.float64(ty[k])
                wy[k] = T(squared / math.sqrt(_dmul(r, r) + squared)) * cy[k]
            k = u32(k + step)

    @kernel(A, A, T, A, i64, F64A, U32A, A)
    def circular_offset(phi, wrapped, tau, out, n, partials, counter, offset):
        """ADMM: out = wrap(phi - wrapped); offset[0] = its circular mean.

        angle(sum exp(i out)): the offset removed before rounding onto the
        branches, in one pass (the median, which IRLS removes, takes three
        selection passes for float32).
        """
        s_acc = 0.0
        c_acc = 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            v = phi[k] - wrapped[k]
            v = v - mul(rint(v / tau), tau)
            out[k] = v
            s_acc += math.sin(numba.float64(v))
            c_acc += math.cos(numba.float64(v))
            k = u32(k + step)
        last, s_tot, c_tot = _last_block2(s_acc, c_acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            offset[0] = T(math.atan2(s_tot, c_tot))

    @kernel(F64A, F64A, f64)
    def scale_into(x, out, factor):
        """out[0] = x[0] * factor (as a node of a captured graph)."""
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            out[0] = x[0] * factor

    @kernel(A, i64, F64A, U32A, F64A, I32A)
    def mean(x, n, partials, counter, s, f):
        """Compute the mean of x into s[S_MEAN], once CG has stopped."""
        if not f[F_DONE]:
            return
        acc = 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            acc += numba.float64(x[k])
            k = u32(k + step)
        last, total = _last_block(acc, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_MEAN] = total / numba.float64(n)

    @kernel(A, i64, F64A, I32A)
    def subtract_mean_if_iterated(x, n, s, f):
        """Subtract the mean from x, once CG has stopped, if it iterated.

        As `if iterated: phi -= phi.mean()` in `irls._WeightedProblem.solve`.
        """
        if not f[F_DONE] or not f[F_ITERATED]:
            return
        m = T(s[S_MEAN])
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            x[k] -= m
            k = u32(k + step)

    @kernel(A, A, A, A, A, F64A, A, A, i32, i32, I32A)
    def charbonnier_weights(phi, tx, ty, cx, cy, params, wx, wy, rows, cols, f):
        """Compute w = C eps^2 / sqrt((grad phi - g)^2 + eps^2), eps^2 = params[0].

        As `_irls_numba.charbonnier_weights`. With params[1] != 0 (as
        `unless_settled` of `irls._WeightedProblem.reweight`), nothing happens if
        the last CG solve ran no iteration (the IRLS fixed point), decided on the
        device. The parameters are in device memory, so that a captured graph can
        take new ones each round.
        """
        if params[1] != 0.0 and f[F_ITER] == 0:
            return
        squared = T(params[0])
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            if j < cm1:
                e = u32(i * cm1 + j)
                r = (phi[u32(k + U1)] - phi[k]) - tx[e]
                wx[e] = (squared / math.sqrt(mul(r, r) + squared)) * cx[e]
            if i < rm1:
                r = (phi[u32(k + cols_u)] - phi[k]) - ty[k]
                wy[k] = (squared / math.sqrt(mul(r, r) + squared)) * cy[k]
            k = u32(k + step)

    @kernel(A, A, T, A, i64)
    def wrapped_difference(phi, wrapped, tau, out, n):
        """Compute out = phi - wrapped, wrapped into [-pi, pi] (in T)."""
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            v = phi[k] - wrapped[k]
            out[k] = v - mul(rint(v / tau), tau)
            k = u32(k + step)

    @kernel(A, A, A, T, I32A, F64A, i64)
    def branches_and_phase(phi, wrapped, offset, tau, branches, phase, n):
        """Round phi - offset[0] onto the branches of `wrapped`, and their phase.

        branches = rint((phi - offset - wrapped) / tau) (int32), and phase =
        wrapped + 2 pi branches in float64.
        """
        off = offset[0]
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            b = numba.int32(rint((phi[k] - off - wrapped[k]) / tau))
            branches[k] = b
            turns = _dmul(2.0 * math.pi, numba.float64(b))
            phase[k] = numba.float64(wrapped[k]) + turns
            k = u32(k + step)

    @cuda.jit(device=True, inline=True)
    def cut_sums(br, jx, jy, cx, cy, cols_u, cm1, rm1, n, k, step):
        """Return this thread's sums of the cut turns and their cost, over all edges."""
        turns = 0.0
        cost = 0.0
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            if j < cm1:
                e = u32(i * cm1 + j)
                t = abs(jx[e] + (br[u32(k + U1)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cx[e]))
            if i < rm1:
                t = abs(jy[k] + (br[u32(k + cols_u)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cy[k]))
            k = u32(k + step)
        return turns, cost

    @cuda.jit(device=True, inline=True)
    def cut_sums_masked(br, jx, jy, cx, cy, invalid, cols_u, cm1, rm1, n, k, step):
        """`cut_sums` without the edges that touch an invalid pixel."""
        turns = 0.0
        cost = 0.0
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            here = invalid[k] != 0
            if j < cm1 and not (here or invalid[u32(k + U1)] != 0):
                e = u32(i * cm1 + j)
                t = abs(jx[e] + (br[u32(k + U1)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cx[e]))
            if i < rm1 and not (here or invalid[u32(k + cols_u)] != 0):
                t = abs(jy[k] + (br[u32(k + cols_u)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cy[k]))
            k = u32(k + step)
        return turns, cost

    @kernel(I32A, I32A, I32A, A, A, U8A, i32, i32, F64A, U32A, F64A)
    def cuts(br, jx, jy, cx, cy, invalid, rows, cols, partials, counter, s):
        """Sum the cut turns |jumps + diff(branches)| and their cut cost.

        The cost weights each edge's turns by its edge weight C. Summed into
        s[S_TURNS] and s[S_COST] over all edges except those that touch an invalid
        pixel (if s[S_INVALID_PIXELS] > 0).
        """
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        if s[S_INVALID_PIXELS] > 0.0:
            turns, cost = cut_sums_masked(br, jx, jy, cx, cy, invalid, cols_u, cm1,
                                          rm1, n, k, step)
        else:
            turns, cost = cut_sums(br, jx, jy, cx, cy, cols_u, cm1, rm1, n, k, step)
        last, t0, t1 = _last_block2(turns, cost, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_TURNS] = t0
            s[S_COST] = t1

    @kernel(A, i32, i32, u64, u64, U64A, U32A, U32A, i32, f64, A, F64A, i32)
    def select_pass(x, n, pass_, k0, k1, st, hist, counter, mode, t, out_t, out_d,
                    device_ranks):
        """One pass of the selection of two order statistics (see `_find_bucket`).

        The k-th smallest value, by its bit pattern: keys that sort like the
        floats (`_key32`, `_key64`), and one pass per 11-bit digit, most
        significant first. Each pass takes the histogram of the digit among the
        values that match the digits found so far; the last block to finish finds
        the bucket that holds rank k, and narrows the prefix and the rank to it.
        Two ranks at once, k0 <= k1: the two middle values of a median, or the two
        neighbours of a percentile. No host synchronization. `st` holds prefix0,
        prefix1, rank0, rank1 (set from k0, k1 in pass 0). `hist` (2 * BUCKETS)
        must be zero, and is zero again after each pass. After the last pass,
        mode 0 writes the median (v0 + v1) / 2 in T to out_t[0] (v0 == v1 for odd
        n, so that is v0), mode 1 the percentile by linear interpolation with
        weight t, as numpy's `_lerp`, in float64 to out_d[0]; both modes write v0
        and v1 to
        out_d[1:3]. With `device_ranks`, k0, k1 and t come from st[2], st[3] and
        out_d[3] instead (see `percentile_ranks`).
        """
        h0 = cuda.shared.array(BUCKETS, numba.uint32)
        h1 = cuda.shared.array(BUCKETS, numba.uint32)
        sh = cuda.shared.array(BLOCK, numba.uint64)
        found = cuda.shared.array(2, numba.int32)
        found_rank = cuda.shared.array(2, numba.uint64)
        tid = cuda.threadIdx.x
        fixed = DIGIT_BITS * pass_
        width = min(DIGIT_BITS, BITS - fixed)
        shift = u64(BITS - fixed - width)
        digit_mask = (u64(1) << u64(width)) - u64(1)
        high_mask = u64(0) if fixed == 0 else (
            all_bits ^ ((u64(1) << u64(BITS - fixed)) - u64(1)))
        p0 = u64(0) if pass_ == 0 else st[0]
        p1 = u64(0) if pass_ == 0 else st[1]
        b = tid
        while b < BUCKETS:
            h0[b] = U0
            h1[b] = U0
            b += BLOCK
        cuda.syncthreads()
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            kk = key(x[k])
            high = kk & high_mask
            digit = numba.int32((kk >> shift) & digit_mask)
            if high == p0:
                cuda.atomic.add(h0, digit, U1)
            if high == p1:
                cuda.atomic.add(h1, digit, U1)
            k = u32(k + step)
        cuda.syncthreads()
        b = tid
        while b < BUCKETS:
            if h0[b]:
                cuda.atomic.add(hist, b, h0[b])
            if h1[b]:
                cuda.atomic.add(hist, BUCKETS + b, h1[b])
            b += BLOCK
        cuda.threadfence()
        cuda.syncthreads()
        if not _ticket_last(counter):
            return
        cuda.threadfence()
        r0 = k0 if pass_ == 0 and not device_ranks else st[2]
        r1 = k1 if pass_ == 0 and not device_ranks else st[3]
        _find_bucket(hist, 0, r0, found, found_rank, 0, sh)
        _find_bucket(hist, BUCKETS, r1, found, found_rank, 1, sh)
        b = tid
        while b < 2 * BUCKETS:
            hist[b] = U0
            b += BLOCK
        if tid == 0:
            np0 = p0 | (u64(found[0]) << shift)
            np1 = p1 | (u64(found[1]) << shift)
            st[0] = np0
            st[1] = np1
            st[2] = found_rank[0]
            st[3] = found_rank[1]
            counter[0] = U0
            if shift == u64(0):  # the last pass: the keys are complete
                v0 = value(np0)
                v1 = value(np1)
                out_d[1] = numba.float64(v0)
                out_d[2] = numba.float64(v1)
                if mode == 0:
                    out_t[0] = (v0 + v1) / T(2)
                else:
                    w = out_d[3] if device_ranks else t
                    a = numba.float64(v0)
                    diff = numba.float64(v1) - a
                    out_d[0] = (numba.float64(v1) - _dmul(diff, 1.0 - w) if w >= 0.5
                                else a + _dmul(diff, w))

    @cuda.jit(device=True, inline=True)
    def percentile_ranks(invalid_count, n, q, st, out_d):
        """Set the ranks of the q-th percentile of the valid values.

        For `select_pass` with `device_ranks`. Of n values, `invalid_count` are
        invalid, and sort above all valid ones. As numpy's "linear" method on the
        valid values: the ranks of the two neighbours into st[2] and st[3], the
        interpolation weight into out_d[3]. Without valid values, as if there
        were one.
        """
        valid = n - numba.int64(invalid_count)
        if valid < 1:
            valid = numba.int64(1)
        virtual = numba.float64(valid - 1) * (q / 100.0)
        low = math.floor(virtual)
        rank = numba.int64(low)
        st[2] = u64(rank)
        st[3] = u64(rank + 1 if rank + 1 < valid else valid - 1)
        out_d[3] = virtual - low

    @kernel(
        A, A, U8A, A, A, I32A, I32A, A, A, A, A, i32, i32, i32, i32, f64, F64A, U32A,
        F64A,
    )
    def prepare_targets(in_, wrapped, invalid, tx, ty, jx, jy, cx, cy, wx, wy, rows,
                        cols, unit_edges, detect, invalid_weight, partials, counter,
                        s):
        """Copy the wrapped phase, and compute the targets and their jumps.

        `in_` may be `wrapped` without `detect`. As `_WeightedProblem`: the
        wrapped target gradients g = d - rint(d / tau) tau and their whole-turn
        jumps rint((d - g) / tau), in T. With `unit_edges`, also C = w = 1 on
        every edge. With `detect`, the pixels of `in_` that are not finite are
        invalid: 1 in `invalid` (0 elsewhere), their number in
        s[S_INVALID_PIXELS], C = w = invalid_weight on the edges that touch
        them, and 0 in `wrapped` and the targets. The mask is only cleared where
        an earlier input set it (s[S_MASK_SET]): without invalid pixels, it is
        not written.
        """
        tau = T(6.283185307179586)
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        count = 0.0
        mask_set = detect and s[S_MASK_SET] > 0.0
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            c = in_[k]
            bad = False
            if detect:
                bad = not math.isfinite(c)
                if bad:
                    c = T(0)
                    count += 1.0
                    invalid[k] = numba.uint8(1)
                elif mask_set:
                    invalid[k] = numba.uint8(0)
            wrapped[k] = c
            if j < cm1:
                e = u32(i * cm1 + j)
                other = in_[u32(k + U1)]
                bad_other = detect and not math.isfinite(other)
                if bad_other:
                    other = T(0)
                d = other - c
                g = d - mul(rint(d / tau), tau)
                tx[e] = g
                jx[e] = numba.int32(rint((d - g) / tau))
                if unit_edges:
                    cx[e] = T(1)
                    wx[e] = T(1)
                if bad or bad_other:
                    cx[e] = T(invalid_weight)
                    wx[e] = T(invalid_weight)
            if i < rm1:
                other = in_[u32(k + cols_u)]
                bad_other = detect and not math.isfinite(other)
                if bad_other:
                    other = T(0)
                d = other - c
                g = d - mul(rint(d / tau), tau)
                ty[k] = g
                jy[k] = numba.int32(rint((d - g) / tau))
                if unit_edges:
                    cy[k] = T(1)
                    wy[k] = T(1)
                if bad or bad_other:
                    cy[k] = T(invalid_weight)
                    wy[k] = T(invalid_weight)
            k = u32(k + step)
        if detect:
            last, total = _last_block(count, partials, counter)
            if last and cuda.threadIdx.x == 0:
                s[S_INVALID_PIXELS] = total
                s[S_MASK_SET] = 1.0 if total > 0.0 else 0.0

    @kernel(U8A, F64A, I32A, i64)
    def restore_invalid(invalid, phase, branches, n):
        """Set the result at the invalid pixels: phase NaN, branch 0."""
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            if invalid[k] != 0:
                phase[k] = math.nan
                branches[k] = numba.int32(0)
            k = u32(k + step)

    @kernel(A, F64A, I32A, i32, A, i64)
    def starting_phase(wrapped, prior_phase, prior_branches, mode, phi, n):
        """Compute the starting phase from a prior, as `irls._starting_phase`.

        mode 0 (a phase prior): the branch of the wrapped value nearest to the
        prior, in float64, and the wrapped value where the prior is not finite;
        mode 1 (a branch prior): wrapped + 2 pi branches, in T.
        """
        tau = 6.283185307179586
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            if mode == 0:
                raw = numba.float64(wrapped[k])
                b = libdevice.rint((prior_phase[k] - raw) / tau)
                if not math.isfinite(b):
                    b = 0.0
                phi[k] = T(raw + _dmul(tau, b))
            else:
                phi[k] = wrapped[k] + mul(T(tau), T(prior_branches[k]))
            k = u32(k + step)

    @kernel(CA, A, A, U8A, i32, F64A, U32A, F64A, f64, U64A, F64A)
    def wave_phase_amplitude(wave, phase, amplitude, invalid, n, partials, counter, s,
                             q, st, out_d):
        """Compute phase and amplitude of a complex wave, as cupy's angle and abs.

        Pixels whose amplitude is not finite are invalid: phase 0, amplitude +inf
        (above all valid ones), 1 in `invalid` (0 elsewhere, written as in
        `prepare_targets`), and their number into s[S_INVALID_PIXELS]. The last
        block also sets the ranks of the q-th percentile of the valid amplitudes
        (`percentile_ranks`).
        """
        count = 0.0
        mask_set = s[S_MASK_SET] > 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            z = _cload(wave, k)
            a = math.hypot(z.real, z.imag)
            if math.isfinite(a):
                phase[k] = math.atan2(z.imag, z.real)
                amplitude[k] = a
                if mask_set:
                    invalid[k] = numba.uint8(0)
            else:
                phase[k] = T(0)
                amplitude[k] = T(math.inf)
                invalid[k] = numba.uint8(1)
                count += 1.0
            k = u32(k + step)
        last, total = _last_block(count, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_INVALID_PIXELS] = total
            s[S_MASK_SET] = 1.0 if total > 0.0 else 0.0
            percentile_ranks(total, numba.int64(n), q, st, out_d)

    @kernel(A, U8A, F64A, f64, f64, i32, i32, i32, F64A, U32A, F64A, A, A, A, A)
    def amplitude_edges(amp, invalid, params, floor, invalid_weight, rows, cols, mode,
                        partials, counter, s, cx, cy, wx, wy):
        """Compute edge weights from the amplitude, ref = params[0] (the percentile).

        Invalid pixels count as amplitude 0. mode 0: the largest edge weight into
        s[S_LARGEST] (1 without edges), and the number of edges whose weight is
        not finite and positive into s[S_INVALID]. mode 1: C = w =
        T(weight / s[S_LARGEST]), as `irls._check_edge_weights` normalizes, and
        invalid_weight on the edges that touch an invalid pixel.
        """
        ref = params[0]
        masked = s[S_INVALID_PIXELS] > 0.0
        largest = s[S_LARGEST] if mode == 1 else 0.0
        top = -math.inf
        bad = 0.0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            here = masked and invalid[k] != 0
            a = _relative_amplitude(
                0.0 if here else numba.float64(amp[k]), ref, floor)
            if j < cm1:
                right = masked and invalid[u32(k + U1)] != 0
                w = _nan_minimum(_relative_amplitude(
                    0.0 if right else numba.float64(amp[u32(k + U1)]), ref, floor), a)
                if mode == 0:
                    top = libdevice.fmax(top, w)
                    if not (math.isfinite(w) and w > 0):
                        bad += 1.0
                else:
                    e = u32(i * cm1 + j)
                    cx[e] = T(invalid_weight) if here or right else T(w / largest)
                    wx[e] = cx[e]
            if i < rm1:
                down = masked and invalid[u32(k + cols_u)] != 0
                below = 0.0 if down else numba.float64(amp[u32(k + cols_u)])
                w = _nan_minimum(_relative_amplitude(below, ref, floor), a)
                if mode == 0:
                    top = libdevice.fmax(top, w)
                    if not (math.isfinite(w) and w > 0):
                        bad += 1.0
                else:
                    cy[k] = T(invalid_weight) if here or down else T(w / largest)
                    wy[k] = cy[k]
            k = u32(k + step)
        if mode == 1:
            return
        bmax = _block_max(top)
        bsum = _block_sum(bad)
        if cuda.threadIdx.x == 0:
            partials[u32(cuda.blockIdx.x)] = bmax
            partials[u32(u32(cuda.gridDim.x) + u32(cuda.blockIdx.x))] = bsum
        if not _ticket_last(counter):
            return
        cuda.threadfence()
        m = -math.inf
        c = 0.0
        kk = u32(cuda.threadIdx.x)
        blocks = u32(cuda.gridDim.x)
        while kk < blocks:
            m = libdevice.fmax(m, cuda.ldcg(partials, kk))
            c += cuda.ldcg(partials, u32(blocks + kk))
            kk = u32(kk + UBLOCK)
        m = _block_max(m)
        c = _block_sum(c)
        if cuda.threadIdx.x == 0:
            s[S_LARGEST] = 1.0 if m == -math.inf else m
            s[S_INVALID] = c
            counter[0] = U0

    return compiled, signatures
