"""The kernels of the GPU IRLS solver, in numba CUDA.

The same algorithms as the CUDA C++ kernels of the same names in `_irls_cupy.py`,
with the same arguments, except that an absent array is an empty array, not a NULL
pointer. They give bit-identical results, and whole solves run about as fast (on a
GH200, from -3 % at 2048^2 to +1 % for float64 at ~500^2, cold and warm starts).
What that took, found by comparing Nsight Compute profiles and SASS with the CUDA
C++ kernels:

* **32-bit index arithmetic, truncated after every operation.** numba computes
  integer arithmetic at the machine width: `uint32 + uint32` is a 64-bit addition,
  and a `//` of 64-bit values takes a slow path. Every integer result that meets
  an index is therefore wrapped in `u32(...)`: truncated, so that LLVM narrows the
  arithmetic back to 32 bits. Unsigned, because with signed integers numba also
  gives every array access Python's handling of negative indices, and `//` and `%`
  Python's rounding. Plain Python integer constants are int64 and would widen the
  arithmetic again: hence the typed constants `U0`, `U1`, `U2`, `UBLOCK`. Without
  this, the streaming kernels executed 2.4-2.6x the instructions and ran up to 2x
  as long; with it, `apply_operator` and `cg_update` execute fewer instructions
  than the CUDA C++ ones. Narrowing cannot repair a comparison, `//` or `%` of a
  widened value, so loop bounds such as `u32(u32(rows) * cols)` are truncated too
  (a 64-bit loop test and a 64-bit `%` per element cost ~10 % in `apply_operator`
  and `makhoul_spectrum`).
* **Vector loads and stores of complex values** (`_cload`, `_cstore`): numba
  aligns a complex element as its components and moves the real and imaginary
  parts separately, which made `makhoul_spectrum` touch ~40 % more L2 sectors.
* **No fused multiply-adds.** NVVM contracts `a * b + c` into an FMA, and
  numba-cuda has no switch for that. Products that are added or subtracted are
  computed with round-to-nearest intrinsics (`mul`, `_dmul`: libdevice's
  `fmul_rn`, `dmul_rn`), which are never fused: so the results match the CPU
  solver's and the CUDA C++ kernels' bit for bit. Allowing FMAs gained at most ~1 %.
* **One loop per mode.** numba does not hoist a loop-invariant test out of a loop;
  `apply_operator` has a loop for each of its modes.
* **`ld.global.cg` for the last block's sum of partial sums** (`cuda.ldcg`): L1 is
  not coherent between blocks, L2 is. The CUDA C++ kernels use `volatile` loads,
  which numba lacks; atomic reads, the other way, cost ~1 us per reduction.
* **Short reductions without loops.** The deferred sums of partial sums have at
  most two per thread, added in straight-line code (`_sum_partials`): LLVM's
  unrolled loop cost branches, and with a redundant barrier in `_ticket_last`,
  ~0.3 us per kernel, a few percent at ~500^2.
* **No launch bounds.** Before the 32-bit arithmetic, a few more registers than
  the CUDA C++ kernels cost a wave on large images, and launch bounds helped
  `apply_operator`; now they only cost (`makhoul_spectrum` spills under them).

Launching a numba kernel from Python costs ~40 us of host time (cupy: ~3 us), in
numba's argument handling, even for a compiled kernel launched directly. So these
kernels run only inside captured CUDA graphs, including the setup of a solve
(`CupyWeightedProblem.cold_start`, `warm_start`): host launches made warm starts
~70 % slower than with the CUDA C++ kernels. Capturing still pays it, once per
workspace: ~25 ms for the first solve of a shape (CUDA C++: ~8 ms).

Compiling all kernels for one dtype takes ~7 s, so they are cached on disk
(`cache=True`); a fresh process then loads them in ~0.2 s, mostly numba's own
initialization. (The cache key hashes the bytecode, which differs between a module
compiled from source and one loaded from its .pyc: after an edit, the first two
processes compile.)
"""
# Kernels take their arrays as positional arguments, and T is a type. Modes are
# numbered as the CUDA C++ kernels' arguments, and plain comparisons keep numba's
# code simple (no tuple membership tests, no min/max with their NaN handling):
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
F_DONE, F_ITER, F_ITERATED, F_ZERO, F_PENDING = range(5)
(S_NORM, S_RZ, S_ALPHA, S_BETA, S_MEAN, S_TURNS, S_COST, S_LARGEST, S_INVALID,
 S_NAN) = range(10)

u32 = numba.uint32
u64 = numba.uint64
U0 = u32(0)
U1 = u32(1)
U2 = u32(2)
UBLOCK = u32(BLOCK)

# EXPERIMENT: True: products are rounded before they are added (no FMAs), which
# keeps the results bit-identical with the CPU; False: NVVM may contract them.
EXACT = True

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

    The first two per thread unrolled by hand (deferred sums have at most two
    blocks' worth): LLVM's own unrolling of the loop costs a few branches.
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
    """Take a ticket: whether this block is the last one to finish (all threads)."""
    flag = cuda.shared.array(1, numba.int32)
    if cuda.threadIdx.x == 0:
        cuda.threadfence()
        ticket = cuda.atomic.add(counter, 0, U1)
        flag[0] = 1 if ticket == u32(u32(cuda.gridDim.x) - U1) else 0
    cuda.syncthreads()
    return flag[0] != 0  # written once per launch: no barrier needed after


@cuda.jit(device=True, inline=True)
def _coherent_sum(partials, offset):
    """In the last block: the sum of all blocks' partial sums.

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
    """As `last_block` in the CUDA source, for one value: (last block, total)."""
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
    """Where the Makhoul reordering puts index n of a length-N axis (unsigned)."""
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
    return _kernels(np.dtype(dtype), exact=EXACT)


@cuda.jit(device=True, inline=True)
def _plain_mul(a, b):
    return a * b


@functools.cache
def _kernels(dtype: np.dtype, *, exact: bool) -> dict:
    single = dtype == np.float32
    T = numba.float32 if single else numba.float64
    BITS = 32 if single else 64
    if exact:
        mul = libdevice.fmul_rn if single else libdevice.dmul_rn
        _dmul = libdevice.dmul_rn
    else:
        mul = _plain_mul
        _dmul = _plain_mul
    rint = libdevice.rintf if single else libdevice.rint
    key = _key32 if single else _key64
    value = _value32 if single else _value64
    all_bits = u64(0xFFFFFFFF) if single else u64(0xFFFFFFFFFFFFFFFF)
    compiled = {}

    def kernel(func):
        # Cached on disk: the index key covers the source file and the closure
        # (dtype, `exact`), but not options of `cuda.jit`, which must not vary.
        compiled[func.__name__] = cuda.jit(cache=True)(func)
        return compiled[func.__name__]

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
        """Run `stencil_plain` for the next direction p' = beta p + z (to p_new)."""
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

    @kernel
    def reset_flags(f):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            f[F_DONE] = 0
            f[F_ITER] = 0
            f[F_ITERATED] = 0
            f[F_ZERO] = 0
            f[F_PENDING] = 0

    @kernel
    def set_params(p, a, b):
        if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
            p[0] = a
            p[1] = b

    @kernel
    def apply_operator(p, z, p_new, wx, wy, out, rows, cols, partials, counter, s, f,
                       mode, tiny, rz_new, rz_old, rz_count, pap_out):
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

    @kernel
    def right_hand_side(wx, tx, wy, ty, b, rows, cols, partials, counter, s):
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

    @kernel
    def center_and_norm(b, n, partials, counter, s, f, history):
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

    @kernel
    def initial_residual(b, q, r, n, partials, counter, s, f, history, tol):
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

    @kernel
    def cg_update(phi, d, r, q, n, partials, counter, s, f, history, tol, limit,
                  v, cols, p1, p2, pap, rz_cur, count, tiny, rr_out):
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
        reorder = v.size > 0
        acc = 0.0
        cols_u, p1_u, p2_u, n_u = u32(cols), u32(p1), u32(p2), u32(n)
        k = start()
        step = stride()
        while k < n_u:
            phi[k] += mul(alpha, d[k])
            value = r[k] - mul(alpha, q[k])
            r[k] = value
            if reorder:
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

    @kernel
    def reorder(r, v, rows, cols, p1, p2, f):
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

    @kernel
    def makhoul_spectrum(half, out, inverse, tw1, tw2, p1, p2, f,
                         rr, count, s, history, tol, limit):
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

    @kernel
    def crop(e, r, z, d, rows, cols, p1, p2, width, unreorder, partials, counter, s,
             f, mode, count_iteration):
        if (count_iteration and cuda.blockIdx.x == 0 and cuda.threadIdx.x == 0
                and f[F_PENDING]):
            f[F_ITER] += 1
            f[F_PENDING] = 0
        if f[F_DONE]:
            return
        cols_u, p1_u, p2_u, width_u = u32(cols), u32(p1), u32(p2), u32(width)
        n = u32(u32(rows) * cols_u)
        acc = 0.0
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            if unreorder:
                i = _reordered(i, p1_u)
                j = _reordered(j, p2_u)
            value = e[u32(i * width_u + j)]
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

    @kernel
    def mean(x, n, partials, counter, s, f):
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

    @kernel
    def subtract_mean_if_iterated(x, n, s, f):
        if not f[F_DONE] or not f[F_ITERATED]:
            return
        m = T(s[S_MEAN])
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            x[k] -= m
            k = u32(k + step)

    @kernel
    def charbonnier_weights(phi, tx, ty, cx, cy, params, wx, wy, rows, cols, f):
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

    @kernel
    def wrapped_difference(phi, wrapped, tau, out, n):
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            v = phi[k] - wrapped[k]
            out[k] = v - mul(rint(v / tau), tau)
            k = u32(k + step)

    @kernel
    def branches_and_phase(phi, wrapped, offset, tau, branches, phase, n):
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

    @kernel
    def cuts(br, jx, jy, cx, cy, rows, cols, partials, counter, s):
        turns = 0.0
        cost = 0.0
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
                t = abs(jx[e] + (br[u32(k + U1)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cx[e]))
            if i < rm1:
                t = abs(jy[k] + (br[u32(k + cols_u)] - br[k]))
                turns += t
                cost += _dmul(numba.float64(t), numba.float64(cy[k]))
            k = u32(k + step)
        last, t0, t1 = _last_block2(turns, cost, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_TURNS] = t0
            s[S_COST] = t1

    @kernel
    def select_pass(x, n, pass_, k0, k1, st, hist, counter, mode, t, out_t, out_d):
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
        r0 = k0 if pass_ == 0 else st[2]
        r1 = k1 if pass_ == 0 else st[3]
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
                    a = numba.float64(v0)
                    diff = numba.float64(v1) - a
                    out_d[0] = (numba.float64(v1) - _dmul(diff, 1.0 - t) if t >= 0.5
                                else a + _dmul(diff, t))

    @kernel
    def prepare_targets(in_, wrapped, tx, ty, jx, jy, cx, cy, wx, wy, rows, cols,
                        unit_edges):
        tau = T(6.283185307179586)
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            c = in_[k]
            wrapped[k] = c
            if j < cm1:
                e = u32(i * cm1 + j)
                d = in_[u32(k + U1)] - c
                g = d - mul(rint(d / tau), tau)
                tx[e] = g
                jx[e] = numba.int32(rint((d - g) / tau))
                if unit_edges:
                    cx[e] = T(1)
                    wx[e] = T(1)
            if i < rm1:
                d = in_[u32(k + cols_u)] - c
                g = d - mul(rint(d / tau), tau)
                ty[k] = g
                jy[k] = numba.int32(rint((d - g) / tau))
                if unit_edges:
                    cy[k] = T(1)
                    wy[k] = T(1)
            k = u32(k + step)

    @kernel
    def starting_phase(wrapped, prior_phase, prior_branches, mode, phi, n):
        tau = 6.283185307179586
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            if mode == 0:
                raw = numba.float64(wrapped[k])
                b = libdevice.rint((prior_phase[k] - raw) / tau)
                phi[k] = T(raw + _dmul(tau, b))
            else:
                phi[k] = wrapped[k] + mul(T(tau), T(prior_branches[k]))
            k = u32(k + step)

    @kernel
    def wave_phase_amplitude(wave, phase, amplitude, n, partials, counter, s):
        nans = 0.0
        n_u = u32(n)
        k = start()
        step = stride()
        while k < n_u:
            z = _cload(wave, k)
            phase[k] = math.atan2(z.imag, z.real)
            a = math.hypot(z.real, z.imag)
            amplitude[k] = a
            if math.isnan(a):
                nans += 1.0
            k = u32(k + step)
        last, total = _last_block(nans, partials, counter)
        if last and cuda.threadIdx.x == 0:
            s[S_NAN] = total

    @kernel
    def amplitude_edges(amp, params, floor, rows, cols, mode, partials, counter, s,
                        cx, cy, wx, wy):
        ref = math.nan if s[S_NAN] > 0 else params[0]
        largest = s[S_LARGEST] if mode == 1 else 0.0
        top = -math.inf
        invalid = 0.0
        cols_u = u32(cols)
        cm1 = u32(cols_u - U1)
        rm1 = u32(u32(rows) - U1)
        n = u32(u32(rows) * cols_u)
        k = start()
        step = stride()
        while k < n:
            i = u32(k // cols_u)
            j = u32(k - i * cols_u)
            a = _relative_amplitude(numba.float64(amp[k]), ref, floor)
            if j < cm1:
                w = _nan_minimum(
                    _relative_amplitude(numba.float64(amp[u32(k + U1)]), ref, floor), a)
                if mode == 0:
                    top = libdevice.fmax(top, w)
                    if not (math.isfinite(w) and w > 0):
                        invalid += 1.0
                else:
                    e = u32(i * cm1 + j)
                    cx[e] = T(w / largest)
                    wx[e] = cx[e]
            if i < rm1:
                below = numba.float64(amp[u32(k + cols_u)])
                w = _nan_minimum(_relative_amplitude(below, ref, floor), a)
                if mode == 0:
                    top = libdevice.fmax(top, w)
                    if not (math.isfinite(w) and w > 0):
                        invalid += 1.0
                else:
                    cy[k] = T(w / largest)
                    wy[k] = cy[k]
            k = u32(k + step)
        if mode == 1:
            return
        bmax = _block_max(top)
        bsum = _block_sum(invalid)
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
            kk = u32(k + UBLOCK)
        m = _block_max(m)
        c = _block_sum(c)
        if cuda.threadIdx.x == 0:
            s[S_LARGEST] = 1.0 if m == -math.inf else m
            s[S_INVALID] = c
            counter[0] = U0

    return compiled


# Which arguments may be absent (NULL in the CUDA C++ kernels), and their dtype:
# "T" for the solve's dtype, else a numpy dtype.
OPTIONAL = {
    "apply_operator": {17: np.float64},
    "cg_update": {12: "T", 16: np.float64, 17: np.float64, 20: np.float64},
    "makhoul_spectrum": {8: np.float64},
}
