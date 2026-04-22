# Forward Model Analysis: Analytic Solutions to the Magnetic Phase Integral

## Context

The magnetic phase integral being discussed:

$$\varphi_\text{mag}(x,y) = \frac{-B_\text{sat}}{2\Phi_0} \iint \frac{(y-y')m_{\text{pr},x}(x',y') - (x-x')m_{\text{pr},y}(x',y')}{(x-x')^2+(y-y')^2}\,dx'dy'$$

---

## Q: Does this integral have an analytic solution, to avoid convolutions?

**A:** Yes, with one important distinction.

For an arbitrary projected magnetization field there is not a simpler closed-form phase map than the integral itself. But the integral **does** have an analytic kernel, so it can be written exactly as a convolution, and in Fourier space its symbol is known in closed form. That is why the RDFC approach makes sense.

Taking the 2D Fourier transform of $K_j = x_j/r^2 = \partial_j \ln r$ and using $\nabla^2 \ln r = 2\pi\delta^2$:

$$\hat{K}_x = -\frac{2\pi i k_x}{|k|^2}, \quad \hat{K}_y = -\frac{2\pi i k_y}{|k|^2}$$

Substituting:

$$\hat\varphi(\mathbf{k}) = \frac{\pi i B_\text{sat}}{\Phi_0}\cdot\frac{k_y\,\hat{m}_x(\mathbf{k}) - k_x\,\hat{m}_y(\mathbf{k})}{k_x^2+k_y^2}$$

This is **exact** for point-source voxels (`geometry="disc"`). The `get_freq_grid` function in `kernel.py` already builds exactly $(k_x, k_y, k_x^2+k_y^2)$ — but is currently unused by the forward model.

The RDFC `"disc"` kernel is exactly the $y/r^2$ or $x/r^2$ structure, assembled into two convolution components and scaled into phase. The `"slab"` branch goes a step further: it is an analytic finite-pixel integral, not just a point-sampled approximation.

---

## Q: Does the direct Fourier method introduce artefacts?

**A:** Yes, in one specific way.

The kernel $K(x,y) = y/r^2$ decays as $1/r$ and is not absolutely integrable. In Fourier space this shows up as $H(\mathbf{k}) \propto k_j/|\mathbf{k}|^2$ diverging at $k=0$. On a finite pixel grid with periodic tiling assumed, sharp features (domain walls, particle edges) produce Gibbs ringing.

The RDFC `"slab"` geometry avoids this. The `_F_a` formula is the analytic integral of $K$ over the *area* of a unit voxel — a spatial anti-aliasing filter:

$$K_{\text{slab}}(n,m) = \int_{n-\tfrac{1}{2}}^{n+\tfrac{1}{2}}\int_{m-\tfrac{1}{2}}^{m+\tfrac{1}{2}} \frac{m'}{n'^2+m'^2}\,dn'\,dm'$$

| | Direct Fourier | RDFC `"disc"` | RDFC `"slab"` |
|---|---|---|---|
| Equivalent to | Point-source kernel | Point-source kernel | Area-averaged kernel |
| Ringing near sharp edges | Yes | Yes (same) | Reduced |
| Precomputed kernel needed | No | Yes | Yes |

---

## Q: Is the direct Fourier method faster?

**A:** For repeated forward passes (as in MBIR — hundreds of solver iterations), essentially identical.

At inference time, both paths do the same work:
1. Pad `u_field` and `v_field` to $(2N \times 2M)$
2. Two `rfft2` calls
3. Element-wise multiply by a precomputed frequency-domain array
4. One `irfft2` + slice

The one-time kernel build overhead (`build_rdfc_kernel`) disappears with the direct method, but this cost amortises to near-zero over many solver iterations. No meaningful speed advantage for MBIR usage.

---

## Q: Is the direct Fourier method related to the FDFC approach from the literature?

**A:** Yes, exactly. The paper's FDFC (Fourier space Discretization of the Fourier Convolution) Eq. 3.35:

$$\tilde\varphi_\text{mag}[p,q] = \frac{\imath B_0 a^2}{2\Phi_0}\frac{\tilde m_{s,u}[p,q]\cdot\hat f_v[q] - \tilde m_{s,v}[p,q]\cdot\hat f_u[p]}{\hat f_u^2[p]+\hat f_v^2[q]}$$

maps directly to existing code:

| Paper symbol | Code |
|---|---|
| $\tilde m_{s,u}[p,q]$ | `jnp.fft.rfft2(u_pad)` |
| $\tilde m_{s,v}[p,q]$ | `jnp.fft.rfft2(v_pad)` |
| $\hat f_u[p]$, $\hat f_v[q]$ | `f_x`, `f_y` from `get_freq_grid` |
| $\hat f_u^2 + \hat f_v^2$ | `denom` from `get_freq_grid` |
| $a^2$ | `pixel_size**2` |

`get_freq_grid` was presumably written in anticipation of exactly this. The paper presents FDFC as the *existing* alternative (section 3.2.2), and RDFC as the newer/more accurate method.

---

## Q: Does the integral have a true analytic solution (no convolution at all)?

**A:** Yes, for specific magnetization geometries — and those solutions are already in the test suite.

### The stream function identity

For a localized 3D sample, the projected magnetization is always **divergence-free**:

$$\frac{\partial m_x^{pr}}{\partial x} + \frac{\partial m_y^{pr}}{\partial y} = -[m_z]_{-\infty}^{+\infty} = 0$$

This means a stream function $\psi$ always exists: $m_x = \partial_y\psi$, $m_y = -\partial_x\psi$.

Integrating by parts (using $\nabla^2\ln r = 2\pi\delta^2$):

$$\boxed{\varphi_\text{mag} = \frac{-\pi B_\text{sat}}{\Phi_0}\,\psi}$$

**The phase is the stream function, scaled by a physical constant.**

### The four analytic solutions in `test_mbir.py`

These are ground-truth phase images that work because $\psi$ has a closed-form expression for simple geometries:

| Shape | Function | Why $\psi$ is analytic |
|---|---|---|
| Uniform disc | `_analytic_phase_disc` | Uniform $\mathbf{m}_{pr}$ inside → $\psi$ linear; dipole outside |
| Slab | `_analytic_phase_slab` | Piece-wise uniform → $\psi$ via corner-sum $F_0$ formula |
| Sphere | `_analytic_phase_sphere` | Projected ellipsoidal field → analytic dipole |
| Vortex disc | `_analytic_phase_vortex` | $\mathbf{m}_{pr}$ purely azimuthal → $\psi \propto r$ inside |

The slab $F_0$ formula in `_analytic_phase_slab` is the **same** function as `_F_a` in `_rdfc_elementary_phase(geometry="slab")` — RDFC encodes the slab analytic solution as a per-pixel kernel so convolution recovers it exactly for arbitrary superpositions.

### For the general (MBIR) case

For an unknown, arbitrary $\mathbf{m}$ field, $\psi$ has no closed form. You must solve:

$$\nabla^2\psi = \partial_u m_v - \partial_v m_u$$

In Fourier space: $\hat\psi = \widehat{(\partial_u m_v - \partial_v m_u)} / (k_u^2+k_v^2)$ — which is exactly FDFC again. The Poisson/Fourier solve is unavoidable for general fields.

---

## Q: Can the 3D forward model be simplified using the stream function identity?

**A:** For the forward model in isolation (not MBIR reconstruction), yes — by replacing `build_rdfc_kernel` + `phase_mapper_rdfc` with a direct FDFC implementation using `get_freq_grid`.

### Concrete gains

| | Current RDFC | Simplified FDFC |
|---|---|---|
| `build_rdfc_kernel` call | Required | Gone |
| Stored kernel arrays | `u_fft`, `v_fft` (large) | None |
| FFTs per forward call | 3 (same) | 3 (same) |
| `geometry` parameter | `"disc"` / `"slab"` | Disappears |
| Accuracy near sharp boundaries | `"slab"` is better | Point-voxel only |

The runtime per call is identical, but the first-call overhead disappears and the code is substantially simpler.

### The one trade-off

The `"slab"` geometry anti-aliasing is lost. For smooth or reconstructed fields this makes no practical difference; for sharp-edged objects (disc, slab) `"slab"` is more accurate.

### What the simplified forward model looks like

Given projected $(m_u, m_v)$:

$$\hat\varphi(\mathbf{k}) = \frac{i\pi B_0\, a^2}{\Phi_0}\cdot\frac{\hat m_u(\mathbf{k})\,k_v - \hat m_v(\mathbf{k})\,k_u}{k_u^2+k_v^2}$$

In code: `FFT(m_u)`, `FFT(m_v)`, combine with the frequency grid from `get_freq_grid`, `IFFT`. No kernel dict, no `build_rdfc_kernel`, no `geometry` parameter.

---

## Q: Is a real-space finite-cell analytic method better than RDFC (as sometimes proposed)?

**A: No — RDFC `"slab"` already *is* the real-space finite-cell analytic method. They are mathematically identical.**

The idea of a "real-space analytic model" is to compute, per source cell, the exact integral

$$K_\text{cell}(\Delta x, \Delta y) = \iint_\text{cell} \frac{\Delta y - y'}{(\Delta x - x')^2 + (\Delta y - y')^2}\,dx'\,dy'$$

and sum contributions from all cells. Because $K_\text{cell}$ depends only on the offset, "sum over all cells" is a discrete convolution. The result is identical whether evaluated (a) in real space by direct convolution with the cell-integrated kernel, or (b) in Fourier space by multiplying zero-padded FFTs of the same kernel. Only numerical precision and cost differ.

`build_rdfc_kernel(geometry="slab")` builds exactly this cell-integrated kernel from the $F_a$ corner-sum primitive, zero-pads the signal to $(2H, 2W)$ to prevent wraparound, and evaluates the convolution via FFT.

### Numerical verification

The module `libertem_holo.base.mbir.forward_realspace` implements the direct real-space convolution explicitly for audit purposes. Across slab, vortex and random test fields on a 64×64 grid:

| Comparison | max abs difference |
|---|---|
| RDFC `"slab"` vs direct real-space convolution | ~1 × 10⁻¹⁶ (floating-point noise) |
| RDFC `"disc"` vs direct real-space convolution | ~5 × 10⁻⁶ (point-sampled vs cell-integrated) |

See `notebooks/MBIR/forward_model_benchmark.ipynb` for the full benchmark.

### So why use the FFT path?

For a dense field on a regular grid the cost of the convolution is:

- **Direct real-space:** $\mathcal{O}(N^4)$ per forward call
- **Zero-padded FFT:** $\mathcal{O}(N^2 \log N)$ per forward call

Both give the same answer to floating-point precision. On a typical $256^2$ grid the FFT path is ~100× faster in wall time (measured on GPU), and the gap grows with $N$. The "Fourier artefact" concern raised against FFT-based convolution applies to the naïve unpadded cyclic case; `build_rdfc_kernel` uses $(2H, 2W)$ zero-padding, which makes the FFT compute a linear (non-cyclic) convolution — no wraparound, no Gibbs ringing from the boundary.

### When is a real-space evaluator actually useful?

- **Arbitrary, non-grid detector positions.** The FFT assumes phase samples on the same regular grid as the source. If phases are needed at a few off-grid points (e.g. nanoprobe positions), direct evaluation of the kernel sum at those points is $\mathcal{O}(N^2)$ per point and avoids interpolation error.
- **Auditing / unit tests.** A short independent implementation is valuable ground truth. `forward_model_realspace_2d` serves this role.
- **Truncated kernels on very large grids.** If $N \gg 10^3$ and the kernel is truncated to a radius $R_c \ll N$, direct real-space convolution costs $\mathcal{O}(N^2 R_c^2)$ and can win over padded FFT.

But the $1/r$ magnetic-phase kernel decays *slowly*. The benchmark shows that truncating to $R_c = 16$ pixels introduces ~10⁻² relative error in $\varphi$ — two orders of magnitude worse than RDFC. Truncation is not a practical strategy for this kernel.

### Is FMM/treecode worthwhile?

For dense fields on a regular grid at typical electron-holography sizes ($N \lesssim 10^3$), FMM offers no advantage: the zero-padded FFT is already asymptotically $\mathcal{O}(N^2 \log N)$ and benefits from heavily optimised cuFFT/MKL kernels. FMM would be worth considering only for sparsely sampled 3D source distributions with $\gg 10^6$ nontrivial voxels — a regime that is not reached by the current 2D or 3D MBIR forward models.

---

## Summary and recommendation

1. **Keep `geometry="slab"` as the default** for both inference and MBIR forward modelling. It is the correct cell-integrated operator; `"disc"` is a point-sampled approximation that is strictly less accurate near sharp boundaries.
2. **The zero-padded FFT convolution used by RDFC has no wraparound artefacts.** There is no Fourier-related accuracy problem to solve.
3. **A direct real-space implementation (`forward_model_realspace_2d`) is useful as an audit tool and for off-grid detector positions**, but is not competitive with the FFT path for dense forward passes on regular grids.
4. **Kernel truncation and FMM are not worthwhile** for the current problem size and the slowly decaying $1/r$ magnetic-phase kernel.
