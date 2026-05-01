# NeuralMag Vortex LLG Demo

This standalone demo repo contains a single notebook that:

- builds a cylindrical vortex initial state;
- relaxes it with NeuralMag LLG at two saturation magnetizations;
- projects the relaxed magnetization through the MBIR phase model; and
- fits `Msat` back from the target phase image with differentiable LLG.

## Fresh Environment Setup

Python 3.12 is the recommended baseline.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name vortex-llg-demo --display-name "Python (vortex-llg-demo)"
jupyter lab
```

Then open `notebooks/neuralmag_vortex_fit_llg.ipynb`, switch to the `Python (vortex-llg-demo)` kernel, and run the notebook from the top.

## Notes

- This repo assumes a normal package install of `neuralmag[jax]`; it does not rely on any vendored checkout.
- The notebook sets `JAX_ENABLE_X64=1` before importing JAX. Always start from a fresh kernel and run the first code cell before anything else.
- The demo uses CPU or GPU JAX depending on what your environment provides. If you want a CUDA-specific JAX build, install the matching `jax`/`jaxlib` wheel pair before installing `requirements.txt`, then reinstall `neuralmag[jax]` if needed.

## Files

- `notebooks/neuralmag_vortex_fit_llg.ipynb`: main demo notebook
- `requirements.txt`: core runtime and notebook dependencies
