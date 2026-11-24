# Dev container

Ubuntu 26.04 with uv, Python 3.10–3.14, neovim, Claude Code, the Nsight
profilers and pandoc (for the docs build), on amd64 and arm64 hosts. When the
container is created, `post-create.sh` sets up the editable dev install in `.venv`,
with `cupy-cuda13x` and the CUDA toolkit from pip. The GPU is passed through if the
host has the NVIDIA container toolkit; otherwise the container starts without a GPU.

`.venv`, `.tox`, `~/.claude`, `~/.cache` and the neovim plugin directory are Docker
volumes: they stay out of the host checkout and survive container rebuilds.

## VS Code

Install the "Dev Containers" extension, open the repo and run
"Dev Containers: Reopen in Container". For a remote GPU machine, connect with Remote-SSH
first.

## neovim / terminal

Use the [devcontainer CLI](https://github.com/devcontainers/cli). It needs Node.js;
without it, run the CLI as `uvx --from nodejs-wheel npx -y @devcontainers/cli ...`:

```sh
devcontainer up --workspace-folder .
# optional: bring your config along, e.g.
#   --dotfiles-repository https://github.com/<you>/dotfiles
#   --mount "type=bind,source=$HOME/.config/nvim,target=/home/vscode/.config/nvim"
devcontainer exec --workspace-folder . bash      # then: nvim, claude, pytest, ...
```

Run `devcontainer up --workspace-folder . --remove-existing-container` after
changing the container config.

## Claude Code

Run `claude` inside the container and log in once. The login is kept in the
`~/.claude` volume.

## Profiling CUDA code

Nsight Systems (`nsys`, timeline) and Nsight Compute (`ncu`, per-kernel metrics) are
installed from NVIDIA's apt repo, matching CUDA 13.5 (build arg `CUDA_VERSION`):

```sh
nsys profile -o holo --trace=cuda,nvtx .venv/bin/python script.py
nsys stats --report cuda_gpu_kern_sum holo.nsys-rep
ncu --set full -k regex:my_kernel -c 1 -o holo .venv/bin/python script.py
```

Open the `.nsys-rep` / `.ncu-rep` files in the Nsight GUIs on your desktop (free
download from NVIDIA). The GUIs are installed in the container, too, but need a
display.

Host settings that limit profiling inside the container:

- `nsys` CPU sampling and context switch tracing need
  `kernel.perf_event_paranoid <= 2` on the host (Ubuntu's default is 4). CUDA/NVTX
  tracing works without it.
- `ncu` needs access to the GPU performance counters. If it fails with
  `ERR_NVGPUCTRPERM`, the host driver restricts them to admin users
  (`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`). Either allow it on the
  host (`options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`) or add
  `"--cap-add=SYS_ADMIN"` to `runArgs` in `devcontainer.json`.

## Test data

To run the data tests, add a bind mount and set `TESTDATA_BASE_PATH`, e.g. with
`--mount "type=bind,source=/path/to/data,target=/data"` on `devcontainer up`, then
`TESTDATA_BASE_PATH=/data uvx --with tox-uv --python=3.14 tox -e py314-data` (see
`CLAUDE.md`).
