# Dev container

Ubuntu 24.04 with uv, Python 3.10–3.14, neovim and Claude Code. The editable dev
install (`.venv`, with `cupy-cuda13x` and the pip `cuda-toolkit`) is set up on first
start by `post-create.sh`. The GPU is passed through if the host has the NVIDIA
container toolkit, and is optional otherwise. Works on amd64 and arm64 hosts.

`.venv`, `.tox`, `~/.claude`, `~/.cache` and the neovim plugin dir live in docker
volumes. They are kept separate from the host checkout and survive container rebuilds.

## VS Code

Install the "Dev Containers" extension, open the repo and run
"Dev Containers: Reopen in Container". For a remote GPU box, connect with Remote-SSH
first.

## neovim / terminal

Use the [devcontainer CLI](https://github.com/devcontainers/cli) (needs node; if you
don't have it, `uvx --from nodejs-wheel npx -y @devcontainers/cli ...` works without
installing anything):

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

## Test data

To run the data tests, add a bind mount and set `TESTDATA_BASE_PATH`, e.g. with
`--mount "type=bind,source=/path/to/data,target=/data"` on `devcontainer up`, then
`TESTDATA_BASE_PATH=/data tox -e py314-data` (see `CLAUDE.md`).
