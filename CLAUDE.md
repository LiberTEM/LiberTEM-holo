# LiberTEM-holo

Electron holography reconstruction on top of LiberTEM. Source in `src/libertem_holo/`:
`base/` holds plain numpy/cupy functions (reconstruction, alignment, filters, unwrapping,
I/O); `udf/` holds the LiberTEM UDFs (`HoloReconstructUDF`).

## Running tests

### tox (authoritative, matches CI)

Without a local tox install, use `uvx`:

```sh
uvx --with tox-uv --python=3.14 tox -e py314                  # unit tests + doctests, data tests skip
uvx --with tox-uv --python=3.14 tox -e py314 -- tests/base/   # posargs replace the default `tests/`
```

Each env builds and installs the package as a wheel (catches packaging problems), then runs
pytest with coverage on `{posargs:tests/}`, then `pytest --doctest-modules src/libertem_holo/`
(the doctest pass runs even when posargs are given).

### Fast loop (editable venv)

For iterating, use an editable install in `.venv` (one-time setup):

```sh
uv venv --python 3.14
uv pip install -e ".[gui]" -r test_requirements.txt "numba>=0.63"
PY_IGNORE_IMPORTMISMATCH=1 .venv/bin/pytest -q tests/base/test_base_io.py   # ~3s
```

Don't run `.tox/*/bin/pytest` directly for this: tox installs the package non-editable, so
that tests the copy from the last tox run, not the current `src/`. Avoid `uv run`, which
creates an untracked `uv.lock`. Finish with a tox run before handing off.

### Test data

Tests that need real data use the `dm_testdata_path` fixture (`conftest.py`), which is
`$TESTDATA_BASE_PATH/dm` and skips the test when the variable is unset.

- Only the `py3XX-data` envs pass `TESTDATA_BASE_PATH` through (on purpose); the plain
  `py3XX` envs always skip data tests:
  `TESTDATA_BASE_PATH=/path/to/data uvx --with tox-uv --python=3.14 tox -e py314-data`
- CI with data runs on Azure (`azure-pipelines-data.yml`), triggered on demand for relevant
  changes. GitHub CI (`.github/workflows/ci.yml`) runs the plain envs.
- Every code path where it is feasible needs a **quick** test on synthetic data (in
  `tests/base/`) for fast iteration, in addition to any data test (in `tests/data/`).

### Fixtures and synthetic data

- `conftest.py` provides `lt_ctx` (inline executor), `dask_ctx`, synthetic holograms
  (`holo_data`, `holo_stack`), and injects `ctx`, `dataset` and `dataset_ref` into the
  doctest namespace.
- Outside of UDF tests, build input with `InputData.from_array(...)`.
- For tests that need a LiberTEM dataset, use `libertem.io.dataset.memory.MemoryDataSet`.
- pytest deselects markers `slow`, `dist`, `flaky`, `compilation`, `need_cupy` by default
  (`pytest.ini`).

## Other tox envs

- `docs-check`: Sphinx build with `-W` plus `sphinx-build -b doctest` (runs the
  `.. testcode::` blocks in `docs/source/*.rst`, which skip without test data)
- `docs`: live-reloading docs server on port 8008
- `mypy`, `numba_coverage` (`NUMBA_DISABLE_JIT=1`, `-m with_numba`)
- `notebooks` / `notebooks_gen`: run or regenerate `notebooks/*.ipynb`

## Style and linting

- Linting is not enforced at the moment (to be re-enabled later). Ruff is configured in
  `pyproject.toml` with `select = ["ALL"]`; existing code has many findings.
  **New files must pass `uvx ruff check <file>`**; in existing files, don't add new findings.
- pre-commit runs flake8 (max line length 100, `setup.cfg`); pyright is configured for `src/`
  (`pyrightconfig.json`, Python 3.10 baseline).
- Docstrings use numpy style; doctests that need data or files use `# doctest: +SKIP`.
- API naming (e.g. `load_from_*` vs `from_*` constructors) is not settled; follow the
  surrounding code and mention the choice.

## Changelog

Add one `.rst` file per change under `docs/source/changelog/{features,bugfix,documentation,misc,obsolescence}/`,
formatted like:

```rst
[Feature] Short title
=====================

 * What changed, with :meth:`...` references
```

## Git

Ask before committing.
