# AGENTS.md

Reusable guidance for PySCF development. Paths below are relative to the repository root.

## Repository

- PySCF is primarily written in Python, with C extensions for computational hotspots.
- Codebase layout:
  - `pyscf/lib/`: shared utilities and compiled C extensions. The CMake entry point is `pyscf/lib/CMakeLists.txt`.
  - `pyscf/data/`: common data and physical constants.
  - `pyscf/__config__.py`: global configuration defaults.
  - `pyscf/pbc/`: periodic implementations.
  - Other method packages under `pyscf/`: molecular implementations.
- Tests generally live in `pyscf/<module>/test/`, including nested packages.
- Runnable examples live in `examples/`, grouped by package.
- Documentation is maintained separately at https://github.com/pyscf/github.pyscf.io. The local `doc/` directory is legacy and can be ignored unless the task explicitly concerns it.
- Consult `pyproject.toml` for supported Python versions and dependencies, `pytest.ini` for test selection, and `.github/workflows/` for CI behavior.
- Do not add mandatory dependencies to the core package. Declare optional dependencies in `pyproject.toml` and import them only in modules that need them.

## Development Environment

- Develop with any supported Python version, but keep syntax and standard-library usage compatible with the minimum version in `pyproject.toml`.
- Prefer serial BLAS libraries: BLAS functions may be called from within OpenMP threads. When using multithreaded OpenBLAS or MKL, configure their thread counts to one, for example with `OPENBLAS_NUM_THREADS=1` or `MKL_NUM_THREADS=1`. Avoid nested multithreading.
- Set `PYSCF_TMPDIR` or `TMPDIR` to a writable scratch directory.
- `uv` is supported for developing PySCF alone. Prefer `virtualenv` when also developing namespace packages such as PySCF extensions or `pyscf-forge`; that combination has not been tested with `uv`.
- To develop or debug C extensions, configure a native build with `cmake -S pyscf/lib -B pyscf/lib/build`, then compile with `cmake --build pyscf/lib/build`. Reuse an existing configured build and its dependency options where possible.

## Code Style

- Keep method-specific logic in the corresponding package and reusable infrastructure in the existing shared modules.
- Make small, complete changes. Follow nearby code and existing APIs before introducing abstractions or dependencies.
- Preserve unrelated tracked changes, untracked experiments, calculation outputs, checkpoints, and local configuration.
- Follow the existing formatting and lint configuration. Avoid unrelated reformatting and preserve license notices.

## Validation

* Prefer focused regression tests for bug fixes and numerical changes, using small systems and nearby test conventions.
* Prefer lightweight, deterministic tests, with minimal mocking.
- Run the smallest relevant test selection. Documentation-only changes do not require numerical tests.
- Investigate failures near numerical tolerances. Do not substantially relax tolerances or replace reference values merely to make tests pass.

## AI Disclosure

- When preparing a pull request or issue with AI assistance, include a `Notes`
  section briefly describing how AI was used. Mention the AI tool or agent and
  the nature of its assistance, such as code generation, testing, debugging, or
  drafting.
- Do not create pull requests against the upstream repository unless explicitly
  asked.

## Skills

- When a task explicitly selects a skill, read the corresponding template in `agent-recipes/skills/` or the user's customized version before applying it.
