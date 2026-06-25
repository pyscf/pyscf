# Windows Packaging Workflow For PySCF

This README is the entry point for the Windows packaging work under `conda/windows/`.
It is written to match the current Windows workflow as it exists in this repository,
not as an abstract future design. A new contributor should be able to read this file
from top to bottom and understand what the Windows packaging effort is doing, which
script owns which responsibility, and where to continue the work safely.

## 0. Workflow Overview

The current Windows logic is intentionally layered.

1. **Prepare the Windows build environment**
   - Script and file: `conda/windows/environment.yml`
   - Purpose: define the recommended local `conda` environment used for build and verification.

2. **Build the wheel**
   - Script: `conda/windows/build-wheel.ps1`
   - Purpose: resolve the Python interpreter, validate the toolchain, stage runtime DLLs, and build the wheel.

3. **Install the wheel into the target environment**
   - Script: `conda/windows/verify/verify-wheel.ps1`
   - Purpose: optionally install the built wheel into the intended interpreter before running verification.

4. **Run package-level verification**
   - Scripts: `conda/windows/verify/verify-wheel.ps1`, `conda/windows/verify/verify-wheel.py`, `conda/windows/verify/wheel_utils.py`
   - Purpose: check the wheel artifact itself, then verify imports and minimal runtime behavior.

5. **Run a small, targeted example set**
   - Scripts: `conda/windows/verify/verify-wheel.py`, `conda/windows/verify/verify-wheel-manifest.json`
   - Purpose: guard the known packaging regressions and keep a small, high-signal installed-wheel gate.

6. **Run the full installed-wheel example sweep**
   - Script: `conda/windows/examples/run-installed-examples.py`
   - Purpose: execute the broad installed-wheel example suite and classify outcomes into `PASS`, `FAIL`, `TIMEOUT`, `MISSING_DEP`, `IMPORT_ERROR`, and `MISSING_FILE`.

7. **Extract and summarize the full-sweep results**
   - Current state: partial
   - Purpose: turn raw full-sweep JSONL output into an indexed, queryable, repeatable triage layer.
   - Current repository status: raw JSONL output exists and the README records high-level counts, but the dedicated indexing/summarization layer has not been implemented yet.

8. **Maintain the workflow contract**
   - Tests: `conda/windows/tests/`
   - Decision note: `conda/windows/docs/cibuildwheel-evaluation.md`
   - Purpose: keep the build/verify entry points stable and document why Windows currently stays on its native runner path.

If you only need the shortest mental model, it is:

- `environment.yml` prepares the interpreter
- `build-wheel.ps1` creates the wheel
- `verify-wheel.ps1` installs the wheel and drives layered verification
- `verify-wheel.py` owns the targeted installed-wheel checks
- `run-installed-examples.py` owns the full example sweep
- the future indexing layer will sit on top of the full-sweep JSONL output

## 1. Directory Map

The Windows packaging directory is organized by responsibility:

- `conda/windows/README.md`
  - This document. Read this first.
- `conda/windows/environment.yml`
  - Recommended local `conda` environment.
- `conda/windows/build-wheel.ps1`
  - Wheel build orchestration entry point.
- `conda/windows/verify/`
  - Post-build installation and layered verification.
- `conda/windows/examples/`
  - Full installed-wheel example sweep entry point.
- `conda/windows/tests/`
  - Unit tests that protect the Windows packaging and verification helpers.
- `conda/windows/docs/`
  - Supporting design and policy notes that are referenced from this README.

## 2. Environment Preparation

The recommended environment file is:

- `conda/windows/environment.yml`

It currently describes:

- Python `3.13`
- `pip`
- `setuptools`
- `wheel`
- `cmake<4`
- `ninja`
- Python package `build`

Create the environment with:

```powershell
$RepoRoot = (Get-Location).Path
conda env create -f (Join-Path $RepoRoot "conda\windows\environment.yml")
```

Then resolve the interpreter explicitly:

```powershell
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
```

The explicit interpreter path matters because the Windows workflow is designed
to keep build and verification tied to the same known environment rather than
to whichever Python happens to be active in the shell.

## 3. Layer 1: Build The Wheel

The authoritative build entry point is:

- `conda/windows/build-wheel.ps1`

This script is responsible for more than just calling `python -m build`. It:

- resolves the target Python interpreter
- checks that `git`, `cmake`, `ninja`, `gcc`, and `g++` are available
- validates the Python build frontend
- locates the runtime DLL source directory
- stages runtime DLLs into `pyscf/lib/deps/win64/bin`
- sets the environment needed by the native build
- runs `python -m build --wheel --no-isolation`

Recommended command:

```powershell
$RepoRoot = (Get-Location).Path
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
$RuntimeDllDir = (Split-Path (Get-Command gcc).Source -Parent)

powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "conda\windows\build-wheel.ps1") `
  -PythonExe $PythonExe `
  -RepoRoot $RepoRoot `
  -RuntimeDllDir $RuntimeDllDir `
  -ParallelLevel 8
```

Use `-Clean` for a release rebuild or when you need to reset cached build state.
For normal local iteration, omit `-Clean` so the external project build tree can be reused.

## 4. Layer 2: Install The Wheel

The Windows flow treats wheel installation as an explicit step, not as an
implicit side effect of verification.

The installation entry point is:

- `conda/windows/verify/verify-wheel.ps1`

When called with `-InstallWheel`, it runs:

```powershell
& $PythonExe -m pip install --force-reinstall $WheelPath
```

This layer is intentionally separate from build so that:

- wheel creation and wheel consumption stay distinct
- the same verification entry point can be used with an already-built wheel
- failures can be attributed to build, install, or runtime more cleanly

## 5. Layer 3: Package-Level Verification

The package-level verification stack is:

- `conda/windows/verify/verify-wheel.ps1`
- `conda/windows/verify/verify-wheel.py`
- `conda/windows/verify/wheel_utils.py`

This layer answers a narrower question than the full example sweep:

> Does the built wheel look structurally correct, import correctly, and survive a few minimal runtime checks?

The verification phases are:

- `artifact`
  - inspect the wheel payload and metadata before installation-time reasoning
- `import`
  - verify that `pyscf` and core modules import from the installed wheel
- `smoke`
  - run minimal RHF, DFT, DF, and export checks

Recommended command:

```powershell
$RepoRoot = (Get-Location).Path
$TmpDir = Join-Path $RepoRoot ".tmp\windows"
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
$WheelPath = Join-Path $RepoRoot "dist\pyscf-2.13.1-py3-none-win_amd64.whl"

New-Item -ItemType Directory -Force -Path (Join-Path $TmpDir "verify") | Out-Null

powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "conda\windows\verify\verify-wheel.ps1") `
  -PythonExe $PythonExe `
  -RepoRoot $RepoRoot `
  -InstallWheel `
  -WheelPath $WheelPath `
  -Phase all `
  -OutputJson (Join-Path $TmpDir "verify\verify-wheel-all.json")
```

Current targeted package-level status on this branch:

- `artifact`: PASS
- `import`: PASS
- `smoke`: PASS

The latest targeted verification JSON in the workspace is:

- `.tmp/windows/verify/verify-wheel-all.json`

## 6. Layer 4: Targeted Example Verification

The targeted example layer is still driven by `verify-wheel.py`, but it should
be thought of as a different gate from the package-level checks above.

Its inputs are:

- `conda/windows/verify/verify-wheel.py`
- `conda/windows/verify/verify-wheel-manifest.json`

Its phases are:

- `examples`
  - representative installed-wheel examples
- `packaging`
  - historically sensitive packaging-regression examples
- `diagnostics`
  - expected non-packaging failures

The current branch keeps `diagnostics` empty because the previously tracked
optional-dependency cases now short-circuit cleanly under `--pyscf-verify-windows`.

Current targeted example status on this branch:

- `examples`: PASS
- `packaging`: PASS
- `diagnostics`: no active cases

This layer is the fastest installed-wheel gate that still protects real
regressions. It should stay small, explicit, and stable.

## 7. Layer 5: Full Installed-Wheel Example Sweep

The full sweep entry point is:

- `conda/windows/examples/run-installed-examples.py`

This script is intentionally broader and noisier than the targeted verification
layer. It exists to surface the full installed-wheel compatibility picture, not
to act as the release gate by itself.

Recommended command:

```powershell
$RepoRoot = (Get-Location).Path
$TmpDir = Join-Path $RepoRoot ".tmp\windows"
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"

New-Item -ItemType Directory -Force -Path (Join-Path $TmpDir "examples") | Out-Null

& $PythonExe (Join-Path $RepoRoot "conda\windows\examples\run-installed-examples.py") `
  --repo-root $RepoRoot `
  --examples-root (Join-Path $RepoRoot "examples") `
  --output (Join-Path $TmpDir "examples\wheel-examples.jsonl")
```

For chunked execution:

```powershell
$RepoRoot = (Get-Location).Path
$TmpDir = Join-Path $RepoRoot ".tmp\windows"
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"

New-Item -ItemType Directory -Force -Path (Join-Path $TmpDir "examples") | Out-Null

& $PythonExe (Join-Path $RepoRoot "conda\windows\examples\run-installed-examples.py") `
  --repo-root $RepoRoot `
  --examples-root (Join-Path $RepoRoot "examples") `
  --output (Join-Path $TmpDir "examples\wheel-examples-part1.jsonl") `
  --limit 228

& $PythonExe (Join-Path $RepoRoot "conda\windows\examples\run-installed-examples.py") `
  --repo-root $RepoRoot `
  --examples-root (Join-Path $RepoRoot "examples") `
  --output (Join-Path $TmpDir "examples\wheel-examples-part2.jsonl") `
  --start-at examples\mcscf\13-restart.py
```

The classifier currently distinguishes:

- `PASS`
- `FAIL`
- `TIMEOUT`
- `MISSING_DEP`
- `IMPORT_ERROR`
- `MISSING_FILE`

That classification is useful because the full-sweep failures are not all
Windows wheel defects. Some are optional dependencies, some are heavy examples,
some are example/API drift, and some are genuine issues worth fixing in the
installed-wheel path.

The latest complete full-sweep counts recorded in this README are:

- `PASS 323`
- `FAIL 62`
- `TIMEOUT 25`
- `IMPORT_ERROR 4`
- `MISSING_DEP 41`
- `MISSING_FILE 1`

These counts are still an upper bound on the remaining problem set because the
targeted `packaging` layer has already revalidated several historically failing
examples after that full sweep was captured.

## 8. Layer 6: Full-Sweep Result Extraction And Summarization

This is the only layer in the current Windows logic that is still intentionally
incomplete.

What already exists:

- raw JSONL full-sweep artifacts under `.tmp/`
- human-readable aggregate counts in this README
- a stable status vocabulary from the example runner

What does not exist yet:

- a dedicated indexed results store
- repeatable error clustering
- historical run-to-run diff reporting
- a script-owned summary extraction layer for the full sweep

In other words:

- the full sweep can already produce facts
- the repository does not yet have the final extraction/indexing layer that
  turns those facts into a queryable triage database

That future work should sit on top of the JSONL artifacts rather than replacing
the current full-sweep runner.

## 9. Supporting Tests

Windows-specific helper tests live in:

- `conda/windows/tests/test_build_wheel.py`
- `conda/windows/tests/test_verify_wheel.py`
- `conda/windows/tests/test_examples_compat.py`

These tests protect the contract of the Windows packaging layer itself. They do
not replace real build or verification runs, but they are useful when editing
the local helper scripts and README paths.

Examples:

```powershell
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"

& $PythonExe conda\windows\tests\test_build_wheel.py
& $PythonExe conda\windows\tests\test_verify_wheel.py
& $PythonExe conda\windows\tests\test_examples_compat.py
```

## 10. Supporting Notes And Policy Documents

The main supplementary note currently tracked with this workflow is:

- `conda/windows/docs/cibuildwheel-evaluation.md`

Read it when you want to answer:

- why Windows still uses the native runner path
- why the project did not immediately migrate Windows to `cibuildwheel`
- what conditions would justify revisiting that decision later

This note is not the workflow entry point. It is background context that should
be read after this README, not instead of it.

## 11. How The Windows Release Job Maps To The Layers

The GitHub Actions Windows release job follows the same layered model:

1. set up Python and MSYS2
2. run `conda/windows/build-wheel.ps1`
3. install and verify through `conda/windows/verify/verify-wheel.ps1`
4. collect the wheel
5. upload the artifact

This is important because local debugging should stay aligned with the same
build/verify split used by the release workflow. Local commands should explain
release behavior, not invent a separate Windows path.

## 12. Recommended Reading Order For New Contributors

If you are new to this Windows packaging work, use this reading order:

1. read this `README.md`
2. inspect `environment.yml`
3. inspect `build-wheel.ps1`
4. inspect `verify/verify-wheel.ps1`
5. inspect `verify/verify-wheel.py`
6. inspect `examples/run-installed-examples.py`
7. inspect `tests/`
8. read `docs/cibuildwheel-evaluation.md` if you need policy context

That order matches the actual workflow and keeps the mental model consistent.
