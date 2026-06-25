# Windows Wheel Packaging and Verification

This directory contains the Windows-specific build and verification entry points for PySCF wheel packaging.

## Layout

- `build-wheel.ps1`
  Builds a Windows wheel and prepares the required external runtime dependencies in the local staging directory `pyscf/lib/deps/win64/bin`.
- `verify-wheel.ps1`
  Installs a wheel if requested and runs the targeted Windows verification phases.
- `verify-wheel.py`
  Python implementation of the targeted verification phases.
- `verify-wheel-manifest.json`
  Stable data file for the targeted example, packaging regression, and diagnostic expected-failure cases.
- `cibuildwheel-evaluation.md`
  Recorded go/no-go decision for future Windows wheel unification work.
- `test_verify_wheel.py`
  Unit tests for the verification helpers.
- `run-installed-examples.py`
  Full installed-wheel example sweep for Windows.

The Windows external runtime dependencies are not committed to the repository.
They are prepared under `pyscf/lib/deps/win64/bin` as local build artifacts by
`build-wheel.ps1` before wheel creation.

## Prerequisites

The local build flow assumes:

- Python 3.13
- `build`
- Git for Windows
- CMake >= 3.22
- Ninja
- MSYS2 UCRT64 GCC
- MSYS2 UCRT64 OpenBLAS runtime files

`build-wheel.ps1` now enforces the critical parts of this contract directly:

- Python must be 3.13
- `build`, `cmake`, `ninja`, `git`, `gcc`, and `g++` must resolve on `PATH`
- the runtime source directory must contain the required external runtime files

During packaging the script stages those runtime files into `pyscf/lib/deps/win64/bin`, sets `PYSCF_WINDOWS_RUNTIME_DLL_DIR` to that location, and invokes `python -m build --wheel --no-isolation` from outside the repository root so the repo-local `build/` directory does not shadow the Python build frontend.

It also prepends these directories to `PATH`:

- `pyscf/lib`
- `pyscf/lib/deps/bin`
- `pyscf/lib/deps/lib`
- `pyscf/lib/deps/win64/bin`

This ensures that runtime library resolution works both while building the wheel and while
running local source-tree validation after the build.

The staged runtime files are local build outputs and should remain ignored by git.

## Environment Setup

If Miniforge is not installed yet, install it to a directory of your choice, for example:

```powershell
$MiniforgeRoot = "C:\Tools\Miniforge3"
```

Create the recommended local environment from `conda/windows/environment.yml`:

```powershell
$RepoRoot = (Get-Location).Path

conda env create -f (Join-Path $RepoRoot "conda\windows\environment.yml")
```

Resolve the environment interpreter explicitly:

```powershell
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
```

If Miniforge is installed elsewhere, adjust the path accordingly.

The scripts in this directory assume that:

- `-PythonExe` points at the intended environment interpreter
- GCC/OpenBLAS runtime files are available from MSYS2 UCRT64, typically under a path such as `$RuntimeDllDir`

Why the explicit interpreter path is recommended:

- it makes local runs match the successful verification path exactly
- it avoids non-interactive PowerShell sessions accidentally falling back to `base`
- it lets build and verification use the same Python without relying on shell activation state

## Build Command

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

If `third_party/libcint`, `third_party/libxc-7.0.0`, or `third_party/xcfun` are absent, the CMake ExternalProject configuration in `pyscf/lib/CMakeLists.txt` will download them during the build.

Use `-Clean` for a release rebuild or when you need to reset cached build state.
For normal local iteration, omit `-Clean` so the ExternalProject build tree can be reused.

The build script prints the resolved parallel level and the final total build time:

- `parallel: 8`
- `Build time: hh:mm:ss`

## Targeted Verification

Run the targeted Windows wheel verification suite:

```powershell
$RepoRoot = (Get-Location).Path
$TmpDir = Join-Path $RepoRoot ".tmp\windows"
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
$WheelPath = Join-Path $RepoRoot "dist\pyscf-2.13.1-py3-none-win_amd64.whl"

New-Item -ItemType Directory -Force -Path (Join-Path $TmpDir "verify") | Out-Null

powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "conda\windows\verify-wheel.ps1") `
  -PythonExe $PythonExe `
  -RepoRoot $RepoRoot `
  -InstallWheel `
  -WheelPath $WheelPath `
  -Phase all `
  -OutputJson (Join-Path $TmpDir "verify\verify-wheel-all.json")
```

Available phases:

- `artifact`
- `import`
- `smoke`
- `examples`
- `packaging`
- `diagnostics`
- `all`

At the end of each run, `verify-wheel.py` prints a compact summary grouped by phase and lists failing cases separately. This is the primary CI triage view.

Both `verify-wheel.py` and `run-installed-examples.py` force the child processes to use:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

This keeps the Windows verification path stable and avoids spurious `CVHFallocate_JKArray` allocation failures in otherwise valid packaging checks.

## Unit Tests

```powershell
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"
& $PythonExe conda\windows\test_verify_wheel.py
```

## Full Installed-Wheel Example Sweep

Use the full example runner when a complete installed-wheel report is needed:

```powershell
$RepoRoot = (Get-Location).Path
$TmpDir = Join-Path $RepoRoot ".tmp\windows"
$PythonExe = "D:\ProgramData\miniforge3\envs\pyscf-win313\python.exe"

New-Item -ItemType Directory -Force -Path (Join-Path $TmpDir "examples") | Out-Null

& $PythonExe conda\windows\run-installed-examples.py `
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

& $PythonExe conda\windows\run-installed-examples.py `
  --repo-root $RepoRoot `
  --examples-root (Join-Path $RepoRoot "examples") `
  --output (Join-Path $TmpDir "examples\wheel-examples-part1.jsonl") `
  --limit 228

& $PythonExe conda\windows\run-installed-examples.py `
  --repo-root $RepoRoot `
  --examples-root (Join-Path $RepoRoot "examples") `
  --output (Join-Path $TmpDir "examples\wheel-examples-part2.jsonl") `
  --start-at examples\mcscf\13-restart.py
```

## Validation Results

### Targeted wheel verification

The current Windows targeted verification result is:

- `artifact`: PASS
- `import`: PASS
- `smoke`: PASS
- `examples`: PASS
- `packaging`: PASS
The aggregated `all` report contains:

- `passed = 26`
- `failed = 0`
- `skipped = 0`

This report confirms that the Windows wheel packaging issues addressed in this branch are fixed:

- `pyscf.cc.MomGFCCSD` is exported correctly
- `libxc` and `xcfun` DLL lookup works from an installed wheel
- representative CC, DFT, DF, ADC, AGF2, AO2MO, and MCSCF examples run outside the repository tree

The additional `artifact` phase can now be run before installation to verify wheel contents directly.

There are currently no active targeted `diagnostics` cases in
`conda/windows/verify-wheel-manifest.json`. Optional-dependency examples that
were previously tracked as expected failures now short-circuit cleanly during
Windows verification and are therefore treated as ordinary PASS cases.

The latest full targeted verification report in the workspace is:

- `.tmp/windows/verify/verify-wheel-all.json`

Recommended transient output layout for Windows packaging work:

- verification JSON: `.tmp/windows/verify/`
- installed-wheel example runs: `.tmp/windows/examples/`
- final wheel artifacts: `dist/`

The stable targeted case lists now live in:

- `conda/windows/verify-wheel-manifest.json`

The current `cibuildwheel` decision note lives in:

- `conda/windows/cibuildwheel-evaluation.md`

### Latest complete installed-wheel example sweep

The latest complete installed-wheel sweep currently available in the workspace is the two-part report stored in:

- `.tmp/wheel-examples-part1.jsonl`
- `.tmp/wheel-examples-part2.jsonl`

That complete sweep reported:

- `PASS 323`
- `FAIL 62`
- `TIMEOUT 25`
- `IMPORT_ERROR 4`
- `MISSING_DEP 41`
- `MISSING_FILE 1`

Important note:

- Those counts are from the latest complete sweep, but they still include historical packaging failures that this branch has since revalidated individually.
- In particular, the targeted `packaging` phase now passes for:
  - `examples/cc/50-simple_momgfccsd.py`
  - `examples/cc/51-momgfccsd_hermiticity.py`
  - `examples/cc/52-momgfccsd_moment_input.py`
  - `examples/cc/53-momgfccsd_weight_threshold.py`
  - `examples/cc/54-momgfccsd_self_energy.py`
  - `examples/df/11-get_j_io_free.py`
  - `examples/dft/12-camb3lyp.py`
  - `examples/dft/13-rsh_dft.py`
  - `examples/dft/15-nlc_functionals.py`

Because the full 456-example sweep has not been rerun after those targeted fixes, the `FAIL 62` figure should be treated as an upper bound on the remaining full-suite failures.

## Remaining Issues Observed In The Full Sweep

The remaining issues are dominated by non-packaging categories:

### Optional dependencies not installed

Examples that require optional packages still fail until those packages are installed:

- `geometric`
- `ase`
- `berny`
- `numba`
- `mcfun`
- `pyscf.dispersion`
- selected extension modules such as `dmrgscf`

### Example/API compatibility drift

Some examples fail because their example code does not match current public APIs:

- `examples/1-advanced/033-constrained_dft.py`
  - `get_fock()` does not accept `level_shift_factor`
- `examples/agf2/06-adc2_solver.py`
  - `pyscf.adc.radc.RADCIP` is not available
- `examples/gw/00-simple_gw.py`
  - `GWAC.kernel()` does not accept `orbs`
- `examples/2-benchmark/fock_multigrid.py`
  - `pyscf.pbc.dft.multigrid.MultiGridFFTDF` is not available

### Memory-allocation failures on large examples

Several examples still fail with:

- `malloc(...) failed in CVHFallocate_JKArray`

Representative files include:

- `examples/1-advanced/002-input_script.py`
- `examples/2-benchmark/bz.py`
- `examples/2-benchmark/c60.py`
- `examples/2-benchmark/ccsd_iteration.py`
- `examples/ao2mo/10-diff_orbs_for_ijkl.py`
- `examples/dft/20-density_fitting.py`

These failures do not currently point to wheel packaging defects. They are more consistent with runtime memory pressure and workload size on Windows.

### Long-running examples

Some examples still hit the example-runner timeout:

- large FCI examples
- large CC examples
- selected MCSCF and benchmark examples

## GitHub Release Workflow

`.github/workflows/publish.yml` now includes a Windows release job that:

1. checks out the repository
2. sets up Python 3.13
3. installs MSYS2 UCRT64 GCC and OpenBLAS
4. installs `build`, `cmake<4.0`, and Ninja
5. restores cached `third_party` source directories when available
6. restores cached ExternalProject source/download state under `build/temp.win-amd64/deps/` when available
7. runs `conda/windows/build-wheel.ps1`
8. writes verification JSON under `.tmp/windows/verify/` and runs `conda/windows/verify-wheel.ps1 -Phase all`
9. uploads the resulting wheel to PyPI

This keeps the Windows packaging path aligned with the existing release workflow structure already used for Linux and macOS.

## Future Unification

The current decision is to keep Windows on the native runner path for now and defer `cibuildwheel` migration.

See:

- `conda/windows/cibuildwheel-evaluation.md`
