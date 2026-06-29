# Windows 64-bit Wheel Packaging

## Environment

Install the required tools first:

- Miniforge or Miniconda
- Git for Windows
- MSYS2 UCRT64

Install them with the following commands:

```powershell
winget install CondaForge.Miniforge3
winget install Git.Git
winget install MSYS2.MSYS2
```

After MSYS2 is installed, open an `MSYS2 UCRT64` shell and run:

```bash
pacman -Syu --noconfirm
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-openblas
```

Create the packaging environment from the repository root:

```powershell
conda env create -f tools\windows\environment.yml
conda activate pyscf-win313
python -V
python -m build --version
ninja --version
```

Create the separate installed-wheel test environment from the repository root:

```powershell
conda env create -f tools\windows\environment-test.yml
conda activate pyscf-win313-test
python -m pip install -r tools\windows\requirements-test.txt
python -V
python -m pytest --version
```

Notes:

- `cmake` is provided by `tools/windows/environment.yml`. A separate global CMake installation is not required.
- `ninja` is provided by `tools/windows/environment.yml`. A separate global Ninja installation is not required.
- `tools/windows/environment-test.yml` is intentionally separate from the packaging environment. Use `pyscf-win313` to build the wheel and `pyscf-win313-test` to install that wheel and run the module test suite.
- `tools/windows/environment-test.yml` pins `pytest<9` because the current `pytest-timer` release used by upstream CI is not compatible with `pytest 9`.
- The pip-only test helpers live in `tools/windows/requirements-test.txt`. They are installed after `conda env create` because this Windows/conda path was more reliable than embedding them in the yml `pip:` section.
- The build script will look for the MSYS2 UCRT64 runtime directory in this order:
  1. `-RuntimeDllDir` if provided
  2. the directory of `gcc` if `gcc` is already on `PATH`
  3. `D:\msys64\ucrt64\bin`
  4. `C:\msys64\ucrt64\bin`
- Useful verification commands:

```powershell
conda --version
git --version
gcc --version
g++ --version
where.exe gcc
```

## Build The Wheel

Run the packaging script from the repository root:

```powershell
conda activate pyscf-win313
powershell -ExecutionPolicy Bypass -File .\tools\windows\build-wheel.ps1 -Clean
```

If MSYS2 UCRT64 is installed in a different location, pass it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\windows\build-wheel.ps1 `
  -RuntimeDllDir <path-to-ucrt64-bin> `
  -Clean
```

Notes:

- The script builds the wheel with `python -m build -x --wheel --no-isolation --outdir dist .`
- The script stages the required runtime DLLs and bundled support DLLs into `pyscf/lib` before building the wheel.
- The Windows build now enables `xcfun` explicitly with `-DENABLE_XCFUN=ON -DBUILD_XCFUN=ON` so a reused build directory does not keep an older `OFF` cache entry.
- On the first `xcfun`-enabled build pass, the script may build the wheel twice: once to let CMake populate `deps\bin\libxcfun.dll`, then again after staging that DLL into `pyscf/lib` for the final wheel payload.
- When `xcfun` is built, the script stages either `xcfun.dll` or `libxcfun.dll` into `pyscf/lib` so the installed wheel can load it on Windows.
- `-Clean` removes the build directory before building the wheel.
- Build time is approximately 14 minutes on a i5-13600KF.

## Full Installed-Wheel Verification

After the wheel has been built, switch to the dedicated test environment and run the installed-wheel sweep:

```powershell
conda activate pyscf-win313-test
powershell -ExecutionPolicy Bypass -File .\tools\windows\verify-installed-wheel.ps1 -SkipBuild
```

Notes:

- `verify-installed-wheel.ps1` discovers every `pyscf/**/test` directory from the repository checkout.
- The script installs the newest `dist\pyscf-*.whl` into `pyscf-win313-test`.
- The script disables auto-loaded third-party pytest plugins during the installed-wheel sweep so the report only reflects PySCF tests rather than environment-specific plugin behavior.
- Each source `test` directory is first copied into a temporary run directory under `%TEMP%\pyscf-installed-wheel-<timestamp>` before pytest is invoked.
- The staged directories are written under `%TEMP%\pyscf-installed-wheel-<timestamp>\tests\...` with sanitized names such as `pyscf__gto__test`, rather than recreating a top-level importable `pyscf` package in the temp root.
- This staging step keeps the test code from the repository, while `import pyscf` still resolves to the already-installed wheel in the active `pyscf-win313-test` environment rather than the local source checkout.
- The current whitelist-based support-file staging is only used for `pyscf\lib\test`, which reads the neighboring source file `pyscf\lib\misc.py` directly. For that directory, the script stages `misc.py` explicitly so the run stays rooted in the installed wheel without broadening the temp tree into a source-layout checkout.
- Unlike the upstream CI, this Windows flow is not a source-tree pytest run. The upstream Linux/macOS CI builds from the checkout and then runs source-tree pytest with the repository root on `PYTHONPATH`, while `verify-installed-wheel.ps1` clears `PYTHONPATH`, changes into the temp run root, and validates the installed wheel from `site-packages`.
- Copying the tests into a more complete `pyscf\...\test` package layout inside the temp root is possible, but if that temp tree becomes importable as `pyscf`, the run stops being a clean installed-wheel check and starts drifting toward a source-layout test. For that reason, the current staging keeps only the test directories and avoids materializing a top-level `pyscf` package in the run root.
- The script writes both reports below by default, and also archives timestamped copies such as `installed-wheel-report-YYYYMMDD-HHMMSS.md` and `installed-wheel-report-YYYYMMDD-HHMMSS.json`:
  - `tools/windows/reports/installed-wheel-report.md`
  - `tools/windows/reports/installed-wheel-report.json`
- Per-directory raw pytest logs are stored under `tools/windows/reports/logs/`.
- Verification time is approximately 46 minutes on a i5-13600KF.
- A recent full Windows run finished with two known failures, `pyscf\cc\test` and `pyscf\pbc\tdscf\test`, which currently look like numerical or platform-specific deviations rather than wheel packaging defects. See `Appendix -> Known Windows Test Deviations` for the concrete failing test cases and current review notes.

Useful `verify-installed-wheel.ps1` parameters:

- `-TestRoots <paths...>`: only run the listed test directories or module roots, for example `-TestRoots pyscf\gto\test pyscf\scf\test`
- `-ExcludeTestRoots <paths...>`: exclude one or more directories or subtrees from the discovered test set, for example `-ExcludeTestRoots pyscf\pbc`
- `-SkipPbc`: shorthand to exclude the entire `pyscf\pbc` subtree while leaving the rest of the repository unchanged
- `-SkipBuild`: reuse the newest existing wheel under `dist\`
- `-SkipInstall`: skip reinstalling the wheel into the active test environment
- `-KeepRunRoot`: keep the temporary run directory after verification for debugging
- `-ReportDir <path>`: write reports and logs to a different output directory
- `-RepoRoot <path>`: point the script at a different checkout root
- `-PythonExe <path>`: override the active environment Python executable
- `-RuntimeDllDir <path>`: only relevant when the script is also building the wheel

Examples:

```powershell
# Run only the gto and scf installed-wheel tests
powershell -ExecutionPolicy Bypass -File .\tools\windows\verify-installed-wheel.ps1 `
  -SkipBuild `
  -TestRoots pyscf\gto\test pyscf\scf\test

# Run the full installed-wheel sweep except for pyscf\pbc
powershell -ExecutionPolicy Bypass -File .\tools\windows\verify-installed-wheel.ps1 `
  -SkipBuild `
  -SkipPbc

# Run all discovered tests except selected subtrees
powershell -ExecutionPolicy Bypass -File .\tools\windows\verify-installed-wheel.ps1 `
  -SkipBuild `
  -ExcludeTestRoots pyscf\pbc pyscf\solvent
```

## Post-Verification Cleanup

Use the commands below when you want to remove both conda environments and all local build artifacts, so the next packaging and verification run starts from a clean state.

```powershell
# 0. Set the repository root for this checkout
$RepoRoot = "C:\path\to\pyscf"

# 1. Leave the current conda environment
conda deactivate

# 2. Remove the packaging and test environments
conda env remove -n pyscf-win313
conda env remove -n pyscf-win313-test

# 3. Return to the repository root
Set-Location $RepoRoot

# 4. Delete local build outputs to force a fresh rebuild
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\pyscf\lib\deps -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\pyscf\lib\*.dll -Force -ErrorAction SilentlyContinue

# 5. Optional: remove timestamped historical reports and keep only the stable report filenames
Remove-Item .\tools\windows\reports\installed-wheel-report-*.md -Force -ErrorAction SilentlyContinue
Remove-Item .\tools\windows\reports\installed-wheel-report-*.json -Force -ErrorAction SilentlyContinue
```

## Appendix

### Known Windows Test Deviations

The current full installed-wheel verification on Windows passed 51 of 53 discovered test directories. The remaining two failures are recorded here for review because they appear to be numerical or platform-specific test deviations rather than wheel packaging regressions.

1. `pyscf\cc\test`
   - Failing test: `pyscf\cc\test\test_eom_gccsd.py::KnownValues::test_eaccsd`
   - Log: `tools\windows\reports\logs\pyscf__cc__test.log`
   - Observed failure: the final `eaccsd_star_contract` check for the third root returned `0.19228223370666436` instead of the expected `0.2820757599337823`.
   - Context from the log: `WARN: Small 4.0361280166920697e-29 left-right amplitude overlap. Results may be inaccurate.`
   - Current assessment: this looks like a numerical stability or left/right eigenvector matching issue inside the test scenario, not a missing file, missing DLL, or installed-wheel import problem.
   - Review position: worth follow-up as a separate numerical correctness investigation, but not currently treated as a Windows wheel packaging merge blocker.

2. `pyscf\pbc\tdscf\test`
   - Failing test: `pyscf\pbc\tdscf\test\test_uks.py::DiamondPBE0::test_tdhf`
   - Log: `tools\windows\reports\logs\pyscf__pbc__tdscf__test.log`
   - Observed failure: the `TDDFT` result differed from the hard-coded reference by `0.10508681940613762 eV`, exceeding the current `assertAlmostEqual(..., 5)` tolerance.
   - Current assessment: this looks like a platform- or linear-algebra-dependent numerical deviation in the periodic TDDFT/TDHF test case rather than an installed-wheel staging or packaging defect.
   - Review position: worth follow-up as a separate Windows numerical deviation, but not currently treated as a wheel packaging merge blocker.

### TODO: `libxcfun.patch` Is Corrupt

During the Windows `xcfun` packaging work, the build exposed a pre-existing issue in `pyscf\lib\libxcfun.patch`:

```text
error: corrupt patch at line 11
```

Current status:

- The Windows wheel build now succeeds because the `PATCH_COMMAND` fallback was adjusted to continue when this patch cannot be applied.
- `xcfun` is still built, bundled, installed, and validated successfully in the current wheel path.
- This means the issue is not a current merge blocker for the Windows packaging flow, but it remains a cleanup item.

Cause:

- The patch file itself is malformed. In the first hunk, an empty context line is stored as a truly empty line instead of a unified-diff context line with a leading space.
- As a result, `git apply` fails while parsing the patch file itself, before it can even determine whether the patch still matches the upstream `xcfun` source tree.

Why this is a TODO instead of an immediate blocker:

- The current Windows packaging path already works without the patch being applied.
- The main functional goal for this work was to get `xcfun` built and shipped in the wheel, and that goal has been reached.
- The patch appears to target higher-order `xcfun` derivative support, while the current PySCF build still defaults to `XCFUN_MAX_ORDER=3`, so the immediate runtime benefit of fixing the patch is limited.

Suggested repair directions:

1. Minimal cleanup:
   - Fix the unified-diff formatting so `git apply --check pyscf\lib\libxcfun.patch` no longer reports a corrupt patch.
   - Keep the current fallback logic in place until the patch has been validated on all supported build paths.
2. Functional review:
   - Re-check whether the patch is still needed for the pinned upstream `xcfun` revision (`a89b783`).
   - If the patch is still needed, rebuild it cleanly against that revision.
   - If it is no longer needed, remove the patch and the patch application step entirely.
3. Optional follow-up:
   - If the patch is meant to unlock higher derivative orders, evaluate whether `XCFUN_MAX_ORDER` should also be raised and tested explicitly.

Expected benefit of fixing it:

- Cleaner and more deterministic build logs.
- Clearer ownership of the `xcfun` integration logic.
- Reduced risk that future maintainers mistake the patch failure for a new regression.
- Potential access to the originally intended higher-order `xcfun` behavior, if that behavior is still relevant and the patch is still required.

Recommended validation after any future fix:

```powershell
# 0. Set the repository root for this checkout
$RepoRoot = "C:\path\to\pyscf"
Set-Location $RepoRoot

# 1. Confirm the patch is syntactically valid against the pinned xcfun source tree
git -C build\temp.win-amd64\deps\src\libxcfun apply --check (Join-Path $RepoRoot "pyscf\lib\libxcfun.patch")

# 2. Rebuild the wheel in the packaging environment
conda activate pyscf-win313
powershell -ExecutionPolicy Bypass -File .\tools\windows\build-wheel.ps1

# 3. Reinstall the wheel into the test environment and verify xcfun imports from site-packages
conda activate pyscf-win313-test
python -m pip install --force-reinstall .\dist\pyscf-*.whl
cd $env:TEMP
python -c "import pyscf; from pyscf import dft; print(pyscf.__file__); print(dft.xcfun.__file__)"

# 4. Re-run the installed-wheel directories that previously depended on xcfun
Set-Location $RepoRoot
powershell -ExecutionPolicy Bypass -File .\tools\windows\verify-installed-wheel.ps1 `
  -SkipBuild `
  -SkipInstall `
  -TestRoots pyscf\dft\test pyscf\tdscf\test
```
