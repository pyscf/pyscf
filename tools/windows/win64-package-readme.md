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

Notes:

- `cmake` is provided by `tools/windows/environment.yml`. A separate global CMake installation is not required.
- `ninja` is provided by `tools/windows/environment.yml`. A separate global Ninja installation is not required.
- The build script will look for the MSYS2 UCRT64 runtime directory in this order:
  1. `-RuntimeDllDir` if provided
  2. the directory of `gcc` if `gcc` is already on `PATH`
  3. `D:\msys64\ucrt64\bin`
  4. `C:\msys64\ucrt64\bin`
- Useful verification commands:

```powershell
conda --version
git --version
cmake --version
gcc --version
g++ --version
where.exe gcc
where.exe ninja
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
- The current Windows packaging path disables `xcfun` with `-DENABLE_XCFUN=OFF -DBUILD_XCFUN=OFF`

## Install The Wheel

After the build finishes, install the newest wheel into the same conda environment:

```powershell
$wheel = Get-ChildItem .\dist\pyscf-*.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -m pip install --force-reinstall $wheel.FullName
```

## Shortest Verification

Run the import check outside the repository tree:

```powershell
cd $env:TEMP
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
python -c "import pyscf, pyscf.lib; print(pyscf.__file__); print(pyscf.__version__)"
```

Run a minimal calculation smoke test:

```powershell
python -c "from pyscf import gto, dft; mol=gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g'); mf=dft.RKS(mol, xc='lda,vwn'); print(round(mf.kernel(), 8))"
```
