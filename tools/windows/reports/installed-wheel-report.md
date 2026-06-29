# Installed Wheel Verification Report

- Generated at: 2026-06-29 13:57:50 +08:00
- Conda environment: pyscf-win313-test
- pytest: pytest 8.4.2
- Wheel: dist\pyscf-2.13.1-py3-none-win_amd64.whl
- Run root: pyscf-installed-wheel-20260629-131155
- Report JSON: tools\windows\reports\installed-wheel-report.json
- Report JSON (timestamped): tools\windows\reports\installed-wheel-report-20260629-135750.json
- Report Markdown: tools\windows\reports\installed-wheel-report.md
- Report Markdown (timestamped): tools\windows\reports\installed-wheel-report-20260629-135750.md
- Summary: passed 51 / total 53, failed 2

## Installed Import

- Conda environment: pyscf-win313-test
- pyscf==2.13.1
- numpy==2.5.0
- scipy==1.18.0
- h5py==3.16.0
- pytest==8.4.2
- pytest-cov==7.1.0
- pytest-timer==1.0.0
- geometric==1.1.1
- spglib==2.7.0
- pyberny==0.6.3.post77.dev0+36a4be9

## Per-Directory Results

| Directory | Status | Seconds | Pytest Summary | Log |
| --- | --- | ---: | --- | --- |
| pyscf\adc\test | passed | 293.872 | 176 passed, 27 deselected in 292.68s (0:04:52) | tools\windows\reports\logs\pyscf__adc__test.log |
| pyscf\agf2\test | passed | 10.106 | 23 passed in 9.55s | tools\windows\reports\logs\pyscf__agf2__test.log |
| pyscf\ao2mo\test | passed | 2.048 | 23 passed in 1.53s | tools\windows\reports\logs\pyscf__ao2mo__test.log |
| pyscf\cc\test | failed | 191.79 | 1 failed, 300 passed, 4 deselected in 191.03s (0:03:11) | tools\windows\reports\logs\pyscf__cc__test.log |
| pyscf\ci\test | passed | 4.065 | 43 passed in 3.48s | tools\windows\reports\logs\pyscf__ci__test.log |
| pyscf\df\test | passed | 28.219 | 53 passed, 1 deselected, 42 warnings in 27.09s | tools\windows\reports\logs\pyscf__df__test.log |
| pyscf\dft\test | passed | 41.394 | 164 passed, 32 deselected, 2 warnings in 40.12s | tools\windows\reports\logs\pyscf__dft__test.log |
| pyscf\eph\test | passed | 3.066 | 4 passed, 4 deselected in 2.34s | tools\windows\reports\logs\pyscf__eph__test.log |
| pyscf\fci\test | passed | 41.379 | 118 passed, 1 deselected, 2 warnings in 40.02s | tools\windows\reports\logs\pyscf__fci__test.log |
| pyscf\geomopt\test | passed | 2.05 | 6 passed, 2 skipped, 1 deselected in 1.27s | tools\windows\reports\logs\pyscf__geomopt__test.log |
| pyscf\grad\test | passed | 425.579 | 184 passed, 18 deselected, 47 warnings in 424.87s (0:07:04) | tools\windows\reports\logs\pyscf__grad__test.log |
| pyscf\gto\test | passed | 11.119 | 126 passed, 2 deselected, 12 warnings in 10.71s | tools\windows\reports\logs\pyscf__gto__test.log |
| pyscf\gw\test | passed | 5.069 | 20 passed in 3.77s | tools\windows\reports\logs\pyscf__gw__test.log |
| pyscf\hessian\test | passed | 52.461 | 27 passed, 26 deselected in 51.23s | tools\windows\reports\logs\pyscf__hessian__test.log |
| pyscf\lib\ao2mo\test | passed | 1.026 | 1 passed in 0.55s | tools\windows\reports\logs\pyscf__lib__ao2mo__test.log |
| pyscf\lib\dft\test | passed | 1.05 | 10 passed in 0.67s | tools\windows\reports\logs\pyscf__lib__dft__test.log |
| pyscf\lib\gto\test | passed | 5.075 | 29 passed in 4.41s | tools\windows\reports\logs\pyscf__lib__gto__test.log |
| pyscf\lib\test | passed | 3.054 | 104 passed, 1 deselected, 1 warning in 1.78s | tools\windows\reports\logs\pyscf__lib__test.log |
| pyscf\lib\vhf\test | passed | 2.052 | 4 passed in 1.49s | tools\windows\reports\logs\pyscf__lib__vhf__test.log |
| pyscf\lo\test | passed | 4.055 | 28 passed, 1 warning in 3.58s | tools\windows\reports\logs\pyscf__lo__test.log |
| pyscf\mcpdft\test | passed | 87.816 | 62 passed in 87.00s (0:01:27) | tools\windows\reports\logs\pyscf__mcpdft__test.log |
| pyscf\mcscf\test | passed | 38.411 | 135 passed, 11 deselected in 37.22s | tools\windows\reports\logs\pyscf__mcscf__test.log |
| pyscf\md\test | passed | 26.269 | 7 passed in 25.37s | tools\windows\reports\logs\pyscf__md__test.log |
| pyscf\mp\test | passed | 8.081 | 80 passed, 1 deselected, 1 warning in 7.22s | tools\windows\reports\logs\pyscf__mp__test.log |
| pyscf\mrpt\test | passed | 3.059 | 13 passed in 1.92s | tools\windows\reports\logs\pyscf__mrpt__test.log |
| pyscf\nac\test | passed | 17.179 | 12 passed in 16.64s | tools\windows\reports\logs\pyscf__nac__test.log |
| pyscf\pbc\adc\test | passed | 10.125 | 14 passed, 11 deselected in 8.87s | tools\windows\reports\logs\pyscf__pbc__adc__test.log |
| pyscf\pbc\cc\test | passed | 328.024 | 62 passed, 22 deselected in 326.80s (0:05:26) | tools\windows\reports\logs\pyscf__pbc__cc__test.log |
| pyscf\pbc\ci\test | passed | 2.043 | 5 passed, 2 deselected in 0.96s | tools\windows\reports\logs\pyscf__pbc__ci__test.log |
| pyscf\pbc\df\test | passed | 68.549 | 228 passed, 25 deselected in 67.32s (0:01:07) | tools\windows\reports\logs\pyscf__pbc__df__test.log |
| pyscf\pbc\dft\test | passed | 174.436 | 127 passed, 15 deselected, 1 warning in 173.72s (0:02:53) | tools\windows\reports\logs\pyscf__pbc__dft__test.log |
| pyscf\pbc\grad\test | passed | 20.202 | 57 passed, 8 deselected in 19.77s | tools\windows\reports\logs\pyscf__pbc__grad__test.log |
| pyscf\pbc\gto\pseudo\test | passed | 3.05 | 9 passed in 2.06s | tools\windows\reports\logs\pyscf__pbc__gto__pseudo__test.log |
| pyscf\pbc\gto\test | passed | 3.052 | 37 passed, 2 warnings in 2.71s | tools\windows\reports\logs\pyscf__pbc__gto__test.log |
| pyscf\pbc\gw\test | passed | 74.615 | 15 passed, 2 deselected in 74.22s (0:01:14) | tools\windows\reports\logs\pyscf__pbc__gw__test.log |
| pyscf\pbc\lib\test | passed | 23.228 | 10 passed in 22.09s | tools\windows\reports\logs\pyscf__pbc__lib__test.log |
| pyscf\pbc\mp\test | passed | 24.25 | 28 passed, 7 deselected, 3 warnings in 23.06s | tools\windows\reports\logs\pyscf__pbc__mp__test.log |
| pyscf\pbc\scf\test | passed | 136.125 | 134 passed, 6 deselected, 8 warnings in 135.71s (0:02:15) | tools\windows\reports\logs\pyscf__pbc__scf__test.log |
| pyscf\pbc\symm\test | passed | 8.124 | 14 passed, 32 warnings in 6.88s | tools\windows\reports\logs\pyscf__pbc__symm__test.log |
| pyscf\pbc\tdscf\test | failed | 101.882 | 1 failed, 76 passed, 1 deselected in 100.76s (0:01:40) | tools\windows\reports\logs\pyscf__pbc__tdscf__test.log |
| pyscf\pbc\tools\test | passed | 25.266 | 19 passed, 3 warnings in 24.24s | tools\windows\reports\logs\pyscf__pbc__tools__test.log |
| pyscf\pbc\x2c\test | passed | 2.043 | 4 passed, 4 deselected in 0.83s | tools\windows\reports\logs\pyscf__pbc__x2c__test.log |
| pyscf\qmmm\pbc\test | passed | 2.04 | 1 passed in 0.81s | tools\windows\reports\logs\pyscf__qmmm__pbc__test.log |
| pyscf\qmmm\test | passed | 3.057 | 10 passed in 2.57s | tools\windows\reports\logs\pyscf__qmmm__test.log |
| pyscf\scf\test | passed | 18.197 | 241 passed, 12 deselected, 35 warnings in 17.72s | tools\windows\reports\logs\pyscf__scf__test.log |
| pyscf\sgx\grad\test | passed | 47.402 | 6 passed, 118 warnings in 46.60s | tools\windows\reports\logs\pyscf__sgx__grad__test.log |
| pyscf\sgx\test | passed | 26.256 | 10 passed, 24 warnings in 25.80s | tools\windows\reports\logs\pyscf__sgx__test.log |
| pyscf\solvent\test | passed | 114.966 | 92 passed, 53 skipped, 2 deselected, 1 warning in 114.23s (0:01:54) | tools\windows\reports\logs\pyscf__solvent__test.log |
| pyscf\soscf\test | passed | 5.074 | 23 passed in 4.07s | tools\windows\reports\logs\pyscf__soscf__test.log |
| pyscf\symm\test | passed | 2.051 | 85 passed in 1.09s | tools\windows\reports\logs\pyscf__symm__test.log |
| pyscf\tdscf\test | passed | 203.948 | 79 passed, 14 deselected, 2 warnings in 203.13s (0:03:23) | tools\windows\reports\logs\pyscf__tdscf__test.log |
| pyscf\tools\test | passed | 2.052 | 25 passed in 1.47s | tools\windows\reports\logs\pyscf__tools__test.log |
| pyscf\x2c\test | passed | 12.144 | 28 passed, 5 deselected in 11.27s | tools\windows\reports\logs\pyscf__x2c__test.log |
