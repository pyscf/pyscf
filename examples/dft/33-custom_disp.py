#!/usr/bin/env python
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################
#  Example of DFT with custom dispersion correction (dftd3/dftd4)
###########################################################

"""
This example demonstrates the updated dispersion convention (mf.disp) in PySCF.

To run the D3 and D4 examples, install the optional dispersion dependencies:

    pip install 'pyscf-dispersion>1.5.0'

Key knobs
1) mf.xc
   The XC functional for the underlying DFT calculation (e.g. 'b3lyp', 'wb97x-v').

2) mf.disp
   The dispersion correction to apply (e.g. D3BJ or D4).

   Two common forms are supported:
   a) Version only: 'd3bj', 'd3zero', 'd3bjm', 'd3zerom', 'd3op', 'd4'
      The code will infer the dispersion parameter "method keyword" from mf.xc.

   b) Explicit version:method: 'd4:wb97x' / 'd4:wb97x-rev' / 'd4:wb97x-3c'
      - version: dispersion engine/version tag (d3bj, d3zero, d4, ...)
      - method:  the keyword used by dftd3 (https://github.com/dftd3/simple-dftd3/blob/main/assets/parameters.toml) 
                    or dftd4 (https://github.com/dftd4/dftd4/blob/main/assets/parameters.toml) to select parameters

3) mf.nlc
   Non-local correlation (e.g. VV10).
   You do not need to set this if you would like to use *-V functional since
   they will invoke VV10 by default. If you want the wB97X-V/wB97M-V XC form, but
   without VV10, and with D3/D4 instead (e.g. wB97X-3c, wB97M-D4), explicitly disable VV10 via:
       mf.nlc = 0

Below we run six minimal single-point examples for H2O. Each block creates an
SCF object, then sets mf.xc / mf.disp / mf.nlc explicitly.
"""

import pyscf
from pyscf import dft

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-svp')

print('Dispersion convention examples (tutorial)')
print('------------------------------------------------')

print()
print('Example 1: B3LYP + D3BJ')
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.disp = 'd3bj'
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')

print()
print("Example 2: B3LYP + D3BJ (explicit version:method)")
print("  'd3bj:b3lyp' means: use D3BJ, and force the D3BJ parameters of method='b3lyp'")
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.disp = 'd3bj:b3lyp'
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')

print()
print('Example 3: wB97X-V (VV10 nonlocal correlation)')
print("  Here we demonstrate mf.nlc='vv10' (and no extra dispersion via mf.disp)")
mf = dft.RKS(mol)
mf.xc = 'wb97x-v'
mf.nlc = 'vv10'
mf.disp = None
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.nlc  = {mf.nlc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')

print()
print('Example 4: wB97X-D4 (explicit D4 parameters for method=wb97x, VV10 disabled)')
print("  Key point: mf.xc='wb97x-v' + mf.nlc=0 + mf.disp='d4:wb97x'")
mf = dft.RKS(mol)
mf.xc = 'wb97x-v'
mf.nlc = 0
mf.disp = 'd4:wb97x'
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.nlc  = {mf.nlc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')

print()
print('Example 5: wB97X-D4rev (explicit D4 parameters for method=wb97x-rev, VV10 disabled)')
print("  Key point: mf.xc='wb97x-v' + mf.nlc=0 + mf.disp='d4:wb97x-rev'")
mf = dft.RKS(mol)
mf.xc = 'wb97x-v'
mf.nlc = 0
mf.disp = 'd4:wb97x-rev'
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.nlc  = {mf.nlc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')

print()
print('Example 6: wB97X-3c (use wB97X-V form but disable VV10, then add D4 parameters for wb97x-3c)')
print("  Key point: mf.xc='wb97x-v' + mf.nlc=0 + mf.disp='d4:wb97x-3c'")
print("  basis = 'Grimme vDZP'")
print("  ecp   = 'Grimme vDZP', please specify it for each element that needs ecp")
print("  To load the Grimme vDZP basis/ECP, install basis-set-exchange:")
print("      pip install basis-set-exchange")

mol_3c = pyscf.M(
    atom=atom,
    basis='Grimme vDZP',
    ecp={'O': 'Grimme vDZP'},  # H does not have ecp in Grimme vDZP.
)
mf = dft.RKS(mol_3c)
mf.xc = 'wb97x-v'
mf.nlc = 0
mf.disp = 'd4:wb97x-3c'
mf.grids.level = 5
mf.direct_scf_tol = 1e-14
mf.conv_tol = 1e-12
mf.max_cycle = 50
e_tot = mf.kernel()
print(f'  mf.xc   = {mf.xc}')
print(f'  mf.nlc  = {mf.nlc}')
print(f'  mf.disp = {mf.disp}')
print(f'  e_tot   = {e_tot}')
