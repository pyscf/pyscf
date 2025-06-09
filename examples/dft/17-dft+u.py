#!/usr/bin/env python

'''
DFT+U method for molecular systems.

Following the widely used DFT+U method in PBC calculations, the U parameter is
applied to stabilize the DFT calculations for molecular systems.

See also examples/pbc/22-dft+u.py
'''

import pyscf
from pyscf.dft.rkspu import RKSpU
from pyscf.dft.ukspu import UKSpU
mol = pyscf.M(
    atom='Fe 0. 0. 0.; O 1.8 0 0',
    basis='def2-svp',
    verbose=4
)
# Add U term to 3d orbitals
U_idx = ["Fe 3d"]
U_val = [2.0]
mf = RKSpU(mol, xc='svwn', U_idx=U_idx, U_val=U_val, minao_ref='minao')
print(mf.U_idx)
print(mf.U_val)
mf.run()

U_idx = ["Fe 3d"]
U_val = [2.0]
mf = UKSpU(mol, xc='svwn', U_idx=U_idx, U_val=U_val, minao_ref='minao')
print(mf.U_idx)
print(mf.U_val)
mf.run()
