#!/usr/bin/env python

'''
Linear response DFT calculation to optimize Hubbard U for DFT+U

References:
    [1] M. Cococcioni and S. de Gironcoli, Phys. Rev. B 71, 035105 (2005)
    [2] https://hjkgrp.mit.edu/tutorials/2011-05-31-calculating-hubbard-u/
    [3] https://hjkgrp.mit.edu/tutorials/2011-06-28-hubbard-u-multiple-sites/
'''

from pyscf import gto
from pyscf.dft.ukspu import UKSpU, linear_response_u

mol = gto.M(atom='Mn 0 0 0; O 0 0 1.6',
            basis='ccpvdz-dk',
            spin=5,
            verbose=4)
# Hubbard U on Mn 3d shells only.
# Scalar relativisitic corrections are considered in this system.
mf = UKSpU(mol, xc='pbe', U_idx=['Mn 3d'], U_val=[4.5]).x2c()
mf.run()

u0 = linear_response_u(mf)
print(f'linear response U = {u0} eV')
