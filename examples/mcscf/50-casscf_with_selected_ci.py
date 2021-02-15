#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Use selected CI as the active space solver for CASSCF (New in PySCF-1.3)
'''

from pyscf import gto, scf, mcscf, fci

b = 1.2
mol = gto.M(
    atom = 'N 0 0 0; N 0 0 %f'%b,
    basis = 'cc-pvdz',
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 8, 6).run()

#
# Use SCI (Selected CI) to replace fcisolver
#
mc = mcscf.CASSCF(mf, 8, 6)
mc.fcisolver = fci.SCI(mol)
mc.kernel()
