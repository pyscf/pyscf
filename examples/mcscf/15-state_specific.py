#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CASCI/CASSCF excited state with mcscf.state_specific_ decorator
'''

from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g',
    symmetry = 1)

mf = scf.RHF(mol)
mf.kernel()

#
# 0 is ground state,  state_id = 2 means the second excited state
#
state_id = 2
mc = mcscf.CASSCF(mf, 4, 4).state_specific_(state_id)
mc.verbose = 4
mc.kernel()
mo = mc.mo_coeff

#
# For CASCI object, there are two options to compute excited state
#
# 1. Change fcisolver.nroots to compute multiple CI roots
#
mc = mcscf.CASCI(mf, 4, 4)
mc.fcisolver.nroots = 4
emc = mc.casci(mo)[0]

#
# 2. Use state_specific_ decorator to solve a specific state
#
mc = mcscf.CASCI(mf, 4, 4).state_specific_(7)
emc = mc.casci(mo)[0]

