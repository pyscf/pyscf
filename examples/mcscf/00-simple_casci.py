#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CASCI calculation.
'''

import pyscf

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASCI(6, 8).run()
#
# Note this mycas object can also be created using the APIs of mcscf module:
#
# from pyscf import mcscf
# mycas = mcscf.CASCI(myhf, 6, 8).run()

# Natural occupancy in CAS space, Mulliken population etc.
mycas.verbose = 4
mycas.analyze()

#
# By default, the output of analyze() method has 6 parts.
#
# First two parts are the natural orbital analysis of the active space. The
# natural orbitals by default was expanded on the "meta-Lowdin" atomic orbitals.
# Meta-lowdin AO is one type of orthogonal orbital, which largely keeps the
# atomic nature of the core and valence space. The character of each orbitals
# can be roughly read based on the square of the coefficients.
#
# Natural occ [1.98127707 1.95671369 1.95671369 1.04270854 1.04270854 0.01987847]
# Natural orbital (expansion on meta-Lowdin AOs) in CAS space
#                #1        #2        #3        #4        #5
#   0 O 1s       0.00063   0.00000   0.00000  -0.00000   0.00000
#   0 O 2s       0.30447   0.00000  -0.00000   0.00000  -0.00000
#   0 O 3s       0.04894  -0.00000  -0.00000  -0.00000   0.00000
#   0 O 2px     -0.00000   0.05038   0.70413  -0.70572   0.04213
#   0 O 2py     -0.00000   0.70413  -0.05038  -0.04213  -0.70572
#   0 O 2pz     -0.63298  -0.00000  -0.00000   0.00000   0.00000
# ...

#
# Next part prints the overlap between the canonical MCSCF orbitals and
# HF orbitals of the initial guess. It can be used to measure how close the
# initial guess and the MCSCF results are.
# ...
# <mo_coeff-mcscf|mo_coeff-hf>  12  12    0.60371478
# <mo_coeff-mcscf|mo_coeff-hf>  12  13    0.79720035
# <mo_coeff-mcscf|mo_coeff-hf>  13  12    0.79720035
# <mo_coeff-mcscf|mo_coeff-hf>  13  13   -0.60371478
# <mo_coeff-mcscf|mo_coeff-hf>  14  14    0.99998785
# <mo_coeff-mcscf|mo_coeff-hf>  15  15   -0.61646818
# ...

#
# Next session is the analysis for CI coefficients. This part is not available
# for external FCI solver (such as DMRG, QMC).
#
# ** Largest CI components **
#   [alpha occ-orbitals] [beta occ-orbitals]            CI coefficient
#   [0 1 2 3 4]          [0 1 2]                        0.973574063441
#   [0 1 2 3 4]          [0 3 4]                        -0.187737433798

#
# The last two parts of the output are the Mulliken population analysis. To
# obtain better transferability, the electron population was computed based on
# meta-Lowdin orthogonal orbitals (than the input raw basis which may not
# possess AO character)
#
#  ** Mulliken pop on meta-lowdin orthogonal AOs  **
#  ** Mulliken pop  **
# pop of  0 O 1s        1.99999
# pop of  0 O 2s        1.78300
# pop of  0 O 3s        0.00789
# pop of  0 O 2px       1.49626
# pop of  0 O 2py       1.49626
# pop of  0 O 2pz       1.19312
# ...
