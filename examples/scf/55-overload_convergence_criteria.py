#!/usr/bin/env python

# Author: Qiming Sun <osirpt.sun@gmail.com>

'''
Change the default SCF convergence criteria iterations.

SCF object has a hook check_convergence to redefine the convergence criteria.
If a convergence testing function is assigned to the attribute
check_convergence of an SCF object, the testing function will be called to
check whether SCF iteration is converged.
'''

import os
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft


mol = gto.Mole()
mol.verbose = 4
mol.atom = [
 ["C",  ( 0.000000,  0.418626, 0.000000)],
 ["H",  (-0.460595,  1.426053, 0.000000)],
 ["O",  ( 1.196516,  0.242075, 0.000000)],
 ["N",  (-0.936579, -0.568753, 0.000000)],
 ["H",  (-0.634414, -1.530889, 0.000000)],
 ["H",  (-1.921071, -0.362247, 0.000000)]
]
mol.basis = '631g**'
mol.build()

#
# Use RMS(delta-DM) as convergence criteria than the default criteria
#
mf = dft.RKS(mol)
nao = mol.nao_nr()
def check_convergence(envs):
    e_tot = envs['e_tot']
    last_hf_e = envs['last_hf_e']
    norm_ddm = envs['norm_ddm']
    rms = norm_ddm / nao
    return abs(e_tot-last_hf_e) < 1e-7 and rms < 1e-3

mf.check_convergence = check_convergence
mf.verbose = 4
mf.kernel()
