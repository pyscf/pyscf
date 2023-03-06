#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to control DIIS parameters and how to use different
DIIS schemes (CDIIS, ADIIS, EDIIS) in SCF calculations.

Note the calculations in this example is served as a demonstration for the use
of DIIS.  Without other convergence technique, none of them can converge.
'''

from pyscf import gto, scf, dft
from pyscf.ao2mo import addons
import scipy
import scipy.linalg
import time
import numpy.linalg
import numpy
import sys

atom_str = 'F 0 0 0; F 0 0 1.0'
mol = gto.M(atom=atom_str, basis='6-31g')

mf = scf.HF(mol)
mf.diis = scf.EDIIS()

mf.kernel()

converged_dm = mf.make_rdm1()

int_2e = addons.restore('s1', mol.intor('int2e'), mol.nao_nr())
asymm_int2e = 2 * int_2e - numpy.einsum('ijkl->ikjl', int_2e)

inv = numpy.linalg.tensorinv(asymm_int2e)

print("Shape: " + str(inv.shape))

print(inv)


