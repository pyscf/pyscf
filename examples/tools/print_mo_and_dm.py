#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import numpy
from pyscf import gto, scf, tools

'''
Formatted output for 2D array
'''

mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')
nf = mol.nao_nr()
orb = numpy.random.random((nf,4))

#
# Print orbital coefficients
#
tools.dump_mat.dump_mo(mol, orb)

#
# Print lower triangular part of an array
#
dm = numpy.eye(3)
tools.dump_mat.dump_tri(sys.stdout, dm)

#
# Print rectangular matrix
#
mol = gto.M(atom='C 0 0 0',basis='6-31g')
dm = numpy.eye(mol.nao_nr())
tools.dump_mat.dump_rec(sys.stdout, dm, label=mol.ao_labels(), ncol=9, digits=2)


#
# Change the default output format of .analyze function.
#
mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz')
mf = scf.RHF(mol).run()
mf.analyze(verbose=5, ncol=10, digits=9)
