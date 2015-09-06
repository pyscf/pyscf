#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import re
from pyscf import tools
from pyscf.tools import mo_mapping

'''
Read localized orbitals from molden, then find out C 2py and 2pz orbitals
'''


mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
        tools.molden.load('benzene-631g-boys.molden')

if mol._cart_gto:
    # If molden file does not have 5d,9g label, it's in Cartesian Gaussian
    label = mol.cart_labels(True)
    comp = mo_mapping.mo_comps(lambda x: re.search('C.*2p[yz]', x),
                               mol, mo_coeff, cart=True)
else:
    label = mol.spheric_labels(True)
    comp = mo_mapping.mo_comps(lambda x: re.search('C.*2p[yz]', x),
                               mol, mo_coeff)

#tools.dump_mat.dump_rec(mol.stdout, mo_coeff, label, start=1)

print('rank   MO-id    components')
for i,j in enumerate(numpy.argsort(-comp)):
    print('%3d    %3d      %.10f' % (i, j, comp[j]))
