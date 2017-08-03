#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf import lib
from pyscf import tools
from pyscf.tools import mo_mapping

'''
Read localized orbitals from molden or chkfile, then find out C 2py and 2pz orbitals
'''


mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
        tools.molden.load('benzene-631g-boys.molden')

label = mol.ao_labels(True)
comp = mo_mapping.mo_comps('C.*2p[yz]',  # regular expression
                           mol, mo_coeff)

mol = lib.chkfile.load_mol('benzene-631g.chk')
mo = lib.chkfile.load('benzene-631g.chk', 'scf/mo_coeff')
comp = mo_mapping.mo_comps('C 2p', mol, mo)

#tools.dump_mat.dump_rec(mol.stdout, mo_coeff, label, start=1)

print('rank   MO-id    components')
for i,j in enumerate(numpy.argsort(-comp)):
    print('%3d    %3d      %.10f' % (i, j, comp[j]))
