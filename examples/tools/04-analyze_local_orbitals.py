#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Read localized orbitals from molden or chkfile, then find out C 2py and 2pz orbitals
'''

import numpy
from pyscf import gto, scf
from pyscf import lo
from pyscf.tools import molden

mol = gto.M(
    atom = '''
C    0.000000000000     1.398696930758     0.000000000000
C    0.000000000000    -1.398696930758     0.000000000000
C    1.211265339156     0.699329968382     0.000000000000
C    1.211265339156    -0.699329968382     0.000000000000
C   -1.211265339156     0.699329968382     0.000000000000
C   -1.211265339156    -0.699329968382     0.000000000000
H    0.000000000000     2.491406946734     0.000000000000
H    0.000000000000    -2.491406946734     0.000000000000
H    2.157597486829     1.245660462400     0.000000000000
H    2.157597486829    -1.245660462400     0.000000000000
H   -2.157597486829     1.245660462400     0.000000000000
H   -2.157597486829    -1.245660462400     0.000000000000''',
    basis = '6-31g')
mf = scf.RHF(mol)
mf.chkfile = 'benzene-631g.chk'
mf.kernel()

pz_idx = numpy.array([17,20,21,22,23,30,36,41,42,47,48,49])-1
loc_orb = lo.Boys(mol, mf.mo_coeff[:,pz_idx]).kernel()
molden.from_mo(mol, 'benzene-631g-boys.molden', loc_orb)



import numpy
from pyscf import lib
from pyscf import tools
from pyscf.tools import mo_mapping

mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
        tools.molden.load('benzene-631g-boys.molden')

comp = mo_mapping.mo_comps('C.*2p[yz]',  # regular expression
                           mol, mo_coeff)

mol = lib.chkfile.load_mol('benzene-631g.chk')
mo = lib.chkfile.load('benzene-631g.chk', 'scf/mo_coeff')
comp = mo_mapping.mo_comps('C 2p', mol, mo)

#label = mol.ao_labels()
#tools.dump_mat.dump_rec(mol.stdout, mo_coeff, label, start=1)

print('rank   MO-id    components')
for i,j in enumerate(numpy.argsort(-comp)):
    print('%3d    %3d      %.10f' % (i, j, comp[j]))
