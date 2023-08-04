#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
TDDFT NTO analysis.
'''

from pyscf import gto, dft, tddft

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '631g',
    symmetry = True,
)

mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mytd = tddft.TDDFT(mf)
mytd.kernel()

weights_1, nto_1 = mytd.get_nto(state=1, verbose=4)
weights_2, nto_2 = mytd.get_nto(state=2, verbose=4)
weights_3, nto_3 = mytd.get_nto(state=3, verbose=4)

from pyscf.tools import molden
molden.from_mo(mol, 'nto-td-3.molden', nto_3)
