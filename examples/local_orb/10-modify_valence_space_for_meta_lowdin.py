#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import gto
from pyscf import lo
from pyscf.tools import ring, molden

'''
Meta-lowdin orthogonalization takes the minimal AOs as the occupied sets.
The valence space might not be large enough for many elements eg
* Li and Be may need include 2p as valence orbitals, that is the extra p
  shells for s-elements;
* Al, Si, P, S, Cl may need include 3d for valence space, which is the extra
  d shells for p-elements;
* extra f shells for some of the transition metal
The extra valence shell is particular useful to identify the localized orbitals.
'''

#
# Modify the default core valence settings for Be and Al
#
# 1 s-shell as core, 1 s-shell + 1 p-shell as valence
lo.set_atom_conf('Be', ('1s', '1s1p'))
# 2 s-shells + 1 p-shell as core, 1 s-shell + 1 p-shell + 1 d-shell as valence
lo.set_atom_conf('Al', ('2s1p', '1s1p1d'))
# double-d shell for Fe, ie taking 3d and 4d orbitals as valence
lo.set_atom_conf('Fe', 'double d')
# double-d shell for Mo, ie taking 4d and 5d orbitals as valence
lo.set_atom_conf('Mo', 'double d')
# Put 3d orbital in valence space for Si
lo.set_atom_conf('Si', 'polarize')

#
# Localize Be12 ring
#
mol = gto.M(atom = [('Be', x) for x in ring.make(12, 2.4)], basis='ccpvtz')
c = lo.orth.orth_ao(mol)
molden.from_mo(mol, 'be12.molden', c)

#
# Localize Al12 ring
#
mol = gto.M(atom = [('Al', x) for x in ring.make(12, 2.4)], basis='ccpvtz')
c = lo.orth.orth_ao(mol)
molden.from_mo(mol, 'al12.molden', c)

