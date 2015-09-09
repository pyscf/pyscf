#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

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
#                     Core        Valence
lo.nao.AOSHELL[4 ] = ['1s0p0d0f', '2s1p0d0f'] # redefine the valence shell for Be
lo.nao.AOSHELL[13] = ['2s1p0d0f', '3s2p1d0f'] # for Al

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

