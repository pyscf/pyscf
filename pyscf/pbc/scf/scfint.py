#!/usr/bin/env python
#
# Authors: Garnet Chan <gkc1000@gmail.com>
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF (Hartree-Fock and DFT) tools for periodic systems at a *single* k-point,
    using analytical GTO integrals instead of PWs.

See Also:
    kscf.py : SCF tools for periodic systems with k-point *sampling*.

'''

import numpy as np
import pyscf.pbc.scf
from pyscf.pbc import gto as pgto

print('This module is deporacted and will be removed in future release.  '
      'Please use cell.pbc_intor and pbc.hf.get_hcore function instead.')

def get_hcore(cell, kpt=None):
    '''Get the core Hamiltonian AO matrix, following :func:`dft.rks.get_veff_`.'''
    if kpt is None:
        kpt = np.zeros(3)

    # TODO: these are still on grid
    if cell.pseudo is None:
        hcore = pyscf.pbc.scf.hf.get_nuc(cell, kpt)
    else:
        hcore = pyscf.pbc.scf.hf.get_pp(cell, kpt)
    hcore += get_t(cell, kpt)

    return hcore

def get_int1e_cross(intor, cell1, cell2, kpt=None, comp=1):
    r'''1-electron integrals from two molecules like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    return pgto.intor_cross(intor, cell1, cell2, comp, 0, kpt)

def get_int1e(intor, cell, kpt=None):
    '''Get the one-electron integral defined by `intor` using lattice sums.'''
    return cell.pbc_intor(intor, kpts=kpt)

def get_ovlp(cell, kpt=None):
    '''Get the overlap AO matrix.'''
    return cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpt)

def get_t(cell, kpt=None):
    '''Get the kinetic energy AO matrix.'''
    return cell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=kpt)

