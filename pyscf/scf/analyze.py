#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Wavefunction analysis based on SCF density matrices
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger

def mayer_bond_order(mol, dm, s=None, verbose=logger.DEBUG):
    r'''Mayer bond orders

    .. math::

        B_{AB} = 2 \sum_{\mu\in A} \sum_{\nu\in B}
        [(P^\alpha S)_{\mu\nu} (P^\alpha S)_{\nu\mu}
        + (P^\beta S)_{\mu\nu} (P^\beta S)_{\nu\mu}]

    which is :math:`\sum (PS)_{\mu\nu} (PS)_{\nu\mu}` for a closed-shell
    density matrix P.

    Ref: Mayer, Chem. Phys. Lett. 97, 270 (1983)

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray
            Closed-shell density matrix, a pair of alpha and beta density
            matrices, or a GHF density matrix

    Returns:
        A (natm, natm) array of bond orders, zero on the diagonal. The sum
        of a row is the valence of the atom.
    '''
    if s is None: s = mol.intor_symmetric('int1e_ovlp')
    bo = _bond_order_matrix(mol, dm, s, 'mayer')
    bo[numpy.diag_indices(mol.natm)] = 0
    _dump_bond_order(mol, bo, 'Mayer bond orders', verbose)
    return bo

def wiberg_bond_order(mol, dm, s=None, verbose=logger.DEBUG):
    r'''Wiberg bond indices in the basis of Lowdin orthogonalized AOs

    .. math::

        W_{AB} = 2 \sum_{\mu\in A} \sum_{\nu\in B}
        [(\tilde P^\alpha_{\mu\nu})^2 + (\tilde P^\beta_{\mu\nu})^2],
        \quad \tilde P = S^{1/2} P S^{1/2}

    Ref: Wiberg, Tetrahedron 24, 1083 (1968)

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray
            Closed-shell density matrix, a pair of alpha and beta density
            matrices, or a GHF density matrix

    Returns:
        A (natm, natm) array of bond indices, zero on the diagonal.
    '''
    if s is None: s = mol.intor_symmetric('int1e_ovlp')
    bo = _bond_order_matrix(mol, dm, s, 'wiberg')
    bo[numpy.diag_indices(mol.natm)] = 0
    _dump_bond_order(mol, bo, 'Wiberg bond indices (Lowdin basis)', verbose)
    return bo

def _bond_order_matrix(mol, dm, s, method):
    '''Bond orders between all pairs of atoms, the diagonal holding the
    contribution of each atom with itself'''
    nao = mol.nao_nr()
    atom_of_ao = numpy.zeros(nao, dtype=int)
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:,2:]):
        atom_of_ao[p0:p1] = ia
    dm = numpy.asarray(dm)
    if dm.ndim == 2 and dm.shape[0] == nao:
        dms = (dm * .5, dm * .5)
    elif dm.ndim == 2:  # GHF, alpha AOs followed by beta AOs
        dms = (dm,)
        if s.shape[0] == nao:
            s = scipy.linalg.block_diag(s, s)
        atom_of_ao = numpy.hstack([atom_of_ao, atom_of_ao])
    else:
        dms = dm

    if method == 'mayer':
        xs = [numpy.dot(d, s) for d in dms]
    else:
        e, u = scipy.linalg.eigh(s)
        s_half = numpy.dot(u * numpy.sqrt(e), u.conj().T)
        xs = [reduce(numpy.dot, (s_half, d, s_half)) for d in dms]
    pair = sum((x * x.T).real for x in xs) * 2

    proj = numpy.zeros((atom_of_ao.size, mol.natm))
    proj[numpy.arange(atom_of_ao.size), atom_of_ao] = 1
    return reduce(numpy.dot, (proj.T, pair, proj))

def _dump_bond_order(mol, bo, title, verbose, threshold=0.1):
    log = logger.new_logger(mol, verbose)
    if log.verbose < logger.INFO:
        return
    log.info(' ** %s  **', title)
    for ia in range(mol.natm):
        for ja in range(ia):
            if abs(bo[ia,ja]) > threshold:
                log.info('%3d%-2s - %3d%-2s %10.5f', ja, mol.atom_symbol(ja),
                         ia, mol.atom_symbol(ia), bo[ia,ja])
    for ia in range(mol.natm):
        log.info('valence of  %3d%s =   %10.5f', ia, mol.atom_symbol(ia),
                 bo[ia].sum())
