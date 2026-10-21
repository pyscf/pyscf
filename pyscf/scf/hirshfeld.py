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
Hirshfeld and CM5 atomic charges

Refs:
    Hirshfeld, Theor. Chim. Acta 44, 129 (1977)
    Marenich, Jerome, Cramer and Truhlar, J. Chem. Theory Comput. 8, 527 (2012)
'''

import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger

# CM5 parameters, as in CM5PAC (version 2015) of Marenich, Jerome, Cramer
# and Truhlar. Covalent radii in Angstrom for Z = 1..118, from Mantina,
# Valero, Cramer and Truhlar, CRC Handbook of Chemistry and Physics, 91st ed.
CM5_RADII = numpy.array([
    0.32, 0.37, 1.30, 0.99, 0.84, 0.75, 0.71, 0.64, 0.60, 0.62,
    1.60, 1.40, 1.24, 1.14, 1.09, 1.04, 1.00, 1.01, 2.00, 1.74,
    1.59, 1.48, 1.44, 1.30, 1.29, 1.24, 1.18, 1.17, 1.22, 1.20,
    1.23, 1.20, 1.20, 1.18, 1.17, 1.16, 2.15, 1.90, 1.76, 1.64,
    1.56, 1.46, 1.38, 1.36, 1.34, 1.30, 1.36, 1.40, 1.42, 1.40,
    1.40, 1.37, 1.36, 1.36, 2.38, 2.06, 1.94, 1.84, 1.90, 1.88,
    1.86, 1.85, 1.83, 1.82, 1.81, 1.80, 1.79, 1.77, 1.77, 1.78,
    1.74, 1.64, 1.58, 1.50, 1.41, 1.36, 1.32, 1.30, 1.30, 1.32,
    1.44, 1.45, 1.50, 1.42, 1.48, 1.46, 2.42, 2.11, 2.01, 1.90,
    1.84, 1.83, 1.80, 1.80, 1.73, 1.68, 1.68, 1.68, 1.65, 1.67,
    1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28,
    1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
])

# Atomwise parameters D_Z for Z = 1..118. D_{ZZ'} = D_Z - D_Z' except for the
# pairs in CM5_PAIRWISE.
CM5_ATOMWISE = numpy.array([
    0.0056, -0.1543, 0.0000, 0.0333, -0.1030, -0.0446, -0.1072, -0.0802, -0.0629, -0.1088,
    0.0184, 0.0000, -0.0726, -0.0790, -0.0756, -0.0565, -0.0444, -0.0767, 0.0130, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -0.0512, -0.0557, -0.0533, -0.0399, -0.0313, -0.0541, 0.0092, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0361, -0.0393,
    -0.0376, -0.0281, -0.0220, -0.0381, 0.0065, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    -0.0255, -0.0277, -0.0265, -0.0198, -0.0155, -0.0269, 0.0046, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, -0.0179, -0.0195, -0.0187, -0.0140, -0.0110, -0.0189,
])

CM5_PAIRWISE = {
    (1, 6): 0.0502,
    (1, 7): 0.1747,
    (1, 8): 0.1671,
    (6, 7): 0.0556,
    (6, 8): 0.0234,
    (7, 8): -0.0346,
}

CM5_ALPHA = 2.474  # Angstrom^-1


def _spin_summed_dm(mol, dm):
    dm = numpy.asarray(dm)
    nao = mol.nao_nr()
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    elif dm.shape[0] == nao * 2:  # GHF
        dm = dm[:nao,:nao] + dm[nao:,nao:]
    return dm.real

def hirshfeld_charges(mol, dm, grids=None, verbose=logger.DEBUG):
    r'''Hirshfeld charges

    .. math::

        q_A = Z_A - \int \frac{\rho^0_A(r)}{\sum_B \rho^0_B(r)} \rho(r) dr

    The free-atom densities :math:`\rho^0_A` are the spherically averaged
    atomic HF densities in the basis of mol, as used by the 'atom' initial
    guess (scf.hf.init_guess_by_atom).

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray
            Closed-shell density matrix, a pair of alpha and beta density
            matrices, or a GHF density matrix

    Kwargs:
        grids : an instance of :class:`dft.gen_grid.Grids`
            The integration grid. By default, a grid of level 3 is built.

    Returns:
        Hirshfeld charges, a 1D array of length mol.natm
    '''
    from pyscf.dft import gen_grid, numint
    from pyscf.scf import hf
    log = logger.new_logger(mol, verbose)
    if grids is None:
        grids = gen_grid.Grids(mol)
    if grids.coords is None:
        grids.build(with_non0tab=True)

    dm = _spin_summed_dm(mol, dm)
    dm0 = hf.init_guess_by_atom(mol)
    aoslices = mol.aoslice_by_atom()[:,2:]
    nao = mol.nao_nr()
    ni = numint.NumInt()
    max_memory = max(2000, mol.max_memory - lib.current_memory()[0])
    pop = numpy.zeros(mol.natm)
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 0, max_memory):
        rho = ni.eval_rho(mol, ao, dm, mask, 'LDA')
        rho0 = numpy.empty((mol.natm, weight.size))
        for ia, (p0, p1) in enumerate(aoslices):
            c0 = numpy.dot(ao[:,p0:p1], dm0[p0:p1,p0:p1])
            rho0[ia] = numpy.einsum('gi,gi->g', c0, ao[:,p0:p1])
        rho0_tot = rho0.sum(axis=0)
        mask = rho0_tot > 1e-20
        rho_w = numpy.zeros_like(rho)
        rho_w[mask] = rho[mask] * weight[mask] / rho0_tot[mask]
        pop += numpy.dot(rho0, rho_w)

    charges = mol.atom_charges() - pop
    log.note(' ** Hirshfeld atomic charges  **')
    for ia in range(mol.natm):
        log.note('charge of  %3d%s =   %10.5f', ia, mol.atom_symbol(ia), charges[ia])
    return charges

def cm5_charges(mol, dm=None, grids=None, hirshfeld=None, verbose=logger.DEBUG):
    r'''Charge Model 5 (CM5) charges

    .. math::

        q_k = q_k^{Hirshfeld} + \sum_{k' \ne k} T_{Z_k Z_{k'}}
        \exp[-\alpha (r_{kk'} - R_{Z_k} - R_{Z_{k'}})]

    Ref: Marenich, Jerome, Cramer and Truhlar, J. Chem. Theory Comput. 8,
    527 (2012)

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray
            Density matrix, from which the Hirshfeld charges are computed when
            they are not given

    Kwargs:
        hirshfeld : 1D array
            Hirshfeld charges, as returned by hirshfeld_charges

    Returns:
        CM5 charges, a 1D array of length mol.natm
    '''
    log = logger.new_logger(mol, verbose)
    if hirshfeld is None:
        hirshfeld = hirshfeld_charges(mol, dm, grids, verbose=logger.QUIET)
    charges = numpy.array(hirshfeld, dtype=float)
    # Atomic numbers of the elements, 0 for ghost atoms
    z = numpy.array([gto.charge(mol.atom_pure_symbol(ia)) if mol.atom_charge(ia) != 0
                     else 0 for ia in range(mol.natm)])
    coords = mol.atom_coords(unit='Angstrom')
    for k in range(mol.natm):
        for k1 in range(mol.natm):
            zk, zk1 = z[k], z[k1]
            if zk == 0 or zk1 == 0 or zk == zk1:
                continue
            r = numpy.linalg.norm(coords[k] - coords[k1])
            b = numpy.exp(-CM5_ALPHA * (r - CM5_RADII[zk-1] - CM5_RADII[zk1-1]))
            if (zk, zk1) in CM5_PAIRWISE:
                d = CM5_PAIRWISE[(zk, zk1)]
            elif (zk1, zk) in CM5_PAIRWISE:
                d = -CM5_PAIRWISE[(zk1, zk)]
            else:
                d = CM5_ATOMWISE[zk-1] - CM5_ATOMWISE[zk1-1]
            charges[k] += b * d

    log.note(' ** CM5 atomic charges  **')
    for ia in range(mol.natm):
        log.note('charge of  %3d%s =   %10.5f', ia, mol.atom_symbol(ia), charges[ia])
    return charges
