#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Claude Code

'''
Quasi-Atomic Orbitals (QUAOs)

Ruedenberg's Quasi-Atomic Orbital analysis decomposes molecular orbitals into
atom-centered orbitals that optimally represent the molecular wavefunction in
terms of free-atom references.  The analysis yields:

- QUAO coefficients: atom-centered orbitals spanning the molecular MO space
- Occupations: diagonal of the density matrix in the QUAO basis
- KEI-BO (Kinetic Energy Interaction Bond Orders): off-diagonal kinetic energy
  weighted by density, a measure of covalent bonding
- Hybridization: s/p/d/f character of each QUAO

References:
    West, Schmidt, Gordon, Ruedenberg, JCP 139, 234107 (2013)
    Ruedenberg, Schmidt, JPCA 113, 1954 (2009)
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf import gto
from pyscf.scf import atom_hf
from pyscf.lo.orth import vec_lowdin


def quao(mol, mo_coeff, mo_occ=None, s=None, orient=True, sigma_thresh=0.1,
         verbose=None):
    '''Compute Quasi-Atomic Orbitals via SVD projection onto free-atom references.

    Args:
        mol : gto.Mole
            Molecule object.
        mo_coeff : ndarray of shape (nao, nmo)
            MO coefficients. Can be from HF, DFT, or CASSCF.
        mo_occ : ndarray of shape (nmo,), optional
            MO occupation numbers. If given, only MOs with nonzero occupation
            are used. If None, all columns of mo_coeff are used.
        s : ndarray of shape (nao, nao), optional
            AO overlap matrix. Computed if not given.
        orient : bool
            If True, rotate QUAOs on each atom to align with bond directions.
        sigma_thresh : float
            SVD singular values below this threshold are discarded (the
            corresponding free-atom AO has no molecular counterpart).
        verbose : int or Logger, optional
            Verbosity level.

    Returns:
        quao_coeff : ndarray of shape (nao, n_quao)
            QUAO coefficients in the AO basis.
        quao_labels : list of tuples
            Each entry is (atom_idx, element, l, m) identifying the QUAO.
    '''
    log = logger.new_logger(mol, verbose)

    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')

    # Select MOs to include
    if mo_occ is not None:
        idx = numpy.where(numpy.abs(mo_occ) > 1e-15)[0]
        mo = mo_coeff[:, idx]
    else:
        mo = mo_coeff

    nao_mol, nmo = mo.shape
    log.info('QUAO: %d AOs, %d MOs included', nao_mol, nmo)

    # Get free-atom reference orbitals
    atm_scf = atom_hf.get_atm_nrhf(mol)
    aoslice = mol.aoslice_by_atom()

    # Build per-atom QUAO blocks
    quao_blocks = []
    label_blocks = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in atm_scf:
            symb = mol.atom_pure_symbol(ia)

        e_hf, mo_e, ref_coeff, ref_occ = atm_scf[symb]
        # ref_coeff is (nao_atom, nao_atom) from the atomic calc

        ao0, ao1 = aoslice[ia, 2], aoslice[ia, 3]
        nao_atom = ao1 - ao0

        if nao_atom == 0:
            continue

        # Overlap block: <mol_MO | atom_AO>
        # S_cross(mu, nu) with mu in mol, nu in atom's AOs
        # We only need the atom's AO block of the full overlap
        # S_block = mo.T @ s[:, ao0:ao1] @ ref_coeff
        s_mo_atom = mo.T @ s[:, ao0:ao1]  # (nmo, nao_atom)

        # Project through free-atom MOs for better reference
        # ref_coeff columns are atomic MOs; use all of them
        s_block = s_mo_atom @ ref_coeff  # (nmo, nao_atom)

        u, sigma, vt = scipy.linalg.svd(s_block, full_matrices=False)
        # u: (nmo, min(nmo,nao_atom)), sigma: (min,), vt: (min, nao_atom)

        # Keep singular vectors above threshold
        mask = sigma > sigma_thresh
        n_keep = numpy.sum(mask)
        if n_keep == 0:
            log.warn('Atom %d (%s): all singular values below threshold', ia, symb)
            continue

        log.info('Atom %d (%s): %d QUAOs (sigma: %s)',
                 ia, symb, n_keep, numpy.array2string(sigma[:n_keep], precision=3))

        # QUAOs in AO basis: transform back from MO space
        # quao_A = mo @ u[:, :n_keep]  -- these are orthonormal in MO metric
        quao_atom = mo @ u[:, mask]  # (nao_mol, n_keep)

        # Build labels from the atomic MO composition
        # vt rows tell us which atomic MOs contribute
        labels = _atom_quao_labels(mol, ia, ref_coeff, vt[mask], ref_occ)
        quao_blocks.append(quao_atom)
        label_blocks.extend(labels)

    if not quao_blocks:
        raise RuntimeError('No QUAOs could be constructed')

    quao_coeff = numpy.hstack(quao_blocks)

    # Orthogonalize across atoms (QUAOs from different atoms may overlap)
    quao_coeff = vec_lowdin(quao_coeff, s)

    if orient:
        quao_coeff = _orient_quaos(mol, quao_coeff, label_blocks, s, log)

    log.info('QUAO: %d total quasi-atomic orbitals constructed', quao_coeff.shape[1])
    return quao_coeff, label_blocks


def kei_bo(mol, quao_coeff, dm, s=None):
    '''Kinetic Energy Interaction Bond Order in the QUAO basis.

    The KEI-BO between QUAOs A and B is defined as:
        KEI_AB = 2 * T_AB * D_AB
    where T is the kinetic energy matrix and D is the density matrix,
    both in the QUAO basis. Negative off-diagonal values indicate
    covalent bonding (kinetic energy lowering).

    Args:
        mol : gto.Mole
        quao_coeff : ndarray of shape (nao, n_quao)
            From :func:`quao`.
        dm : ndarray of shape (nao, nao)
            One-particle density matrix in the AO basis.
        s : ndarray of shape (nao, nao), optional
            AO overlap matrix.

    Returns:
        kei : ndarray of shape (n_quao, n_quao)
            KEI-BO matrix. Diagonal elements are self-interaction terms.
            Off-diagonal elements are bond orders.
    '''
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')

    t_ao = mol.intor_symmetric('int1e_kin')

    # Transform kinetic energy to QUAO basis
    t_quao = reduce(numpy.dot, (quao_coeff.T, t_ao, quao_coeff))

    # Transform density to QUAO basis
    # D_quao = C^T S D S C  (for orthogonal QUAOs, simplifies to C^T S D S C)
    # But since QUAOs are orthonormal w.r.t. S: C^T S C = I
    # So D_quao = C^T S dm S C
    sd = s @ dm
    d_quao = reduce(numpy.dot, (quao_coeff.T, sd, s, quao_coeff))

    kei = 2.0 * t_quao * d_quao
    return kei


def occupations(quao_coeff, dm, s=None, mol=None):
    '''QUAO occupation numbers (diagonal of density in QUAO basis).

    Args:
        quao_coeff : ndarray of shape (nao, n_quao)
        dm : ndarray of shape (nao, nao)
            1-RDM in AO basis.
        s : ndarray of shape (nao, nao), optional
        mol : gto.Mole, optional (needed if s is not given)

    Returns:
        occ : ndarray of shape (n_quao,)
    '''
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    sd = s @ dm
    d_quao = reduce(numpy.dot, (quao_coeff.T, sd, s, quao_coeff))
    return numpy.diag(d_quao)


def hybridization(mol, quao_coeff, quao_labels=None, s=None):
    '''Decompose each QUAO into s/p/d/f angular momentum character.

    For each QUAO, computes the fraction of the orbital's norm that
    comes from AO basis functions of each angular momentum.

    Args:
        mol : gto.Mole
        quao_coeff : ndarray of shape (nao, n_quao)
        quao_labels : list, optional (unused, for API consistency)
        s : ndarray of shape (nao, nao), optional

    Returns:
        hybrid : ndarray of shape (n_quao, 4)
            Columns are [S%, P%, D%, F%]. Each row sums to ~1.
    '''
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')

    nao_mol = mol.nao_nr()
    n_quao = quao_coeff.shape[1]
    ao_ang = _angular_momentum_for_each_ao(mol)

    hybrid = numpy.zeros((n_quao, 4))
    for l in range(4):
        mask_l = (ao_ang == l)
        if not numpy.any(mask_l):
            continue
        # Project QUAO onto l-subspace
        # Character_l = C_l^T S_l C  where C_l zeroes out non-l rows
        c_l = quao_coeff.copy()
        c_l[~mask_l] = 0.0
        # <quao_i | P_l | quao_i> = sum_mu,nu c_mu,i * s_mu,nu * c_nu,i
        # where mu,nu restricted to l-type
        proj_l = numpy.einsum('pi,pq,qi->i', c_l, s, quao_coeff)
        hybrid[:, l] = proj_l

    # Normalize rows (they should already sum to ~1 for orthonormal QUAOs)
    row_sums = hybrid.sum(axis=1)
    mask = row_sums > 1e-10
    hybrid[mask] /= row_sums[mask, None]
    return hybrid


def analyze(mol, mo_coeff, dm, mo_occ=None, orient=True, verbose=None):
    '''Run full QUAO analysis: QUAOs, occupations, KEI-BO, hybridization.

    Convenience function that calls :func:`quao`, :func:`occupations`,
    :func:`kei_bo`, and :func:`hybridization`.

    Args:
        mol : gto.Mole
        mo_coeff : ndarray (nao, nmo)
        dm : ndarray (nao, nao)
            1-RDM in AO basis.
        mo_occ : ndarray (nmo,), optional
        orient : bool
        verbose : int or Logger

    Returns:
        results : dict with keys 'quao_coeff', 'labels', 'occupations',
                  'kei_bo', 'hybridization'
    '''
    log = logger.new_logger(mol, verbose)
    s = mol.intor_symmetric('int1e_ovlp')

    quao_coeff, labels = quao(mol, mo_coeff, mo_occ=mo_occ, s=s,
                              orient=orient, verbose=verbose)
    occ = occupations(quao_coeff, dm, s=s)
    kei = kei_bo(mol, quao_coeff, dm, s=s)
    hyb = hybridization(mol, quao_coeff, s=s)

    log.info('')
    log.info('*** QUAO Analysis ***')
    log.info('')
    log.info('%-6s %-4s %-4s  %8s  %6s %6s %6s %6s',
             'Atom', 'Elem', 'QUAO', 'Occup', 'S%', 'P%', 'D%', 'F%')
    log.info('-' * 60)
    for i, (ia, elem, l, m) in enumerate(labels):
        lchar = 'spdf'[l] if l < 4 else '?'
        log.info('%-6d %-4s %s%-3s  %8.4f  %6.1f %6.1f %6.1f %6.1f',
                 ia, elem, lchar, m, occ[i],
                 hyb[i, 0]*100, hyb[i, 1]*100, hyb[i, 2]*100, hyb[i, 3]*100)

    log.info('')
    log.info('Total QUAO occupation: %.4f (should equal number of electrons)',
             numpy.sum(occ))

    # Print significant bond orders
    log.info('')
    log.info('Significant KEI Bond Orders (|KEI| > 0.01):')
    log.info('%-20s  %10s', 'Bond', 'KEI-BO')
    log.info('-' * 32)
    n_quao = len(labels)
    for i in range(n_quao):
        for j in range(i+1, n_quao):
            if abs(kei[i, j]) > 0.01:
                ia_i, elem_i = labels[i][0], labels[i][1]
                ia_j, elem_j = labels[j][0], labels[j][1]
                log.info('%s(%d)-%s(%d)  %10.4f',
                         elem_i, ia_i, elem_j, ia_j, kei[i, j])

    return {
        'quao_coeff': quao_coeff,
        'labels': labels,
        'occupations': occ,
        'kei_bo': kei,
        'hybridization': hyb,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angular_momentum_for_each_ao(mol):
    '''Return angular momentum quantum number for each AO.'''
    ao_ang = numpy.zeros(mol.nao_nr(), dtype=int)
    ao_loc = mol.ao_loc_nr()
    for i in range(mol.nbas):
        p0, p1 = ao_loc[i], ao_loc[i+1]
        ao_ang[p0:p1] = mol.bas_angular(i)
    return ao_ang


def _atom_quao_labels(mol, atom_idx, ref_coeff, vt_kept, ref_occ):
    '''Assign (atom_idx, element, l, m) labels to QUAOs of one atom.

    Uses the dominant atomic MO character from the SVD right singular vectors
    to determine the angular momentum label.
    '''
    elem = mol.atom_pure_symbol(atom_idx)
    ao0, ao1 = mol.aoslice_by_atom()[atom_idx, 2:4]
    ao_ang = _angular_momentum_for_each_ao(mol)
    atom_ao_ang = ao_ang[ao0:ao1]

    labels = []
    l_counters = {}
    for k in range(vt_kept.shape[0]):
        # vt_kept[k] is a vector in the atomic MO space
        # ref_coeff columns are atomic MOs; transform back to atomic AOs
        ao_vec = ref_coeff @ vt_kept[k]  # (nao_atom,)

        # Determine dominant angular momentum
        best_l = 0
        best_weight = 0.0
        for l in range(4):
            mask_l = (atom_ao_ang == l)
            weight = numpy.sum(ao_vec[mask_l]**2)
            if weight > best_weight:
                best_weight = weight
                best_l = l

        l_counters.setdefault(best_l, 0)
        l_counters[best_l] += 1
        m_label = str(l_counters[best_l])
        labels.append((atom_idx, elem, best_l, m_label))

    return labels


def _orient_quaos(mol, quao_coeff, labels, s, log):
    '''Rotate QUAOs on each atom to align with bond directions.

    For each atom, the QUAOs are rotated so that directional orbitals
    (p, d, f character) point toward bonded neighbors. This is done by
    maximizing the overlap between each QUAO and the direction to the
    nearest bonded atom.
    '''
    coords = mol.atom_coords()  # (natm, 3) in Bohr
    aoslice = mol.aoslice_by_atom()
    ao_ang = _angular_momentum_for_each_ao(mol)
    nao_mol = mol.nao_nr()

    # Group QUAOs by atom
    atom_quao_indices = {}
    for i, (ia, elem, l, m) in enumerate(labels):
        atom_quao_indices.setdefault(ia, []).append(i)

    # Compute inter-atom overlap weights to identify bonded neighbors
    # Use density-independent metric: sum of |<quao_i|S|quao_j>|^2
    sq = s @ quao_coeff
    q_overlap = quao_coeff.T @ sq  # should be ~identity for orthonormal QUAOs

    quao_out = quao_coeff.copy()

    for ia, indices in atom_quao_indices.items():
        if len(indices) <= 1:
            continue

        # Find neighboring atoms with significant QUAO overlap
        neighbor_atoms = set()
        for j_idx in indices:
            for jb, j_indices_b in atom_quao_indices.items():
                if jb == ia:
                    continue
                for k_idx in j_indices_b:
                    if abs(q_overlap[j_idx, k_idx]) > 0.05:
                        neighbor_atoms.add(jb)

        if not neighbor_atoms:
            continue

        # Build bond direction vectors from atom ia to neighbors
        bond_dirs = []
        for jb in neighbor_atoms:
            d = coords[jb] - coords[ia]
            norm = numpy.linalg.norm(d)
            if norm > 1e-10:
                bond_dirs.append(d / norm)

        if not bond_dirs:
            continue
        bond_dirs = numpy.array(bond_dirs)  # (n_bonds, 3)

        # Separate QUAOs into s-type (isotropic) and directional (p/d/f)
        idx_arr = numpy.array(indices)
        s_type = []
        dir_type = []
        for i, qi in enumerate(indices):
            _, _, l, _ = labels[qi]
            if l == 0:
                s_type.append(i)
            else:
                dir_type.append(i)

        if len(dir_type) <= 1 or len(bond_dirs) == 0:
            continue

        # For directional QUAOs, compute their "direction" via dipole-like
        # moment: <quao| r - R_A |quao> gives the orbital's center of charge
        # relative to atom A
        ao0, ao1 = aoslice[ia, 2], aoslice[ia, 3]

        # Use the AO coefficient pattern to determine directionality
        # For p-orbitals: the coefficients on px, py, pz AOs give direction
        # We rotate the subspace of directional QUAOs to maximize alignment
        # with bond directions

        dir_quao_idx = idx_arr[dir_type]
        n_dir = len(dir_quao_idx)

        # Compute direction vectors for each directional QUAO
        # using the p-component of coefficients as a proxy for direction
        quao_dirs = numpy.zeros((n_dir, 3))
        for k, qi in enumerate(dir_quao_idx):
            c = quao_coeff[:, qi]
            # Extract p-orbital coefficients on this atom
            p_mask = (ao_ang == 1)
            atom_mask = numpy.zeros(nao_mol, dtype=bool)
            atom_mask[ao0:ao1] = True
            combined = p_mask & atom_mask
            p_coeffs = c[combined]
            # p orbitals come in groups of 3 (px, py, pz or p-1, p0, p+1)
            if len(p_coeffs) >= 3:
                # For spherical harmonics: p-1, p0, p+1 -> y, z, x mapping
                if not mol.cart:
                    # Real spherical: m=-1 -> y, m=0 -> z, m=1 -> x
                    for g in range(len(p_coeffs) // 3):
                        quao_dirs[k, 1] += p_coeffs[g*3]**2    # y (m=-1)
                        quao_dirs[k, 2] += p_coeffs[g*3+1]**2  # z (m=0)
                        quao_dirs[k, 0] += p_coeffs[g*3+2]**2  # x (m=1)
                        # Use signed version for direction
                        quao_dirs[k, 1] = p_coeffs[g*3]
                        quao_dirs[k, 2] = p_coeffs[g*3+1]
                        quao_dirs[k, 0] = p_coeffs[g*3+2]
                else:
                    for g in range(len(p_coeffs) // 3):
                        quao_dirs[k, 0] = p_coeffs[g*3]
                        quao_dirs[k, 1] = p_coeffs[g*3+1]
                        quao_dirs[k, 2] = p_coeffs[g*3+2]

        # Normalize direction vectors
        norms = numpy.linalg.norm(quao_dirs, axis=1, keepdims=True)
        norms = numpy.where(norms > 1e-10, norms, 1.0)
        quao_dirs = quao_dirs / norms

        # Apply Procrustes-like rotation: rotate the directional QUAO subspace
        # to best align with bond directions
        if len(bond_dirs) >= n_dir:
            target = bond_dirs[:n_dir]
        else:
            # Pad with orthogonal directions
            target = _pad_directions(bond_dirs, n_dir)

        # Compute optimal rotation of the subspace
        # M = target.T @ quao_dirs -> SVD -> R = V @ U.T
        m_mat = target.T @ quao_dirs  # (3, 3) or smaller
        try:
            u_r, _, vt_r = scipy.linalg.svd(m_mat, full_matrices=False)
            rot = u_r @ vt_r  # rotation in direction space
        except scipy.linalg.LinAlgError:
            continue

        # Apply rotation to the QUAO coefficient subspace
        c_sub = quao_coeff[:, dir_quao_idx]  # (nao, n_dir)
        # We need a rotation in the QUAO index space, not in 3D space
        # Use the overlap with bond directions to define a unitary in QUAO space
        overlap_bd = numpy.zeros((n_dir, len(bond_dirs)))
        for k, qi in enumerate(dir_quao_idx):
            for b, bd in enumerate(bond_dirs):
                overlap_bd[k, b] = numpy.dot(quao_dirs[k], bd)

        # Maximize alignment via SVD of the overlap matrix
        if overlap_bd.shape[0] >= 2 and overlap_bd.shape[1] >= 1:
            u_o, _, vt_o = scipy.linalg.svd(overlap_bd, full_matrices=True)
            # u_o is the rotation in QUAO space
            c_rotated = c_sub @ u_o
            # Re-orthogonalize
            c_rotated = vec_lowdin(c_rotated, s)
            quao_out[:, dir_quao_idx] = c_rotated

    # Final re-orthogonalization of the full set
    quao_out = vec_lowdin(quao_out, s)
    return quao_out


def _pad_directions(bond_dirs, n_target):
    '''Pad bond direction vectors with orthogonal directions to reach n_target.'''
    dirs = list(bond_dirs)
    # Add cardinal directions not parallel to existing bonds
    candidates = [numpy.array([1., 0., 0.]),
                  numpy.array([0., 1., 0.]),
                  numpy.array([0., 0., 1.])]
    for c in candidates:
        if len(dirs) >= n_target:
            break
        # Check that candidate is not parallel to existing directions
        is_parallel = False
        for d in dirs:
            if abs(numpy.dot(c, d)) > 0.95:
                is_parallel = True
                break
        if not is_parallel:
            dirs.append(c)

    # If still not enough, just add random orthogonal directions
    while len(dirs) < n_target:
        v = numpy.random.randn(3)
        v /= numpy.linalg.norm(v)
        dirs.append(v)

    return numpy.array(dirs[:n_target])
