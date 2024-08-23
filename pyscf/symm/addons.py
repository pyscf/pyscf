#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf.symm import basis
from pyscf.symm import param
from pyscf import __config__


MULTI_IRREPS = -1

def label_orb_symm(mol, irrep_name, symm_orb, mo, s=None,
                   check=getattr(__config__, 'symm_addons_label_orb_symm_check', True),
                   tol=getattr(__config__, 'symm_addons_label_orb_symm_tol', 1e-9)):
    '''Label the symmetry of given orbitals

    irrep_name can be either the symbol or the ID of the irreducible
    representation.  If the ID is provided, it returns the numeric code
    associated with XOR operator, see :py:meth:`symm.param.IRREP_ID_TABLE`

    Args:
        mol : an instance of :class:`Mole`

        irrep_name : list of str or int
            A list of irrep ID or name,  it can be either mol.irrep_id or
            mol.irrep_name.  It can affect the return "label".
        symm_orb : list of 2d array
            the symmetry adapted basis
        mo : 2d array
            the orbitals to label

    Returns:
        list of symbols or integers to represent the irreps for the given
        orbitals

    Examples:

    >>> from pyscf import gto, scf, symm
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz',verbose=0, symmetry=1)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff)
    ['Ag', 'B1u', 'Ag', 'B1u', 'B2u', 'B3u', 'Ag', 'B2g', 'B3g', 'B1u']
    >>> symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
    [0, 5, 0, 5, 6, 7, 0, 2, 3, 5]
    '''
    nmo = mo.shape[1]
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    s_mo = numpy.dot(s, mo)
    norm = numpy.zeros((len(irrep_name), nmo))
    for i, csym in enumerate(symm_orb):
        moso = numpy.dot(csym.T.conj(), s_mo)
        ovlpso = reduce(numpy.dot, (csym.T.conj(), s, csym))
        try:
            s_moso = lib.cho_solve(ovlpso, moso)
        except numpy.linalg.LinAlgError:
            ovlpso[numpy.diag_indices(csym.shape[1])] += 1e-12
            s_moso = lib.cho_solve(ovlpso, moso)
        norm[i] = numpy.einsum('ki,ki->i', moso.conj(), s_moso).real
    norm /= numpy.sum(norm, axis=0)  # for orbitals which are not normalized
    iridx = numpy.argmax(norm, axis=0)
    orbsym = numpy.asarray([irrep_name[i] for i in iridx])
    logger.debug(mol, 'irreps of each MO %s', orbsym)

    if check:
        largest_norm = norm[iridx,numpy.arange(nmo)]
        orbidx = numpy.where(largest_norm < 1-tol)[0]
        if orbidx.size > 0:
            idx = numpy.where(largest_norm < 1-tol*1e2)[0]
            if idx.size > 0:
                raise ValueError('orbitals %s not symmetrized, norm = %s' %
                                 (idx, largest_norm[idx]))
            else:
                logger.warn(mol, 'orbitals %s not strictly symmetrized.',
                            numpy.unique(orbidx))
                logger.warn(mol, 'They can be symmetrized with '
                            'pyscf.symm.symmetrize_space function.')
                logger.debug(mol, 'norm = %s', largest_norm[orbidx])
    return orbsym

def symmetrize_orb(mol, mo, orbsym=None, s=None,
                   check=getattr(__config__, 'symm_addons_symmetrize_orb_check', False)):
    '''Symmetrize the given orbitals.

    This function is different to the :func:`symmetrize_space`:  In this
    function, each orbital is symmetrized by removing non-symmetric components.
    :func:`symmetrize_space` symmetrizes the entire space by mixing different
    orbitals.

    Note this function might return non-orthogonal orbitals.
    Call :func:`symmetrize_space` to find the symmetrized orbitals that are
    close to the given orbitals.

    Args:
        mo : 2D float array
            The orbital space to symmetrize

    Kwargs:
        orbsym : integer list
            Irrep id for each orbital.  If not given, the irreps are guessed
            by calling :func:`label_orb_symm`.
        s : 2D float array
            Overlap matrix.  If given, use this overlap than the the overlap
            of the input mol.

    Returns:
        2D orbital coefficients

    Examples:

    >>> from pyscf import gto, symm, scf
    >>> mol = gto.M(atom = 'C  0  0  0; H  1  1  1; H -1 -1  1; H  1 -1 -1; H -1  1 -1',
    ...             basis = 'sto3g')
    >>> mf = scf.RHF(mol).run()
    >>> mol.build(0, 0, symmetry='D2')
    >>> mo = symm.symmetrize_orb(mol, mf.mo_coeff)
    >>> print(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))
    ['A', 'A', 'B1', 'B2', 'B3', 'A', 'B1', 'B2', 'B3']
    '''
    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    if orbsym is None:
        orbsym = label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                mo, s=s, check=check)
    orbsym = numpy.asarray(orbsym)
    s_mo = numpy.dot(s, mo)
    mo1 = numpy.empty_like(mo)

    if orbsym[0] in mol.irrep_name:
        irrep_id = mol.irrep_name
    else:
        irrep_id = mol.irrep_id

    for i, ir in enumerate(irrep_id):
        idx = orbsym == ir
        csym = mol.symm_orb[i]
        ovlpso = reduce(numpy.dot, (csym.T.conj(), s, csym))
        sc = lib.cho_solve(ovlpso, numpy.dot(csym.T.conj(), s_mo[:,idx]))
        mo1[:,idx] = numpy.dot(csym, sc)
    return mo1


def find_symmetric_mo(moso, ovlpso, thr=1e-8):
    '''Find the list of MO of that thransform like particular irrep

    Args:
        moso : 2D float array
            Overlap matrix of symmetry-adapted AO and MO, it can be obtained
            by reduce(numpy.dot, (csym.T.conj(), s, mo)), where csym taken from
            mol.symm_orb, and s is the AO overlap matrix
        ovlpso : 2D float array
            Overlap matrix between symmetry-adapted AO, it can be obtained
            by reduce(numpy.dot, (csym.T.conj(), s, csym))

    Kwargs:
        thr : float
            Threshold to consider MO symmetry-adapted

    Returns:
        1D bool array to select symmetry adapted MO
'''
    irrep_dim = ovlpso.shape[0]
    try:
        diag = numpy.einsum('ki,ki->i', moso.conj(), lib.cho_solve(ovlpso, moso))
    except numpy.linalg.LinAlgError:
        ovlpso[numpy.diag_indices(irrep_dim)] += 1e-12
        diag = numpy.einsum('ki,ki->i', moso.conj(), lib.cho_solve(ovlpso, moso))
    idx = abs(1-diag) < thr
    return idx


def symmetrize_space(mol, mo, s=None,
                     check=getattr(__config__, 'symm_addons_symmetrize_space_check', True),
                     tol=getattr(__config__, 'symm_addons_symmetrize_space_tol', 1e-9),
                     clean=getattr(__config__, 'symm_addons_symmetrize_space_clean', False)):
    '''Symmetrize the given orbital space.

    This function is different to the :func:`symmetrize_orb`:  In this function,
    the given orbitals are mixed to reveal the symmetry; :func:`symmetrize_orb`
    projects out non-symmetric components for each orbital.

    Args:
        mol : an instance of :class:`Mole`

        mo : 2D float array
            The orbital space to symmetrize

    Kwargs:
        s : 2D float array
            Overlap matrix.  If not given, overlap is computed with the input mol.
        check : bool
            Whether to check orthogonality of input orbitals and try to fix it
        tol : float
            Orthogonality tolerance
        clean : bool
            Whether to zero out symmetry forbidden orbital coefficients

    Returns:
        2D orbital coefficients

    Examples:

    >>> from pyscf import gto, symm, scf
    >>> mol = gto.M(atom = 'C  0  0  0; H  1  1  1; H -1 -1  1; H  1 -1 -1; H -1  1 -1',
    ...             basis = 'sto3g')
    >>> mf = scf.RHF(mol).run()
    >>> mol.build(0, 0, symmetry='D2')
    >>> mo = symm.symmetrize_space(mol, mf.mo_coeff)
    >>> print(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))
    ['A', 'A', 'A', 'B1', 'B1', 'B2', 'B2', 'B3', 'B3']
    '''
    from pyscf.tools import mo_mapping
    from pyscf.lo import orth

    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    nmo = mo.shape[1]
    s_mo = numpy.dot(s, mo)
    if check:
        moso = reduce(numpy.dot, (mo.conj().T, s, mo))
        max_non_orth = abs(moso - numpy.eye(nmo)).max()
        logger.debug(mol, 'Non-orthogonality of input orbitals before symmetrization %8.2e', max_non_orth)
        if max_non_orth > tol:
            logger.info(mol, 'Input orbitals are not orthogonal, perform Lowdin orthogonalization')
            mo = numpy.dot(mo, orth.lowdin(moso))
            s_mo = numpy.dot(s, mo)
            max_non_orth = abs(reduce(numpy.dot, (mo.conj().T, s, mo)) - numpy.eye(nmo)).max()
            logger.debug(mol, 'Non-orthogonality of orbitals before symmetrization %8.2e', max_non_orth)
            if max_non_orth > tol:
                raise ValueError('Input orbitals are not orthogonalized')

    mo1 = []
    for i, csym in enumerate(mol.symm_orb):
        moso = numpy.dot(csym.T.conj(), s_mo)
        ovlpso = reduce(numpy.dot, (csym.T.conj(), s, csym))

        # excluding orbitals which are already symmetrized
        idx = find_symmetric_mo(moso, ovlpso)
        orb_irrep = mo[:, idx]
        if clean:
            # rotate MO to symm-adapted AO and back to zero-out non-symmetric part
            SALC_mo = numpy.dot(csym.T.conj(), orb_irrep)
            orb_irrep = numpy.dot(csym, SALC_mo)
        moso1 = moso[:, ~idx]
        dm = numpy.dot(moso1, moso1.T.conj())

        if dm.trace() > 1e-8 and sum(idx) < csym.shape[1]:
            logger.debug(mol, '%5d orbital(s) symmetrized in irrep %3d', csym.shape[1] - sum(idx), i)
            e, u = scipy.linalg.eigh(dm, ovlpso)
            orb_symmetrized = numpy.dot(csym, u[:, abs(1-e) < 1e-6])
            orb_irrep = numpy.hstack([orb_irrep, orb_symmetrized])

        moso = numpy.dot(orb_irrep.T.conj(), numpy.dot(s, orb_irrep))
        if moso.shape[0] == 0:
            continue
        max_non_orth = abs(moso - numpy.eye(moso.shape[0])).max()
        logger.debug(mol, 'Non-orthogonality in irrep %3d after symmetrization: %8.2e', i, max_non_orth)
        if max_non_orth > tol:
            logger.info(mol, 'Symmetrized orbitals in irrep %3d not orthogonal, perform Lowdin orthogonalization', i)
            orb_irrep = numpy.dot(orb_irrep, orth.lowdin(moso))
            max_non_orth = abs(numpy.dot(orb_irrep.T.conj(), numpy.dot(s, orb_irrep))
                               - numpy.eye(orb_irrep.shape[1])).max()
            logger.debug(mol, 'Non-orthogonality in irrep %3d after symmetrization and orthogonalization: %8.2e',
                         i, max_non_orth)
        mo1.append(orb_irrep)
    mo1 = numpy.hstack(mo1)
    if mo1.shape[1] != nmo:
        raise ValueError('mo1.shape[1] != nmo: %d != %d The input orbital space is not symmetrized.\n One '
                         'possible reason is that the input mol and orbitals '
                         'are of different orientation.' % (mo1.shape[1], nmo))
    if check:
        moso1 = reduce(numpy.dot, (mo1.conj().T, s, mo1))
        max_non_orth = abs(moso1 - numpy.eye(nmo)).max()
        logger.debug(mol, 'Non-orthogonality of output orbitals after symmetrization %8.2e', max_non_orth)
        if max_non_orth > tol:
            logger.info(mol, 'Symmetrized output orbitals are not orthogonalized, perform Lowdin orthogonalization')
            mo1 = numpy.dot(mo1, orth.lowdin(moso1))
            max_non_orth = abs(reduce(numpy.dot, (mo1.conj().T, s, mo1)) - numpy.eye(nmo)).max()
            logger.debug(mol, 'Non-orthogonality of output orbitals after orthogonalization %8.2e', max_non_orth)
            if max_non_orth > tol:
                raise ValueError('Output orbitals are not orthogonalized')
    idx = mo_mapping.mo_1to1map(reduce(numpy.dot, (mo.T.conj(), s, mo1)))
    return mo1[:, idx]


def symmetrize_multidim(mol, mo, s=None,
                        check=getattr(__config__, 'symm_addons_symmetrize_space_check', True),
                        tol=getattr(__config__, 'symm_addons_symmetrize_space_tol', 1e-10),
                        keep_phase=getattr(__config__, 'symm_addons_symmetrize_multidim_keep_phase', True)):
    '''Symmetrize orbitals with respect to multidimensional irreps.

    Make coefficients of partner functions of multidimensional irreps to be the same.
    The functions uses the convention of the libmsym interface, that introduces underscores to
    the labels of multidimensional irreps partners.

    Args:
        mol : an instance of :class:`Mole`
            Symmetry-adapted basis with multidimensional irreps should be generated by libmsym
        mo : 2D float array
            The orbital space to symmetrize

    Kwargs:
        s : 2D float array
            Overlap matrix.  If not given, overlap is computed with the input mol.
        check : bool
            Whether to check orthogonality of input orbitals and try to fix it
        tol : float
            Orthogonality tolerance
        keep_phase : bool
            Whether to keep original orbital phases, rather then make them coherent with the first partner

    Returns:
        2D orbital coefficients

    '''
    from pyscf.tools import mo_mapping
    from pyscf.lo import orth

    if s is None:
        s = mol.intor_symmetric('int1e_ovlp')
    nmo = mo.shape[1]
    s_mo = numpy.dot(s, mo)
    mo1 = []
    if check:
        moso = reduce(numpy.dot, (mo.conj().T, s, mo))
        max_non_orth = abs(moso - numpy.eye(nmo)).max()
        logger.info(mol, 'Non-orthogonality of input orbitals %8.2e', max_non_orth)
        if max_non_orth > tol:
            mo_lowdin = numpy.dot(mo, orth.lowdin(moso))
            max_non_orth_lowdin = abs(reduce(numpy.dot, (mo_lowdin.conj().T, s, mo_lowdin)) -
                                      numpy.eye(nmo)).max()
            logger.info(mol, 'Non-orthogonality after Lowdin orthogonalization %8.2e', max_non_orth)
            if (max_non_orth_lowdin - max_non_orth) > tol/100:
                mo = mo_lowdin
                s_mo = numpy.dot(s, mo)
                logger.info(mol, 'Use Lowdin-orthogonalized input orbitals')
            else:
                logger.info(mol, 'Use original input orbitals')
    irreps_mdim = []
    irreps_1dim = []
    for irrep in mol.irrep_name:
        if "_" in irrep:
            added = False
            base = irrep.split("_")[0]
            for partners in irreps_mdim:
                if any(base in elem for elem in partners):
                    partners.append(irrep)
                    added = True
            if not added:
                # new base
                irreps_mdim.append([irrep])
        else:
            irreps_1dim.append(irrep)
    for irrep in irreps_1dim:
        i = mol.irrep_name.index(irrep)
        csym = mol.symm_orb[i]
        moso = numpy.dot(csym.T.conj(), s_mo)
        ovlpso = reduce(numpy.dot, (csym.T.conj(), s, csym))
        # find MO that thransform like irrep
        idx = find_symmetric_mo(moso, ovlpso)
        if sum(idx) != csym.shape[1]:
            raise ValueError('Number of symmetry-adapted MOs in not equal to dimensionality '
                             'of irrep %s: %d != %d' % (mol.irrep_name[irrep], csym.shape[1], sum(idx)))
        mo1.append(mo[:, idx])
    for partners in irreps_mdim:
        SALC_ao_partners = []
        SALC_ov_partners = []
        SALC_mo_partners = []
        for irrep in partners:
            csym = mol.symm_orb[mol.irrep_name.index(irrep)]
            SALC_ao_partners.append(csym)
            moso = numpy.dot(csym.T.conj(), s_mo)
            ovlpso = reduce(numpy.dot, (csym.T.conj(), s, csym))
            SALC_ov_partners.append(ovlpso)
            # find MO that thransform like irrep
            idx = find_symmetric_mo(moso, ovlpso)
            mo_irrep = mo[:, idx]
            if sum(idx) != csym.shape[1]:
                raise ValueError('Number of symmetry-adapted MOs in not equal to dimensionality '
                                 'of irrep %s: %d != %d' % (mol.irrep_name[irrep], csym.shape[1], sum(idx)))
            # rotate MO to symm-adapted AO
            SALC_mo = numpy.dot(csym.T.conj(), mo_irrep)
            SALC_mo_partners.append(SALC_mo)
        irrep_dim = SALC_mo_partners[0].shape[1]
        phases = [numpy.ones(irrep_dim)]
        idxes = [numpy.array(range(irrep_dim))]
        SALC_mo_average = SALC_mo_partners[0]
        for i in range(1, len(partners)):
            if not numpy.allclose(SALC_ov_partners[0], SALC_ov_partners[i]):
                raise ValueError('Symmetry-adapted AO of partner functions %s and %s are not compatible'
                                 % (partners[0], partners[i]))
            ov_0i = reduce(numpy.dot, (SALC_mo_partners[0].T.conj(),
                                       SALC_ov_partners[0],
                                       SALC_mo_partners[i]))
            idx = mo_mapping.mo_1to1map(ov_0i)
            idxes.append(idx)
            phase = numpy.rint(numpy.diagonal(ov_0i[:, idx]))
            phases.append(phase)
            partners_mo_i = phase*SALC_mo_partners[i][:, idx]
            SALC_mo_average += partners_mo_i
        SALC_mo_average /= len(partners)
        for csym, phase, SALC_mo in zip(SALC_ao_partners, phases, SALC_mo_partners):
            if keep_phase:
                SALC_mo = phase*SALC_mo_average
            else:
                SALC_mo = SALC_mo_average
            orb_done = numpy.dot(csym, SALC_mo)
            moso1 = reduce(numpy.dot, (orb_done.T.conj(), s, orb_done))
            max_non_orth = abs(moso1 - numpy.eye(moso1.shape[0])).max()
            logger.debug(mol, 'Non-orthogonality after symmetrization of %s: %s', partners[i], max_non_orth)
            if max_non_orth > tol:
                orb_done = numpy.dot(orb_done, orth.lowdin(moso1))
                max_non_orth = abs(numpy.dot(orb_done.T.conj(), numpy.dot(s, orb_done))
                                   - numpy.eye(orb_done.shape[1])).max()
                logger.debug(mol, 'After additional orthogonalization: %s', max_non_orth)
            mo1.append(orb_done)
    mo1 = numpy.hstack(mo1)
    if check:
        moso1 = reduce(numpy.dot, (mo1.conj().T, s, mo1))
        max_non_orth = abs(moso1 - numpy.eye(nmo)).max()
        logger.debug(mol, 'Non-orthogonality of output orbitals after symmetrization %8.2e', max_non_orth)
        if max_non_orth > tol:
            logger.info(mol, 'Output orbitals are not orthogonalized, perform Lowdin orthogonalization')
            mo1 = numpy.dot(mo1, orth.lowdin(moso1))
            max_non_orth = abs(reduce(numpy.dot, (mo1.conj().T, s, mo1)) - numpy.eye(nmo)).max()
            logger.info(mol, 'Non-orthogonality of output orbitals after orthogonalization %8.2e', max_non_orth)
            if max_non_orth > tol:
                raise ValueError('Output orbitals are not orthogonalized')
    idx = mo_mapping.mo_1to1map(reduce(numpy.dot, (mo.T.conj(), s, mo1)))
    return mo1[:, idx]


def std_symb(gpname):
    '''std_symb('d2h') returns D2h; std_symb('D2H') returns D2h'''
    if gpname == 'SO3':
        return gpname
    else:
        return str(gpname[0].upper() + gpname[1:].lower())

def irrep_name2id(gpname, symb):
    '''Convert the irrep symbol to internal irrep ID

    Args:
        gpname : str
            The point group symbol
        symb : str
            Irrep symbol

    Returns:
        Irrep ID, int
    '''
    gpname = std_symb(gpname)
    symb = std_symb(symb)
    if gpname == 'SO3':
        return basis.so3_irrep_symb2id(symb)
    elif gpname in ('Dooh', 'Coov'):
        return basis.linearmole_irrep_symb2id(gpname, symb)
    else:
        return param.IRREP_ID_TABLE[gpname][symb]

def irrep_id2name(gpname, irrep_id):
    '''Convert the internal irrep ID to irrep symbol

    Args:
        gpname : str
            The point group symbol
        irrep_id : int
            See IRREP_ID_TABLE in pyscf/symm/param.py

    Returns:
        Irrep symbol, str
    '''
    gpname = std_symb(gpname)
    if gpname == 'SO3':
        return basis.so3_irrep_id2symb(irrep_id)
    elif gpname in ('Dooh', 'Coov'):
        return basis.linearmole_irrep_id2symb(gpname, irrep_id)
    else:
        # irrep_id may be obtained from high symmetry (Dooh, Coov)
        irrep_id_in_d2h = irrep_id % 10
        return param.CHARACTER_TABLE[gpname][irrep_id_in_d2h][0]

def irrep_name(pgname, irrep_id):
    raise PointGroupSymmetryError('This function was obsoleted. Use irrep_id2name')

def route(target, nelec, orbsym):
    '''Pick orbitals to form a determinant which has the right symmetry.
    If solution is not found, return []
    '''
    def riter(target, nelec, orbsym):
        if nelec == 1:
            if target in orbsym:
                return [orbsym.index(target)]
            else:
                return []
        else:
            for i, ir in enumerate(orbsym):
                off = i + 1
                orb_left = orbsym[off:]
                res = riter(target ^ ir, nelec-1, orb_left)
                if res:
                    return [i] + [off+x for x in res]
            return []
    if isinstance(orbsym, numpy.ndarray):
        orbsym = orbsym.tolist()
    return riter(target, nelec, orbsym)

def eigh(h, orbsym):
    '''Solve eigenvalue problem based on the symmetry information for basis.
    See also pyscf/lib/linalg_helper.py :func:`eigh_by_blocks`

    Examples:

    >>> from pyscf import gto, symm
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz', symmetry=True)
    >>> c = numpy.hstack(mol.symm_orb)
    >>> vnuc_so = reduce(numpy.dot, (c.T, mol.intor('int1e_nuc_sph'), c))
    >>> orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
    >>> symm.eigh(vnuc_so, orbsym)
    (array([-4.50766885, -1.80666351, -1.7808565 , -1.7808565 , -1.74189134,
            -0.98998583, -0.98998583, -0.40322226, -0.30242374, -0.07608981]),
     ...)
    '''
    return lib.eigh_by_blocks(h, labels=orbsym)

def direct_prod(orbsym1, orbsym2, groupname='D2h'):
    if groupname == 'SO3':
        prod = orbsym1[:,None] ^ orbsym2
        orbsym1_not_s = orbsym1 != 0
        orbsym2_not_s = orbsym2 != 0
        prod[orbsym1_not_s[:,None] & orbsym2_not_s != 0] = MULTI_IRREPS
        prod[orbsym1[:,None] == orbsym2] = 0
    elif groupname == 'Dooh':
        orbsym1_octa = (orbsym1 // 10) * 8 + orbsym1 % 10
        orbsym2_octa = (orbsym2 // 10) * 8 + orbsym2 % 10
        prod = orbsym1_octa[:,None] ^ orbsym2_octa
        prod = (prod % 8) + (prod // 8) * 10
        orbsym1_irrepE = (orbsym1 >= 2) & (orbsym1 != 4) & (orbsym1 != 5)
        orbsym2_irrepE = (orbsym2 >= 2) & (orbsym2 != 4) & (orbsym2 != 5)
        prod[orbsym1_irrepE[:,None] & orbsym2_irrepE] = MULTI_IRREPS
        prod[orbsym1[:,None] == orbsym2] = 0
    elif groupname == 'Coov':
        prod = orbsym1[:,None] ^ orbsym2
        orbsym1_irrepE = orbsym1 >= 2
        orbsym2_irrepE = orbsym2 >= 2
        prod[orbsym1_irrepE[:,None] & orbsym2_irrepE] = MULTI_IRREPS
        prod[orbsym1[:,None] == orbsym2] = 0
    else:  # D2h and subgroup
        prod = orbsym1[:,None] ^ orbsym2
    return prod


if __name__ == "__main__":
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.build(
        atom = [['H', (0,0,0)], ['H', (0,0,1)]],
        basis = {'H': 'cc-pvdz'},
        symmetry = 1
    )
    mf = scf.RHF(mol)
    mf.scf()

    nao, nmo = mf.mo_coeff.shape
    print(label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff))
    numpy.random.seed(1)
    u = numpy.random.random((nmo,nmo))*1e-2
    u = scipy.linalg.expm(u - u.T)
    mo = symmetrize_orb(mol, numpy.dot(mf.mo_coeff, u))
    print(label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo))

    orbsym = [0, 3, 0, 2, 5, 6]
    res = route(7, 3, orbsym)
    print(res, reduce(lambda x, y: x ^ y, [orbsym[i] for i in res]))
