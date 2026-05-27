#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
TREXIO interface.

Read/write PySCF objects from/to TREXIO files. TREXIO is a standardized
file format developed by the TREX Center of Excellence for quantum
chemistry data interchange, especially with QMC codes.

https://github.com/TREX-CoE/trexio

Saved data
----------
- nucleus group (geometry, labels, point group, repulsion energy)
- electron group (electron counts)
- basis / ao groups (Gaussian basis with spherical or Cartesian AOs)
- ecp group (when ``mol.has_ecp()``)
- mo group (when a mean-field object is supplied)
- ao_1e_int (overlap, kinetic, nuc-electron, core Hamiltonian, dipoles)
- ao_2e_int_eri (electron-repulsion integrals, stored sparsely)

Example
-------
>>> from pyscf import gto, scf
>>> from pyscf.tools import trexio
>>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz')
>>> mf = scf.RHF(mol).run()
>>> trexio.to_trexio(mf, 'h2.h5')
>>> mf2 = trexio.scf_from_trexio('h2.h5')
'''

import os
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

try:
    import trexio
except ImportError as err:
    raise ImportError(
        "The 'trexio' Python package is required by pyscf.tools.trexio. "
        "Install it via `pip install trexio`."
    ) from err

# Default behaviour and a few configurable knobs
DEFAULT_ERI_BUFFER = getattr(__config__, 'trexio_eri_buffer_size', 1_000_000)
DEFAULT_ERI_THRESHOLD = getattr(__config__, 'trexio_eri_threshold', 1e-12)

_BACKENDS = {'HDF5': trexio.TREXIO_HDF5, 'TEXT': trexio.TREXIO_TEXT}


def _backend(spec):
    if isinstance(spec, str):
        try:
            return _BACKENDS[spec.upper()]
        except KeyError:
            raise ValueError(f"Unknown TREXIO backend '{spec}'. "
                             f"Choose from {list(_BACKENDS)}.")
    return spec


def _open(filename, mode, backend='HDF5'):
    return trexio.File(filename, mode=mode, back_end=_backend(backend))


# -- Spherical AO order conversion -----------------------------------------
#
# PySCF spherical order for a shell of angular momentum l: m = -l, ..., +l
# TREXIO spherical order:                                m =  0, +1, -1, +2, -2, ...
# Cartesian ordering is the same in both codes (lex order: xx, xy, xz, yy, ...).

def _pyscf_to_trexio_sph(l):
    '''Permutation P such that ao_trexio[i] = ao_pyscf[P[i]] within a 2l+1 shell.'''
    p = numpy.empty(2 * l + 1, dtype=int)
    p[0] = l
    for k in range(1, l + 1):
        p[2 * k - 1] = l + k   # +k
        p[2 * k]     = l - k   # -k
    return p


def _cart_self_overlap_ratio(l):
    '''For Cartesian shell, ratio of self-overlap of each component (x^a y^b z^c)
    to that of z^l: (2a-1)!! (2b-1)!! (2c-1)!! / (2l-1)!!.

    Returns an array of length (l+1)(l+2)/2 in PySCF/TREXIO Cartesian order.
    '''
    def dfact(n):
        if n <= 0:
            return 1
        out = 1
        for k in range(n, 0, -2):
            out *= k
        return out
    denom = dfact(2 * l - 1)
    out = []
    for a in range(l, -1, -1):
        for b in range(l - a, -1, -1):
            c = l - a - b
            out.append(dfact(2*a - 1) * dfact(2*b - 1) * dfact(2*c - 1) / denom)
    return numpy.asarray(out)


def _ao_permutation(mol):
    '''Permutation taking PySCF AO order -> TREXIO AO order.

    Returns array P such that ao_trexio[i] = ao_pyscf[P[i]].
    '''
    perm = []
    ao_loc = mol.ao_loc_nr() if not mol.cart else mol.ao_loc_nr(cart=True)
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nctr = mol.bas_nctr(ib)
        i0 = ao_loc[ib]
        if mol.cart:
            ncart = (l + 1) * (l + 2) // 2
            for ic in range(nctr):
                perm.extend(range(i0 + ic * ncart, i0 + (ic + 1) * ncart))
        else:
            sub = _pyscf_to_trexio_sph(l)
            for ic in range(nctr):
                base = i0 + ic * (2 * l + 1)
                perm.extend(base + sub)
    return numpy.asarray(perm, dtype=int)


def _ao_normalization(mol):
    '''Per-AO factor ``N_i`` (in PySCF AO order) that maps PySCF's AOs to
    unit-normalised AOs (TREXIO canonical convention).

    Computed directly from ``diag(S)`` so the file's AOs are unit-normalised
    regardless of PySCF's internal scaling. For spherical AOs this is 1; for
    Cartesians it absorbs both the ``(2a-1)!!`` angular ratio and any
    contraction-level prefactor PySCF baked into ``z^l`` (e.g. the ``4π/5``
    for d shells).
    '''
    s = mol.intor('int1e_ovlp').diagonal()
    return 1.0 / numpy.sqrt(s)


def _reorder_to_trexio(mat, perm, axes=(0,)):
    '''Apply ``perm`` along the listed axes of ``mat``.'''
    out = mat
    for ax in axes:
        out = numpy.take(out, perm, axis=ax)
    return out


def _reorder_from_trexio(mat, perm, axes=(0,)):
    inv = numpy.argsort(perm)
    return _reorder_to_trexio(mat, inv, axes)


# -- Writers ---------------------------------------------------------------

def _write_metadata(tf):
    # package_version is set automatically by trexio on file open.
    trexio.write_metadata_code_num(tf, 1)
    trexio.write_metadata_code(tf, ['PySCF'])
    trexio.write_metadata_description(tf, 'File written by pyscf.tools.trexio')
    trexio.write_metadata_unsafe(tf, 0)


def _write_nucleus(tf, mol):
    coords = mol.atom_coords()  # Bohr
    charges = numpy.asarray(mol.atom_charges(), dtype=float)
    labels = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    trexio.write_nucleus_num(tf, mol.natm)
    trexio.write_nucleus_charge(tf, charges)
    trexio.write_nucleus_coord(tf, coords)
    trexio.write_nucleus_label(tf, labels)
    pg = mol.topgroup if getattr(mol, 'topgroup', None) else 'C1'
    trexio.write_nucleus_point_group(tf, pg)
    trexio.write_nucleus_repulsion(tf, float(mol.energy_nuc()))


def _write_electron(tf, mol):
    nelec_a, nelec_b = mol.nelec
    trexio.write_electron_num(tf, int(nelec_a + nelec_b))
    trexio.write_electron_up_num(tf, int(nelec_a))
    trexio.write_electron_dn_num(tf, int(nelec_b))


def _write_basis(tf, mol):
    '''Decompose PySCF basis into one TREXIO shell per (l, contraction).

    PySCF allows general contractions (nctr > 1 sharing exponents); TREXIO
    stores one contraction per shell, so each PySCF segment is unfolded into
    ``nctr`` TREXIO shells repeating the primitives.
    '''
    nuc_idx = []
    shell_l = []
    shell_factor = []
    r_power = []
    prim_shell_idx = []
    exponents = []
    coefficients = []
    prim_factor = []

    shell_count = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        ia = mol.bas_atom(ib)
        exps = mol.bas_exp(ib)
        # libcint stores per-primitive coefficients already multiplied by
        # gto_norm (and by any contraction-level renormalisation pyscf
        # applied at build time). Factor gto_norm back out so TREXIO
        # ``prim_factor`` carries the primitive normalisation and
        # ``coefficient`` is what would be fed back into ``gto.M``.
        gnorms = numpy.asarray([gto.mole.gto_norm(l, e) for e in exps])
        libcint_coefs = mol._libcint_ctr_coeff(ib)  # shape (nprim, nctr)
        user_coefs = libcint_coefs / gnorms[:, None]
        nctr = mol.bas_nctr(ib)
        for ic in range(nctr):
            nuc_idx.append(ia)
            shell_l.append(l)
            shell_factor.append(1.0)
            r_power.append(0)
            for ip, e in enumerate(exps):
                prim_shell_idx.append(shell_count)
                exponents.append(float(e))
                coefficients.append(float(user_coefs[ip, ic]))
                prim_factor.append(float(gnorms[ip]))
            shell_count += 1

    trexio.write_basis_type(tf, 'Gaussian')
    trexio.write_basis_shell_num(tf, shell_count)
    trexio.write_basis_prim_num(tf, len(exponents))
    trexio.write_basis_nucleus_index(tf, numpy.asarray(nuc_idx, dtype=numpy.int64))
    trexio.write_basis_shell_ang_mom(tf,
                                     numpy.asarray(shell_l, dtype=numpy.int64))
    trexio.write_basis_shell_factor(tf, numpy.asarray(shell_factor))
    trexio.write_basis_r_power(tf, numpy.asarray(r_power, dtype=numpy.int64))
    trexio.write_basis_shell_index(tf,
                                   numpy.asarray(prim_shell_idx, dtype=numpy.int64))
    trexio.write_basis_exponent(tf, numpy.asarray(exponents))
    trexio.write_basis_coefficient(tf, numpy.asarray(coefficients))
    trexio.write_basis_prim_factor(tf, numpy.asarray(prim_factor))


def _write_ao(tf, mol):
    '''AO group following the canonical TREXIO convention.

    For Cartesians, ``ao.normalization[i]`` is the per-component factor that
    unit-normalises the AO relative to PySCF's ``z^l`` normalisation; MOs and
    AO integrals are rescaled accordingly elsewhere so the file represents
    unit-normalised Cartesian AOs. For spherical AOs the factor is 1.
    '''
    ao_shell = []
    shell_count = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nctr = mol.bas_nctr(ib)
        n_ao_per_contr = (l + 1) * (l + 2) // 2 if mol.cart else 2 * l + 1
        for _ in range(nctr):
            ao_shell.extend([shell_count] * n_ao_per_contr)
            shell_count += 1

    perm = _ao_permutation(mol)
    ao_norm = _ao_normalization(mol)[perm]
    trexio.write_ao_cartesian(tf, 1 if mol.cart else 0)
    trexio.write_ao_num(tf, len(ao_shell))
    trexio.write_ao_shell(tf, numpy.asarray(ao_shell, dtype=numpy.int64))
    trexio.write_ao_normalization(tf, ao_norm)


def _write_ecp(tf, mol):
    if not mol.has_ecp():
        return
    # Per-atom z_core and max-ang-mom (TREXIO requires arrays of length natm).
    z_core = numpy.zeros(mol.natm, dtype=numpy.int64)
    max_lp1 = numpy.zeros(mol.natm, dtype=numpy.int64)
    for ia in range(mol.natm):
        z_core[ia] = mol.atom_nelec_core(ia)
    # Determine max l per atom from _ecpbas (column 1; -1 == local)
    atom_l_max = {}
    for row in mol._ecpbas:
        ia, l = int(row[0]), int(row[1])
        if l >= 0:
            atom_l_max[ia] = max(atom_l_max.get(ia, -1), l)
    for ia, lmax in atom_l_max.items():
        max_lp1[ia] = lmax + 1

    ang_mom = []
    nucleus_index = []
    powers = []
    exponents = []
    coefficients = []
    # Each row of _ecpbas may pack several primitives; iterate accordingly.
    for row in mol._ecpbas:
        ia, l, nprim, radi_pow = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        ptr_exp, ptr_coef = int(row[5]), int(row[6])
        tx_l = int(max_lp1[ia]) if l < 0 else l
        for k in range(nprim):
            ang_mom.append(tx_l)
            nucleus_index.append(ia)
            powers.append(radi_pow)
            exponents.append(float(mol._env[ptr_exp + k]))
            coefficients.append(float(mol._env[ptr_coef + k]))

    trexio.write_ecp_max_ang_mom_plus_1(tf, max_lp1)
    trexio.write_ecp_z_core(tf, z_core)
    trexio.write_ecp_num(tf, len(ang_mom))
    trexio.write_ecp_ang_mom(tf, numpy.asarray(ang_mom, dtype=numpy.int64))
    trexio.write_ecp_nucleus_index(tf,
                                   numpy.asarray(nucleus_index, dtype=numpy.int64))
    trexio.write_ecp_power(tf, numpy.asarray(powers, dtype=numpy.int64))
    trexio.write_ecp_exponent(tf, numpy.asarray(exponents))
    trexio.write_ecp_coefficient(tf, numpy.asarray(coefficients))


def _to_trexio_2d(mat, perm, N_trexio):
    '''Reorder a (nao, nao) PySCF-AO matrix into TREXIO AO order and apply
    canonical unit-normalisation rescaling ``M_trexio = N_i N_j M_pyscf``.
    '''
    out = mat[numpy.ix_(perm, perm)]
    return out * numpy.outer(N_trexio, N_trexio)


def _from_trexio_2d(mat, perm, N_trexio):
    inv = numpy.argsort(perm)
    return (mat / numpy.outer(N_trexio, N_trexio))[numpy.ix_(inv, inv)]


def _write_ao_1e_integrals(tf, mol):
    perm = _ao_permutation(mol)
    N = _ao_normalization(mol)[perm]
    def w(name, m):
        getattr(trexio, f'write_ao_1e_int_{name}')(tf, _to_trexio_2d(m, perm, N))
    s = mol.intor('int1e_ovlp')
    t = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    w('overlap', s)
    w('kinetic', t)
    w('potential_n_e', v)
    h = t + v
    if mol.has_ecp():
        ecp = mol.intor('ECPscalar')
        w('ecp', ecp)
        h = h + ecp
    w('core_hamiltonian', h)
    try:
        dip = mol.intor('int1e_r', comp=3)
        w('dipole_x', dip[0])
        w('dipole_y', dip[1])
        w('dipole_z', dip[2])
    except Exception:  # pragma: no cover
        pass


def _write_sparse_4index(tf, write_fn, eri_full, perm, N_trexio,
                         buffer_size=DEFAULT_ERI_BUFFER,
                         threshold=DEFAULT_ERI_THRESHOLD):
    '''Write a 4-index PySCF-AO/MO tensor sparsely into TREXIO, applying the
    AO permutation and the canonical normalisation rescaling
    ``(ij|kl)_trexio = N_i N_j N_k N_l (ij|kl)_pyscf``.

    The input ``eri_full`` is the dense 4-index tensor in PySCF index order
    with 8-fold permutational symmetry, and ``write_fn`` is one of
    ``trexio.write_ao_2e_int_eri`` / ``trexio.write_mo_2e_int_eri``.
    '''
    n = eri_full.shape[0]
    inv = numpy.argsort(perm)  # pyscf index -> trexio index

    indices_buf = numpy.empty((buffer_size, 4), dtype=numpy.int32)
    values_buf = numpy.empty(buffer_size, dtype=numpy.float64)
    offset = 0
    count = 0

    for i in range(n):
        for j in range(i + 1):
            for k in range(i + 1):
                lmax = k if k < i else j
                for l in range(lmax + 1):
                    v = (eri_full[i, j, k, l]
                         * N_trexio[inv[i]] * N_trexio[inv[j]]
                         * N_trexio[inv[k]] * N_trexio[inv[l]])
                    if abs(v) < threshold:
                        continue
                    indices_buf[count, 0] = inv[i]
                    indices_buf[count, 1] = inv[j]
                    indices_buf[count, 2] = inv[k]
                    indices_buf[count, 3] = inv[l]
                    values_buf[count] = v
                    count += 1
                    if count == buffer_size:
                        write_fn(tf, offset, count,
                                 indices_buf[:count], values_buf[:count])
                        offset += count
                        count = 0
    if count > 0:
        write_fn(tf, offset, count,
                 indices_buf[:count], values_buf[:count])


def _write_ao_2e_integrals(tf, mol,
                           buffer_size=DEFAULT_ERI_BUFFER,
                           threshold=DEFAULT_ERI_THRESHOLD):
    '''Write 8-fold-symmetry AO ERIs sparsely in chemist's notation (ij|kl).'''
    from pyscf import ao2mo
    perm = _ao_permutation(mol)
    N = _ao_normalization(mol)[perm]
    eri = ao2mo.restore(1, mol.intor('int2e', aosym='s8'), mol.nao)
    _write_sparse_4index(tf, trexio.write_ao_2e_int_eri,
                         eri, perm, N, buffer_size, threshold)


def _write_mo_2e_integrals(tf, mol, mo_coeff,
                           buffer_size=DEFAULT_ERI_BUFFER,
                           threshold=DEFAULT_ERI_THRESHOLD):
    '''Write 8-fold-symmetry MO ERIs sparsely in chemist's notation (ij|kl).

    For UHF MOs, all spin-mixed integrals are not supported in TREXIO's
    single ``mo_2e_int_eri`` group; this writer expects a 2D ``mo_coeff``
    array (RHF/ROHF). Pass alpha or beta coefficients explicitly for UHF.
    '''
    from pyscf import ao2mo
    nmo = mo_coeff.shape[1]
    eri = ao2mo.restore(1, ao2mo.full(mol, mo_coeff, compact=False), nmo)
    # MO ERIs are in MO index order; identity permutation, normalisation = 1.
    identity = numpy.arange(nmo)
    ones = numpy.ones(nmo)
    _write_sparse_4index(tf, trexio.write_mo_2e_int_eri,
                         eri, identity, ones, buffer_size, threshold)


def _write_mo(tf, mf):
    '''Write MOs in canonical TREXIO convention: AO axis is reordered to
    TREXIO order and each component is divided by ``N_i`` so that the file
    represents coefficients in the unit-normalised AO basis.
    '''
    mol = mf.mol
    perm = _ao_permutation(mol)
    N = _ao_normalization(mol)[perm]
    mo_coeff = numpy.asarray(mf.mo_coeff)
    mo_energy = numpy.asarray(mf.mo_energy)
    mo_occ = numpy.asarray(mf.mo_occ)

    if mo_coeff.ndim == 3:
        # UHF: stack alpha then beta along the MO axis
        ca, cb = mo_coeff[0], mo_coeff[1]
        ea, eb = mo_energy[0], mo_energy[1]
        oa, ob = mo_occ[0], mo_occ[1]
        coeff = numpy.hstack([ca, cb])
        energy = numpy.concatenate([ea, eb])
        occ = numpy.concatenate([oa, ob])
        spin = numpy.concatenate([numpy.zeros(ca.shape[1], dtype=numpy.int64),
                                  numpy.ones(cb.shape[1], dtype=numpy.int64)])
        mo_type = 'UHF'
    else:
        coeff = mo_coeff
        energy = mo_energy
        occ = mo_occ
        spin = numpy.zeros(coeff.shape[1], dtype=numpy.int64)
        mo_type = 'RHF'

    coeff = coeff[perm] / N[:, None]
    # TREXIO stores MO coefficients as [mo_num, ao_num]; PySCF has [ao, mo].
    trexio.write_mo_type(tf, mo_type)
    trexio.write_mo_num(tf, coeff.shape[1])
    trexio.write_mo_coefficient(tf, numpy.ascontiguousarray(coeff.T))
    trexio.write_mo_energy(tf, energy)
    trexio.write_mo_occupation(tf, occ)
    trexio.write_mo_spin(tf, spin)


# -- Public writer ----------------------------------------------------------

def to_trexio(obj, filename, backend='HDF5', with_ao_ints=False,
              with_eri=False, with_mo_eri=False,
              eri_threshold=DEFAULT_ERI_THRESHOLD,
              overwrite=True):
    '''Write a PySCF object (``Mole`` or SCF result) to a TREXIO file in
    canonical TREXIO conventions (spherical AO ordering ``0,+1,-1,...``;
    unit-normalised Cartesian AOs via ``ao.normalization``).

    Args:
        obj: either a ``gto.Mole`` instance, or an SCF object exposing
            ``mol``, ``mo_coeff``, ``mo_energy``, ``mo_occ``.
        filename: target path.
        backend: ``'HDF5'`` (default) or ``'TEXT'``.
        with_ao_ints: also write overlap/kinetic/V_ne/core-H/dipole integrals.
        with_eri: also write the 4-index AO ERIs sparsely.
        with_mo_eri: also write the 4-index MO ERIs sparsely (RHF/ROHF only;
            for UHF, only the alpha block is written — pass UHF alpha
            coefficients separately if you need spin-resolved MO ERIs).
        eri_threshold: drop ERIs below this absolute value when writing.
        overwrite: replace any existing file at ``filename``.
    '''
    if isinstance(obj, gto.Mole):
        mol = obj
        mf = None
    else:
        mol = obj.mol
        mf = obj

    if overwrite and os.path.exists(filename):
        # TREXIO refuses to open existing files for write; remove first.
        if os.path.isdir(filename):
            import shutil
            shutil.rmtree(filename)
        else:
            os.remove(filename)

    with _open(filename, 'w', backend) as tf:
        _write_metadata(tf)
        _write_nucleus(tf, mol)
        _write_electron(tf, mol)
        _write_basis(tf, mol)
        _write_ao(tf, mol)
        if mol.has_ecp():
            _write_ecp(tf, mol)
        if mf is not None:
            _write_mo(tf, mf)
        if with_ao_ints:
            _write_ao_1e_integrals(tf, mol)
        if with_eri:
            _write_ao_2e_integrals(tf, mol, threshold=eri_threshold)
        if with_mo_eri:
            if mf is None:
                raise ValueError("with_mo_eri requires an SCF object with "
                                 "mo_coeff, not a bare Mole.")
            mo_coeff = numpy.asarray(mf.mo_coeff)
            # For UHF take alpha block only; user can extract beta manually.
            if mo_coeff.ndim == 3:
                mo_coeff = mo_coeff[0]
            _write_mo_2e_integrals(tf, mol, mo_coeff, threshold=eri_threshold)


# -- Readers ---------------------------------------------------------------

def _read_nucleus(tf):
    natm = trexio.read_nucleus_num(tf)
    charges = trexio.read_nucleus_charge(tf)
    coords = trexio.read_nucleus_coord(tf)
    labels = trexio.read_nucleus_label(tf)
    return natm, charges, coords, labels


def _read_basis(tf):
    n_shell = trexio.read_basis_shell_num(tf)
    n_prim = trexio.read_basis_prim_num(tf)
    nuc_idx = numpy.asarray(trexio.read_basis_nucleus_index(tf))
    shell_l = numpy.asarray(trexio.read_basis_shell_ang_mom(tf))
    shell_factor = numpy.asarray(trexio.read_basis_shell_factor(tf))
    shell_index = numpy.asarray(trexio.read_basis_shell_index(tf))
    exponents = numpy.asarray(trexio.read_basis_exponent(tf))
    coefficients = numpy.asarray(trexio.read_basis_coefficient(tf))
    prim_factor = numpy.asarray(trexio.read_basis_prim_factor(tf))
    return {
        'shell_num': n_shell,
        'prim_num': n_prim,
        'nucleus_index': nuc_idx,
        'shell_ang_mom': shell_l,
        'shell_factor': shell_factor,
        'shell_index': shell_index,
        'exponent': exponents,
        'coefficient': coefficients,
        'prim_factor': prim_factor,
    }


def _read_ecp(tf, natm):
    if not trexio.has_ecp_num(tf):
        return None
    n = trexio.read_ecp_num(tf)
    max_lp1 = numpy.asarray(trexio.read_ecp_max_ang_mom_plus_1(tf))
    z_core = numpy.asarray(trexio.read_ecp_z_core(tf))
    ang_mom = numpy.asarray(trexio.read_ecp_ang_mom(tf))
    nucleus_index = numpy.asarray(trexio.read_ecp_nucleus_index(tf))
    powers = numpy.asarray(trexio.read_ecp_power(tf))
    exponents = numpy.asarray(trexio.read_ecp_exponent(tf))
    coefficients = numpy.asarray(trexio.read_ecp_coefficient(tf))
    return {
        'num': n,
        'max_ang_mom_plus_1': max_lp1,
        'z_core': z_core,
        'ang_mom': ang_mom,
        'nucleus_index': nucleus_index,
        'power': powers,
        'exponent': exponents,
        'coefficient': coefficients,
    }


def _build_mol(tf):
    natm, charges, coords, labels = _read_nucleus(tf)
    basis_info = _read_basis(tf)
    cart = bool(trexio.read_ao_cartesian(tf)) if trexio.has_ao_cartesian(tf) else False

    # Per-atom list of shells in original order, each as [l, [exp, coef], ...]
    # The 'coefficient' fed to pyscf must reproduce stored = c*gto_norm(l,exp);
    # since stored should equal trexio_coefficient * trexio_prim_factor, we
    # pass c = trexio_coefficient * trexio_prim_factor / gto_norm(l, exp).
    per_atom_shells = [[] for _ in range(natm)]
    prim_by_shell = {}
    for k in range(basis_info['prim_num']):
        sh = int(basis_info['shell_index'][k])
        prim_by_shell.setdefault(sh, []).append(k)

    for ish in range(basis_info['shell_num']):
        l = int(basis_info['shell_ang_mom'][ish])
        ia = int(basis_info['nucleus_index'][ish])
        sf = float(basis_info['shell_factor'][ish])
        contraction = [l]
        for k in prim_by_shell.get(ish, []):
            e = float(basis_info['exponent'][k])
            c = float(basis_info['coefficient'][k])
            pf = float(basis_info['prim_factor'][k])
            gn = gto.mole.gto_norm(l, e)
            contraction.append([e, sf * c * pf / gn])
        per_atom_shells[ia].append(contraction)

    # Build per-atom basis dict with unique keys.  Reuse element symbols when
    # all atoms of that element share the basis; otherwise tag them per atom.
    atom_keys = [None] * natm
    per_atom_basis = {}
    by_symbol = {}
    for ia in range(natm):
        sym = labels[ia]
        shells = per_atom_shells[ia]
        if sym not in by_symbol:
            by_symbol[sym] = shells
            atom_keys[ia] = sym
            per_atom_basis[sym] = shells
        elif by_symbol[sym] == shells:
            atom_keys[ia] = sym
        else:
            tag = f"{sym}{ia}"
            atom_keys[ia] = tag
            per_atom_basis[tag] = shells

    # ECP, if present.  PySCF expects:
    #   {atom_tag: [z_core, [[l_pyscf, [<slot 0>, <slot 1>, ...]], ...]]}
    # where each slot is a list of [exp, coef] for r^p, indexed by p, and
    # l_pyscf = -1 marks the local channel.
    ecp_info = _read_ecp(tf, natm)
    ecp_dict = None
    if ecp_info is not None:
        ecp_dict = {}
        atom_channels = {}
        for k in range(ecp_info['num']):
            ia = int(ecp_info['nucleus_index'][k])
            l = int(ecp_info['ang_mom'][k])
            atom_channels.setdefault(ia, {}).setdefault(l, []).append(k)
        for ia, ldict in atom_channels.items():
            lmax_p1 = int(ecp_info['max_ang_mom_plus_1'][ia])
            zcore = int(ecp_info['z_core'][ia])
            ordered_ls = [lmax_p1] + sorted(l for l in ldict if l != lmax_p1)
            channels = []
            for l in ordered_ls:
                if l not in ldict:
                    continue
                max_p = max(int(ecp_info['power'][k]) for k in ldict[l])
                slots = [[] for _ in range(max_p + 1)]
                for k in ldict[l]:
                    p = int(ecp_info['power'][k])
                    slots[p].append([float(ecp_info['exponent'][k]),
                                     float(ecp_info['coefficient'][k])])
                pyscf_l = -1 if l == lmax_p1 else l
                channels.append([pyscf_l, slots])
            ecp_dict[atom_keys[ia]] = [zcore, channels]

    # Electron counts decide spin/charge.
    na = int(trexio.read_electron_up_num(tf)) if trexio.has_electron_up_num(tf) else None
    nb = int(trexio.read_electron_dn_num(tf)) if trexio.has_electron_dn_num(tf) else None

    atom_lines = []
    for ia in range(natm):
        x, y, z = coords[ia]
        atom_lines.append(f"{atom_keys[ia]} {x} {y} {z}")

    mol = gto.Mole()
    mol.atom = '; '.join(atom_lines)
    mol.basis = per_atom_basis
    if ecp_dict:
        mol.ecp = ecp_dict
    mol.cart = cart
    mol.unit = 'Bohr'
    if na is not None and nb is not None:
        mol.spin = na - nb
        # TREXIO nucleus.charge is Z_eff for ECP cases.
        mol.charge = int(round(charges.sum())) - na - nb
    # Disable contraction renormalisation while building. We have already
    # arranged the coefficients so that pyscf's per-primitive ``gto_norm``
    # step reproduces the original env values; the extra contraction-norm
    # rescaling would otherwise drift the basis between round-trips.
    import pyscf.gto.mole as _molmod
    saved = _molmod.NORMALIZE_GTO
    _molmod.NORMALIZE_GTO = False
    try:
        mol.build(parse_arg=False, verbose=0)
    finally:
        _molmod.NORMALIZE_GTO = saved
    return mol


def mol_from_trexio(filename, backend='HDF5'):
    '''Construct a ``gto.Mole`` from a TREXIO file.'''
    with _open(filename, 'r', backend) as tf:
        return _build_mol(tf)


def _read_mo(tf, mol):
    if not trexio.has_mo_coefficient(tf):
        return None
    coeff = numpy.asarray(trexio.read_mo_coefficient(tf)).T  # to [ao, mo]
    energy = numpy.asarray(trexio.read_mo_energy(tf)) \
        if trexio.has_mo_energy(tf) else None
    occ = numpy.asarray(trexio.read_mo_occupation(tf)) \
        if trexio.has_mo_occupation(tf) else None
    spin = numpy.asarray(trexio.read_mo_spin(tf)) \
        if trexio.has_mo_spin(tf) else None
    mo_type = trexio.read_mo_type(tf) if trexio.has_mo_type(tf) else 'RHF'

    perm = _ao_permutation(mol)
    inv = numpy.argsort(perm)
    # Read TREXIO ao.normalization to convert from unit-normalised AO
    # convention back to PySCF's. Default to 1 if the file omits the field.
    if trexio.has_ao_normalization(tf):
        N_trexio = numpy.asarray(trexio.read_ao_normalization(tf))
    else:
        N_trexio = numpy.ones(coeff.shape[0])
    coeff = (coeff * N_trexio[:, None])[inv]

    if mo_type.upper() == 'UHF' and spin is not None and (spin == 1).any():
        mask_a = spin == 0
        mask_b = spin == 1
        ca = coeff[:, mask_a]
        cb = coeff[:, mask_b]
        mo_coeff = numpy.asarray([ca, cb])
        mo_energy = numpy.asarray([energy[mask_a], energy[mask_b]]) \
            if energy is not None else None
        mo_occ = numpy.asarray([occ[mask_a], occ[mask_b]]) \
            if occ is not None else None
    else:
        mo_coeff = coeff
        mo_energy = energy
        mo_occ = occ
    return mo_type, mo_coeff, mo_energy, mo_occ


def scf_from_trexio(filename, backend='HDF5'):
    '''Reconstruct a PySCF mean-field object from a TREXIO file.

    The returned ``mf`` is not run: ``mo_coeff``, ``mo_energy`` and ``mo_occ``
    are populated and ``mf.e_tot = mf.energy_tot()`` is evaluated from them.
    '''
    from pyscf import scf
    with _open(filename, 'r', backend) as tf:
        mol = _build_mol(tf)
        result = _read_mo(tf, mol)
    if result is None:
        raise ValueError(f"No MO coefficients found in '{filename}'.")
    mo_type, mo_coeff, mo_energy, mo_occ = result
    if mo_type.upper() == 'UHF':
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    try:
        dm = mf.make_rdm1()
        mf.e_tot = mf.energy_tot(dm=dm)
    except Exception as e:  # pragma: no cover
        logger.warn(mol, "Could not recompute mf.e_tot: %s", e)
    return mf


# -- AO integral readers ---------------------------------------------------

def _read_ao_int_1e(tf, mol, name, N_trexio):
    perm = _ao_permutation(mol)
    fn = getattr(trexio, f'read_ao_1e_int_{name}')
    mat = numpy.asarray(fn(tf))
    return _from_trexio_2d(mat, perm, N_trexio)


def read_ao_1e_integrals(filename, mol=None, backend='HDF5'):
    '''Return a dict of AO 1-electron integrals stored in the file.

    The matrices are reordered back to PySCF AO ordering. If ``mol`` is
    omitted, it is rebuilt from the file.
    '''
    with _open(filename, 'r', backend) as tf:
        if mol is None:
            mol = _build_mol(tf)
        if trexio.has_ao_normalization(tf):
            N_trexio = numpy.asarray(trexio.read_ao_normalization(tf))
        else:
            N_trexio = numpy.ones(trexio.read_ao_num(tf))
        out = {}
        for tag, name in [('overlap', 'overlap'),
                          ('kinetic', 'kinetic'),
                          ('potential_n_e', 'nuc'),
                          ('ecp', 'ecp'),
                          ('core_hamiltonian', 'hcore'),
                          ('dipole_x', 'dipole_x'),
                          ('dipole_y', 'dipole_y'),
                          ('dipole_z', 'dipole_z')]:
            has = getattr(trexio, f'has_ao_1e_int_{tag}')
            if has(tf):
                out[name] = _read_ao_int_1e(tf, mol, tag, N_trexio)
    return out


def _read_sparse_4index(tf, read_fn, n, buffer_size):
    eri = numpy.zeros((n, n, n, n))
    offset = 0
    while True:
        idx, vals, n_read, eof = read_fn(tf, offset, buffer_size)
        for k in range(n_read):
            i, j, kk, ll = idx[k]
            v = vals[k]
            eri[i, j, kk, ll] = v
            eri[j, i, kk, ll] = v
            eri[i, j, ll, kk] = v
            eri[j, i, ll, kk] = v
            eri[kk, ll, i, j] = v
            eri[ll, kk, i, j] = v
            eri[kk, ll, j, i] = v
            eri[ll, kk, j, i] = v
        offset += n_read
        if eof:
            break
    return eri


def read_ao_2e_integrals(filename, mol=None, backend='HDF5',
                         buffer_size=DEFAULT_ERI_BUFFER):
    '''Return the dense 4-index AO ERI ``(ij|kl)`` in PySCF AO order and
    PySCF Cartesian convention (unit-normalisation rescaling is undone).'''
    with _open(filename, 'r', backend) as tf:
        if mol is None:
            mol = _build_mol(tf)
        perm = _ao_permutation(mol)
        inv = numpy.argsort(perm)
        if trexio.has_ao_normalization(tf):
            N_trexio = numpy.asarray(trexio.read_ao_normalization(tf))
        else:
            N_trexio = numpy.ones(mol.nao)
        eri = _read_sparse_4index(tf, trexio.read_ao_2e_int_eri,
                                  mol.nao, buffer_size)
        # Undo the N⊗N⊗N⊗N scaling, then permute all four indices.
        scale = N_trexio
        eri = eri / scale[:, None, None, None] / scale[None, :, None, None] \
                  / scale[None, None, :, None] / scale[None, None, None, :]
        eri = eri[numpy.ix_(inv, inv, inv, inv)]
    return eri


def read_mo_2e_integrals(filename, backend='HDF5',
                         buffer_size=DEFAULT_ERI_BUFFER):
    '''Return the dense 4-index MO ERI ``(ij|kl)`` in MO index order. No
    AO permutation or normalisation rescaling is applied (MOs already live
    in the canonical ordering with unit-overlap orbitals).'''
    with _open(filename, 'r', backend) as tf:
        if not trexio.has_mo_2e_int_eri(tf):
            raise ValueError(f"No MO ERIs stored in '{filename}'.")
        nmo = trexio.read_mo_num(tf)
        return _read_sparse_4index(tf, trexio.read_mo_2e_int_eri,
                                   nmo, buffer_size)
