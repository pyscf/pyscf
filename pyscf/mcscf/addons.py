#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import sys
from functools import reduce
import numpy
import scipy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import fci
from pyscf import scf
from pyscf import symm
from pyscf import __config__

BASE = getattr(__config__, 'mcscf_addons_sort_mo_base', 1)
MAP2HF_TOL = getattr(__config__, 'mcscf_addons_map2hf_tol', 0.4)

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range


def sort_mo(casscf, mo_coeff, caslst, base=BASE):
    '''Pick orbitals for CAS space

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        mo_coeff : ndarray or a list of ndarray
            Orbitals for CASSCF initial guess.  In the UHF-CASSCF, it's a list
            of two orbitals, for alpha and beta spin.
        caslst : list of int or nested list of int
            A list of orbital indices to represent the CAS space.  In the UHF-CASSCF,
            it's consist of two lists, for alpha and beta spin.

    Kwargs:
        base : int
            0-based (C-style) or 1-based (Fortran-style) caslst

    Returns:
        An reoreded mo_coeff, which put the orbitals given by caslst in the CAS space

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 4, 4)
    >>> cas_list = [5,6,8,9] # pi orbitals
    >>> mo = mc.sort_mo(cas_list)
    >>> mc.kernel(mo)[0]
    -109.007378939813691
    '''
    ncore = casscf.ncore

    def ext_list(nmo, caslst):
        mask = numpy.ones(nmo, dtype=bool)
        mask[caslst] = False
        idx = numpy.where(mask)[0]
        if len(idx) + casscf.ncas != nmo:
            raise ValueError('Active space size is incompatible with caslist. '
                             'ncas = %d.  caslist %s' % (casscf.ncas, caslst))
        return idx

    if isinstance(ncore, (int, numpy.integer)):
        nmo = mo_coeff.shape[1]
        if base != 0:
            caslst = [i-base for i in caslst]
        idx = ext_list(nmo, caslst)
        mo = numpy.hstack((mo_coeff[:,idx[:ncore]],
                           mo_coeff[:,caslst],
                           mo_coeff[:,idx[ncore:]]))

        if getattr(mo_coeff, 'orbsym', None) is not None:
            orbsym = mo_coeff.orbsym
            orbsym = numpy.hstack((orbsym[idx[:ncore]], orbsym[caslst],
                                   orbsym[idx[ncore:]]))
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo

    else: # UHF-based CASSCF
        if isinstance(caslst[0], (int, numpy.integer)):
            if base != 0:
                caslsta = [i-1 for i in caslst]
                caslst = (caslsta, caslsta)
        else: # two casspace lists, for alpha and beta
            if base != 0:
                caslst = ([i-base for i in caslst[0]],
                          [i-base for i in caslst[1]])
        nmo = mo_coeff[0].shape[1]
        idxa = ext_list(nmo, caslst[0])
        mo_a = numpy.hstack((mo_coeff[0][:,idxa[:ncore[0]]],
                             mo_coeff[0][:,caslst[0]],
                             mo_coeff[0][:,idxa[ncore[0]:]]))
        idxb = ext_list(nmo, caslst[1])
        mo_b = numpy.hstack((mo_coeff[1][:,idxb[:ncore[1]]],
                             mo_coeff[1][:,caslst[1]],
                             mo_coeff[1][:,idxb[ncore[1]:]]))

        if getattr(mo_coeff[0], 'orbsym', None) is not None:
            orbsyma, orbsymb = mo_coeff[0].orbsym, mo_coeff[1].orbsym
            orbsyma = numpy.hstack((orbsyma[idxa[:ncore[0]]], orbsyma[caslst[0]],
                                    orbsyma[idxa[ncore[0]:]]))
            orbsymb = numpy.hstack((orbsymb[idxb[:ncore[1]]], orbsymb[caslst[1]],
                                    orbsymb[idxb[ncore[1]:]]))
            mo_a = lib.tag_array(mo_a, orbsym=orbsyma)
            mo_b = lib.tag_array(mo_b, orbsym=orbsymb)
        return (mo_a, mo_b)

def select_mo_by_irrep(casscf,  cas_occ_num, mo=None, base=BASE):
    raise RuntimeError('This function has been replaced by function caslst_by_irrep')

def caslst_by_irrep(casscf, mo_coeff, cas_irrep_nocc,
                    cas_irrep_ncore=None, s=None, base=BASE):
    '''Given number of active orbitals for each irrep, return the orbital
    indices of active space

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        cas_irrep_nocc : list or dict
            Number of active orbitals for each irrep.  It can be a dict, eg
            {'A1': 2, 'B2': 4} to indicate the active space size based on
            irrep names, or {0: 2, 3: 4} for irrep Id,  or a list [2, 0, 0, 4]
            (identical to {0: 2, 3: 4}) in which the list index is served as
            the irrep Id.

    Kwargs:
        cas_irrep_ncore : list or dict
            Number of closed shells for each irrep.  It can be a dict, eg
            {'A1': 6, 'B2': 4} to indicate the closed shells based on
            irrep names, or {0: 6, 3: 4} for irrep Id,  or a list [6, 0, 0, 4]
            (identical to {0: 6, 3: 4}) in which the list index is served as
            the irrep Id.  If cas_irrep_ncore is not given, the program
            will generate a guess based on the lowest :attr:`CASCI.ncore`
            orbitals.
        s : ndarray
            overlap matrix
        base : int
            0-based (C-like) or 1-based (Fortran-like) caslst

    Returns:
        A list of orbital indices

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvtz', symmetry=True, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> mc = mcscf.CASSCF(mf, 12, 4)
    >>> mcscf.caslst_by_irrep(mc, mf.mo_coeff, {'E1gx':4, 'E1gy':4, 'E1ux':2, 'E1uy':2})
    [5, 7, 8, 10, 11, 14, 15, 20, 25, 26, 31, 32]
    '''
    mol = casscf.mol
    log = logger.Logger(casscf.stdout, casscf.verbose)
    orbsym = numpy.asarray(scf.hf_symm.get_orbsym(mol, mo_coeff))
    ncore = casscf.ncore

    irreps = set(orbsym)

    if cas_irrep_ncore is not None:
        irrep_ncore = {}
        for k, v in cas_irrep_ncore.items():
            if isinstance(k, str):
                irrep_ncore[symm.irrep_name2id(mol.groupname, k)] = v
            else:
                irrep_ncore[k] = v

        ncore_rest = ncore - sum(irrep_ncore.values())
        if ncore_rest > 0:  # guess core configuration
            mask = numpy.ones(len(orbsym), dtype=bool)
            for ir in irrep_ncore:
                mask[orbsym == ir] = False
            core_rest = orbsym[mask][:ncore_rest]
            core_rest = {ir: numpy.count_nonzero(core_rest==ir)
                              for ir in set(core_rest)}
            log.info('Given core space %s < casscf core size %d',
                     cas_irrep_ncore, ncore)
            log.info('Add %s to core configuration', core_rest)
            irrep_ncore.update(core_rest)
        elif ncore_rest < 0:
            raise ValueError('Given core space %s > casscf core size %d'
                             % (cas_irrep_ncore, ncore))
    else:
        irrep_ncore = {ir: sum(orbsym[:ncore]==ir) for ir in irreps}

    if not isinstance(cas_irrep_nocc, dict):
        # list => dict
        cas_irrep_nocc = {ir: n for ir,n in enumerate(cas_irrep_nocc)
                               if n > 0}

    irrep_ncas = {}
    for k, v in cas_irrep_nocc.items():
        if isinstance(k, str):
            irrep_ncas[symm.irrep_name2id(mol.groupname, k)] = v
        else:
            irrep_ncas[k] = v

    ncas_rest = casscf.ncas - sum(irrep_ncas.values())
    if ncas_rest > 0:
        mask = numpy.ones(len(orbsym), dtype=bool)
# remove core and specified active space
        for ir in irrep_ncas:
            mask[orbsym == ir] = False
        for ir, ncore in irrep_ncore.items():
            idx = numpy.where(orbsym == ir)[0]
            mask[idx[:ncore]] = False

        cas_rest = orbsym[mask][:ncas_rest]
        cas_rest = {ir: numpy.count_nonzero(cas_rest==ir)
                         for ir in set(cas_rest)}
        log.info('Given active space %s < casscf active space size %d',
                 cas_irrep_nocc, casscf.ncas)
        log.info('Add %s to active space', cas_rest)
        irrep_ncas.update(cas_rest)
    elif ncas_rest < 0:
        raise ValueError('Given active space %s > casscf active space size %d'
                         % (cas_irrep_nocc, casscf.ncas))

    caslst = []
    for ir, ncas in irrep_ncas.items():
        if ncas > 0:
            if ir in irrep_ncore:
                nc = irrep_ncore[ir]
            else:
                nc = 0
            no = nc + ncas
            idx = numpy.where(orbsym == ir)[0]
            caslst.extend(idx[nc:no])
    caslst = numpy.sort(numpy.asarray(caslst)) + base
    if len(caslst) < casscf.ncas:
        raise ValueError('Not enough orbitals found for core %s, cas %s' %
                         (cas_irrep_ncore, cas_irrep_nocc))

    if log.verbose >= logger.INFO:
        log.info('ncore for each irreps %s',
                 {symm.irrep_id2name(mol.groupname, k): v
                       for k,v in irrep_ncore.items()})
        log.info('ncas for each irreps %s',
                 {symm.irrep_id2name(mol.groupname, k): v
                       for k,v in irrep_ncas.items()})
        log.info('(%d-based) caslst = %s', base, caslst)
    return caslst

def sort_mo_by_irrep(casscf, mo_coeff, cas_irrep_nocc,
                     cas_irrep_ncore=None, s=None):
    '''Given number of active orbitals for each irrep, construct the mo initial
    guess for CASSCF

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        cas_irrep_nocc : list or dict
            Number of active orbitals for each irrep.  It can be a dict, eg
            {'A1': 2, 'B2': 4} to indicate the active space size based on
            irrep names, or {0: 2, 3: 4} for irrep Id,  or a list [2, 0, 0, 4]
            (identical to {0: 2, 3: 4}) in which the list index is served as
            the irrep Id.

    Kwargs:
        cas_irrep_ncore : list or dict
            Number of closed shells for each irrep.  It can be a dict, eg
            {'A1': 6, 'B2': 4} to indicate the closed shells based on
            irrep names, or {0: 6, 3: 4} for irrep Id,  or a list [6, 0, 0, 4]
            (identical to {0: 6, 3: 4}) in which the list index is served as
            the irrep Id.  If cas_irrep_ncore is not given, the program
            will generate a guess based on the lowest :attr:`CASCI.ncore`
            orbitals.
        s : ndarray
            overlap matrix

    Returns:
        sorted orbitals, ordered as [c,..,c,a,..,a,v,..,v]

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvtz', symmetry=True, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> mc = mcscf.CASSCF(mf, 12, 4)
    >>> mo = mc.sort_mo_by_irrep({'E1gx':4, 'E1gy':4, 'E1ux':2, 'E1uy':2})
    >>> # Same to mo = sort_mo_by_irrep(mc, mf.mo_coeff, {2: 4, 3: 4, 6: 2, 7: 2})
    >>> # Same to mo = sort_mo_by_irrep(mc, mf.mo_coeff, [0, 0, 4, 4, 0, 0, 2, 2])
    >>> mc.kernel(mo)[0]
    -108.162863845084
    '''
    caslst = caslst_by_irrep(casscf, mo_coeff, cas_irrep_nocc,
                             cas_irrep_ncore, s, base=0)
    return sort_mo(casscf, mo_coeff, caslst, base=0)

def make_natural_orbitals (method_obj):
    """Make natural orbitals from a general PySCF method object.
    See Eqn. (1) in Keller et al. [DOI:10.1063/1.4922352] for details.


    Args:
        method_obj : Any PySCF method that has the function `make_rdm1` with kwarg
            `ao_repr`. This object can be a restricted OR unrestricted method.

    Returns:
        noons : A 1-D array of the natural orbital occupations (NOONs).

        natorbs : A set of natural orbitals made from method_obj.
    """
    mf = method_obj
    if hasattr(method_obj, "_scf"):
        mf = method_obj._scf
    rdm1 = method_obj.make_rdm1(ao_repr=True)
    S = mf.get_ovlp()

    # Slight difference for restricted vs. unrestriced case
    if isinstance(rdm1, tuple):
        Dm = rdm1[0]+rdm1[1]
    elif isinstance(rdm1, numpy.ndarray):
        if numpy.ndim(rdm1) == 3:
            Dm = rdm1[0]+rdm1[1]
        elif numpy.ndim(rdm1) == 2:
            Dm = rdm1
        else:
            raise ValueError(
                "rdm1 passed to is a numpy array," +
                "but it has the wrong number of dimensions: {}".format(numpy.ndim(rdm1)))
    else:
        raise ValueError(
            "\n\tThe rdm1 generated by method_obj.make_rdm1() was a {}."
            "\n\tThis type is not supported, please select a different method and/or "
            "open an issue at https://github.com/pyscf/pyscf/issues".format(type(rdm1))
        )

    # Diagonalize the DM in AO (using Eqn. (1) referenced above)
    A = reduce(numpy.dot, (S, Dm, S))
    w, v = scipy.linalg.eigh(A, b=S)

    # Flip NOONs (and NOs) since they're in increasing order
    noons = numpy.flip(w)
    natorbs = numpy.flip(v, axis=1)

    return noons, natorbs


def project_init_guess (casscf, mo_init, prev_mol=None, priority=None, use_hf_core=None):
    '''Project the given initial guess to the current CASSCF problem
    giving using a sequence of SVDs on orthogonal orbital subspaces.

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        mo_init : ndarray or list of ndarray
            Initial guess orbitals which are not orthonormal for the
            current molecule.  When the casscf is UHF-CASSCF, mo_init
            needs to be a list of two ndarrays, for alpha and beta
            orbitals. Cannot have linear dependencies (i.e., you cannot
            give more orbitals than the basis of casscf.mol has). Must
            have at least ncore+ncas columns with active orbitals last,
            even if use_hf_core=True. If incomplete, additional virtual
            orbitals will be constructed and appended automatically.

    Kwargs:
        prev_mol : an instance of :class:`Mole`
            If given, the initial guess orbitals are associated to the
            basis of prev_mol. Otherwise, the orbitals are presumed to
            be in the basis of casscf.mol. Beware linear dependencies if
            you are projecting from a LARGER basis to a SMALLER one.

        priority : 'active', 'core', nested idx arrays, or mask array
            If arrays are 3d, UHF-CASSCF must be used; arrays can always
            be 2d. Specifies the order in which groups of orbitals are
            projected. Orbitals orthogonalized earlier are deformed less
            than those orthogonalized later. 'core' means core, then
            active, then virtual; 'active' means active, then core, then
            virtual, and the Gram-Schmidt process is generated by
            [[0],[1],[2],...] or numpy.eye (nmo). Missing orbitals are
            presumed virtual. Defaults to 'active' if you are projecting
            from the same basis set (prev_mol is None or has the same
            basis functions) and 'core' otherwise.

        use_hf_core : logical
            If True, the core orbitals of mo_init are swapped out with
            HF orbitals chosen by maximum overlap. Defaults to True
            if you are projecting from a different basis and False if
            you are projecting from a different geometry.

    Returns:
        New orthonormal initial guess orbitals'''

    from pyscf import lo
    ncore, ncas = casscf.ncore, casscf.ncas
    s0 = casscf._scf.get_ovlp ()
    mf_mo = casscf._scf.mo_coeff
    nmo = numpy.asarray (mf_mo).shape[-1]
    nmo_init = numpy.asarray (mo_init).shape[-1]
    if nmo_init > nmo:
        raise RuntimeError ("Too many orbitals in mo_init (try passing only the occupied orbitals)")

    # Project orbitals from a different basis
    if prev_mol is not None:
        if gto.same_mol(prev_mol, casscf.mol, cmp_basis=False):
            if isinstance(ncore, (int, numpy.integer)):  # RHF
                mo_init = scf.addons.project_mo_nr2nr(prev_mol, mo_init, casscf.mol)
            else:
                mo_init = (scf.addons.project_mo_nr2nr(prev_mol, mo_init[0], casscf.mol),
                           scf.addons.project_mo_nr2nr(prev_mol, mo_init[1], casscf.mol))
        elif gto.same_basis_set(prev_mol, casscf.mol):
            prev_mol = None # Ignore! Serves no purpose unless it's a different basis set
        else:
            raise NotImplementedError('Project initial guess from different system.')
    if priority is None: priority = ('core','active')[prev_mol is None]
    if use_hf_core is None: use_hf_core = (prev_mol is not None)

    # Do the projection
    def _symmcase (mo_basis, mo_target):
        ovlp = reduce (numpy.dot, (mo_basis.conj ().T, s0, mo_target))
        u, s, vh = scipy.linalg.svd (ovlp, full_matrices=True)
        mo_proj = reduce (numpy.dot, (mo_basis, u[:,:len(s)], vh))
        if u.shape[1] > len(s):
            mo_null = numpy.dot (mo_basis, u[:,len(s):])
        else:
            mo_null = numpy.zeros ((mo_basis.shape[0], 0))
        return mo_proj, mo_null

    # Iterate over irreps
    def _rangecase (mo_basis, mo_target):
        if not casscf.mol.symmetry:
            return _symmcase (mo_basis, mo_target)
        orbsym_target = scf.hf_symm.get_orbsym(casscf.mol, mo_target, s0)
        orbsym_basis = scf.hf_symm.get_orbsym(casscf.mol, mo_basis, s0)
        for ir in set (orbsym_target):
            errstr = 'inadequate basis for symmetry {}'.format (ir)
            assert (numpy.count_nonzero (orbsym_basis==ir) >=
                    numpy.count_nonzero (orbsym_target==ir)), errstr
        # Avoid scrambling the order of the target orbitals
        mo_null = []
        mo_proj = numpy.zeros_like (mo_target)
        for ir in set (orbsym_basis):
            mo_basis_ir = mo_basis[:,orbsym_basis==ir]
            idx_ir = orbsym_target == ir
            if numpy.count_nonzero (idx_ir) == 0:
                mo_null.append (mo_basis_ir)
                continue
            mo_proj_ir, mo_null_ir = _symmcase (mo_basis_ir, mo_target[:,idx_ir])
            mo_null.append (mo_null_ir)
            mo_proj[:,idx_ir] = mo_proj_ir
        mo_null = numpy.hstack (mo_null)
        return mo_proj, mo_null

    # Iterate over orbital ranges
    def _spincase (mo_basis, mo_target, range_idx, sort_idx, ncore):
        # Swap out HF core orbitals (making a copy for safety)
        if use_hf_core:
            ovlp = numpy.dot (mo_target.conj ().T, s0.dot (mo_basis))
            ix = numpy.argmax (numpy.abs (ovlp[:ncore,:]), axis=1)
            mo_target = numpy.append (mo_basis[:,ix], mo_target[:,ncore:], axis=1)
        # Do the iteration
        mo = numpy.zeros_like (mo_target)
        for idx in range_idx:
            mo[:,idx], mo_basis = _rangecase (mo_basis, mo_target[:,idx])
        idx = numpy.any (range_idx, axis=0)
        mo = mo[:,idx]
        mo_target = mo_target[:,idx]
        # Fix sign
        sgn = numpy.einsum ('pi,pi->i', mo.conj (), s0.dot (mo_target))
        mo[:,sgn<0] *= -1
        # Append remaining virtual orbitals
        if mo_basis.shape[-1] > 0:
            mo = numpy.append (mo, mo_basis, axis=1)
        # Sort and debug print
        mo[:,:nmo_init] = mo[:,:nmo_init][:,sort_idx]
        if casscf.verbose >= logger.DEBUG:
            mocc = mo[:,:ncore+ncas]
            s1 = reduce(numpy.dot, (mocc.T, s0, mo_target))
            tnorm = numpy.einsum ('pi,pi->i', mo_target.conj (), s0.dot (mo_target))
            s1_norm = s1 / numpy.sqrt (tnorm) [None,:]
            idx = numpy.argmax (numpy.abs (s1), axis=1)
            for i, j in enumerate (idx):
                logger.debug(casscf, 'Init guess <mo-orth|mo-init>  %d  %d  %10.8f (%10.8f after norm)',
                             i+1, j+1, s1[i,j], s1_norm[i,j])
        return mo

    # Interpret "priority" keyword
    def _interpret (priority, ncore):
        # Interpret priority keyword
        nocc = ncore + ncas
        if isinstance (priority, str):
            ridx = numpy.zeros ((2, nmo_init), dtype=bool)
            ridx[0,:ncore] = ridx[1,ncore:nocc] = True
            if priority.lower () == 'active': ridx = ridx[::-1,:]
            elif not priority.lower () == 'core':
                raise RuntimeError ("Invalid priority keyword: string must be either 'active' or 'core'")
            # Edge case: ncore == 0 or ncas == 0 -> remove zero rows from ridx
            ridx = ridx[ridx.sum (1).astype (bool)]
        else:
            ridx = numpy.zeros ((len (priority), nmo), dtype=bool)
            for row, idx in zip (ridx, priority):
                try:
                    row[idx] = True
                except IndexError:
                    raise RuntimeError ("Invalid priority keyword: index array cannot address shape (*,nmo_init)")
            ridx_counts = ridx.astype (int).sum (0)
            if numpy.any (ridx_counts > 1):
                raise RuntimeError ("Invalid priority keyword: index array has repeated elements")
        incl = numpy.any (ridx, axis=0)
        sidx = numpy.append (numpy.where (incl)[0], numpy.where (~incl)[0])
        return ridx, numpy.argsort (sidx)

    # Iterate over spin cases
    if isinstance(ncore, (int, numpy.integer)):
        # errstr = 'Invalid priority keyword (3-dim is valid for UHF-CAS only)'
        range_idx, sort_idx = _interpret (priority, ncore)
        mo = _spincase (mf_mo, mo_init, range_idx, sort_idx, ncore)
    else: # UHF-based CASSCF
        if (isinstance (priority, str)  # single string
            or (isinstance (priority, numpy.ndarray) and priority.ndim == 2)  # single mask array
            or isinstance (priority[0][0], (int, numpy.integer))): # 2d nested list
            priority = [priority, priority]
        idx = [_interpret (p, n) for p, n in zip (priority, ncore)]
        mo = (_spincase (mf_mo[0], mo_init[0], idx[0][0], idx[0][1], ncore[0]),
              _spincase (mf_mo[1], mo_init[1], idx[1][0], idx[1][1], ncore[1]))

    return mo


def project_init_guess_old(casscf, init_mo, prev_mol=None):
    '''Project the given initial guess to the current CASSCF problem.  The
    projected initial guess has two parts.  The core orbitals are directly
    taken from the Hartree-Fock orbitals, and the active orbitals are
    projected from the given initial guess.

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

        init_mo : ndarray or list of ndarray
            Initial guess orbitals which are not orth-normal for the current
            molecule.  When the casscf is UHF-CASSCF, the init_mo needs to be
            a list of two ndarrays, for alpha and beta orbitals

    Kwargs:
        prev_mol : an instance of :class:`Mole`
            If given, the initial guess orbitals are associated to the geometry
            and basis of prev_mol.  Otherwise, the orbitals are based of
            the geometry and basis of casscf.mol

    Returns:
        New orthogonal initial guess orbitals with the core taken from
        Hartree-Fock orbitals and projected active space from original initial
        guess orbitals

    Examples:

    .. code:: python

        import numpy
        from pyscf import gto, scf, mcscf
        mol = gto.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz', verbose=0)
        mf = scf.RHF(mol)
        mf.scf()
        mc = mcscf.CASSCF(mf, 6, 6)
        mo = mcscf.sort_mo(mc, mf.mo_coeff, [3,4,5,6,8,9])
        print('E(0.8) = %.12f' % mc.kernel(mo)[0])
        init_mo = mc.mo_coeff
        for b in numpy.arange(1.0, 3., .2):
            mol.atom = [['H', (0, 0, 0)], ['F', (0, 0, b)]]
            mol.build(0, 0)
            mf = scf.RHF(mol)
            mf.scf()
            mc = mcscf.CASSCF(mf, 6, 6)
            mo = mcscf.project_init_guess(mc, init_mo)
            print('E(%2.1f) = %.12f' % (b, mc.kernel(mo)[0]))
            init_mo = mc.mo_coeff
    '''
    from pyscf import lo

    def project(mfmo, init_mo, ncore, s):
        s_init_mo = numpy.einsum('pi,pi->i', init_mo.conj(), s.dot(init_mo))
        if abs(s_init_mo - 1).max() < 1e-7 and mfmo.shape[1] == init_mo.shape[1]:
            # Initial guess orbitals are orthonormal
            return init_mo
# TODO: test whether the canonicalized orbitals are better than the projected orbitals
# Be careful that the ordering of the canonicalized orbitals may be very different
# to the CASSCF orbitals.
#        else:
#            fock = casscf.get_fock(mc, init_mo, casscf.ci)
#            return casscf._scf.eig(fock, s)[1]

        nocc = ncore + casscf.ncas
        if ncore > 0:
            mo0core = init_mo[:,:ncore]
            s1 = reduce(numpy.dot, (mfmo.T, s, mo0core))
            s1core = reduce(numpy.dot, (mo0core.T, s, mo0core))
            coreocc = numpy.einsum('ij,ji->i', s1, lib.cho_solve(s1core, s1.T))
            coreidx = numpy.sort(numpy.argsort(-coreocc)[:ncore])
            logger.debug(casscf, 'Core indices %s', coreidx)
            logger.debug(casscf, 'Core components %s', coreocc[coreidx])
            # take HF core
            mocore = mfmo[:,coreidx]

            # take projected CAS space
            mocas = init_mo[:,ncore:nocc] \
                  - reduce(numpy.dot, (mocore, mocore.T, s, init_mo[:,ncore:nocc]))
            mocc = lo.orth.vec_lowdin(numpy.hstack((mocore, mocas)), s)
        else:
            mocc = lo.orth.vec_lowdin(init_mo[:,:nocc], s)

        # remove core and active space from rest
        if mocc.shape[1] < mfmo.shape[1]:
            if casscf.mol.symmetry:
                restorb = []
                orbsym = scf.hf_symm.get_orbsym(casscf.mol, mfmo, s)
                for ir in set(orbsym):
                    mo_ir = mfmo[:,orbsym==ir]
                    rest = mo_ir - reduce(numpy.dot, (mocc, mocc.T, s, mo_ir))
                    e, u = numpy.linalg.eigh(reduce(numpy.dot, (rest.T, s, rest)))
                    restorb.append(numpy.dot(rest, u[:,e>1e-7]))
                restorb = numpy.hstack(restorb)
            else:
                rest = mfmo - reduce(numpy.dot, (mocc, mocc.T, s, mfmo))
                e, u = numpy.linalg.eigh(reduce(numpy.dot, (rest.T, s, rest)))
                restorb = numpy.dot(rest, u[:,e>1e-7])
            mo = numpy.hstack((mocc, restorb))
        else:
            mo = mocc

        if casscf.verbose >= logger.DEBUG:
            s1 = reduce(numpy.dot, (mo[:,ncore:nocc].T, s, mfmo))
            idx = numpy.argwhere(abs(s1) > 0.4)
            for i,j in idx:
                logger.debug(casscf, 'Init guess <mo-CAS|mo-hf>  %d  %d  %12.8f',
                             ncore+i+1, j+1, s1[i,j])
        return mo

    ncore = casscf.ncore
    mfmo = casscf._scf.mo_coeff
    s = casscf._scf.get_ovlp()
    if prev_mol is None:
        if init_mo.shape[0] != mfmo.shape[0]:
            raise RuntimeError('Initial guess orbitals has wrong dimension')
    elif gto.same_mol(prev_mol, casscf.mol, cmp_basis=False):
        if isinstance(ncore, (int, numpy.integer)):  # RHF
            init_mo = scf.addons.project_mo_nr2nr(prev_mol, init_mo, casscf.mol)
        else:
            init_mo = (scf.addons.project_mo_nr2nr(prev_mol, init_mo[0], casscf.mol),
                       scf.addons.project_mo_nr2nr(prev_mol, init_mo[1], casscf.mol))
    elif gto.same_basis_set(prev_mol, casscf.mol):
        if isinstance(ncore, (int, numpy.integer)):  # RHF
            fock = casscf.get_fock(init_mo, casscf.ci)
            return casscf._scf.eig(fock, s)[1]
        else:
            raise NotImplementedError('Project initial for UHF orbitals.')
    else:
        raise NotImplementedError('Project initial guess from different system.')

# Be careful with the orbital projection. The projection may lead to bad
# initial guess orbitals if the geometry is dramatically changed.
    if isinstance(ncore, (int, numpy.integer)):
        mo = project(mfmo, init_mo, ncore, s)
    else: # UHF-based CASSCF
        mo = (project(mfmo[0], init_mo[0], ncore[0], s),
              project(mfmo[1], init_mo[1], ncore[1], s))
    return mo

# on AO representation
def make_rdm1(casscf, mo_coeff=None, ci=None, **kwargs):
    '''One-particle density matrix in AO representation

    Args:
        casscf : an :class:`CASSCF` or :class:`CASCI` object

    Kwargs:
        ci : ndarray
            CAS space FCI coefficients. If not given, take casscf.ci.
        mo_coeff : ndarray
            Orbital coefficients. If not given, take casscf.mo_coeff.

    Examples:

    >>> import scipy.linalg
    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='sto-3g', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> res = mc.kernel()
    >>> natocc = numpy.linalg.eigh(mcscf.make_rdm1(mc), mf.get_ovlp(), type=2)[0]
    >>> print(natocc)
    [ 0.0121563   0.0494735   0.0494735   1.95040395  1.95040395  1.98808879
      2.          2.          2.          2.        ]
    '''
    return casscf.make_rdm1(mo_coeff, ci, **kwargs)

# make both alpha and beta density matrices
def make_rdm1s(casscf, mo_coeff=None, ci=None, **kwargs):
    '''Alpha and beta one-particle density matrices in AO representation
    '''
    return casscf.make_rdm1s(mo_coeff, ci, **kwargs)

def get_spin_square(casdm1, casdm2):
    # DOI:10.1021/acs.jctc.1c00589 Eq (49)
    spin_square = (0.75*numpy.einsum("ii", casdm1)
                   - 0.5*numpy.einsum("ijji", casdm2)
                   - 0.25*numpy.einsum("iijj", casdm2))
    return spin_square

def make_spin_casdm1(casdm1, casdm2, spin=None, nelec=None):
    # DOI: 10.1002/qua.22320 Eq (3)
    if spin is None:
        spin = numpy.sqrt(get_spin_square(casdm1, casdm2) + 0.25) - 0.5
    if nelec is None:
        nelec = numpy.einsum("ii", casdm1)
    spin_casdm1 = ((2. - nelec/2.)*casdm1 - numpy.einsum('ikkj->ij', casdm2))/(spin + 1)
    return spin_casdm1

def _is_uhf_mo(mo_coeff):
    return not (isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2)

def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    dm2 = numpy.zeros((nmo,nmo,nmo,nmo))
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] += -2
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] =2*casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2

# In AO representation
def make_rdm12(casscf, mo_coeff=None, ci=None):
    if ci is None: ci = casscf.ci
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    assert (not _is_uhf_mo(mo_coeff))
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nmo = mo_coeff.shape[1]
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(ci, ncas, nelecas)
    rdm1, rdm2 = _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo)
    rdm1 = reduce(numpy.dot, (mo_coeff, rdm1, mo_coeff.T))
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo).transpose(2,3,0,1)
    rdm2 = numpy.dot(mo_coeff, rdm2.reshape(nmo,-1))
    rdm2 = numpy.dot(rdm2.reshape(-1,nmo), mo_coeff.T)
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

def get_fock(casscf, mo_coeff=None, ci=None):
    '''Generalized Fock matrix in AO representation
    '''
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        raise RuntimeError('TODO: UCAS general fock')
    else:
        return casscf.get_fock(mo_coeff, ci)

def cas_natorb(casscf, mo_coeff=None, ci=None, sort=False):
    '''Natural orbitals in CAS space
    '''
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        raise RuntimeError('TODO: UCAS natural orbitals')
    else:
        return casscf.cas_natorb(mo_coeff, ci, sort=sort)

def map2hf(casscf, mf_mo=None, base=BASE, tol=MAP2HF_TOL):
    '''The overlap between the CASSCF optimized orbitals and the canonical HF orbitals.
    '''
    if mf_mo is None: mf_mo = casscf._scf.mo_coeff
    s = casscf.mol.intor_symmetric('int1e_ovlp')
    s = reduce(numpy.dot, (casscf.mo_coeff.T, s, mf_mo))
    idx = numpy.argwhere(abs(s) > tol)
    for i,j in idx:
        logger.info(casscf, '<mo_coeff-mcscf|mo_coeff-hf>  %-5d  %-5d  % 12.8f',
                    i+base, j+base, s[i,j])
    return idx

def spin_square(casscf, mo_coeff=None, ci=None, ovlp=None):
    '''Spin square of the UHF-CASSCF wavefunction

    Returns:
        A list of two floats.  The first is the expectation value of S^2.
        The second is the corresponding 2S+1

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='sto-3g', spin=2, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> res = mf.scf()
    >>> mc = mcscf.CASSCF(mf, 4, 6)
    >>> res = mc.kernel()
    >>> print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))
    S^2 = 3.9831589, 2S+1 = 4.1149284
    '''
    if ci is None: ci = casscf.ci
    ncore    = casscf.ncore
    ncas     = casscf.ncas
    nelecas  = casscf.nelecas
    if isinstance(ncore, (int, numpy.integer)):
        return fci.spin_op.spin_square0(ci, ncas, nelecas)
    else:
        if mo_coeff is None: mo_coeff = casscf.mo_coeff
        if ovlp is None: ovlp = casscf._scf.get_ovlp()
        nocc = (ncore[0] + ncas, ncore[1] + ncas)
        mocas = (mo_coeff[0][:,ncore[0]:nocc[0]], mo_coeff[1][:,ncore[1]:nocc[1]])
        if isinstance(ci, (list, tuple, RANGE_TYPE)):
            sscas = numpy.array([fci.spin_op.spin_square(c, ncas, nelecas, mocas, ovlp)[0]
                                 for c in ci])
        else:
            sscas = fci.spin_op.spin_square(ci, ncas, nelecas, mocas, ovlp)[0]
        mocore = (mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
        sscore = casscf._scf.spin_square(mocore, ovlp)[0]
        logger.debug(casscf, 'S^2 of core %s  S^2 of cas %s', sscore, sscas)
        ss = sscas+sscore
        s = numpy.sqrt(ss+.25) - .5
        return ss, s*2+1

# A tag to label the derived MCSCF class
class StateAverageMCSCFSolver:
    pass

def state_average(casscf, weights=(0.5,0.5), wfnsym=None):
    ''' State average over the energy.  The energy functional is
    E = w1<psi1|H|psi1> + w2<psi2|H|psi2> + ...

    Note we may need change the FCI solver to

    mc.fcisolver = fci.solver(mol, False)

    before calling state_average_(mc), to mix the singlet and triplet states

    MRH, 04/08/2019: Instead of turning casscf._finalize into an instance attribute
    that points to the previous casscf object, I'm going to make a whole new child class.
    This will have the added benefit of making state_average and state_average_
    actually behave differently for the first time (until now they *both* modified the
    casscf object inplace). I'm also going to assign the weights argument as a member
    of the mc child class because an accurate second-order CASSCF algorithm for state-averaged
    calculations requires that the gradient and Hessian be computed for CI vectors of each root
    individually and then multiplied by that root's weight. The second derivatives computed
    by newton_casscf.py need to be extended to state-averaged calculations in order to be
    used as intermediates for calculations of the gradient of a single root in the context
    of the SA-CASSCF method; see: Mol. Phys. 99, 103 (2001).
    '''
    assert (abs(sum(weights)-1) < 1e-3)
    fcisolver = casscf.fcisolver

    # No recursion is allowed!
    if isinstance (fcisolver, StateAverageFCISolver):
        fcisolver.nroots = len(weights)
        fcisolver.weights = weights
    else:
        fcisolver = lib.set_class(StateAverageFCISolver(fcisolver, weights, wfnsym),
                                  (StateAverageFCISolver, fcisolver.__class__))
        fcisolver_cls = fcisolver.__class__
        if getattr(fcisolver, 'spin_square', None):
            def spin_square(self, ci0, norb, nelec, *args, **kwargs):
                ss, multip = self.states_spin_square(ci0, norb, nelec, *args, **kwargs)
                weights = self.weights
                return numpy.dot(ss, weights), numpy.dot(multip, weights)

            def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
                fcibase = super(StateAverageFCISolver, self)
                s = [fcibase.spin_square(ci0[i], norb, nelec, *args, **kwargs)
                     for i, wi in enumerate(self.weights)]
                return [x[0] for x in s], [x[1] for x in s]

            fcisolver_cls.spin_square = spin_square
            fcisolver_cls.states_spin_square = states_spin_square

    return _state_average_mcscf_solver(casscf, fcisolver)

class StateAverageFCISolver:
    __name_mixin__ = 'StateAverage'

    _keys = {'weights', 'e_states'}

    def __init__(self, fcibase, weights, wfnsym):
        self.__dict__.update (fcibase.__dict__)
        self.nroots = len(weights)
        self.weights = weights
        if wfnsym is not None:
            self.wfnsym = wfnsym
        self.e_states = [None]
        # MRH 09/09/2022: I turned the _base_class property into an
        # attribute to prevent conflict with fix_spin_ dynamic class
        self._base_class = fcibase.__class__

    def undo_state_average(self):
        obj = lib.view(self, lib.drop_class(self.__class__, StateAverageFCISolver))
        del obj.weights
        del obj.e_states
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('State-average over %d states with weights %s',
                 len(self.weights), self.weights)
        return self

    def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        if 'nroots' not in kwargs:
            kwargs['nroots'] = self.nroots

        fcibase = super()
        # call fcibase_class.kernel function because the attribute orbsym
        # is available in self but undefined in fcibase object
        e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0,
                              wfnsym=self.wfnsym, **kwargs)
        self.e_states = e

        log = logger.new_logger(self, kwargs.get('verbose'))
        if log.verbose >= logger.DEBUG:
            if getattr(fcibase, 'spin_square', None):
                ss = self.states_spin_square(c, norb, nelec)[0]
                for i, ei in enumerate(e):
                    log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
            else:
                for i, ei in enumerate(e):
                    log.debug('state %d  E = %.15g', i, ei)
        return numpy.einsum('i,i->', e, self.weights), c

    def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        fcibase = super()
        if hasattr(fcibase, 'approx_kernel'):
            e, c = fcibase.approx_kernel(h1, h2, norb, nelec,
                                         ci0=ci0, nroots=self.nroots,
                                         wfnsym=self.wfnsym, **kwargs)
        else:
            e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0,
                                  nroots=self.nroots, wfnsym=self.wfnsym, **kwargs)
        return numpy.einsum('i,i->', e, self.weights), c

    def states_make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
        fcibase = super()
        dm1 = [fcibase.make_rdm1(c, norb, nelec, *args, **kwargs) for c in ci0]
        return dm1

    def make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
        return sum ([w * dm for w, dm in zip(self.weights,
                                             self.states_make_rdm1(ci0, norb, nelec, *args, **kwargs))])

    def states_make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
        fcibase = super()
        dm1a = []
        dm1b = []
        for c in ci0:
            dm1s = fcibase.make_rdm1s(c, norb, nelec, *args, **kwargs)
            dm1a.append (dm1s[0])
            dm1b.append (dm1s[1])
        return dm1a, dm1b

    def make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
        dm1s = self.states_make_rdm1s(ci0, norb, nelec, *args, **kwargs)
        dm1s = numpy.einsum ('r,srpq->spq', self.weights, dm1s)
        return dm1s[0], dm1s[1]

    def states_make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
        fcibase = super()
        rdm1 = []
        rdm2 = []
        for c in ci0:
            dm1, dm2 = fcibase.make_rdm12(c, norb, nelec, *args, **kwargs)
            rdm1.append (dm1)
            rdm2.append (dm2)
        return rdm1, rdm2

    def make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
        rdm1, rdm2 = self.states_make_rdm12(ci0, norb, nelec, *args, **kwargs)
        rdm1 = numpy.einsum ('r,rpq->pq', self.weights, rdm1)
        rdm2 = numpy.einsum ('r,rpqst->pqst', self.weights, rdm2)
        return rdm1, rdm2

    def states_make_rdm12s(self, ci0, norb, nelec, *args, **kwargs):
        fcibase = super()
        dm1a, dm1b = [], []
        dm2aa, dm2ab, dm2bb = [], [], []
        for c in ci0:
            dm1s, dm2s = fcibase.make_rdm12s(c, norb, nelec, *args, **kwargs)
            dm1a.append(dm1s[0])
            dm1b.append(dm1s[1])
            dm2aa.append(dm2s[0])
            dm2ab.append(dm2s[1])
            dm2bb.append(dm2s[2])
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

    def make_rdm12s(self, ci0, norb, nelec, *args, **kwargs):
        rdm1s, rdm2s = self.states_make_rdm12s(ci0, norb, nelec, *args, **kwargs)
        rdm1s = numpy.einsum ('r,srpq->spq', self.weights, rdm1s)
        rdm2s = numpy.einsum ('r,srpqtu->spqtu', self.weights, rdm2s)
        return rdm1s, rdm2s

    def states_trans_rdm12 (self, ci1, ci0, norb, nelec, *args, **kwargs):
        fcibase = super()
        tdm1 = []
        tdm2 = []
        for c1, c0 in zip (ci1, ci0):
            dm1, dm2 = fcibase.trans_rdm12 (c1, c0, norb, nelec)
            tdm1.append (dm1)
            tdm2.append (dm2)
        return tdm1, tdm2

    def trans_rdm12 (self, ci1, ci0, norb, nelec, *args, **kwargs):
        tdm1, tdm2 = self.states_trans_rdm12 (ci1, ci0, norb, nelec, *args, **kwargs)
        tdm1 = numpy.einsum ('r,rpq->pq', self.weights, tdm1)
        tdm2 = numpy.einsum ('r,rpqst->pqst', self.weights, tdm2)
        return tdm1, tdm2

def _state_average_mcscf_solver(casscf, fcisolver):
    '''A common routine for function state_average and state_average_mix to
    generate state-average MCSCF solver.
    '''
    if isinstance (casscf, StateAverageMCSCFSolver):
        casscf = casscf.undo_state_average()

    return lib.set_class(StateAverageMCSCF(casscf, fcisolver),
                         (StateAverageMCSCF, casscf.__class__))

class StateAverageMCSCF(StateAverageMCSCFSolver):
    __name_mixin__ = 'StateAverage'

    def __init__(self, my_mc, fcisolver):
        self.__dict__.update (my_mc.__dict__)
        self.fcisolver = fcisolver

    def undo_state_average(self):
        obj = lib.view(self, lib.drop_class(self.__class__, StateAverageMCSCF))
        if isinstance(self.fcisolver, StateAverageFCISolver):
            obj.fcisolver = self.fcisolver.undo_state_average()
        return obj

    @property
    def _base_class (self):
        ''' for convenience; this is equal to mcscfbase_class '''
        return self.__class__.__bases__[1]

    @property
    def weights (self):
        ''' I want these to be accessible but not separable from fcisolver.weights '''
        return self.fcisolver.weights

    @weights.setter
    def weights (self, x):
        self.fcisolver.weights = x
        return self.fcisolver.weights

    @property
    def e_average(self):
        return numpy.dot(self.fcisolver.weights, self.fcisolver.e_states)

    @property
    def e_states(self):
        return self.fcisolver.e_states

    def _finalize(self):
        super()._finalize()
        # Do not overwrite self.e_tot because .e_tot needs to be used in
        # geometry optimization. self.e_states can be used to access the
        # energy of each state
        #self.e_tot = self.fcisolver.e_states
        logger.note(self, 'CASCI state-averaged energy = %.15g', self.e_average)
        logger.note(self, 'CASCI energy for each state')
        if getattr(self.fcisolver, 'states_spin_square', None):
            ss = self.fcisolver.states_spin_square(self.ci, self.ncas,
                                                   self.nelecas)[0]
            for i, ei in enumerate(self.e_states):
                logger.note(self, '  State %d weight %g  E = %.15g S^2 = %.7f',
                            i, self.weights[i], ei, ss[i])
        else:
            for i, ei in enumerate(self.e_states):
                logger.note(self, '  State %d weight %g  E = %.15g',
                            i, self.weights[i], ei)
        return self

    def nuc_grad_method (self, state=None):
        if callable (getattr (self, '_state_average_nuc_grad_method', None)):
            return self._state_average_nuc_grad_method (state=state)
        else: # Avoid messing up state-average CASCI
            return self._base_class.nuc_grad_method (self)

    Gradients = nuc_grad_method

    def nac_method(self):
        if callable(getattr(self, '_state_average_nac_method', None)):
            return self._state_average_nac_method()
        else:
            raise NotImplementedError("NAC method")

    NACs = nac_method

def state_average_(casscf, weights=(0.5,0.5), wfnsym=None):
    ''' Inplace version of state_average '''
    sacasscf = state_average (casscf, weights, wfnsym)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__ = sacasscf.__dict__
    return casscf


def state_specific_(casscf, state=1, wfnsym=None):
    '''For excited state

    Kwargs:
        state : int
        0 for ground state; 1 for first excited state.
    '''
    fcisolver = casscf.fcisolver
    if isinstance(fcisolver, StateSpecificFCISolver):
        fcisolver.nroots = state+1
        fcisolver._civec = None
        if wfnsym is not None:
            fcisolver.wfnsym = wfnsym
    elif isinstance(fcisolver, StateAverageFCISolver):
        fcisolver = fcisolver.undo_state_average()
    else:
        fcisolver = lib.set_class(StateSpecificFCISolver(fcisolver, state, wfnsym),
                                  (StateSpecificFCISolver, fcisolver.__class__))
    casscf.fcisolver = fcisolver
    return casscf
state_specific = state_specific_

class StateSpecificFCISolver:
    __name_mixin__ = 'StateSpecific'

    _keys = {'state', 'nroots'}

    def __init__(self, fcibase, state, wfnsym):
        self.__dict__.update(fcibase.__dict__)
        self.state = state
        self.nroots = state+1
        self._civec = None
        if wfnsym is not None:
            self.wfnsym = wfnsym

    def undo_state_specific(self):
        obj = lib.view(self, lib.drop_class(self.__class__, StateSpecificFCISolver))
        del obj._civec
        return obj

    def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        if self._civec is not None:
            ci0 = self._civec
        fcibase = super()
        e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0,
                              nroots=self.nroots, wfnsym=self.wfnsym, **kwargs)
        state = self.state
        if state == 0:
            e = [e]
            c = [c]
        self._civec = c
        log = logger.new_logger(self, kwargs.get('verbose'))
        if log.verbose >= logger.DEBUG:
            if getattr(fcibase, 'spin_square', None):
                ss = fcibase.spin_square(c[state], norb, nelec)
                log.debug('state %d  E = %.15g S^2 = %.7f',
                          state, e[state], ss[0])
            else:
                log.debug('state %d  E = %.15g', state, e[state])
        return e[state], c[state]

    def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        if self._civec is not None:
            ci0 = self._civec
        fcibase = super()
        if hasattr(fcibase, 'approx_kernel'):
            e, c = fcibase.approx_kernel(h1, h2, norb, nelec,
                                         ci0=ci0, nroots=self.nroots,
                                         wfnsym=self.wfnsym, **kwargs)
        else:
            e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0,
                                  nroots=self.nroots, wfnsym=self.wfnsym, **kwargs)
        state = self.state
        if state == 0:
            self._civec = [c]
            return e, c
        else:
            self._civec = c
            return e[state], c[state]

class StateAverageMixFCISolver_solver_args:
    def __init__(self, data):
        self._data = data
    def __getitem__(self, key): # Handle data = None
        try:
            return self._data[key]
        except TypeError as e:
            assert (self._data is None), e
            return None
class StateAverageMixFCISolver_state_args (StateAverageMixFCISolver_solver_args):
    pass

class StateAverageMixFCISolver(StateAverageFCISolver):
    __name_mixin__ = 'StateAverageMix'

    _keys = {'weights','e_states','fcisolvers'}

    _solver_args = StateAverageMixFCISolver_solver_args
    _state_args = StateAverageMixFCISolver_state_args

    def __init__(self, fcisolvers, weights):
        self.__dict__.update(fcisolvers[0].__dict__)
        self.nroots = len(weights)
        self.weights = weights
        self.e_states = [None]
        self.fcisolvers = fcisolvers

    def undo_state_average(self):
        obj = lib.view(self, lib.drop_class(self.__class__, StateAverageMixFCISolver))
        del obj.weights
        del obj.e_states
        del obj.fcisolvers
        return obj

    @property
    def _base_class (self):
        return self.fcisolvers[0].__base__

    # MRH 06/24/2020: I need these functions in newton_casscf!
    # TODO: handle things like linkstr somehow (variables that
    # have to be different for different solvers or ci vecs)
    def _loop_solver(self, *args, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        p0 = 0
        for ix, solver in enumerate (self.fcisolvers):
            my_args = []
            for arg in args:
                if isinstance (arg, _state_args):
                    my_arg = arg[p0:p0+solver.nroots]
                    if solver.nroots == 1 and my_arg is not None: my_arg = my_arg[0]
                    my_args.append (my_arg)
                elif isinstance (arg, _solver_args):
                    my_args.append (arg[ix])
                else:
                    my_args.append (arg)
            my_kwargs = {}
            for key, item in kwargs.items ():
                if isinstance (item, _state_args):
                    my_arg = item[p0:p0+solver.nroots]
                    if solver.nroots == 1 and my_arg is not None: my_arg = my_arg[0]
                    my_kwargs[key] = my_arg
                elif isinstance (item, _solver_args):
                    my_kwargs[key] = item[ix]
                else:
                    my_kwargs[key] = item
            yield solver, my_args, my_kwargs
            p0 += solver.nroots

    def _loop_civecs(self, *args, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        p0 = 0
        for i, solver in enumerate (self.fcisolvers):
            for j in range(p0, p0+solver.nroots):
                my_args = []
                for arg in args:
                    if isinstance (arg, _state_args):
                        my_args.append (arg[j])
                    elif isinstance (arg, _solver_args):
                        my_args.append (arg[i])
                    else:
                        my_args.append (arg)
                my_kwargs = {}
                for key, item in kwargs.items ():
                    if isinstance (item, _state_args):
                        my_kwargs[key] = item[j]
                    elif isinstance (item, _solver_args):
                        my_kwargs[key] = item[i]
                    else:
                        my_kwargs[key] = item
                yield solver, my_args, my_kwargs
            p0 += solver.nroots

    def _get_nelec(self, solver, nelec):
        # FCISolver does not need this function. Some external solver may not
        # have the function to handle nelec and spin
        # MRH 06/24/2020: Yes, FCISolver DOES need this function!
        if solver.spin is not None:
            nelec = numpy.sum(nelec)
            nelec = (nelec+solver.spin)//2, (nelec-solver.spin)//2
        return nelec

    def _collect(self, fname, *args, **kwargs):
        for solver, args, kwargs in self._loop_civecs(*args, **kwargs):
            fn = getattr(solver, fname)
            yield fn(*args, **kwargs)

    def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, **kwargs):
        # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
        _state_args = self._state_args
        log = logger.new_logger(self, verbose)
        es = []
        cs = []
        for solver, my_args, my_kwargs in self._loop_solver(_state_args (ci0)):
            c0 = my_args[0]
            e, c = solver.kernel(h1, h2, norb, self._get_nelec(solver, nelec), ci0=c0,
                                 orbsym=self.orbsym, verbose=log, **kwargs)
            if solver.nroots == 1:
                es.append(e)
                cs.append(c)
            else:
                es.extend(e)
                cs.extend(c)
        self.e_states = es
        self.converged = numpy.all(getattr(sol, 'converged', True)
                                   for sol in self.fcisolvers)

        if log.verbose >= logger.DEBUG:
            if all(getattr(solver, 'spin_square', None) for solver in self.fcisolvers):
                ss = self.states_spin_square(cs, norb, nelec)[0]
                for i, ei in enumerate(es):
                    log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
            else:
                for i, ei in enumerate(es):
                    log.debug('state %d  E = %.15g', i, ei)
        return numpy.einsum('i,i', numpy.array(es), self.weights), cs

    def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        _state_args = self._state_args
        es = []
        cs = []
        for ix, (solver, my_args, my_kwargs) in enumerate (self._loop_solver(_state_args (ci0))):
            c0 = my_args[0]
            if hasattr(solver, 'approx_kernel'):
                e, c = solver.approx_kernel(h1, h2, norb, self._get_nelec(solver, nelec), ci0=c0,
                                            orbsym=self.orbsym, **kwargs)
            else:
                e, c = solver.kernel(h1, h2, norb, self._get_nelec(solver, nelec), ci0=c0,
                                     orbsym=self.orbsym, **kwargs)
            if solver.nroots == 1:
                es.append(e)
                cs.append(c)
            else:
                es.extend(e)
                cs.extend(c)
        return numpy.einsum('i,i->', es, self.weights), cs

    def states_make_rdm1 (self, ci0, norb, nelec, link_index=None, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        ci0 = _state_args (ci0)
        link_index = _solver_args (link_index)
        nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
        return list(self._collect ('make_rdm1', ci0, norb, nelec, link_index=link_index, **kwargs))

    def make_rdm1(self, ci0, norb, nelec, link_index=None, **kwargs):
        dm1 = self.states_make_rdm1 (ci0, norb, nelec, link_index=link_index, **kwargs)
        return numpy.einsum ('r,rpq->pq', self.weights, dm1)

    def states_make_rdm1s (self, ci0, norb, nelec, link_index=None, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        ci0 = _state_args (ci0)
        link_index = _solver_args (link_index)
        nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
        dm1a = []
        dm1b = []
        for dm1s in self._collect ('make_rdm1s', ci0, norb, nelec, link_index=link_index, **kwargs):
            dm1a.append (dm1s[0])
            dm1b.append (dm1s[1])
        return dm1a, dm1b

    def make_rdm1s(self, ci0, norb, nelec, link_index=None, **kwargs):
        dm1a, dm1b = self.states_make_rdm1s (ci0, norb, nelec, link_index=link_index, **kwargs)
        dm1s = numpy.einsum ('r,srpq->spq', self.weights, [dm1a, dm1b])
        return dm1s[0], dm1s[1]

    def states_make_rdm12 (self, ci0, norb, nelec, link_index=None, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        ci0 = _state_args (ci0)
        link_index = _solver_args (link_index)
        nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
        rdm1 = []
        rdm2 = []
        for dm1, dm2 in self._collect ('make_rdm12', ci0, norb, nelec, link_index=link_index, **kwargs):
            rdm1.append (dm1)
            rdm2.append (dm2)
        return rdm1, rdm2

    def make_rdm12(self, ci0, norb, nelec, link_index=None, **kwargs):
        rdm1, rdm2 = self.states_make_rdm12 (ci0, norb, nelec, link_index=link_index, **kwargs)
        rdm1 = numpy.einsum ('r,rpq->pq', self.weights, rdm1)
        rdm2 = numpy.einsum ('r,rpqst->pqst', self.weights, rdm2)
        return rdm1, rdm2

    def states_make_rdm12s(self, ci0, norb, nelec, link_index=None, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        ci0 = _state_args (ci0)
        link_index = _solver_args (link_index)
        nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
        dm1a, dm1b = [], []
        dm2aa, dm2ab, dm2bb = [], [], []
        for dm1s, dm2s in self._collect ('make_rdm12s', ci0, norb, nelec, link_index=link_index, **kwargs):
            dm1a.append(dm1s[0])
            dm1b.append(dm1s[1])
            dm2aa.append(dm2s[0])
            dm2ab.append(dm2s[1])
            dm2bb.append(dm2s[2])
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

    def make_rdm12s(self, ci0, norb, nelec, link_index=None, **kwargs):
        rdm1s, rdm2s = self.states_make_rdm12s(ci0, norb, nelec, link_index=link_index, **kwargs)
        rdm1s = numpy.einsum ('r,srpq->spq', self.weights, rdm1s)
        rdm2s = numpy.einsum ('r,srpqtu->spqtu', self.weights, rdm2s)
        return rdm1s, rdm2s

    # TODO: linkstr support
    def states_trans_rdm12 (self, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        _solver_args = self._solver_args
        _state_args = self._state_args
        ci1 = _state_args (ci1)
        ci0 = _state_args (ci0)
        link_index = _solver_args (link_index)
        nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
        tdm1 = []
        tdm2 = []
        for dm1, dm2 in self._collect ('trans_rdm12', ci1, ci0, norb, nelec, link_index=link_index, **kwargs):
            tdm1.append (dm1)
            tdm2.append (dm2)
        return tdm1, tdm2

    def trans_rdm12 (self, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        tdm1, tdm2 = self.states_trans_rdm12 (ci1, ci0, norb, nelec, link_index=link_index, **kwargs)
        tdm1 = numpy.einsum ('r,rpq->pq', self.weights, tdm1)
        tdm2 = numpy.einsum ('r,rpqst->pqst', self.weights, tdm2)
        return tdm1, tdm2

    spin_square = None
    large_ci = None
    transform_ci_for_orbital_rotation = None

def state_average_mix(casscf, fcisolvers, weights=(0.5,0.5)):
    '''State-average CASSCF over multiple FCI solvers.
    '''
    nroots = sum(solver.nroots for solver in fcisolvers)
    assert (nroots == len(weights))

    fcisolver = lib.set_class(StateAverageMixFCISolver(fcisolvers, weights),
                              (StateAverageMixFCISolver, fcisolvers[0].__class__))
    fcisolver_cls = fcisolver.__class__

    has_spin_square = all(getattr(solver, 'spin_square', None)
                          for solver in fcisolvers)
    has_large_ci = all(getattr(solver, 'large_ci', None)
                       for solver in fcisolvers)
    has_transform_ci = all(getattr(solver, 'transform_ci_for_orbital_rotation', None)
                           for solver in fcisolvers)

    if has_spin_square:
        def spin_square(self, ci0, norb, nelec, *args, **kwargs):
            ss, multip = self.states_spin_square(ci0, norb, nelec, *args, **kwargs)
            weights = self.weights
            return numpy.dot(ss, weights), numpy.dot(multip, weights)

        def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
            _solver_args = self._solver_args
            _state_args = self._state_args
            ci0 = _state_args (ci0)
            nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            res = list(self._collect('spin_square', ci0, norb, nelec, *args, **kwargs))
            ss = [x[0] for x in res]
            multip = [x[1] for x in res]
            return ss, multip

        fcisolver_cls.spin_square = spin_square
        fcisolver_cls.states_spin_square = states_spin_square

    if has_large_ci:
        def states_large_ci(self, fcivec, norb, nelec, *args, **kwargs):
            _solver_args = self._solver_args
            _state_args = self._state_args
            fcivec = _state_args (fcivec)
            nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            return list(self._collect('large_ci', fcivec, norb, nelec, *args, **kwargs))

        fcisolver_cls.states_large_ci = states_large_ci

    if has_transform_ci:
        def states_transform_ci_for_orbital_rotation(self, fcivec, norb, nelec,
                                                     *args, **kwargs):
            _solver_args = self._solver_args
            _state_args = self._state_args
            fcivec = _state_args (fcivec)
            nelec = _solver_args ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            return list(self._collect('transform_ci_for_orbital_rotation',
                                      fcivec, norb, nelec, *args, **kwargs))

        fcisolver_cls.states_transform_ci_for_orbital_rotation = states_transform_ci_for_orbital_rotation

    mc = _state_average_mcscf_solver(casscf, fcisolver)
    return mc

def state_average_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    ''' Inplace version of state_average '''
    sacasscf = state_average_mix(casscf, fcisolvers, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__ = sacasscf.__dict__
    return casscf


del (BASE, MAP2HF_TOL)
