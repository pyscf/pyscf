#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
            core_rest = dict([(ir, numpy.count_nonzero(core_rest==ir))
                              for ir in set(core_rest)])
            log.info('Given core space %s < casscf core size %d',
                     cas_irrep_ncore, ncore)
            log.info('Add %s to core configuration', core_rest)
            irrep_ncore.update(core_rest)
        elif ncore_rest < 0:
            raise ValueError('Given core space %s > casscf core size %d'
                             % (cas_irrep_ncore, ncore))
    else:
        irrep_ncore = dict([(ir, sum(orbsym[:ncore]==ir)) for ir in irreps])

    if not isinstance(cas_irrep_nocc, dict):
        # list => dict
        cas_irrep_nocc = dict([(ir, n) for ir,n in enumerate(cas_irrep_nocc)
                               if n > 0])

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
        cas_rest = dict([(ir, numpy.count_nonzero(cas_rest==ir))
                         for ir in set(cas_rest)])
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
                 dict([(symm.irrep_id2name(mol.groupname, k), v)
                       for k,v in irrep_ncore.items()]))
        log.info('ncas for each irreps %s',
                 dict([(symm.irrep_id2name(mol.groupname, k), v)
                       for k,v in irrep_ncas.items()]))
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


def project_init_guess(casscf, init_mo, prev_mol=None):
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
            If given, the inital guess orbitals are associated to the geometry
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
    '''One-particle densit matrix in AO representation

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
    '''Alpha and beta one-particle densit matrices in AO representation
    '''
    return casscf.make_rdm1s(mo_coeff, ci, **kwargs)

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

# on AO representation
def make_rdm12(casscf, mo_coeff=None, ci=None):
    if ci is None: ci = casscf.ci
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    assert(not _is_uhf_mo(mo_coeff))
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
    '''Natrual orbitals in CAS space
    '''
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if _is_uhf_mo(mo_coeff):
        raise RuntimeError('TODO: UCAS natrual orbitals')
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
        logger.info(casscf, '<mo_coeff-mcscf|mo_coeff-hf>  %d  %d  %12.8f',
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

# A tag to label the derived FCI class
class StateAverageFCISolver:
    pass
class StateSpecificFCISolver:
    pass
# A tag to label the derived MCSCF class
class StateAverageMCSCFSolver:
    pass

def state_average(casscf, weights=(0.5,0.5), wfnsym=None):
    ''' State average over the energy.  The energy funcitonal is
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
    assert(abs(sum(weights)-1) < 1e-3)
    fcibase_class = casscf.fcisolver.__class__
    has_spin_square = getattr(casscf.fcisolver, 'spin_square', None)

    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self, fcibase):
            self.__dict__.update (fcibase.__dict__)
            self.nroots = len(weights)
            self.weights = weights
            self.wfnsym = wfnsym
            self.e_states = [None]
            keys = set (('weights','e_states','_base_class'))
            self._keys = self._keys.union (keys)

        def dump_flags(self, verbose=None):
            if hasattr(fcibase_class, 'dump_flags'):
                fcibase_class.dump_flags(self, verbose)
            log = logger.new_logger(self, verbose)
            log.info('State-average over %d states with weights %s',
                     len(self.weights), self.weights)
            return self

        @property
        def _base_class (self):
            ''' for convenience; this is equal to fcibase_class '''
            return self.__class__.__bases__[0]

        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if 'nroots' not in kwargs:
                kwargs['nroots'] = self.nroots

            # call fcibase_class.kernel function because the attribute orbsym
            # is available in self but undefined in fcibase object
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        wfnsym=self.wfnsym, **kwargs)
            self.e_states = e

            log = logger.new_logger(self, kwargs.get('verbose'))
            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss = self.states_spin_square(c, norb, nelec)[0]
                    for i, ei in enumerate(e):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(e):
                        log.debug('state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i->', e, self.weights), c

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            try:
                e, c = fcibase_class.approx_kernel(self, h1, h2, norb, nelec,
                                                   ci0, nroots=self.nroots,
                                                   wfnsym=self.wfnsym,
                                                   **kwargs)
            except AttributeError:
                e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                            nroots=self.nroots,
                                            wfnsym=self.wfnsym, **kwargs)
            return numpy.einsum('i,i->', e, self.weights), c

        def make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
            dm1 = 0
            for i, wi in enumerate(self.weights):
                dm1 += wi * fcibase_class.make_rdm1(self, ci0[i], norb, nelec, *args, **kwargs)
            return dm1

        def make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
            dm1a, dm1b = 0, 0
            for i, wi in enumerate(self.weights):
                dm1s = fcibase_class.make_rdm1s(self, ci0[i], norb, nelec, *args, **kwargs)
                dm1a += wi * dm1s[0]
                dm1b += wi * dm1s[1]
            return dm1a, dm1b

        def make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
            rdm1 = 0
            rdm2 = 0
            for i, wi in enumerate(self.weights):
                dm1, dm2 = fcibase_class.make_rdm12(self, ci0[i], norb, nelec, *args, **kwargs)
                rdm1 += wi * dm1
                rdm2 += wi * dm2
            return rdm1, rdm2

        if has_spin_square:
            def spin_square(self, ci0, norb, nelec, *args, **kwargs):
                ss, multip = self.states_spin_square(ci0, norb, nelec, *args, **kwargs)
                weights = self.weights
                return numpy.dot(ss, weights), numpy.dot(multip, weights)

            def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
                s = [fcibase_class.spin_square(self, ci0[i], norb, nelec, *args, **kwargs)
                     for i, wi in enumerate(self.weights)]
                return [x[0] for x in s], [x[1] for x in s]

    # No recursion of FakeCISolver is allowed!
    if isinstance (casscf.fcisolver, StateAverageFCISolver):
        fcisolver = casscf.fcisolver
        fcisolver.nroots = len(weights)
        fcisolver.weights = weights
    else:
        fcisolver = FakeCISolver(casscf.fcisolver)
    return _state_average_mcscf_solver(casscf, fcisolver)

def _state_average_mcscf_solver(casscf, fcisolver):
    '''A common routine for function state_average and state_average_mix to
    generate state-average MCSCF solver.
    '''
    mcscfbase_class = casscf.__class__
    if isinstance (casscf, StateAverageMCSCFSolver):
        raise TypeError('mc is not base MCSCF solver\n'
                        'state_average function cannot work with decorated '
                        'MCSCF solver %s.\nYou can restore the base MCSCF '
                        'then call state_average function, eg\n'
                        '    mc = %s.%s(mc._scf, %s, %s)\n'
                        '    mc.state_average_()\n' %
                        (mcscfbase_class, mcscfbase_class.__base__.__module__,
                         mcscfbase_class.__base__.__name__, casscf.ncas, casscf.nelecas))

    has_spin_square = getattr(fcisolver, 'spin_square', None)

    class StateAverageMCSCF(mcscfbase_class, StateAverageMCSCFSolver):
        def __init__(self, my_mc):
            self.__dict__.update (my_mc.__dict__)
            self.fcisolver = fcisolver
            keys = set (('weights', '_base_class'))
            self._keys = self._keys.union (keys)

        @property
        def _base_class (self):
            ''' for convenience; this is equal to mcscfbase_class '''
            return self.__class__.__bases__[0]

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
            mcscfbase_class._finalize(self)
            # Do not overwrite self.e_tot because .e_tot needs to be used in
            # geometry optimization. self.e_states can be used to access the
            # energy of each state
            #self.e_tot = self.fcisolver.e_states
            logger.note(self, 'CASCI state-averaged energy = %.15g', self.e_average)
            logger.note(self, 'CASCI energy for each state')
            if has_spin_square:
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

    return StateAverageMCSCF(casscf)

def state_average_(casscf, weights=(0.5,0.5)):
    ''' Inplace version of state_average '''
    sacasscf = state_average (casscf, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__.update (sacasscf.__dict__)
    return casscf


def state_specific_(casscf, state=1, wfnsym=None):
    '''For excited state

    Kwargs:
        state : int
        0 for ground state; 1 for first excited state.
    '''
    fcibase_class = casscf.fcisolver.__class__
    if fcibase_class.__name__ == 'FakeCISolver':
        raise TypeError('mc.fcisolver is not base FCI solver\n'
                        'state_specific function cannot work with decorated '
                        'fcisolver %s.\nYou can restore the base fcisolver '
                        'then call state_specific function, eg\n'
                        '    mc.fcisolver = %s.%s(mc.mol)\n'
                        '    mc.state_specific_()\n' %
                        (casscf.fcisolver, fcibase_class.__base__.__module__,
                         fcibase_class.__base__.__name__))

    class FakeCISolver(fcibase_class, StateSpecificFCISolver):
        def __init__(self):
            self.nroots = state+1
            self._civec = None
            self.wfnsym = wfnsym

        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, wfnsym=self.wfnsym,
                                        **kwargs)
            if state == 0:
                e = [e]
                c = [c]
            self._civec = c
            log = logger.new_logger(self, kwargs.get('verbose'))
            if log.verbose >= logger.DEBUG:
                if getattr(fcibase_class, 'spin_square', None):
                    ss = fcibase_class.spin_square(self, c[state], norb, nelec)
                    log.debug('state %d  E = %.15g S^2 = %.7f',
                              state, e[state], ss[0])
                else:
                    log.debug('state %d  E = %.15g', state, e[state])
            return e[state], c[state]

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            try:
                e, c = fcibase_class.approx_kernel(self, h1, h2, norb, nelec,
                                                   ci0, nroots=self.nroots,
                                                   wfnsym=self.wfnsym,
                                                   **kwargs)
            except AttributeError:
                e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                            nroots=self.nroots,
                                            wfnsym=self.wfnsym, **kwargs)
            if state == 0:
                self._civec = [c]
                return e, c
            else:
                self._civec = c
                return e[state], c[state]

    fcisolver = FakeCISolver()
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.nroots = state+1
    casscf.fcisolver = fcisolver
    return casscf
state_specific = state_specific_

def state_average_mix(casscf, fcisolvers, weights=(0.5,0.5)):
    '''State-average CASSCF over multiple FCI solvers.
    '''
    fcibase_class = fcisolvers[0].__class__
    nroots = sum(solver.nroots for solver in fcisolvers)
    assert(nroots == len(weights))
    has_spin_square = all(getattr(solver, 'spin_square', None)
                          for solver in fcisolvers)
    has_large_ci = all(getattr(solver, 'large_ci', None)
                       for solver in fcisolvers)
    has_transform_ci = all(getattr(solver, 'transform_ci_for_orbital_rotation', None)
                           for solver in fcisolvers)

    def loop_solver(solvers, ci0):
        p0 = 0
        for solver in solvers:
            if ci0 is None:
                yield solver, None
            elif solver.nroots == 1:
                yield solver, ci0[p0]
            else:
                yield solver, ci0[p0:p0+solver.nroots]
            p0 += solver.nroots

    def loop_civecs(solvers, ci0):
        p0 = 0
        for solver in solvers:
            for i in range(p0, p0+solver.nroots):
                yield solver, ci0[i]
            p0 += solver.nroots

    def get_nelec(solver, nelec):
        # FCISolver does not need this function. Some external solver may not
        # have the function to handle nelec and spin
        if solver.spin is not None:
            nelec = numpy.sum(nelec)
            nelec = (nelec+solver.spin)//2, (nelec-solver.spin)//2
        return nelec

    def collect(fname, ci0, norb, nelec, *args, **kwargs):
        for solver, c in loop_civecs(fcisolvers, ci0):
            fn = getattr(solver, fname)
            yield fn(c, norb, get_nelec(solver, nelec), *args, **kwargs)

    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self, mol):
            fcibase_class.__init__(self, mol)
            self.nroots = len(weights)
            self.weights = weights
            self.e_states = [None]
            keys = set (('weights','e_states','_base_class'))
            self._keys = self._keys.union (keys)

        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, **kwargs):
            # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            for solver, c0 in loop_solver(fcisolvers, ci0):
                e, c = solver.kernel(h1, h2, norb, get_nelec(solver, nelec), c0,
                                     orbsym=self.orbsym, verbose=log, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            self.e_states = es
            self.converged = numpy.all(getattr(sol, 'converged', True)
                                       for sol in fcisolvers)

            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss = self.states_spin_square(cs, norb, nelec)[0]
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i', numpy.array(es), weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            es = []
            cs = []
            for solver, c0 in loop_solver(fcisolvers, ci0):
                try:
                    e, c = solver.approx_kernel(h1, h2, norb, get_nelec(solver, nelec), c0,
                                                orbsym=self.orbsym, **kwargs)
                except AttributeError:
                    e, c = solver.kernel(h1, h2, norb, get_nelec(solver, nelec), c0,
                                         orbsym=self.orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return numpy.einsum('i,i->', es, weights), cs

        def make_rdm1(self, ci0, norb, nelec, **kwargs):
            dm1 = 0
            for i, dm in enumerate(collect('make_rdm1', ci0, norb, nelec, **kwargs)):
                dm1 += weights[i] * dm
            return dm1

        def make_rdm1s(self, ci0, norb, nelec, **kwargs):
            dm1a, dm1b = 0, 0
            for i, dm1s in enumerate(collect('make_rdm1s', ci0, norb, nelec, **kwargs)):
                dm1a += weights[i] * dm1s[0]
                dm1b += weights[i] * dm1s[1]
            return dm1a, dm1b

        def make_rdm12(self, ci0, norb, nelec, **kwargs):
            rdm1 = 0
            rdm2 = 0
            for i, (dm1, dm2) in enumerate(collect('make_rdm12', ci0, norb, nelec, **kwargs)):
                rdm1 += weights[i] * dm1
                rdm2 += weights[i] * dm2
            return rdm1, rdm2

        if has_spin_square:
            def spin_square(self, ci0, norb, nelec, *args, **kwargs):
                ss, multip = self.states_spin_square(ci0, norb, nelec, *args, **kwargs)
                weights = self.weights
                return numpy.dot(ss, weights), numpy.dot(multip, weights)

            def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
                res = list(collect('spin_square', ci0, norb, nelec, *args, **kwargs))
                ss = [x[0] for x in res]
                multip = [x[1] for x in res]
                return ss, multip
        else:
            spin_square = None

        large_ci = None
        if has_large_ci:
            def states_large_ci(self, fcivec, norb, nelec, *args, **kwargs):
                return list(collect('large_ci', fcivec, norb, nelec, *args, **kwargs))

        transform_ci_for_orbital_rotation = None
        if has_transform_ci:
            def states_transform_ci_for_orbital_rotation(self, fcivec, norb, nelec,
                                                         *args, **kwargs):
                return list(collect('transform_ci_for_orbital_rotation',
                                    fcivec, norb, nelec, *args, **kwargs))

    fcisolver = FakeCISolver(casscf.mol)
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.fcisolvers = fcisolvers
    mc = _state_average_mcscf_solver(casscf, fcisolver)
    return mc

def state_average_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    ''' Inplace version of state_average '''
    sacasscf = state_average_mix(casscf, fcisolvers, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__.update(sacasscf.__dict__)
    return casscf


del(BASE, MAP2HF_TOL)


if __name__ == '__main__':
    from pyscf import mcscf
    from pyscf.tools import ring

    mol = gto.M(verbose=0,
                output=None,
                atom=[['H', c] for c in ring.make(6, 1.2)],
                basis='6-31g')

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mcscf.CASSCF(m, 6, 6)
    mc.verbose = 4
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()
    print(ehf, emc, emc-ehf)
    print(emc - -3.272089958)

    rdm1 = make_rdm1(mc, mo, fcivec)
    rdm1, rdm2 = make_rdm12(mc, mo, fcivec)
    print(rdm1)

    mo1 = cas_natorb(mc)[0]
    numpy.set_printoptions(2)
    print(reduce(numpy.dot, (mo1[:,:6].T, mol.intor('int1e_ovlp_sph'),
                             mo[:,:6])))

# state average
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = '6-31g'
    mol.symmetry = 1
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()


    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mc.fcisolver = fci.solver(mol, False) # to mix the singlet and triplet
    mc = state_average_(mc, (.64,.36))
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()[:5]
    mc = mcscf.CASCI(m, 4, 4)
    emc = mc.casci(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -76.003352190262532)

    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mc = state_average_(mc, (.64,.36))
    emc, e_ci, fcivec, mo, mo_energy = mc.mc1step()[:5]
    mc = mcscf.CASCI(m, 4, 4)
    emc = mc.casci(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.982520334896776)


    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    mc = state_specific_(mc, 2)
    emc = mc.kernel()[0]

