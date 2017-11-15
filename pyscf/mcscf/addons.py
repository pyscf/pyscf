#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
import sys
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import fci
from pyscf import scf
from pyscf import symm


def sort_mo(casscf, mo_coeff, caslst, base=1):
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
    if isinstance(ncore, (int, numpy.integer)):
        assert(casscf.ncas == len(caslst))
        nmo = mo_coeff.shape[1]
        if base != 0:
            caslst = [i-base for i in caslst]
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst])
        return numpy.hstack((mo_coeff[:,idx[:ncore]],
                             mo_coeff[:,caslst],
                             mo_coeff[:,idx[ncore:]]))
    else: # UHF-based CASSCF
        if isinstance(caslst[0], (int, numpy.integer)):
            assert(casscf.ncas == len(caslst))
            if base != 0:
                caslsta = [i-1 for i in caslst]
                caslst = (caslsta, caslsta)
        else: # two casspace lists, for alpha and beta
            assert(casscf.ncas == len(caslst[0]))
            assert(casscf.ncas == len(caslst[1]))
            if base != 0:
                caslst = ([i-base for i in caslst[0]],
                          [i-base for i in caslst[1]])
        nmo = mo_coeff[0].shape[1]
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst[0]])
        mo_a = numpy.hstack((mo_coeff[0][:,idx[:ncore[0]]],
                             mo_coeff[0][:,caslst[0]],
                             mo_coeff[0][:,idx[ncore[0]:]]))
        idx = numpy.asarray([i for i in range(nmo) if i not in caslst[1]])
        mo_b = numpy.hstack((mo_coeff[1][:,idx[:ncore[1]]],
                             mo_coeff[1][:,caslst[1]],
                             mo_coeff[1][:,idx[ncore[1]:]]))
        return (mo_a, mo_b)

def select_mo_by_irrep(casscf,  cas_occ_num, mo = None, base=1):
    raise RuntimeError('This function has been replaced by function caslst_by_irrep')

def caslst_by_irrep(casscf, mo_coeff, cas_irrep_nocc,
                    cas_irrep_ncore=None, s=None, base=1):
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

        ncore_rest = casscf.ncore - sum(irrep_ncore.values())
        if ncore_rest > 0:  # guess core configuration
            mask = numpy.ones(len(orbsym), dtype=bool)
            for ir in irrep_ncore:
                mask[orbsym == ir] = False
            core_rest = orbsym[mask][:ncore_rest]
            core_rest = dict([(ir, numpy.count_nonzero(core_rest==ir))
                              for ir in set(core_rest)])
            log.info('Given core space %s < casscf core size %d',
                     cas_irrep_ncore, casscf.ncore)
            log.info('Add %s to core configuration', core_rest)
            irrep_ncore.update(core_rest)
        elif ncore_rest < 0:
            raise ValueError('Given core space %s > casscf core size %d'
                             % (cas_irrep_ncore, casscf.ncore))
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
                             cas_irrep_ncore, s, 0)
    #FIXME: update mo_coeff.orbsym?
    return sort_mo(casscf, mo_coeff, caslst, 0)


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
            mocc = init_mo[:,:nocc]

        # remove core and active space from rest
        if mocc.shape[1] < mfmo.shape[1]:
            rest = mfmo - reduce(numpy.dot, (mocc, mocc.T, s, mfmo))
            e, u = numpy.linalg.eigh(reduce(numpy.dot, (rest.T, s, rest)))
            restorb = numpy.dot(rest, u[:,e>1e-7])
            if casscf.mol.symmetry:
                t = casscf.mol.intor_symmetric('int1e_kin')
                t = reduce(numpy.dot, (restorb.T, t, restorb))
                e, u = numpy.linalg.eigh(t)
                restorb = numpy.dot(restorb, u)
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
    if isinstance(ncore, (int, numpy.integer)):
        if prev_mol is not None:
            init_mo = scf.addons.project_mo_nr2nr(prev_mol, init_mo, casscf.mol)
        else:
            assert(mfmo.shape[0] == init_mo.shape[0])
        mo = project(mfmo, init_mo, ncore, s)
    else: # UHF-based CASSCF
        if prev_mol is not None:
            init_mo = (scf.addons.project_mo_nr2nr(prev_mol, init_mo[0], casscf.mol),
                       scf.addons.project_mo_nr2nr(prev_mol, init_mo[1], casscf.mol))
        else:
            assert(mfmo[0].shape[0] == init_mo[0].shape[0])
        mo = (project(mfmo[0], init_mo[0], ncore[0], s),
              project(mfmo[1], init_mo[1], ncore[1], s))
    return mo

# on AO representation
def make_rdm1(casscf, mo_coeff=None, ci=None):
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
    return casscf.make_rdm1(mo_coeff, ci)

# make both alpha and beta density matrices
def make_rdm1s(casscf, mo_coeff=None, ci=None):
    '''Alpha and beta one-particle densit matrices in AO representation
    '''
    return casscf.make_rdm1s(mo_coeff, ci)

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

def map2hf(casscf, mf_mo=None, base=1, tol=.5):
    '''The overlap between the CASSCF optimized orbitals and the canonical HF orbitals.
    '''
    if mf_mo is None:
        mf_mo = casscf._scf.mo_coeff
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
        sscas = fci.spin_op.spin_square(ci, ncas, nelecas, mocas, ovlp)
        mocore = (mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
        sscore = scf.uhf.spin_square(mocore, ovlp)
        logger.debug(casscf, 'S^2 of core %f  S^2 of cas %f', sscore[0], sscas[0])
        ss = sscas[0]+sscore[0]
        s = numpy.sqrt(ss+.25) - .5
        return ss, s*2+1

# A tag to label the derived FCI class
class StateAverageFCISolver:
    pass

def state_average_(casscf, weights=(0.5,0.5)):
    ''' State average over the energy.  The energy funcitonal is
    E = w1<psi1|H|psi1> + w2<psi2|H|psi2> + ...

    Note we may need change the FCI solver to

    mc.fcisolver = fci.solver(mol, False)

    before calling state_average_(mc), to mix the singlet and triplet states
    '''
    assert(abs(sum(weights)-1) < 1e-3)
    fcibase_class = casscf.fcisolver.__class__
    if fcibase_class.__name__ == 'FakeCISolver':
        sys.stderr.write('state_average function cannot work with decorated '
                         'fcisolver %s.\nYou can restore the base fcisolver '
                         'then call state_average function, eg\n'
                         '    mc.fcisolver = %s.%s(mc.mol)\n'
                         '    mc.state_average_()\n' %
                         (casscf.fcisolver, fcibase_class.__base__.__module__,
                          fcibase_class.__base__.__name__))
        raise TypeError('mc.fcisolver is not base FCI solver')
    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self, mol=None):
            self.nroots = len(weights)
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
# pass self to fcibase_class.kernel function because orbsym argument is stored in self
# but undefined in fcibase object
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            if casscf.verbose >= logger.DEBUG:
                if hasattr(fcibase_class, 'spin_square'):
                    for i, ei in enumerate(e):
                        ss = fcibase_class.spin_square(self, c[i], norb, nelec)
                        logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                                     i, ei, ss[0])
                else:
                    for i, ei in enumerate(e):
                        logger.debug(casscf, 'state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i->', e, weights), c
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
            return numpy.einsum('i,i->', e, weights), c
        def make_rdm1(self, ci0, norb, nelec):
            dm1 = 0
            for i, wi in enumerate(weights):
                dm1 += wi*fcibase_class.make_rdm1(self, ci0[i], norb, nelec)
            return dm1
        def make_rdm12(self, ci0, norb, nelec):
            rdm1 = 0
            rdm2 = 0
            for i, wi in enumerate(weights):
                dm1, dm2 = fcibase_class.make_rdm12(self, ci0[i], norb, nelec)
                rdm1 += wi * dm1
                rdm2 += wi * dm2
            return rdm1, rdm2

        if hasattr(fcibase_class, 'spin_square'):
            def spin_square(self, ci0, norb, nelec):
                ss = 0
                multip = 0
                for i, wi in enumerate(weights):
                    res = fcibase_class.spin_square(self, ci0[i], norb, nelec)
                    ss += wi * res[0]
                    multip += wi * res[1]
                return ss, multip

    fcisolver = FakeCISolver(casscf.mol)
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.nroots = len(weights)
    casscf.fcisolver = fcisolver
    return casscf
state_average = state_average_


def state_specific_(casscf, state=1):
    '''For excited state

    Kwargs:
        state : int
        0 for ground state; 1 for first excited state.
    '''
    fcibase_class = casscf.fcisolver.__class__
    if fcibase_class.__name__ == 'FakeCISolver':
        sys.stderr.write('state_specific function cannot work with decorated '
                         'fcisolver %s.\nYou can restore the base fcisolver '
                         'then call state_specific function, eg\n'
                         '    mc.fcisolver = %s.%s(mc.mol)\n'
                         '    mc.state_specific_()\n' %
                         (casscf.fcisolver, fcibase_class.__base__.__module__,
                          fcibase_class.__base__.__name__))
        raise TypeError('mc.fcisolver is not base FCI solver')
    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def __init__(self):
            self.nroots = state+1
            self._civec = None
        def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        nroots=self.nroots, **kwargs)
            if state == 0:
                e = [e]
                c = [c]
            self._civec = c
            if casscf.verbose >= logger.DEBUG:
                if hasattr(fcibase_class, 'spin_square'):
                    ss = fcibase_class.spin_square(self, c[state], norb, nelec)
                    logger.debug(casscf, 'state %d  E = %.15g S^2 = %.7f',
                                 state, e[state], ss[0])
                else:
                    logger.debug(casscf, 'state %d  E = %.15g', state, e[state])
            return e[state], c[state]
        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            if self._civec is not None:
                ci0 = self._civec
            e, c = fcibase_class.kernel(self, h1, h2, norb, nelec, ci0,
                                        max_cycle=casscf.ci_response_space,
                                        nroots=self.nroots, **kwargs)
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

def state_average_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    '''State-average CASSCF over multiple FCI solvers.
    '''
    fcibase_class = fcisolvers[0].__class__
#    if fcibase_class.__name__ == 'FakeCISolver':
#        logger.warn(casscf, 'casscf.fcisolver %s is a decorated FCI solver. '
#                    'state_average_mix_ function rolls back to the base solver %s',
#                    fcibase_class, fcibase_class.__base__)
#        fcibase_class = fcibase_class.__base__
    nroots = sum(solver.nroots for solver in fcisolvers)
    assert(nroots == len(weights))
    has_spin_square = all(hasattr(solver, 'spin_square')
                          for solver in fcisolvers)

    def collect(items):
        items = list(items)
        cols = [[item[i] for item in items] for i in range(len(items[0]))]
        return cols
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
        if solver.spin is not None:
            nelec = numpy.sum(nelec)
            nelec = (nelec+solver.spin)//2, (nelec-solver.spin)//2
        return nelec

    class FakeCISolver(fcibase_class, StateAverageFCISolver):
        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, **kwargs):
# Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(sys, verbose)
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
            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss, multip = collect(solver.spin_square(c0, norb, get_nelec(solver, nelec))
                                         for solver, c0 in loop_civecs(fcisolvers, cs))
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(e):
                        log.debug('state %d  E = %.15g', i, ei)
            return numpy.einsum('i,i', numpy.array(es), weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            es = []
            cs = []
            for solver, c0 in loop_solver(fcisolvers, ci0):
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
            for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                dm1 += weights[i]*solver.make_rdm1(c, norb, get_nelec(solver, nelec), **kwargs)
            return dm1
        def make_rdm12(self, ci0, norb, nelec, **kwargs):
            rdm1 = 0
            rdm2 = 0
            for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                dm1, dm2 = solver.make_rdm12(c, norb, get_nelec(solver, nelec), **kwargs)
                rdm1 += weights[i] * dm1
                rdm2 += weights[i] * dm2
            return rdm1, rdm2

        if has_spin_square:
            def spin_square(self, ci0, norb, nelec):
                ss = 0
                multip = 0
                for i, (solver, c) in enumerate(loop_civecs(fcisolvers, ci0)):
                    res = solver.spin_square(c, norb, nelec)
                    ss += weights[i] * res[0]
                    multip += weights[i] * res[1]
                return ss, multip

    fcisolver = FakeCISolver(casscf.mol)
    fcisolver.__dict__.update(casscf.fcisolver.__dict__)
    fcisolver.fcisolvers = fcisolvers
    casscf.fcisolver = fcisolver
    return casscf
state_average_mix = state_average_mix_

def hot_tuning_(casscf, configfile=None):
    '''Allow you to tune CASSCF parameters at the runtime
    '''
    import traceback
    import tempfile
    import json
    #from numpy import array

    if configfile is None:
        fconfig = tempfile.NamedTemporaryFile(suffix='.json')
        configfile = fconfig.name
    logger.info(casscf, 'Function hot_tuning_ dumps CASSCF parameters in config file%s',
                configfile)

    exclude_keys = set(('stdout', 'verbose', 'ci', 'mo_coeff', 'mo_energy',
                        'e_cas', 'e_tot', 'ncore', 'ncas', 'nelecas', 'mol',
                        'callback', 'fcisolver'))

    casscf_settings = {}
    for k, v in casscf.__dict__.items():
        if not (k.startswith('_') or k in exclude_keys):
            if (v is None or
                isinstance(v, (str, bool, int, float, list, tuple, dict))):
                casscf_settings[k] = v
            elif isinstance(v, set):
                casscf_settings[k] = list(v)

    doc = '''# JSON format
# Note the double quote "" around keyword
'''
    conf = {'casscf': casscf_settings}
    with open(configfile, 'w') as f:
        f.write(doc)
        f.write(json.dumps(conf, indent=4, sort_keys=True) + '\n')
        f.write('# Starting from this line, code are parsed as Python script.  The Python code\n'
                '# will be injected to casscf.kernel through callback hook.  The casscf.kernel\n'
                '# function local variables can be directly accessed.  Note, these variables\n'
                '# cannot be directly modified because the environment is generated using\n'
                '# locals() function (see\n'
                '# https://docs.python.org/2/library/functions.html#locals).\n'
                '# You can modify some variables with inplace updating, eg\n'
                '# from pyscf import fci\n'
                '# if imacro > 6:\n'
                '#     casscf.fcislover = fci.fix_spin_(fci.direct_spin1, ss=2)\n'
                '#     mo[:,:3] *= -1\n'
                '# Warning: this runtime modification is unsafe and highly unrecommended.\n')

    old_cb = casscf.callback
    def hot_load(envs):
        try:
            with open(configfile) as f:
# filter out comments
                raw_js = []
                balance = 0
                data = [x for x in f.readlines()
                        if not x.startswith('#') and x.rstrip()]
                for n, line in enumerate(data):
                    if not line.lstrip().startswith('#'):
                        raw_js.append(line)
                        balance += line.count('{') - line.count('}')
                        if balance == 0:
                            break
            raw_py = ''.join(data[n+1:])
            raw_js = ''.join(raw_js)

            logger.debug(casscf, 'Reading CASSCF parameters from config file  %s',
                         os.path.realpath(configfile))
            logger.debug1(casscf, '    Inject casscf settings %s', raw_js)
            conf = json.loads(raw_js)
            casscf.__dict__.update(conf.pop('casscf'))

            # Not yet found a way to update locals() on the runtime
            # https://docs.python.org/2/library/functions.html#locals
            #for k in conf:
            #    if k in envs:
            #        logger.info(casscf, 'Update envs[%s] = %s', k, conf[k])
            #        envs[k] = conf[k]

            logger.debug1(casscf, '    Inject python script\n%s\n', raw_py)
            if len(raw_py.strip()) > 0:
                if sys.version_info >= (3,):
# A hacky call using eval because exec are so different in python2 and python3
                    eval(compile('exec(raw_py, envs, {})', '<str>', 'exec'))
                else:
                    eval(compile('exec raw_py in envs, {}', '<str>', 'exec'))
        except Exception as e:
            logger.warn(casscf, 'CASSCF hot_load error %s', e)
            logger.warn(casscf, ''.join(traceback.format_exc()))

        if callable(old_cb):
            old_cb(envs)

    casscf.callback = hot_load
    return casscf


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import mcscf
    from pyscf.tools import ring

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['H', c] for c in ring.make(6, 1.2)]
    mol.basis = '6-31g'
    mol.build()

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


    mc = mcscf.CASSCF(m, 4, 4)
    mc.verbose = 4
    hot_tuning_(mc, 'config1')
    mc.kernel()
    mc = None  # release for gc
