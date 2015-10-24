#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import numpy
import pyscf.lib
import pyscf.gto
from pyscf.lib import logger
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.tools.mo_mapping import mo_1to1map


def extract_orbs(mo_coeff, ncas, nelecas, ncore):
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    mo_vir = mo_coeff[:,nocc:]
    return mo_core, mo_cas, mo_vir

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''CAS sapce one-electron hamiltonian

    Args:
        casci : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    hcore = casci.get_hcore()
    if mo_core.size == 0:
        corevhf = 0
        energy_core = 0
    else:
        core_dm = numpy.dot(mo_core, mo_core.T) * 2
        corevhf = casci.get_veff(casci.mol, core_dm)
        energy_core = numpy.einsum('ij,ji', core_dm, hcore) \
                    + numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore+corevhf, mo_cas))
    return h1eff, energy_core

def analyze(casscf, mo_coeff=None, ci=None, verbose=logger.INFO):
    from pyscf.tools import dump_mat
    from pyscf.mcscf import addons
    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ci is None: ci = casscf.ci
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(casscf.stdout, verbose)
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncore + ncas
    label = casscf.mol.spheric_labels(True)

    if hasattr(casscf.fcisolver, 'make_rdm1s'):
        casdm1a, casdm1b = casscf.fcisolver.make_rdm1s(ci, ncas, nelecas)
        casdm1 = casdm1a + casdm1b
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:nocc]
        dm1b = numpy.dot(mocore, mocore.T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.T))
        dm1 = dm1a + dm1b
        if log.verbose >= logger.DEBUG1:
            log.info('alpha density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1a, label)
            log.info('beta density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1b, label)
    else:
        casdm1 = casscf.fcisolver.make_rdm1(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:nocc]
        dm1a =(numpy.dot(mocore, mocore.T) * 2
             + reduce(numpy.dot, (mocas, casdm1, mocas.T)))
        dm1b = None
        dm1 = dm1a

    if log.verbose >= logger.INFO:
        # note the last two args of ._eig for mc1step_symm
        occ, ucas = casscf._eig(-casdm1, ncore, nocc)
        log.info('Natural occ %s', str(-occ))
        for i, k in enumerate(numpy.argmax(abs(ucas), axis=0)):
            if ucas[k,i] < 0:
                ucas[:,i] *= -1
        mo_cas = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
        log.info('Natural orbital in CAS space')
        dump_mat.dump_rec(log.stdout, mo_cas, label, start=1)

        if casscf._scf.mo_coeff is not None:
            s = reduce(numpy.dot, (casscf.mo_coeff.T, casscf._scf.get_ovlp(),
                                   casscf._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.4)
            for i,j in idx:
                log.info('<mo-mcscf|mo-hf> %d  %d  %12.8f', i+1, j+1, s[i,j])

        if ci is not None:
            log.info('** Largest CI components **')
            log.info(' string alpha, string beta, CI coefficients')
            for c,ia,ib in fci.addons.large_ci(ci, casscf.ncas, casscf.nelecas):
                log.info('  %9s    %9s    %.12f', ia, ib, c)

        s = casscf._scf.get_ovlp()
        #casscf._scf.mulliken_pop(casscf.mol, dm1, s, verbose=log)
        casscf._scf.mulliken_pop_meta_lowdin_ao(casscf.mol, dm1, verbose=log)
    return dm1a, dm1b

def get_fock(mc, mo_coeff=None, ci=None, eris=None, verbose=None):
    '''Generalized Fock matrix
    '''
    from pyscf.mcscf import mc_ao2mo
    if ci is None: ci = mc.ci
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    h1 = reduce(numpy.dot, (mo_coeff.T, mc.get_hcore(), mo_coeff))
    if eris is not None and hasattr(eris, 'ppaa'):
        vj = numpy.empty((nmo,nmo))
        vk = numpy.empty((nmo,nmo))
        for i in range(nmo):
            vj[i] = numpy.einsum('ij,qij->q', casdm1, eris.ppaa[i])
            vk[i] = numpy.einsum('ij,iqj->q', casdm1, eris.papa[i])
        fock = h1 + eris.vhf_c + vj - vk * .5
    else:
        dm_core = numpy.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].T)
        mocas = mo_coeff[:,ncore:nocc]
        dm = dm_core + reduce(numpy.dot, (mocas, casdm1, mocas.T))
        vj, vk = mc._scf.get_jk(mc.mol, dm)
        fock = h1 + reduce(numpy.dot, (mo_coeff.T, vj-vk*.5, mo_coeff))
    return fock

def cas_natorb(mc, mo_coeff=None, ci=None, eris=None, sort=False,
               verbose=None):
    '''Transform active orbitals to natrual orbitals, and update the CI wfn

    Args:
        mc : a CASSCF/CASCI object or RHF object

    Kwargs:
        sort : bool
            Sort natural orbitals wrt the occupancy.  Be careful with this
            option since the resultant natural orbitals might have the
            different symmetry to the irreps indicated by CASSCF.orbsym

    Returns:
        A tuple, the first item is natural orbitals, the second is updated CI
        coefficients, the third is the natural occupancy associated to the
        natural orbitals.
    '''
    from pyscf.mcscf import mc_ao2mo
    from pyscf.tools import dump_mat
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mc.stdout, mc.verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    occ, ucas = mc._eig(-casdm1, ncore, nocc)
    if sort:
        idx = numpy.argsort(occ)
        occ = occ[idx]
        ucas = ucas[:,idx]
        if hasattr(mc, 'orbsym'): # for casci_symm
            mc.orbsym[ncore:nocc] = mc.orbsym[ncore:nocc][idx]
            mc.fcisolver.orbsym = mc.orbsym[ncore:nocc]

    occ = -occ

# where_natorb gives the location of the natural orbital for the input cas
# orbitals.  gen_strings4orblist map thes sorted strings (on CAS orbital) to
# the unsorted determinant strings (on natural orbital). e.g.  (3o,2e) system
#       CAS orbital      1  2  3
#       natural orbital  3  1  2        <= by mo_1to1map
#       CASorb-strings   0b011, 0b101, 0b110
#                    ==  (1,2), (1,3), (2,3)
#       natorb-strings   (3,1), (3,2), (1,2)
#                    ==  0B101, 0B110, 0B011    <= by gen_strings4orblist
# then argsort to translate the string representation to the address
#       [2(=0B011), 0(=0B101), 1(=0B110)]
# to indicate which CASorb-strings address to be loaded in each natorb-strings slot
    where_natorb = mo_1to1map(ucas)
    #guide_stringsa = fci.cistring.gen_strings4orblist(where_natorb, nelecas[0])
    #guide_stringsb = fci.cistring.gen_strings4orblist(where_natorb, nelecas[1])
    #old_det_idxa = numpy.argsort(guide_stringsa)
    #old_det_idxb = numpy.argsort(guide_stringsb)
    #ci0 = ci[old_det_idxa[:,None],old_det_idxb]
    if isinstance(ci, numpy.ndarray):
        ci0 = fci.addons.reorder(ci, nelecas, where_natorb)
    elif isinstance(ci, (tuple, list)) and isinstance(ci[0], numpy.ndarray):
        # for state-average eigenfunctions
        ci0 = [fci.addons.reorder(x, nelecas, where_natorb) for x in ci]
    else:
        log.info('FCI vector not available, so not using old wavefunction as initial guess')
        ci0 = None

# restore phase, to ensure the reordered ci vector is the correct initial guess
    for i, k in enumerate(where_natorb):
        if ucas[i,k] < 0:
            ucas[:,k] *= -1
    mo_coeff1 = mo_coeff.copy()
    mo_coeff1[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
    if log.verbose >= logger.INFO:
        log.debug('where_natorb %s', str(where_natorb))
        log.info('Natural occ %s', str(occ))
        log.info('Natural orbital in CAS space')
        label = mc.mol.spheric_labels(True)
        dump_mat.dump_rec(log.stdout, mo_coeff1[:,ncore:nocc], label, start=1)

        if mc._scf.mo_coeff is not None:
            s = reduce(numpy.dot, (mo_coeff1[:,ncore:nocc].T,
                                   mc._scf.get_ovlp(), mc._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.4)
            for i,j in idx:
                log.info('<CAS-nat-orb|mo-hf>  %d  %d  %12.8f',
                         ncore+i+1, j+1, s[i,j])

    mocas = mo_coeff1[:,ncore:nocc]
    h1eff = reduce(numpy.dot, (mocas.T, mc.get_hcore(), mocas))
    if eris is not None and hasattr(eris, 'ppaa'):
        h1eff += reduce(numpy.dot, (ucas.T, eris.vhf_c[ncore:nocc,ncore:nocc], ucas))
        aaaa = ao2mo.restore(4, eris.ppaa[ncore:nocc,ncore:nocc,:,:], ncas)
        aaaa = ao2mo.incore.full(aaaa, ucas)
    else:
        dm_core = numpy.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].T)
        vj, vk = mc._scf.get_jk(mc.mol, dm_core)
        h1eff += reduce(numpy.dot, (mocas.T, vj-vk*.5, mocas))
        aaaa = ao2mo.kernel(mc.mol, mocas)
    e_cas, fcivec = mc.fcisolver.kernel(h1eff, aaaa, ncas, nelecas, ci0=ci0)
    log.debug('In Natural orbital, CI energy = %.12g', e_cas)
    return mo_coeff1, fcivec, occ

def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, verbose=logger.NOTE):
    '''Canonicalize CASCI/CASSCF orbitals

    Args:
        mc : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, the first item is natural orbitals, the second is updated CI
        coefficients.
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mc.stdout, mc.verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_coeff.shape[1]
    fock = mc.get_fock(mo_coeff, ci, eris)
    if cas_natorb:
        mo_coeff1, ci, occ = mc.cas_natorb(mo_coeff, ci, eris, sort=sort,
                                           verbose=verbose)
    else:
# Keep the active space unchanged by default.  The rotation in active space
# may cause problem for external CI solver eg DMRG.
        mo_coeff1 = numpy.empty_like(mo_coeff)
        mo_coeff1[:,ncore:nocc] = mo_coeff[:,ncore:nocc]
    if ncore > 0:
        # note the last two args of ._eig for mc1step_symm
        w, c1 = mc._eig(fock[:ncore,:ncore], 0, ncore)
        if sort:
            idx = numpy.argsort(w)
            w = w[idx]
            c1 = c1[:,idx]
            if hasattr(mc, 'orbsym'): # for mc1step_symm
                mc.orbsym[:ncore] = mc.orbsym[:ncore][idx]
        mo_coeff1[:,:ncore] = numpy.dot(mo_coeff[:,:ncore], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(ncore):
                log.debug('i = %d  <i|F|i> = %12.8f', i+1, w[i])
    if nmo-nocc > 0:
        w, c1 = mc._eig(fock[nocc:,nocc:], nocc, nmo)
        if sort:
            idx = numpy.argsort(w)
            w = w[idx]
            c1 = c1[:,idx]
            if hasattr(mc, 'orbsym'): # for mc1step_symm
                mc.orbsym[nocc:] = mc.orbsym[nocc:][idx]
        mo_coeff1[:,nocc:] = numpy.dot(mo_coeff[:,nocc:], c1)
        if log.verbose >= logger.DEBUG:
            for i in range(nmo-nocc):
                log.debug('i = %d  <i|F|i> = %12.8f', nocc+i+1, w[i])
# still return ci coefficients, in case the canonicalization funciton changed
# cas orbitals, the ci coefficients should also be updated.
    return mo_coeff1, ci


def kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''CASCI solver
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(casci.stdout, verbose)
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    t0 = (time.clock(), time.time())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas
    ncore = casci.ncore
    mo_core, mo_cas, mo_vir = extract_orbs(mo_coeff, ncas, nelecas, ncore)

    # 1e
    h1eff, energy_core = casci.h1e_for_cas(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t0)

    # 2e
    eri_cas = casci.ao2mo(mo_cas)
    t1 = log.timer('integral transformation to CAS space', *t1)

    # FCI
    max_memory = max(400, casci.max_memory-pyscf.lib.current_memory()[0])
    e_cas, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory)

    t1 = log.timer('FCI solver', *t1)
    e_tot = e_cas + energy_core + casci.mol.energy_nuc()
    log.timer('CASCI', *t0)
    return e_tot, e_cas, fcivec


class CASCI(object):
    '''CASCI

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        ncas : int
            Active space size.
        nelecas : tuple of int
            Active (nelec_alpha, nelec_beta)
        ncore : int or tuple of int
            Core electron number.  In UHF-CASSCF, it's a tuple to indicate the different core eletron numbers.
        fcisolver : an instance of :class:`FCISolver`
            The pyscf.fci module provides several FCISolver for different scenario.  Generally,
            fci.direct_spin1.FCISolver can be used for all RHF-CASSCF.  However, a proper FCISolver
            can provide better performance and better numerical stability.  One can either use
            :func:`fci.solver` function to pick the FCISolver by the program or manually assigen
            the FCISolver to this attribute, e.g.

            >>> from pyscf import fci
            >>> mc = mcscf.CASSCF(mf, 4, 4)
            >>> mc.fcisolver = fci.solver(mol, singlet=True)
            >>> mc.fcisolver = fci.direct_spin1.FCISolver(mol)

            You can control FCISolver by setting e.g.::

                >>> mc.fcisolver.max_cycle = 30
                >>> mc.fcisolver.conv_tol = 1e-7

            For more details of the parameter for FCISolver, See :mod:`fci`.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        ci : ndarray
            CAS space FCI coefficients

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASCI(mf, 6, 6)
    >>> mc.kernel()[0]
    -108.980200816243354
    '''
    def __init__(self, mf, ncas, nelecas, ncore=None):
        mol = mf.mol
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, numpy.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1])
        if ncore is None:
            ncorelec = mol.nelectron - (self.nelecas[0]+self.nelecas[1])
            assert(ncorelec % 2 == 0)
            self.ncore = ncorelec // 2
        else:
            assert(isinstance(ncore, (int, numpy.integer)))
            self.ncore = ncore
        #self.fcisolver = fci.direct_spin0.FCISolver(mol)
        self.fcisolver = fci.solver(mol, self.nelecas[0]==self.nelecas[1])
# CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = 1e-10
        self.fcisolver.max_cycle = 50
        self.fcisolver.conv_tol = 1e-8

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mf.mo_coeff
        self.ci = None
        self.e_tot = 0

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        log.info('max_memory %d (MB)', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = numpy.dot(mocore, mocore.T) * 2
# don't call self._scf.get_veff, _scf object might be from DFT
        vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        return vj - vk * .5

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_h2cas(self, mo_coeff=None):
        return self.ao2mo(mo_coeff)
    def get_h2eff(self, mo_coeff=None):
        return self.ao2mo(mo_coeff)
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:,self.ncore:self.ncore+self.ncas]
        nao, nmo = mo_coeff.shape
        if self._scf._eri is not None and \
           (nao**2*nmo**2+nmo**4*2+self._scf._eri.size)*8/1e6 < self.max_memory*.95:
            eri = pyscf.ao2mo.incore.full(self._scf._eri, mo_coeff)
        else:
            eri = pyscf.ao2mo.outcore.full_iofree(self.mol, mo_coeff,
                                                  verbose=self.verbose)
        return eri

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    def h1e_for_cas(self, mo_coeff=None, ncas=None, ncore=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return h1e_for_cas(self, mo_coeff, ncas, ncore)

    def casci(self, mo_coeff=None, ci0=None):
        return self.kernel(mo_coeff, ci0)
    def kernel(self, mo_coeff=None, ci0=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci

        if self.verbose > logger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.dump_flags()

        self.e_tot, e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=self.verbose)

        log = logger.Logger(self.stdout, self.verbose)
        if log.verbose >= logger.NOTE and hasattr(self.fcisolver, 'spin_square'):
            ss = self.fcisolver.spin_square(self.ci, self.ncas, self.nelecas)
            if isinstance(e_cas, (float, numpy.number)):
                log.note('CASCI E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                         self.e_tot, e_cas, ss[0])
            else:
                for i, e in enumerate(e_cas):
                    log.note('CASCI root %d  E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                             i, self.e_tot[i], e, ss[0][i])
        else:
            if isinstance(e_cas, (float, numpy.number)):
                log.note('CASCI E = %.15g  E(CI) = %.15g', self.e_tot, e_cas)
            else:
                for i, e in enumerate(e_cas):
                    log.note('CASCI root %d  E = %.15g  E(CI) = %.15g',
                             i, self.e_tot[i], e)
        return self.e_tot, e_cas, self.ci

    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False,
                   verbose=None):
        return cas_natorb(self, mo_coeff, ci, eris, sort, verbose)
    def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                    verbose=None):
        self.mo_coeff, self.ci, occ = cas_natorb(self, mo_coeff, ci, eris,
                                                 sort, verbose)
        return self.mo_coeff, self.ci, occ

    def get_fock(self, mo_coeff=None, ci=None, eris=None, verbose=None):
        return get_fock(self, mo_coeff, ci, eris, verbose)

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     cas_natorb=False, verbose=None):
        return canonicalize(self, mo_coeff, ci, eris, sort, cas_natorb, verbose)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, verbose=None):
        self.mo_coeff, self.ci = canonicalize(self, mo_coeff, ci, eris,
                                              sort, cas_natorb, verbose)
        return self.mo_coeff, self.ci

    def analyze(self, mo_coeff=None, ci=None, verbose=logger.INFO):
        return analyze(self, mo_coeff, ci, verbose)

    def sort_mo(self, caslst, mo_coeff=None, base=1):
        from pyscf.mcscf import addons
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo(self, mo_coeff, caslst, base)

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1a, casdm1b = self.fcisolver.make_rdm1s(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1b = numpy.dot(mocore, mocore.T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.T))
        return dm1a, dm1b

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1 = self.fcisolver.make_rdm1(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1 = numpy.dot(mocore, mocore.T) * 2
        dm1 = dm1 + reduce(numpy.dot, (mocas, casdm1, mocas.T))
        return dm1


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASCI(m, 4, 4)
    mc.fcisolver = fci.solver(mol)
    emc = mc.casci()[0]
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

    mc = CASCI(m, 4, (3,1))
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    emc = mc.casci()[0]
    print(emc - -75.439016172976)

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = "out_casci"
    mol.atom = [
        ["C", (-0.65830719,  0.61123287, -0.00800148)],
        ["C", ( 0.73685281,  0.61123287, -0.00800148)],
        ["C", ( 1.43439081,  1.81898387, -0.00800148)],
        ["C", ( 0.73673681,  3.02749287, -0.00920048)],
        ["C", (-0.65808819,  3.02741487, -0.00967948)],
        ["C", (-1.35568919,  1.81920887, -0.00868348)],
        ["H", (-1.20806619, -0.34108413, -0.00755148)],
        ["H", ( 1.28636081, -0.34128013, -0.00668648)],
        ["H", ( 2.53407081,  1.81906387, -0.00736748)],
        ["H", ( 1.28693681,  3.97963587, -0.00925948)],
        ["H", (-1.20821019,  3.97969587, -0.01063248)],
        ["H", (-2.45529319,  1.81939187, -0.00886348)],]

    mol.basis = {'H': 'sto-3g',
                 'C': 'sto-3g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASCI(m, 9, 8)
    mc.fcisolver = fci.solver(mol)
    emc = mc.casci()[0]
    print(ehf, emc, emc-ehf)
    print(emc - -227.948912536)

    mc = CASCI(m, 9, (5,3))
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    mc.fcisolver.nroots = 3
    emc = mc.casci()[0]
    print(emc[0] - -227.7674519720)
