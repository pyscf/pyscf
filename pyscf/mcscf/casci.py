#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.mcscf import addons
from pyscf import symm


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
    if hasattr(casci, 'energy_nuc'):
        energy_core = casci.energy_nuc()
    else:
        energy_core = casci._scf.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = numpy.dot(mo_core, mo_core.T) * 2
        corevhf = casci.get_veff(casci.mol, core_dm)
        energy_core += numpy.einsum('ij,ji', core_dm, hcore)
        energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore+corevhf, mo_cas))
    return h1eff, energy_core

def analyze(casscf, mo_coeff=None, ci=None, verbose=logger.INFO,
            large_ci_tol=.1, **kwargs):
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    log = logger.new_logger(casscf, verbose)

    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ci is None: ci = casscf.ci
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncore + ncas
    label = casscf.mol.ao_labels()

    if isinstance(ci, (tuple, list)):
        ci0 = ci[0]
        log.info('** Natural natural orbitals are based on the first root **')
    else:
        ci0 = ci
    if ci0 is None and hasattr(casscf, 'casdm1'):
        casdm1 = casscf.casdm1
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:nocc]
        dm1a =(numpy.dot(mocore, mocore.T) * 2
             + reduce(numpy.dot, (mocas, casdm1, mocas.T)))
        dm1b = None
        dm1 = dm1a
    elif hasattr(casscf.fcisolver, 'make_rdm1s'):
        casdm1a, casdm1b = casscf.fcisolver.make_rdm1s(ci0, ncas, nelecas)
        casdm1 = casdm1a + casdm1b
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:nocc]
        dm1b = numpy.dot(mocore, mocore.T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.T))
        dm1 = dm1a + dm1b
        if log.verbose >= logger.DEBUG2:
            log.info('alpha density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1a, label, **kwargs)
            log.info('beta density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1b, label, **kwargs)
    else:
        casdm1 = casscf.fcisolver.make_rdm1(ci0, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:nocc]
        dm1a =(numpy.dot(mocore, mocore.T) * 2
             + reduce(numpy.dot, (mocas, casdm1, mocas.T)))
        dm1b = None
        dm1 = dm1a

    if log.verbose >= logger.INFO:
        ovlp_ao = casscf._scf.get_ovlp()
        # note the last two args of ._eig for mc1step_symm
        occ, ucas = casscf._eig(-casdm1, ncore, nocc)
        log.info('Natural occ %s', str(-occ))
        for i, k in enumerate(numpy.argmax(abs(ucas), axis=0)):
            if ucas[k,i] < 0:
                ucas[:,i] *= -1
        orth_coeff = orth.orth_ao(casscf.mol, 'meta_lowdin', s=ovlp_ao)
        mo_cas = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff[:,ncore:nocc], ucas))
        log.info('Natural orbital (expansion on meta-Lowdin AOs) in CAS space')
        dump_mat.dump_rec(log.stdout, mo_cas, label, start=1, **kwargs)
        if log.verbose >= logger.DEBUG2:
            if not casscf.natorb:
                log.debug2('NOTE: mc.mo_coeff in active space is different to '
                           'the natural orbital coefficients printed in above.')
            log.debug2(' ** CASCI/CASSCF orbital coefficients (expansion on meta-Lowdin AOs) **')
            c = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff))
            dump_mat.dump_rec(log.stdout, c, label, start=1, **kwargs)

        if casscf._scf.mo_coeff is not None:
            s = reduce(numpy.dot, (casscf.mo_coeff.T, ovlp_ao, casscf._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.4)
            for i,j in idx:
                log.info('<mo-mcscf|mo-hf> %d  %d  %12.8f', i+1, j+1, s[i,j])

        if hasattr(casscf.fcisolver, 'large_ci') and ci is not None:
            log.info('** Largest CI components **')
            if isinstance(ci, (tuple, list)):
                for i, civec in enumerate(ci):
                    res = casscf.fcisolver.large_ci(civec, casscf.ncas, casscf.nelecas,
                                                    large_ci_tol, return_strs=False)
                    log.info('  [alpha occ-orbitals] [beta occ-orbitals]  state %-3d CI coefficient', i)
                    for c,ia,ib in res:
                        log.info('  %-20s %-30s %.12f', ia, ib, c)
            else:
                log.info('  [alpha occ-orbitals] [beta occ-orbitals]            CI coefficient')
                res = casscf.fcisolver.large_ci(ci, casscf.ncas, casscf.nelecas,
                                                large_ci_tol, return_strs=False)
                for c,ia,ib in res:
                    log.info('  %-20s %-30s %.12f', ia, ib, c)

        casscf._scf.mulliken_meta(casscf.mol, dm1, s=ovlp_ao, verbose=log)
    return dm1a, dm1b

def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    '''Generalized Fock matrix in AO representation
    '''
    if ci is None: ci = mc.ci
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    if eris is not None and hasattr(eris, 'ppaa'):
        vj = numpy.empty((nmo,nmo))
        vk = numpy.empty((nmo,nmo))
        for i in range(nmo):
            vj[i] = numpy.einsum('ij,qij->q', casdm1, eris.ppaa[i])
            vk[i] = numpy.einsum('ij,iqj->q', casdm1, eris.papa[i])
        mo_inv = numpy.dot(mo_coeff.T, mc._scf.get_ovlp())
        fock =(mc.get_hcore()
             + reduce(numpy.dot, (mo_inv.T, eris.vhf_c+vj-vk*.5, mo_inv)))
    else:
        dm_core = numpy.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].T)
        mocas = mo_coeff[:,ncore:nocc]
        dm = dm_core + reduce(numpy.dot, (mocas, casdm1, mocas.T))
        vj, vk = mc._scf.get_jk(mc.mol, dm)
        fock = mc.get_hcore() + vj-vk*.5
    return fock

def cas_natorb(mc, mo_coeff=None, ci=None, eris=None, sort=False,
               casdm1=None, verbose=None):
    '''Transform active orbitals to natrual orbitals, and update the CI wfn

    Args:
        mc : a CASSCF/CASCI object or RHF object

    Kwargs:
        sort : bool
            Sort natural orbitals wrt the occupancy.

    Returns:
        A tuple, the first item is natural orbitals, the second is updated CI
        coefficients, the third is the natural occupancy associated to the
        natural orbitals.
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    from pyscf.tools.mo_mapping import mo_1to1map
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
    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    # orbital symmetry is reserved in this _eig call
    occ, ucas = mc._eig(-casdm1, ncore, nocc)
    if sort:
        casorb_idx = numpy.argsort(occ)
        occ = occ[casorb_idx]
        ucas = ucas[:,casorb_idx]
# restore phase
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
    for i, k in enumerate(where_natorb):
        if ucas[i,k] < 0:
            ucas[:,k] *= -1

    occ = -occ
    mo_occ = numpy.zeros(mo_coeff.shape[1])
    mo_occ[:ncore] = 2
    mo_occ[ncore:nocc] = occ

    mo_coeff1 = mo_coeff.copy()
    mo_coeff1[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
    if hasattr(mo_coeff, 'orbsym'):
        orbsym = numpy.copy(mo_coeff.orbsym)
        if sort:
            orbsym[ncore:nocc] = orbsym[ncore:nocc][casorb_idx]
        mo_coeff1 = lib.tag_array(mo_coeff1, orbsym=orbsym)

    if isinstance(ci, numpy.ndarray):
        fcivec = fci.addons.transform_ci_for_orbital_rotation(ci, ncas, nelecas, ucas)
    elif isinstance(ci, (tuple, list)) and isinstance(ci[0], numpy.ndarray):
        # for state-average eigenfunctions
        fcivec = [fci.addons.transform_ci_for_orbital_rotation(x, ncas, nelecas, ucas)
                  for x in ci]
    else:
        log.info('FCI vector not available, call CASCI for wavefunction')
        mocas = mo_coeff1[:,ncore:nocc]
        hcore = mc.get_hcore()
        dm_core = numpy.dot(mo_coeff1[:,:ncore]*2, mo_coeff1[:,:ncore].T)
        ecore = mc._scf.energy_nuc()
        ecore+= numpy.einsum('ij,ji', hcore, dm_core)
        h1eff = reduce(numpy.dot, (mocas.T, hcore, mocas))
        if eris is not None and hasattr(eris, 'ppaa'):
            ecore += eris.vhf_c[:ncore,:ncore].trace()
            h1eff += reduce(numpy.dot, (ucas.T, eris.vhf_c[ncore:nocc,ncore:nocc], ucas))
            aaaa = ao2mo.restore(4, eris.ppaa[ncore:nocc,ncore:nocc,:,:], ncas)
            aaaa = ao2mo.incore.full(aaaa, ucas)
        else:
            corevhf = mc.get_veff(mc.mol, dm_core)
            ecore += numpy.einsum('ij,ji', dm_core, corevhf) * .5
            h1eff += reduce(numpy.dot, (mocas.T, corevhf, mocas))
            aaaa = ao2mo.kernel(mc.mol, mocas)

        # See label_symmetry_ function in casci_symm.py which initialize the
        # orbital symmetry information in fcisolver.  This orbital symmetry
        # labels should be reordered to match the sorted active space orbitals.
        if hasattr(mo_coeff1, 'orbsym') and sort:
            mc.fcisolver.orbsym = mo_coeff1.orbsym[ncore:nocc]

        max_memory = max(400, mc.max_memory-lib.current_memory()[0])
        e, fcivec = mc.fcisolver.kernel(h1eff, aaaa, ncas, nelecas, ecore=ecore,
                                        max_memory=max_memory, verbose=log)
        log.debug('In Natural orbital, CASCI energy = %.12g', e)

    if log.verbose >= logger.INFO:
        ovlp_ao = mc._scf.get_ovlp()
        log.debug('where_natorb %s', str(where_natorb))
        log.info('Natural occ %s', str(occ))
        log.info('Natural orbital (expansion on meta-Lowdin AOs) in CAS space')
        label = mc.mol.ao_labels()
        orth_coeff = orth.orth_ao(mc.mol, 'meta_lowdin', s=ovlp_ao)
        mo_cas = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff1[:,ncore:nocc]))
        dump_mat.dump_rec(log.stdout, mo_cas, label, start=1)

        if mc._scf.mo_coeff is not None:
            s = reduce(numpy.dot, (mo_coeff1[:,ncore:nocc].T,
                                   mc._scf.get_ovlp(), mc._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.4)
            for i,j in idx:
                log.info('<CAS-nat-orb|mo-hf>  %d  %d  %12.8f',
                         ncore+i+1, j+1, s[i,j])
    return mo_coeff1, fcivec, mo_occ

def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE):
    '''Canonicalize CASCI/CASSCF orbitals

    Args:
        mc : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, (natural orbitals, CI coefficients, orbital energies)
        The orbital energies are the diagonal terms of general Fock matrix.
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    log = logger.new_logger(mc, verbose)

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_coeff.shape[1]
    fock_ao = mc.get_fock(mo_coeff, ci, eris, casdm1, verbose)
    fock = reduce(numpy.dot, (mo_coeff.T, fock_ao, mo_coeff))
    mo_energy = fock.diagonal().copy()
    if cas_natorb:
        mo_coeff1, ci, occ = mc.cas_natorb(mo_coeff, ci, eris, sort, casdm1,
                                           verbose)
        ma = mo_coeff1[:,ncore:nocc]
        mo_energy[ncore:nocc] = numpy.einsum('ji,ji->i', ma, fock_ao.dot(ma))
    else:
# Keep the active space unchanged by default.  The rotation in active space
# may cause problem for external CI solver eg DMRG.
        mo_coeff1 = numpy.empty_like(mo_coeff)
        mo_coeff1[:,ncore:nocc] = mo_coeff[:,ncore:nocc]
    if ncore > 0:
        # note the last two args of ._eig for mc1step_symm
        # mc._eig function is called to handle symmetry adapated fock
        w, c1 = mc._eig(fock[:ncore,:ncore], 0, ncore)
        if sort:
            idx = numpy.argsort(w.round(9))
            w = w[idx]
            c1 = c1[:,idx]
        mo_coeff1[:,:ncore] = numpy.dot(mo_coeff[:,:ncore], c1)
        mo_energy[:ncore] = w
    if nmo-nocc > 0:
        w, c1 = mc._eig(fock[nocc:,nocc:], nocc, nmo)
        if sort:
            idx = numpy.argsort(w.round(9))
            w = w[idx]
            c1 = c1[:,idx]
        mo_coeff1[:,nocc:] = numpy.dot(mo_coeff[:,nocc:], c1)
        mo_energy[nocc:] = w

    if hasattr(mo_coeff, 'orbsym'):
        if sort:
            orbsym = symm.label_orb_symm(mc.mol, mc.mol.irrep_id,
                                         mc.mol.symm_orb, mo_coeff1)
        else:
            orbsym = mo_coeff.orbsym
        mo_coeff1 = lib.tag_array(mo_coeff1, orbsym=orbsym)

    if log.verbose >= logger.DEBUG:
        for i in range(nmo):
            log.debug('i = %d  <i|F|i> = %12.8f', i+1, mo_energy[i])
# still return ci coefficients, in case the canonicalization funciton changed
# cas orbitals, the ci coefficients should also be updated.
    return mo_coeff1, ci, mo_energy


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

    # 2e
    eri_cas = casci.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = casci.get_h1eff(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t1)

    # FCI
    max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec


class CASCI(lib.StreamObject):
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
        natorb : bool
            Whether to restore the natural orbital in CAS space.  Default is not.
            Be very careful to set this parameter when CASCI/CASSCF are combined
            with DMRG solver because this parameter changes the orbital ordering
            which DMRG relies on.
        canonicalization : bool
            Whether to canonicalize orbitals.  Default is True.
        sorting_mo_energy : bool
            Whether to sort the orbitals based on the diagonal elements of the
            general Fock matrix.  Default is False.
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
        #self.fcisolver = fci.solver(mol, self.nelecas[0]==self.nelecas[1], False)
        self.fcisolver = fci.solver(mol, singlet=False, symm=False)
# CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = 1e-10
        self.fcisolver.max_cycle = 200
        self.fcisolver.conv_tol = 1e-8
        self.natorb = False
        self.canonicalization = True
        self.sorting_mo_energy = False

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        assert(self.ncas > 0)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('max_memory %d (MB)', self.max_memory)
        if self.mo_coeff is None:
            log.warn('Orbital initial guess is not given.\n'
                     'You may need mf.kernel() to generate initial guess form SCF calculation.')
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        if self.mo_coeff is None:
            log.warn('Orbital for CASCI is not specified.  You probably need '
                     'call SCF.kernel() to initialize orbitals.')
        return self

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
        elif mo_coeff.shape[1] != self.ncas:
            mo_coeff = mo_coeff[:,self.ncore:self.ncore+self.ncas]

        if self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    @lib.with_doc(h1e_for_cas.__doc__)
    def h1e_for_cas(self, mo_coeff=None, ncas=None, ncore=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return h1e_for_cas(self, mo_coeff, ncas, ncore)
    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    get_h1cas.__doc__ = h1e_for_cas.__doc__
    get_h1eff.__doc__ = h1e_for_cas.__doc__

    def casci(self, mo_coeff=None, ci0=None):
        return self.kernel(mo_coeff, ci0)
    def kernel(self, mo_coeff=None, ci0=None):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_tot, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=self.verbose)

        log = logger.Logger(self.stdout, self.verbose)

        if self.canonicalization:
            if isinstance(self.e_cas, (float, numpy.number)):
                self.canonicalize_(mo_coeff, self.ci,
                                   sort=self.sorting_mo_energy,
                                   cas_natorb=self.natorb, verbose=log)
            else:
                self.canonicalize_(mo_coeff, self.ci[0],
                                   sort=self.sorting_mo_energy,
                                   cas_natorb=self.natorb, verbose=log)

        if hasattr(self.fcisolver, 'converged'):
            if numpy.all(self.fcisolver.converged):
                log.info('CASCI converged')
            else:
                log.info('CASCI not converged')
        if log.verbose >= logger.NOTE and hasattr(self.fcisolver, 'spin_square'):
            if isinstance(self.e_cas, (float, numpy.number)):
                ss = self.fcisolver.spin_square(self.ci, self.ncas, self.nelecas)
                log.note('CASCI E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                         self.e_tot, self.e_cas, ss[0])
            else:
                for i, e in enumerate(self.e_cas):
                    ss = self.fcisolver.spin_square(self.ci[i], self.ncas, self.nelecas)
                    log.note('CASCI root %d  E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                             i, self.e_tot[i], e, ss[0])
        else:
            if isinstance(self.e_cas, (float, numpy.number)):
                log.note('CASCI E = %.15g  E(CI) = %.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI root %d  E = %.15g  E(CI) = %.15g',
                             i, self.e_tot[i], e)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def _finalize(self):
        pass

    @lib.with_doc(cas_natorb.__doc__)
    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False,
                   casdm1=None, verbose=None):
        return cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose)
    @lib.with_doc(cas_natorb.__doc__)
    def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                    casdm1=None, verbose=None):
        self.mo_coeff, self.ci, occ = cas_natorb(self, mo_coeff, ci, eris,
                                                 sort, casdm1, verbose)
        return self.mo_coeff, self.ci, occ

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None,
                 verbose=None):
        return get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    canonicalize = canonicalize
    @lib.with_doc(canonicalize.__doc__)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None):
        self.mo_coeff, ci, self.mo_energy = \
                canonicalize(self, mo_coeff, ci, eris,
                             sort, cas_natorb, casdm1, verbose)
        if cas_natorb:  # When active space is changed, the ci solution needs to be updated
            self.ci = ci
        return self.mo_coeff, ci, self.mo_energy

    @lib.with_doc(analyze.__doc__)
    def analyze(self, mo_coeff=None, ci=None, verbose=None):
        return analyze(self, mo_coeff, ci, verbose)

    def sort_mo(self, caslst, mo_coeff=None, base=1):
        '''Select active space.  See also :func:`pyscf.mcscf.addons.sort_mo`
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo(self, mo_coeff, caslst, base)

    @lib.with_doc(addons.state_average_.__doc__)
    def state_average_(self, weights=(0.5,0.5)):
        addons.state_average(self, weights)
        return self

    @lib.with_doc(addons.state_specific_.__doc__)
    def state_specific_(self, state=1):
        addons.state_specific(self, state)
        return self

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None):
        '''One-particle density matrices for alpha and beta spin on AO basis
        '''
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
        '''One-particle density matrix in AO representation
        '''
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

    def fix_spin_(self, shift=.2, ss=None):
        r'''Use level shift to control FCI solver spin.

        .. math::

            (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

        Kwargs:
            shift : float
                Level shift for states which have different spin
            ss : number
                S^2 expection value == s*(s+1)
        '''
        fci.addons.fix_spin_(self.fcisolver, shift, ss)
        return self
    fix_spin = fix_spin_

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.density_fit(self, auxbasis, with_df)

    def approx_hessian(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.approx_hessian(self, auxbasis, with_df)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
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
    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = fci.solver(mol)
    mc.natorb = 1
    emc = mc.kernel()[0]
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

#    mc = CASCI(m, 4, (3,1))
#    #mc.fcisolver = fci.direct_spin1
#    mc.fcisolver = fci.solver(mol, False)
#    emc = mc.casci()[0]
#    print(emc - -75.439016172976)
#
#    mol = gto.Mole()
#    mol.verbose = 0
#    mol.output = "out_casci"
#    mol.atom = [
#        ["C", (-0.65830719,  0.61123287, -0.00800148)],
#        ["C", ( 0.73685281,  0.61123287, -0.00800148)],
#        ["C", ( 1.43439081,  1.81898387, -0.00800148)],
#        ["C", ( 0.73673681,  3.02749287, -0.00920048)],
#        ["C", (-0.65808819,  3.02741487, -0.00967948)],
#        ["C", (-1.35568919,  1.81920887, -0.00868348)],
#        ["H", (-1.20806619, -0.34108413, -0.00755148)],
#        ["H", ( 1.28636081, -0.34128013, -0.00668648)],
#        ["H", ( 2.53407081,  1.81906387, -0.00736748)],
#        ["H", ( 1.28693681,  3.97963587, -0.00925948)],
#        ["H", (-1.20821019,  3.97969587, -0.01063248)],
#        ["H", (-2.45529319,  1.81939187, -0.00886348)],]
#
#    mol.basis = {'H': 'sto-3g',
#                 'C': 'sto-3g',}
#    mol.build()
#
#    m = scf.RHF(mol)
#    ehf = m.scf()
#    mc = CASCI(m, 9, 8)
#    mc.fcisolver = fci.solver(mol)
#    emc = mc.casci()[0]
#    print(ehf, emc, emc-ehf)
#    print(emc - -227.948912536)
#
#    mc = CASCI(m, 9, (5,3))
#    #mc.fcisolver = fci.direct_spin1
#    mc.fcisolver = fci.solver(mol, False)
#    mc.fcisolver.nroots = 3
#    emc = mc.casci()[0]
#    print(emc[0] - -227.7674519720)
