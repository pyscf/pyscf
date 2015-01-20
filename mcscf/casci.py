#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import time
from functools import reduce
import numpy
import h5py
import pyscf.lib
from pyscf.lib import logger
import pyscf.scf
import pyscf.ao2mo
from pyscf import fci
from pyscf.mcscf import addons


def extract_orbs(mo_coeff, ncas, nelecas, ncore):
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    mo_vir = mo_coeff[:,nocc:]
    return mo_core, mo_cas, mo_vir

def h1e_for_cas(casci, mo_core, mo_cas):
    '''CAS sapce one-electron hamiltonian

    Args:
        casci : an :class:`CASSCF` or :class:`CASCI` object

        mo_core : ndarray
            Core orbitals
        mo_cas : ndarray
            CAS orbitals

    Returns:
        Effective one-electron hamiltonian defined in CAS space, and the electronic
        energy from core.
    '''
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

def kernel(casci, mo_coeff, ci0=None, verbose=None, **cikwargs):
    '''CASCI solver
    '''
    if verbose is None:
        verbose = casci.verbose
    log = pyscf.lib.logger.Logger(casci.stdout, verbose)
    t0 = (time.clock(), time.time())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas
    ncore = casci.ncore
    mo_core, mo_cas, mo_vir = extract_orbs(mo_coeff, ncas, nelecas, ncore)

    # 1e
    h1eff, energy_core = h1e_for_cas(casci, mo_core, mo_cas)
    t1 = log.timer('effective h1e in CAS space', *t0)

    # 2e
    eri_cas = casci.ao2mo(mo_cas)
    t1 = log.timer('integral transformation to CAS space', *t1)

    # FCI
    e_cas, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, **cikwargs)

    t1 = log.timer('FCI solver', *t1)
    e_tot = e_cas + energy_core + casci.mol.energy_nuc()
    log.info('CASCI E = %.15g, E(CI) = %.15g', e_tot, e_cas)
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
            >>> mc = mcscf.CASSCF(mol, mf, 4, 4)
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
    >>> mol = gto.Mole()
    >>> mol.build(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASCI(mol, mf, 6, 6)
    >>> mc.kernel()[0]
    -108.980200816243354
    '''
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, int):
            assert(nelecas%2 == 0)
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
            assert(isinstance(ncore, int))
            self.ncore = ncore
        #self.fcisolver = fci.direct_spin0.FCISolver(mol)
        self.fcisolver = fci.solver(mol, self.nelecas[0]==self.nelecas[1])
# CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = 1e-10
        self.fcisolver.max_cycle = 50
        self.fcisolver.conv_tol = 1e-8

        self.mo_coeff = mf.mo_coeff
        self.ci = None
        self.e_tot = 0

        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = pyscf.lib.logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        log.info('max_memory %d (MB)', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except:
            pass

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    def get_veff(self, mol, dm, hermi=1):
# don't call self._scf.get_veff, because ROHF return alpha,beta potential separately
        vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        return vj - vk * .5

    def ao2mo(self, mo_coeff):
        nao, nmo = mo_coeff.shape
        if self._scf._eri is not None and \
           (nao**2*nmo**2+nmo**4*2)/4*8/1e6 > self.max_memory:
            eri = pyscf.ao2mo.incore.full(self._scf._eri, mo_coeff)
        else:
            ftmp = tempfile.NamedTemporaryFile()
            pyscf.ao2mo.outcore.full(self.mol, mo_coeff, ftmp.name,
                                     verbose=self.verbose)
            feri = h5py.File(ftmp.name, 'r')
            eri = numpy.array(feri['eri_mo'])
        return eri

    def kernel(self, *args, **kwargs):
        return self.casci(*args, **kwargs)
    def casci(self, mo_coeff=None, ci0=None, **cikwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci

        self.mol.check_sanity(self)

        self.dump_flags()

        self.e_tot, e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=self.verbose, **cikwargs)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci

    def cas_natorb(self, mo_coeff=None, ci0=None):
        return self.cas_natorb(mo_coeff, ci0)
    def cas_natorb_(self, mo_coeff=None, ci0=None):
        self.ci, self.mo_coeff, occ = addons.cas_natorb(self, ci0, mo_coeff)
        return self.ci, self.mo_coeff

    def analyze(self, mo_coeff=None, ci=None, verbose=logger.DEBUG):
        from pyscf.tools import dump_mat
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if verbose >= logger.INFO:
            log = logger.Logger(self.stdout, verbose)
            dm1a, dm1b = addons.make_rdm1s(self, ci, mo_coeff)
            label = ['%d%3s %s%-4s' % x for x in self.mol.spheric_labels()]
            log.info('alpha density matrix (on AO)')
            dump_mat.dump_tri(self.stdout, dm1a, label)
            log.info('beta density matrix (on AO)')
            dump_mat.dump_tri(self.stdout, dm1b, label)

            s = reduce(numpy.dot, (mo_coeff.T, self._scf.get_ovlp(),
                                   self._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.5)
            for i,j in idx:
                log.info('<mo-mcscf|mo-hf> %d, %d, %12.8f' % (i+1,j+1,s[i,j]))
            log.info('** Largest CI components **')
            log.info(' string alpha, string beta, CI coefficients')
            for c,ia,ib in fci.addons.large_ci(ci, self.ncas, self.nelecas):
                log.info('  %9s    %9s    %.12f', ia, ib, c)
        return dm1a, dm1b



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
    mc = CASCI(mol, m, 4, 4)
    mc.fcisolver = fci.solver(mol)
    emc = mc.casci()[0]
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

    mc = CASCI(mol, m, 4, (3,1))
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
    mc = CASCI(mol, m, 9, 8)
    mc.fcisolver = fci.solver(mol)
    emc = mc.casci()[0]
    print(ehf, emc, emc-ehf)
    print(emc - -227.948912536)

    mc = CASCI(mol, m, 9, (5,3))
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    emc = mc.casci()[0]
    print(emc - -227.7674519720)
