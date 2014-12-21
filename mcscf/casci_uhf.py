#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import time
import numpy
import h5py
import pyscf.lib
import pyscf.ao2mo
import pyscf.fci

# TODO, different ncas space for alpha and beta

def extract_orbs(mo_coeff, ncas, nelecas, ncore):
    ncore_a, ncore_b = ncore
    nocc_a = ncore_a + ncas
    nocc_b = ncore_b + ncas
    mo_core = (mo_coeff[0][:,:ncore_a]      , mo_coeff[1][:,:ncore_a]      )
    mo_cas  = (mo_coeff[0][:,ncore_a:nocc_a], mo_coeff[1][:,ncore_a:nocc_a])
    mo_vir  = (mo_coeff[0][:,nocc_a:]       , mo_coeff[1][:,nocc_a:]       )
    return mo_core, mo_cas, mo_vir

def kernel(casci, mo_coeff, ci0=None, verbose=None):
    if verbose is None:
        verbose = casci.verbose
    log = pyscf.lib.logger.Logger(casci.stdout, verbose)
    t0 = (time.clock(), time.time())
    log.debug('Start uhf-based CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas
    ncore = casci.ncore
    mo_core, mo_cas, mo_vir = extract_orbs(mo_coeff, ncas, nelecas, ncore)

    # 1e
    hcore = casci.get_hcore()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
        energy_core = 0
    else:
        core_dm = (numpy.dot(mo_core[0], mo_core[0].T),
                   numpy.dot(mo_core[1], mo_core[1].T))
        corevhf = casci.get_veff(core_dm)
        energy_core = numpy.einsum('ij,ji', core_dm[0], hcore[0]) \
                    + numpy.einsum('ij,ji', core_dm[1], hcore[1]) \
                    + numpy.einsum('ij,ji', core_dm[0], corevhf[0]) * .5 \
                    + numpy.einsum('ij,ji', core_dm[1], corevhf[1]) * .5
    h1eff = (reduce(numpy.dot, (mo_cas[0].T, hcore[0]+corevhf[0], mo_cas[0])),
             reduce(numpy.dot, (mo_cas[1].T, hcore[1]+corevhf[1], mo_cas[1])))
    t1 = log.timer('effective h1e in CAS space', *t0)

    # 2e
    eri_cas = casci.ao2mo(mo_cas)
    t1 = log.timer('integral transformation to CAS space', *t1)

    # FCI
    e_cas, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas, ci0=ci0)

    t1 = log.timer('FCI solver', *t1)
    e_tot = e_cas + energy_core
    log.info('CASCI E = %.15g', e_tot)
    log.timer('CASCI', *t0)
    return e_tot, e_cas, fcivec


class CASCI(object):
    # nelecas is tuple of (nelecas_alpha, nelecas_beta)
    def __init__(self, mol, mf, ncas, nelecas, ncore=None):
#TODO:        assert('RHF' not in str(mf.__class__))
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, int):
            nelecb = (nelecas-mol.spin)/2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0], nelecas[1])
        if ncore is None:
            ncorelec = mol.nelectron - (self.nelecas[0]+self.nelecas[1])
            if ncorelec % 2:
                self.ncore = ((ncorelec+1)/2, (ncorelec-1)/2)
            else:
                self.ncore = (ncorelec/2, ncorelec/2)
        else:
            self.ncore = (ncore[0], ncore[1])

        self.fcisolver = pyscf.fci.direct_uhf.FCISolver(mol)
        self.fcisolver.lindep = 1e-10
        self.fcisolver.max_cycle = 30
        self.fcisolver.conv_threshold = 1e-8

        self.mo_coeff = mf.mo_coeff
        self.ci = None
        self.e_tot = 0

        self._keys = set(self.__dict__.keys() + ['_keys'])

    def dump_flags(self):
        log = pyscf.lib.logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** UHF-CASCI flags ********')
        nmo = self.mo_coeff[0].shape[1]
        nvir_alpha = nmo - self.ncore[0] - self.ncas
        nvir_beta  = nmo - self.ncore[1]  - self.ncas
        log.info('CAS ((%de+%de), %do), ncore = [%d+%d], nvir = [%d+%d]', \
                 self.nelecas[0], self.nelecas[1], self.ncas,
                 self.ncore[0], self.ncore[1], nvir_alpha, nvir_beta)
        log.info('max_memory %d (MB)', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except:
            pass

    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        return (hcore,hcore)

    def get_veff(self, dm):
        return self._scf.get_veff(self.mol, dm)

    def ao2mo(self, mo):
        nao, nmo = mo[0].shape
        if self._scf._eri is not None:
            #and nao*nao*nmo*nmo*3/4*8/1e6 > self.max_memory
            eri_aa = pyscf.ao2mo.incore.full(self._scf._eri, mo[0])
            eri_ab = pyscf.ao2mo.incore.general(self._scf._eri,
                                                (mo[0],mo[0],mo[1],mo[1]))
            eri_bb = pyscf.ao2mo.incore.full(self._scf._eri, mo[1])
        else:
            ftmp = tempfile.NamedTemporaryFile()
            pyscf.ao2mo.outcore.full(self.mol, mo[0], ftmp.name,
                                     verbose=self.verbose)
            with h5py.File(ftmp.name, 'r') as feri:
                eri_aa = numpy.array(feri['eri_mo'])
            pyscf.ao2mo.outcore.general(self.mol, (mo[0],mo[0],mo[1],mo[1]),
                                        ftmp.name, verbose=self.verbose)
            with h5py.File(ftmp.name, 'r') as feri:
                eri_ab = numpy.array(feri['eri_mo'])
            pyscf.ao2mo.outcore.full(self.mol, mo[1], ftmp.name,
                                     verbose=self.verbose)
            with h5py.File(ftmp.name, 'r') as feri:
                eri_bb = numpy.array(feri['eri_mo'])
        return (eri_aa, eri_ab, eri_bb)

    def casci(self, mo=None, ci0=None):
        if mo is None:
            mo = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        self.mol.check_sanity(self)

        self.dump_flags()

        self.e_tot, e_cas, self.ci = \
                kernel(self, mo, ci0=ci0, verbose=self.verbose)
        return self.e_tot, e_cas, self.ci



if __name__ == '__main__':
    import gto
    import scf
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

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = CASCI(mol, m, 4, (2,2))
    emc = mc.casci()[0] + mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    #-75.9577817425 -75.9624554777 -0.00467373522233
    print(emc+75.9624554777)

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

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = CASCI(mol, m, 9, (4,4))
    emc = mc.casci()[0] + mol.nuclear_repulsion()
    print(ehf, emc, emc-ehf)
    print(emc - -227.948912536)
