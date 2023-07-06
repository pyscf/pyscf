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

'''
UCASCI (CASCI with non-degenerated alpha and beta orbitals, typically UHF
orbitals)
'''

import sys

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf import gto
from pyscf import scf
from pyscf import fci
from pyscf.mcscf import addons
from pyscf.mcscf import casci
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'mcscf_analyze_with_meta_lowdin', True)
LARGE_CI_TOL = getattr(__config__, 'mcscf_analyze_large_ci_tol', 0.1)

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range

# TODO, different ncas space for alpha and beta

def extract_orbs(mo_coeff, ncas, nelecas, ncore):
    ncore_a, ncore_b = ncore
    nocc_a = ncore_a + ncas
    mo_core = (mo_coeff[0][:,:ncore_a]      , mo_coeff[1][:,:ncore_a]      )
    mo_cas  = (mo_coeff[0][:,ncore_a:nocc_a], mo_coeff[1][:,ncore_a:nocc_a])
    mo_vir  = (mo_coeff[0][:,nocc_a:]       , mo_coeff[1][:,nocc_a:]       )
    return mo_core, mo_cas, mo_vir

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''CAS sapce one-electron hamiltonian for UHF-CASCI or UHF-CASSCF

    Args:
        casci : a U-CASSCF/U-CASCI object or UHF object

    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core =(mo_coeff[0][:,:ncore[0]], mo_coeff[1][:,:ncore[1]])
    mo_cas = (mo_coeff[0][:,ncore[0]:ncore[0]+ncas],
              mo_coeff[1][:,ncore[1]:ncore[1]+ncas])

    hcore = casci.get_hcore()
    energy_core = casci.energy_nuc()
    if mo_core[0].size == 0 and mo_core[1].size == 0:
        corevhf = (0,0)
    else:
        core_dm = (numpy.dot(mo_core[0], mo_core[0].T),
                   numpy.dot(mo_core[1], mo_core[1].T))
        corevhf = casci.get_veff(casci.mol, core_dm)
        energy_core += numpy.einsum('ij,ji', core_dm[0], hcore[0])
        energy_core += numpy.einsum('ij,ji', core_dm[1], hcore[1])
        energy_core += numpy.einsum('ij,ji', core_dm[0], corevhf[0]) * .5
        energy_core += numpy.einsum('ij,ji', core_dm[1], corevhf[1]) * .5
    h1eff = (reduce(numpy.dot, (mo_cas[0].T, hcore[0]+corevhf[0], mo_cas[0])),
             reduce(numpy.dot, (mo_cas[1].T, hcore[1]+corevhf[1], mo_cas[1])))
    return h1eff, energy_core

def kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''UHF-CASCI solver
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    log = logger.new_logger(casci, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start uhf-based CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas
    ncore = casci.ncore
    mo_core, mo_cas, mo_vir = extract_orbs(mo_coeff, ncas, nelecas, ncore)

    # 1e
    h1eff, energy_core = casci.h1e_for_cas(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t0)

    # 2e
    eri_cas = casci.get_h2eff(mo_cas)
    t1 = log.timer('integral transformation to CAS space', *t1)

    # FCI
    max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec


class UCASCI(casci.CASCI):
    # nelecas is tuple of (nelecas_alpha, nelecas_beta)
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        #assert ('UHF' == mf.__class__.__name__)
        if isinstance(mf_or_mol, gto.Mole):
            mf = scf.UHF(mf_or_mol)
        else:
            mf = mf_or_mol

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
            self.nelecas = (nelecas[0], nelecas[1])
        self.ncore = ncore

        self.fcisolver = fci.direct_uhf.FCISolver(mol)
        self.fcisolver.lindep = getattr(__config__,
                                        'mcscf_ucasci_UCASCI_fcisolver_lindep', 1e-10)
        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_ucasci_UCASCI_fcisolver_max_cycle', 200)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_ucasci_UCASCI_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mf.mo_coeff
        self.ci = None
        self.e_tot = 0
        self.e_cas = 0

        self._keys = set(self.__dict__.keys())

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec >= 0
            if ncorelec % 2:
                ncore = ((ncorelec+1)//2, (ncorelec-1)//2)
            else:
                ncore = (ncorelec//2, ncorelec//2)
            return ncore
        else:
            return self._ncore
    @ncore.setter
    def ncore(self, x):
        if x is None:
            self._ncore = x
        elif isinstance(x, (int, numpy.integer)):
            assert x >= 0
            self._ncore = (x, x)
        else:
            assert x[0] >= 0 and x[1] >= 0
            self._ncore = (x[0], x[1])

    def dump_flags(self, verbose=None):
        log = lib.logger.new_logger(self, verbose)
        log.info('')
        log.info('******** UHF-CASCI flags ********')
        nmo = self.mo_coeff[0].shape[1]
        nvir_alpha = nmo - self.ncore[0] - self.ncas
        nvir_beta  = nmo - self.ncore[1]  - self.ncas
        log.info('CAS ((%de+%de), %do), ncore = [%d+%d], nvir = [%d+%d]',
                 self.nelecas[0], self.nelecas[1], self.ncas,
                 self.ncore[0], self.ncore[1], nvir_alpha, nvir_beta)
        log.info('max_memory %d (MB)', self.max_memory)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
        return self

    def check_sanity(self):
        assert self.ncas > 0
        ncore = self.ncore
        nmo = self.mo_coeff[0].shape[1]
        nvir_alpha = nmo - ncore[0] - self.ncas
        nvir_beta  = nmo - ncore[1]  - self.ncas
        assert ncore[0] >= 0 and ncore[1] >= 0
        assert nvir_alpha >= 0 and nvir_beta >= 0
        assert sum(ncore) + sum(self.nelecas) == self.mol.nelectron
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self

    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        return (hcore,hcore)

    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = (self.mo_coeff[0][:,:self.ncore[0]],
                      self.mo_coeff[1][:,:self.ncore[1]])
            dm = (numpy.dot(mocore[0], mocore[0].T),
                  numpy.dot(mocore[1], mocore[1].T))
# don't call self._scf.get_veff because _scf might be DFT object
        vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        return vj[0]+vj[1] - vk

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_h2cas(self, mo_coeff=None):
        return self.ao2mo(mo_coeff)
    def get_h2eff(self, mo_coeff=None):
        return self.ao2mo(mo_coeff)
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.ncore[0]:self.ncore[0]+self.ncas],
                        self.mo_coeff[1][:,self.ncore[1]:self.ncore[1]+self.ncas])
        nao, nmo = mo_coeff[0].shape
        if self._scf._eri is not None and \
           (nao*nao*nmo*nmo*12+self._scf._eri.size)*8/1e6 < self.max_memory*.95:
            moab = numpy.hstack((mo_coeff[0], mo_coeff[1]))
            na = mo_coeff[0].shape[1]
            nab = moab.shape[1]
            eri = pyscf.ao2mo.incore.full(self._scf._eri, moab)
            eri = pyscf.ao2mo.restore(1, eri, nab)
            eri_aa = eri[:na,:na,:na,:na].copy()
            eri_ab = eri[:na,:na,na:,na:].copy()
            eri_bb = eri[na:,na:,na:,na:].copy()
        else:
            moab = numpy.hstack((mo_coeff[0], mo_coeff[1]))
            eri = pyscf.ao2mo.full(self.mol, moab, verbose=self.verbose)
            na = mo_coeff[0].shape[1]
            nab = moab.shape[1]
            eri = pyscf.ao2mo.restore(1, eri, nab)
            eri_aa = eri[:na,:na,:na,:na].copy()
            eri_ab = eri[:na,:na,na:,na:].copy()
            eri_bb = eri[na:,na:,na:,na:].copy()

        return (eri_aa, eri_ab, eri_bb)

    get_h1cas = h1e_for_cas = h1e_for_cas

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    get_h1eff.__doc__ = h1e_for_cas.__doc__

    def casci(self, mo_coeff=None, ci0=None):
        return self.kernel(mo_coeff, ci0)
    def kernel(self, mo_coeff=None, ci0=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci0 is None:
            ci0 = self.ci

        self.check_sanity()
        self.dump_flags()

        log = logger.Logger(self.stdout, self.verbose)
        self.e_tot, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=log)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            ncore = self.ncore
            ncas = self.ncas
            mocas = (self.mo_coeff[0][:,ncore[0]:ncore[0]+ncas],
                     self.mo_coeff[1][:,ncore[1]:ncore[1]+ncas])
            mocore = (self.mo_coeff[0][:,:ncore[0]],
                      self.mo_coeff[1][:,:ncore[1]])
            ovlp_ao = self._scf.get_ovlp()
            ss_core = self._scf.spin_square(mocore, ovlp_ao)
            if isinstance(self.e_cas, (float, numpy.number)):
                ss = fci.spin_op.spin_square(self.ci, self.ncas, self.nelecas,
                                             mocas, ovlp_ao, self.fcisolver)
                log.note('UCASCI E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                         self.e_tot, self.e_cas, ss[0]+ss_core[0])
            else:
                for i, e in enumerate(self.e_cas):
                    ss = fci.spin_op.spin_square(self.ci[i], self.ncas, self.nelecas,
                                                 mocas, ovlp_ao, self.fcisolver)
                    log.note('UCASCI root %d  E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                             i, self.e_tot[i], e, ss[0]+ss_core[0])
        else:
            if isinstance(self.e_cas, (float, numpy.number)):
                log.note('UCASCI E = %.15g  E(CI) = %.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('UCASCI root %d  E = %.15g  E(CI) = %.15g',
                             i, self.e_tot[i], e)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        return self

    def cas_natorb(self, mo_coeff=None, ci0=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci0 is None: ci0 = self.ci
        return addons.cas_natorb(self, mo_coeff, ci0)
    def cas_natorb_(self, mo_coeff=None, ci0=None):
        self.ci, self.mo_coeff, occ = self.cas_natorb(mo_coeff, ci0)
        return self.ci, self.mo_coeff

    def analyze(self, mo_coeff=None, ci=None, verbose=None,
                large_ci_tol=LARGE_CI_TOL, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        from pyscf.lo import orth
        from pyscf.tools import dump_mat
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        log = logger.new_logger(self, verbose)
        nelecas = self.nelecas
        ncas = self.ncas
        ncore = self.ncore
        mocore_a = mo_coeff[0][:,:ncore[0]]
        mocas_a = mo_coeff[0][:,ncore[0]:ncore[0]+ncas]
        mocore_b = mo_coeff[1][:,:ncore[1]]
        mocas_b = mo_coeff[1][:,ncore[1]:ncore[1]+ncas]

        label = self.mol.ao_labels()
        if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
            not isinstance(self.fcisolver, addons.StateAverageFCISolver)):
            log.warn('Mulitple states found in UCASCI solver. Density '
                     'matrix of first state is generated in .analyze() function.')
            civec = ci[0]
        else:
            civec = ci
        casdm1a, casdm1b = self.fcisolver.make_rdm1s(civec, ncas, nelecas)
        dm1a = numpy.dot(mocore_a, mocore_a.T)
        dm1a += reduce(numpy.dot, (mocas_a, casdm1a, mocas_a.T))
        dm1b = numpy.dot(mocore_b, mocore_b.T)
        dm1b += reduce(numpy.dot, (mocas_b, casdm1b, mocas_b.T))
        if log.verbose >= logger.DEBUG2:
            log.debug2('alpha density matrix (on AO)')
            dump_mat.dump_tri(self.stdout, dm1a, label)
            log.debug2('beta density matrix (on AO)')
            dump_mat.dump_tri(self.stdout, dm1b, label)

        if log.verbose >= logger.INFO:
            ovlp_ao = self._scf.get_ovlp()
            occa, ucasa = self._eig(-casdm1a, ncore[0], ncore[0]+ncas)
            occb, ucasb = self._eig(-casdm1b, ncore[1], ncore[1]+ncas)
            log.info('Natural alpha-occupancy %s', str(-occa))
            log.info('Natural beta-occupancy %s', str(-occb))
            mocas_a = numpy.dot(mocas_a, ucasa)
            mocas_b = numpy.dot(mocas_b, ucasb)
            if with_meta_lowdin:
                orth_coeff = orth.orth_ao(self.mol, 'meta_lowdin', s=ovlp_ao)
                mocas_a = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mocas_a))
                mocas_b = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mocas_b))
                log.info('Natural alpha-orbital (expansion on meta-Lowdin AOs) in CAS space')
                dump_mat.dump_rec(log.stdout, mocas_a, label, start=1, **kwargs)
                log.info('Natural beta-orbital (expansion on meta-Lowdin AOs) in CAS space')
                dump_mat.dump_rec(log.stdout, mocas_b, label, start=1, **kwargs)
            else:
                log.info('Natural alpha-orbital (expansion on AOs) in CAS space')
                dump_mat.dump_rec(log.stdout, mocas_a, label, start=1, **kwargs)
                log.info('Natural beta-orbital (expansion on AOs) in CAS space')
                dump_mat.dump_rec(log.stdout, mocas_b, label, start=1, **kwargs)

            if self._scf.mo_coeff is not None:
                tol = getattr(__config__, 'mcscf_addons_map2hf_tol', 0.4)
                s = reduce(numpy.dot, (mo_coeff[0].T, ovlp_ao, self._scf.mo_coeff[0]))
                idx = numpy.argwhere(abs(s)>tol)
                for i,j in idx:
                    log.info('alpha <mo-mcscf|mo-hf> %d  %d  %12.8f' % (i+1,j+1,s[i,j]))
                s = reduce(numpy.dot, (mo_coeff[1].T, ovlp_ao, self._scf.mo_coeff[1]))
                idx = numpy.argwhere(abs(s)>tol)
                for i,j in idx:
                    log.info('beta <mo-mcscf|mo-hf> %d  %d  %12.8f' % (i+1,j+1,s[i,j]))

            if getattr(self.fcisolver, 'large_ci', None) and ci is not None:
                log.info('\n** Largest CI components **')
                if isinstance(ci, (list, tuple, RANGE_TYPE)):
                    for i, state in enumerate(ci):
                        log.info('  [alpha occ-orbitals] [beta occ-orbitals]  state %-3d CI coefficient', i)
                        res = self.fcisolver.large_ci(state, self.ncas, self.nelecas,
                                                      large_ci_tol, return_strs=False)
                        for c,ia,ib in res:
                            log.info('  %-20s %-30s %.12f', ia, ib, c)
                else:
                    log.info('  [alpha occ-orbitals] [beta occ-orbitals]            CI coefficient')
                    res = self.fcisolver.large_ci(ci, self.ncas, self.nelecas,
                                                  large_ci_tol, return_strs=False)
                    for c,ia,ib in res:
                        log.info('  %-20s %-30s %.12f', ia, ib, c)
        return dm1a, dm1b

    def spin_square(self, fcivec=None, mo_coeff=None, ovlp=None):
        return addons.spin_square(self, mo_coeff, fcivec, ovlp)

    fix_spin_ = fix_spin = lib.invalid_method('fix_spin')

    @lib.with_doc(addons.sort_mo.__doc__)
    def sort_mo(self, caslst, mo_coeff=None, base=1):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo(self, mo_coeff, caslst, base)

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1a, casdm1b = self.fcisolver.make_rdm1s(ci, ncas, nelecas)

        mocore = mo_coeff[0][:,:ncore[0]]
        mocas = mo_coeff[0][:,ncore[0]:ncore[0]+ncas]
        dm1a = numpy.dot(mocore, mocore.T)
        dm1a += reduce(numpy.dot, (mocas, casdm1a, mocas.T))

        mocore = mo_coeff[1][:,:ncore[1]]
        mocas = mo_coeff[1][:,ncore[1]:ncore[1]+ncas]
        dm1b = numpy.dot(mocore, mocore.T)
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.T))
        return dm1a, dm1b

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        dm1a,dm1b = self.make_rdm1s(mo_coeff, ci, ncas, nelecas, ncore)
        return dm1a+dm1b

CASCI = UCASCI

del (WITH_META_LOWDIN, LARGE_CI_TOL)


if __name__ == '__main__':
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
    mc = CASCI(m, 4, (2,2))
    emc = mc.casci()[0]
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
    mc = CASCI(m, 9, (4,4))
    mc.fcisolver.nroots = 2
    emc = mc.casci()[0]
    mc.analyze()
    print(ehf, emc, emc[0]-ehf)
    print(emc[0] - -227.948912536)
