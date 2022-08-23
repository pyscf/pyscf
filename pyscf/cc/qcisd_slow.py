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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
Restricted QCISD implementation
The 4-index integrals are saved on disk entirely (without using any symmetry).

Note MO integrals are treated in chemist's notation

Ref:
'''


import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rccsd_slow as rccsd
from pyscf.cc import rintermediates as imd
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)


def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    '''Same as ccsd.kernel with strings modified to correct the method name'''
    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(QCISD) = %.15g', eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = numpy.linalg.norm(tmpvec)
        tmpvec = None
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E_corr(QCISD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('QCISD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('QCISD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(0*t1,t2,eris)
    Fvv = imd.cc_Fvv(0*t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new = np.asarray(fov).conj().copy()
    t1new +=   lib.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -lib.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*lib.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -lib.einsum('kc,ikca->ia', Fov, t2)
    t1new += 2*lib.einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -lib.einsum('kiac,kc->ia', eris.oovv, t1)
    eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new +=-2*lib.einsum('kilc,klac->ia', eris.ooov, t2)
    t1new +=   lib.einsum('likc,klac->ia', eris.ooov, t2)

    # T2 equation
    t2new = np.asarray(eris.ovov).conj().transpose(0,2,1,3).copy()
    Loo = imd.Loo(0*t1, t2, eris)
    Lvv = imd.Lvv(0*t1, t2, eris)
    Loo -= np.diag(np.diag(foo))
    Lvv -= np.diag(np.diag(fvv))
    Woooo = imd.cc_Woooo(0*t1, t2, eris)
    Wvoov = imd.cc_Wvoov(0*t1, t2, eris)
    Wvovo = imd.cc_Wvovo(0*t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(0*t1, t2, eris)
    t2new += lib.einsum('klij,klab->ijab', Woooo, t2)
    t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, t2)
    tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2 = np.asarray(eris.ovvv).conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2 = np.asarray(eris.ooov).transpose(3,1,2,0).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


class QCISD(rccsd.RCCSD):
    '''restricted QCISD
    '''

    def kernel(self, t1=None, t2=None, eris=None):
        return self.qcisd(t1, t2, eris)
    def qcisd(self, t1=None, t2=None, eris=None):
        assert (self.mo_coeff is not None)
        assert (self.mo_occ is not None)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def energy(self, t1=None, t2=None, eris=None):
        return rccsd.energy(self, t1*0, t2, eris)

    update_amps = update_amps

    def qcisd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import qcisd_t_slow as qcisd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return qcisd_t.kernel(self, eris, t1, t2, self.verbose)

    def density_fit(self, auxbasis=None, with_df=None):
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = """C  0.000  0.000  0.000
                  H  0.637  0.637  0.637
                  H -0.637 -0.637  0.637
                  H -0.637  0.637 -0.637
                  H  0.637 -0.637 -0.637"""
    mol.basis = 'cc-pvdz'
    mol.verbose = 7
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = QCISD(mf, frozen=1)
    ecc, t1, t2 = mycc.kernel()
    print(mycc.e_tot - -40.383989)
    et = mycc.qcisd_t()
    print(mycc.e_tot+et - -40.387679)
