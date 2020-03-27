#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
Restricted hybrid CCSD implementation which supports both real and complex integrals.

Note MO integrals are treated in chemist's notation
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf import __config__

from pyscf.cc import rccsd

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc,nocc:].copy()
    #foo = fock[:nocc,:nocc].copy()
    #fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    #occ_act = np.arange(cc.frozen_occ,nocc)
    occ_act = np.arange(nocc)
    vir_act = np.arange(cc.nvir_act)
    #print("occ_act =", occ_act)
    #print("vir_act =", vir_act)
    ia_act = np.ix_(occ_act,vir_act)
    ijab_act = np.ix_(occ_act,occ_act,vir_act,vir_act)
    #print("ia =", ia_act)
    #print("ijab =", ijab_act)

    # T1 equation
    t1new = np.copy(fov.conj())
    t1new[ia_act] -= 2*np.einsum('kc,ka,ic->ia', fov, t1, t1)[ia_act]
    t1new[ia_act] +=   np.einsum('ac,ic->ia', Fvv, t1)[ia_act]
    t1new[ia_act] +=  -np.einsum('ki,ka->ia', Foo, t1)[ia_act]
    t1new[ia_act] += 2*np.einsum('kc,kica->ia', Fov, t2)[ia_act]
    t1new[ia_act] +=  -np.einsum('kc,ikca->ia', Fov, t2)[ia_act]
    t1new[ia_act] +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)[ia_act]
    t1new[ia_act] += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)[ia_act]
    t1new[ia_act] +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)[ia_act]
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new[ia_act] += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)[ia_act]
    t1new[ia_act] +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)[ia_act]
    t1new[ia_act] += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)[ia_act]
    t1new[ia_act] +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)[ia_act]
    eris_ovoo = np.asarray(eris.ovoo, order='C')
    t1new[ia_act] +=-2*lib.einsum('lcki,klac->ia', eris_ovoo, t2)[ia_act]
    t1new[ia_act] +=   lib.einsum('kcli,klac->ia', eris_ovoo, t2)[ia_act]
    t1new[ia_act] +=-2*lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)[ia_act]
    t1new[ia_act] +=   lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)[ia_act]

    # T2 equation
    t2new = np.copy(np.asarray(eris.ovov).conj().transpose(0,2,1,3))
    tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1,3,0,2)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new[ijab_act] += (tmp + tmp.transpose(1,0,3,2))[ijab_act]
    tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new[ijab_act] -= (tmp + tmp.transpose(1,0,3,2))[ijab_act]

    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    Lvv[np.diag_indices(nvir)] -= mo_e_v

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

    tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
    t2new[ijab_act] += lib.einsum('klij,klab->ijab', Woooo, tau)[ijab_act]
    t2new[ijab_act] += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)[ijab_act]
    tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
    t2new[ijab_act] += (tmp + tmp.transpose(1,0,3,2))[ijab_act]
    tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
    t2new[ijab_act] -= (tmp + tmp.transpose(1,0,3,2))[ijab_act]
    tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
    tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
    t2new[ijab_act] += (tmp + tmp.transpose(1,0,3,2))[ijab_act]
    tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
    t2new[ijab_act] -= (tmp + tmp.transpose(1,0,3,2))[ijab_act]
    tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
    t2new[ijab_act] -= (tmp + tmp.transpose(1,0,3,2))[ijab_act]

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1=None, t2=None, eris=None):
    '''RCCSD correlation energy'''
    if t1 is None: t1 = cc.t1
    if t2 is None: t2 = cc.t2
    if eris is None: eris = cc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in RCCSD energy %s', e)
    return e.real


class HybridRCCSD(rccsd.RCCSD):
    '''restricted hybrid CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities
    '''

    def __init__(self, mf, frozen=0, frozen_occ=0, nvir_act=0, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        #self.frozen_occ = frozen_occ
        self.nvir_act = nvir_act

    update_amps = update_amps


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 5
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = HybridRCCSD(mf, nvir_act=4)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2079029498387684)
