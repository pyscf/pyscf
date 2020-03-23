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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import numpy as np

from pyscf import lib
from pyscf.cc.eom_rccsd import (EOMIP, EOMEA, amplitudes_to_vector_ip,
                                vector_to_amplitudes_ip,
                                amplitudes_to_vector_ea,
                                vector_to_amplitudes_ea)

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ip(vector, nmo, nocc)

    occ_act = np.arange(nocc)
    vir_act = np.arange(eom.nvir_act)
    ija_act = np.ix_(occ_act,occ_act,vir_act)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-2h1p block
    # first, do a diagonal (MP) update
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    FVV_DIAG = False
    if FVV_DIAG:
        Hr2 = lib.einsum('bb,ijb->ijb', fvv, r2)
    else:
        Hr2 = lib.einsum('bd,ijd->ijb', fvv, r2)
    Hr2 += -lib.einsum('ki,kjb->ijb', foo, r2)
    Hr2 += -lib.einsum('lj,ilb->ijb', foo, r2)

    # then zero out the internal amplitudes, to start over
    Hr2[ija_act] = 0.
    # now do a full update of internal amplitudes
    SYM = True
    if SYM:
        r2_in = np.zeros_like(r2)
        r2_in[ija_act] = np.copy(r2)[ija_act]
    else:
        r2_in = np.copy(r2)
    Hr2[ija_act] += lib.einsum('bd,ijd->ijb', imds.Lvv, r2_in)[ija_act]
    Hr2[ija_act] += -lib.einsum('ki,kjb->ijb', imds.Loo, r2_in)[ija_act]
    Hr2[ija_act] += -lib.einsum('lj,ilb->ijb', imds.Loo, r2_in)[ija_act]
    Hr2[ija_act] +=  lib.einsum('klij,klb->ijb', imds.Woooo, r2_in)[ija_act]
    Hr2[ija_act] += 2*lib.einsum('lbdj,ild->ijb', imds.Wovvo, r2_in)[ija_act]
    Hr2[ija_act] +=  -lib.einsum('kbdj,kid->ijb', imds.Wovvo, r2_in)[ija_act]
    Hr2[ija_act] +=  -lib.einsum('lbjd,ild->ijb', imds.Wovov, r2_in)[ija_act]  #typo in Ref
    Hr2[ija_act] +=  -lib.einsum('kbid,kjd->ijb', imds.Wovov, r2_in)[ija_act]
    tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2_in)
    tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2_in)
    Hr2[ija_act] += -np.einsum('c,ijcb->ijb', tmp, imds.t2)[ija_act]
    # 2h1p-1h block
    Hr2 += -np.einsum('kbij,k->ijb', imds.Wovoo, r1)

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector


def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype)
    for i in range(nocc):
        for j in range(nocc):
            for b in range(nvir):
                if eom.partition == 'mp':
                    Hr2[i,j,b] += fvv[b,b]
                    Hr2[i,j,b] += -foo[i,i]
                    Hr2[i,j,b] += -foo[j,j]
                else:
                    Hr2[i,j,b] += imds.Lvv[b,b]
                    Hr2[i,j,b] += -imds.Loo[i,i]
                    Hr2[i,j,b] += -imds.Loo[j,j]
                    Hr2[i,j,b] +=  imds.Woooo[i,j,i,j]
                    Hr2[i,j,b] +=2*imds.Wovvo[j,b,b,j]
                    Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                    Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                    Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                    Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:], t2[i,j,:,b])
                    Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:], t2[i,j,:,b])

    vector = amplitudes_to_vector_ip(Hr1, Hr2)
    return vector


class HybridEOMIP(EOMIP):

    def __init__(self, cc, nvir_act=0):
        EOMIP.__init__(self, cc)
        self.nvir_act = nvir_act

    matvec = ipccsd_matvec
    get_diag = ipccsd_diag


########################################
# EOM-EA-CCSD
########################################


def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1995) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = vector_to_amplitudes_ea(vector, nmo, nocc)

    occ_act = np.arange(nocc)
    vir_act = np.arange(eom.nvir_act)
    iab_act = np.ix_(occ_act,vir_act,vir_act)
    ib_act = np.ix_(occ_act,vir_act)

    # Eq. (37)
    # 1p-1p block
    Hr1 =  np.einsum('ac,c->a', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('ld,lad->a', 2.*imds.Fov, r2)
    Hr1 += np.einsum('ld,lda->a',   -imds.Fov, r2)
    Hr1 += np.einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
    # 2p1h-2p1h block
    # first, do a diagonal (MP) update
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    FVV_DIAG = False
    if FVV_DIAG:
        Hr2 =  lib.einsum('aa,jab->jab', fvv, r2)
        Hr2 +=  lib.einsum('bb,jab->jab', fvv, r2)
    else:
        Hr2 =  lib.einsum('ac,jcb->jab', fvv, r2)
        Hr2 +=  lib.einsum('bd,jad->jab', fvv, r2)
    Hr2 += -lib.einsum('lj,lab->jab', foo, r2)
    # then zero out the internal amplitudes, to start over
    Hr2[iab_act] = 0.
    # now do a full update of internal amplitudes
    SYM = True
    if SYM:
        r2_in = np.zeros_like(r2)
        r2_in[iab_act] = np.copy(r2)[iab_act]
    else:
        r2_in = np.copy(r2)
    Hr2[iab_act] +=  lib.einsum('ac,jcb->jab', imds.Lvv, r2_in)[iab_act]
    Hr2[iab_act] +=  lib.einsum('bd,jad->jab', imds.Lvv, r2_in)[iab_act]
    Hr2[iab_act] += -lib.einsum('lj,lab->jab', imds.Loo, r2_in)[iab_act]
    Hr2[iab_act] += lib.einsum('lbdj,lad->jab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2_in)[iab_act]
    Hr2[iab_act] += -lib.einsum('lajc,lcb->jab', imds.Wovov, r2_in)[iab_act]
    Hr2[iab_act] += -lib.einsum('lbcj,lca->jab', imds.Wovvo, r2_in)[iab_act]
    for a in range(eom.nvir_act):
        Hr2[:,a,:][ib_act] += lib.einsum('bcd,jcd->jb', imds.Wvvvv[a], r2_in)[ib_act]
    tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2_in)
    Hr2[iab_act] += -np.einsum('k,kjab->jab', tmp, imds.t2)[iab_act]
    # Eq. (38)
    # 2p1h-1p block
    Hr2 += np.einsum('abcj,c->jab', imds.Wvvvo, r1)

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector


def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape

    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = np.diag(imds.Lvv)
    Hr2 = np.zeros((nocc,nvir,nvir), dtype)
    for a in range(nvir):
        if eom.partition != 'mp':
            _Wvvvva = np.array(imds.Wvvvv[a])
        for b in range(nvir):
            for j in range(nocc):
                if eom.partition == 'mp':
                    Hr2[j,a,b] += fvv[a,a]
                    Hr2[j,a,b] += fvv[b,b]
                    Hr2[j,a,b] += -foo[j,j]
                else:
                    Hr2[j,a,b] += imds.Lvv[a,a]
                    Hr2[j,a,b] += imds.Lvv[b,b]
                    Hr2[j,a,b] += -imds.Loo[j,j]
                    Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                    Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                    Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                    Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                    Hr2[j,a,b] += _Wvvvva[b,a,b]
                    Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b])
                    Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b])

    vector = amplitudes_to_vector_ea(Hr1,Hr2)
    return vector


class HybridEOMEA(EOMEA):

    def __init__(self, cc, nvir_act=0):
        EOMEA.__init__(self, cc)
        self.nvir_act = nvir_act

    matvec = eaccsd_matvec
    get_diag = eaccsd_diag


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.cc import rccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = rccsd.RCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    myeom = HybridEOMIP(mycc, nvir_act=4)
    print("IP energies... (right eigenvector)")
    e,v = myeom.kernel(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    myeom = HybridEOMEA(mycc)
    print("EA energies... (right eigenvector)")
    e,v = myeom.kernel(nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

