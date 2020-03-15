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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy as np
import pyscf.ao2mo as ao2mo

### Integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

    occ = myadc.mo_coeff[:,:myadc._nocc]
    vir = myadc.mo_coeff[:,myadc._nocc:]

    nocc = occ.shape[1]
    nvir = vir.shape[1]
    #n_oo = nocc * (nocc + 1) // 2
    #n_vv = nvir * (nvir + 1) // 2
    #ind_oo = np.tril_indices(nocc)
    #ind_vv = np.tril_indices(nvir)
   
    n_oo = nocc * (nocc - 1) // 2
    n_vv = nvir * (nvir - 1) // 2
    ind_oo = np.tril_indices(nocc,k=-1)
    ind_vv = np.tril_indices(nvir,k=-1)
    eris = lambda:None

    # TODO: check if myadc._scf._eri is not None

    eris.oooo = ao2mo.general(myadc._scf._eri, (occ, occ, occ, occ), compact=False).reshape(nocc, nocc, nocc, nocc).copy()
    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ, vir, occ, occ), compact=False).reshape(nocc, nvir, nocc, nocc).copy()
    eris.oovo = ao2mo.general(myadc._scf._eri, (occ, occ, vir, occ), compact=False).reshape(nocc, nocc, nvir, nocc).copy()
    eris.ovov = ao2mo.general(myadc._scf._eri, (occ, vir, occ, vir), compact=False).reshape(nocc, nvir, nocc, nvir).copy()
    eris.oovv = ao2mo.general(myadc._scf._eri, (occ, occ, vir, vir), compact=False).reshape(nocc, nocc, nvir, nvir).copy()
    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ, vir, vir, occ), compact=False).reshape(nocc, nvir, nvir, nocc).copy()
    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ, vir, vir, vir), compact=True).reshape(nocc, nvir, -1).copy()

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
        #eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=True)
        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
        eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3)) 
        eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)

    return eris

def unpack_eri_1(eri, norb):

    n_oo = norb * (norb + 1) // 2
    ind_oo = np.tril_indices(norb)

    eri_ = None

    if len(eri.shape) == 3:
        if (eri.shape[0] == n_oo):
            eri_ = np.zeros((norb, norb, eri.shape[1], eri.shape[2]))
            eri_[ind_oo[0], ind_oo[1]] = eri
            eri_[ind_oo[1], ind_oo[0]] = eri

        elif (eri.shape[2] == n_oo):
            eri_ = np.zeros((eri.shape[0], eri.shape[1], norb, norb))
            eri_[:, :, ind_oo[0], ind_oo[1]] = eri
            eri_[:, :, ind_oo[1], ind_oo[0]] = eri
        else:
            raise TypeError("ERI dimensions don't match")

    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_

def unpack_eri_2s(eri, norb):

    n_oo = norb * (norb + 1) // 2
    ind_oo = np.tril_indices(norb)

    eri_ = None

    if len(eri.shape) == 2:
        if (eri.shape[0] != n_oo or eri.shape[1] != n_oo):
            raise TypeError("ERI dimensions don't match")

        temp = np.zeros((n_oo, norb, norb))
        temp[:, ind_oo[0], ind_oo[1]] = eri
        temp[:, ind_oo[1], ind_oo[0]] = eri
        eri_ = np.zeros((norb, norb, norb, norb))
        eri_[ind_oo[0], ind_oo[1]] = temp
        eri_[ind_oo[1], ind_oo[0]] = temp
    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_


def unpack_eri_2sn(eri, norb):

    #n_oo = norb * (norb + 1) // 2
    #ind_oo = np.tril_indices(norb)
    n_oo = norb * (norb - 1) // 2
    ind_oo = np.tril_indices(norb,k=-1)

    eri_ = None

    if len(eri.shape) == 2:
        if (eri.shape[0] != n_oo or eri.shape[1] != n_oo):
            raise TypeError("ERI dimensions don't match")

        temp = np.zeros((n_oo, norb, norb))
        temp[:, ind_oo[0], ind_oo[1]] = eri
        temp[:, ind_oo[1], ind_oo[0]] = -eri
        eri_ = np.zeros((norb, norb, norb, norb))
        eri_[ind_oo[0], ind_oo[1]] = temp
        eri_[ind_oo[1], ind_oo[0]] = -temp
    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_

def unpack_eri_2(eri, norb1, norb2):

    #n_oo1 = norb1 * (norb1 + 1) // 2
    #ind_oo1 = np.tril_indices(norb1)
    #n_oo2 = norb2 * (norb2 + 1) // 2
    #ind_oo2 = np.tril_indices(norb2)

    n_oo1 = norb1 * (norb1 - 1) // 2
    ind_oo1 = np.tril_indices(norb1,k=-1)
    n_oo2 = norb2 * (norb2 - 1) // 2
    ind_oo2 = np.tril_indices(norb2,k=-1)
    eri_ = None

    if len(eri.shape) == 2:
        if (eri.shape[0] != n_oo1 or eri.shape[1] != n_oo2):
            raise TypeError("ERI dimensions don't match")

        temp = np.zeros((n_oo1, norb2, norb2))
        temp[:, ind_oo2[0], ind_oo2[1]] = eri
        temp[:, ind_oo2[1], ind_oo2[0]] = -eri
        eri_ = np.zeros((norb1, norb1, norb2, norb2))
        eri_[ind_oo1[0], ind_oo1[1]] = temp
        eri_[ind_oo1[1], ind_oo1[0]] = -temp
    else: 
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_
