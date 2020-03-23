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

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]
    n_oo = nocc_a * (nocc_a + 1) // 2
    n_OO = nocc_b * (nocc_b + 1) // 2
    n_vv = nvir_a * (nvir_a + 1) // 2
    n_VV = nvir_b * (nvir_b + 1) // 2
    ind_oo = np.tril_indices(nocc_a)
    ind_vv = np.tril_indices(nvir_a)
    ind_OO = np.tril_indices(nocc_b)
    ind_VV = np.tril_indices(nvir_b)

    eris = lambda:None

    # TODO: check if myadc._scf._eri is not None
    eris.oooo = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_a, occ_a), compact=False).reshape(nocc_a, nocc_a, nocc_a, nocc_a).copy()
    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nocc_a, nocc_a).copy()
    eris.ovov = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_a, vir_a), compact=False).reshape(nocc_a, nvir_a, nocc_a, nvir_a).copy()
    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nvir_a, nocc_a).copy()
    eris.oovv = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_a, vir_a), compact=False).reshape(nocc_a, nocc_a, nvir_a, nvir_a).copy()
    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, vir_a), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OOOO = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, occ_b, occ_b), compact=False).reshape(nocc_b, nocc_b, nocc_b, nocc_b).copy()
    eris.OVOO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nocc_b, nocc_b).copy()
    eris.OVOV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_b, vir_b), compact=False).reshape(nocc_b, nvir_b, nocc_b, nvir_b).copy()
    eris.OOVV = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_b, vir_b), compact=False).reshape(nocc_b, nocc_b, nvir_b, nvir_b).copy()
    eris.OVVO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nvir_b, nocc_b).copy()
    eris.OVVV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, vir_b), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    eris.ooOO = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_b, occ_b), compact=False).reshape(nocc_a, nocc_a, nocc_b, nocc_b).copy()
    eris.ovOO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nocc_b, nocc_b).copy()
    eris.ovOV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_b, vir_b), compact=False).reshape(nocc_a, nvir_a, nocc_b, nvir_b).copy()
    eris.ooVV = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_b, vir_b), compact=False).reshape(nocc_a, nocc_a, nvir_b, nvir_b).copy()
    eris.ovVO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nvir_b, nocc_b).copy()
    eris.ovVV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, vir_b), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OVoo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nocc_a, nocc_a).copy()
    eris.OOvv = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_a, vir_a), compact=False).reshape(nocc_b, nocc_b, nvir_a, nvir_a).copy()
    eris.OVov = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_a, vir_a), compact=False).reshape(nocc_b, nvir_b, nocc_a, nvir_a).copy()
    eris.OVvo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nvir_a, nocc_a).copy()
    eris.OVvv = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, vir_a), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_a, vir_a), compact=True)
        eris.VVVV = ao2mo.general(myadc._scf._eri, (vir_b, vir_b, vir_b, vir_b), compact=True)
        eris.vvVV = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_b, vir_b), compact=True)

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

def unpack_eri_2(eri, norb1, norb2):

    n_oo1 = norb1 * (norb1 + 1) // 2
    ind_oo1 = np.tril_indices(norb1)
    n_oo2 = norb2 * (norb2 + 1) // 2
    ind_oo2 = np.tril_indices(norb2)

    eri_ = None

    if len(eri.shape) == 2:
        if (eri.shape[0] != n_oo1 or eri.shape[1] != n_oo2):
            raise TypeError("ERI dimensions don't match")

        temp = np.zeros((n_oo1, norb2, norb2))
        temp[:, ind_oo2[0], ind_oo2[1]] = eri
        temp[:, ind_oo2[1], ind_oo2[0]] = eri
        eri_ = np.zeros((norb1, norb1, norb2, norb2))
        eri_[ind_oo1[0], ind_oo1[1]] = temp
        eri_[ind_oo1[1], ind_oo1[0]] = temp
    else:
            raise RuntimeError("ERI does not have a correct dimension")

    return eri_
