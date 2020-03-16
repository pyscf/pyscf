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
    ind_vv_g = np.tril_indices(nvir_a, k=-1)
    ind_VV_g = np.tril_indices(nvir_b, k=-1)
   
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

        eris.vvvv_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_a, vir_a), compact=False).reshape(nvir_a, nvir_a, nvir_a, nvir_a)
        eris.vvvv_p = eris.vvvv_p.transpose(0,2,1,3)
        eris.vvvv_p -= eris.vvvv_p.transpose(0,1,3,2)
        eris.vvvv_p = eris.vvvv_p[:, :, ind_vv_g[0], ind_vv_g[1]]
        eris.vvvv_p = eris.vvvv_p[ind_vv_g[0], ind_vv_g[1]].copy()

        eris.VVVV_p = ao2mo.general(myadc._scf._eri, (vir_b, vir_b, vir_b, vir_b), compact=False).reshape(nvir_b, nvir_b, nvir_b, nvir_b)
        eris.VVVV_p = eris.VVVV_p.transpose(0,2,1,3)
        eris.VVVV_p -= eris.VVVV_p.transpose(0,1,3,2)
        eris.VVVV_p = eris.VVVV_p[:, :, ind_VV_g[0], ind_VV_g[1]]
        eris.VVVV_p = eris.VVVV_p[ind_VV_g[0], ind_VV_g[1]].copy()

        eris.vVvV_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_b, vir_b), compact=False).reshape(nvir_a, nvir_a, nvir_b, nvir_b)
        eris.vVvV_p = np.ascontiguousarray(eris.vVvV_p.transpose(0,2,1,3)) 
        eris.vVvV_p = eris.vVvV_p.reshape(nvir_a*nvir_b, nvir_a*nvir_b) 

    return eris
