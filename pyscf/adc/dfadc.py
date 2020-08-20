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

import numpy as np

def  get_vvvv_df(myadc, Lvv, p, chnk_size):

    nocc = myadc._nocc
    nvir = myadc._nvir
    naux = myadc._scf.with_df.get_naoaux()

    Lvv = Lvv.reshape(naux,nvir,nvir)

    if chnk_size < nvir:
        Lvv_temp = np.ascontiguousarray(Lvv.T[p:p+chnk_size].reshape(-1,naux))
    else :
        Lvv_temp = np.ascontiguousarray(Lvv.T.reshape(-1,naux))

    Lvv = Lvv.reshape(naux,nvir*nvir)
    vvvv = np.dot(Lvv_temp, Lvv)
    vvvv = vvvv.reshape(-1, nvir, nvir, nvir)
    vvvv = np.ascontiguousarray(vvvv.transpose(0,2,1,3)).reshape(-1, nvir, nvir * nvir)
    return vvvv    


def  get_vvvv_antisym_df(myadc, Lvv, p, nvir, chnk_size):

    naux = myadc._scf.with_df.get_naoaux()

    Lvv = Lvv.reshape(naux,nvir,nvir)
    ind_vv_g = np.tril_indices(nvir, k=-1)

    if chnk_size < nvir:
        Lvv_temp = np.ascontiguousarray(Lvv.T[p:p+chnk_size].reshape(-1,naux))
    else :
        Lvv_temp = np.ascontiguousarray(Lvv.T.reshape(-1,naux))

    Lvv = Lvv.reshape(naux,nvir*nvir)
    vvvv = np.dot(Lvv_temp, Lvv)
    vvvv = vvvv.reshape(-1, nvir, nvir, nvir)
    vvvv = np.ascontiguousarray(vvvv.transpose(0,2,1,3)).reshape(-1, nvir, nvir, nvir)
    vvvv -= np.ascontiguousarray(vvvv.transpose(0,1,3,2))
    vvvv = vvvv[:, :, ind_vv_g[0], ind_vv_g[1]]

    return vvvv    


def  get_vVvV_df(myadc, Lvv, LVV, p, chnk_size):

    naux = myadc._scf.with_df.get_naoaux()
    nvir_1 = Lvv.shape[1]
    nvir_2 = LVV.shape[1]

    if chnk_size < nvir_1:
        Lvv_temp = np.ascontiguousarray(Lvv.T[p:p+chnk_size].reshape(-1,naux))
    else :
        Lvv_temp = np.ascontiguousarray(Lvv.T.reshape(-1,naux))

    LVV = LVV.reshape(naux,nvir_2*nvir_2)
    vvvv = np.dot(Lvv_temp, LVV).reshape(-1,nvir_1,nvir_2,nvir_2)
    vvvv = np.ascontiguousarray(vvvv.transpose(0,2,1,3)).reshape(-1, nvir_2, nvir_1, nvir_2)

    return vvvv    
