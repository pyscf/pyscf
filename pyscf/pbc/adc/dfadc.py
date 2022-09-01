#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

def get_ovvv_df(myadc, Lov, Lvv, p, chnk_size):

    ''' Returns approximate ovvv integrals used in restricted implementation'''

    naux = Lov.shape[0]
    nocc = Lov.shape[1]
    nvir = Lov.shape[2]

    Lvv = Lvv.reshape(naux,nvir*nvir)
    Lov = Lov.reshape(naux,nocc,nvir)

    if chnk_size < nocc:
        Lov_temp = np.ascontiguousarray(Lov.transpose(1,2,0)[p:p+chnk_size].reshape(-1,naux))
    else :
        Lov_temp = np.ascontiguousarray(Lov.transpose(1,2,0).reshape(-1,naux))

    ovvv = np.dot(Lov_temp, Lvv)
    ovvv = ovvv.reshape(-1, nvir, nvir, nvir)
    del Lvv
    del Lov
    del Lov_temp
    return ovvv


def get_vvvv_df(myadc, vv1, vv2, p, chnk_size):

    ''' Returns approximate vvvv integrals used in restricted implementation'''

    naux = vv1.shape[0]
    nvir = vv1.shape[1]

    vv1 = vv1.reshape(naux,nvir,nvir)

    if chnk_size < nvir:
        vv1_temp = np.ascontiguousarray(vv1.transpose(1,2,0)[p:p+chnk_size].reshape(-1,naux))
    else :
        vv1_temp = np.ascontiguousarray(vv1.transpose(1,2,0).reshape(-1,naux))

    vv2 = vv2.reshape(naux,nvir*nvir)
    vvvv = np.dot(vv1_temp, vv2)
    vvvv = vvvv.reshape(-1, nvir, nvir, nvir)
    vvvv = np.ascontiguousarray(vvvv.transpose(0,2,1,3)).reshape(-1, nvir, nvir ,nvir)
    del vv1
    del vv2
    return vvvv
