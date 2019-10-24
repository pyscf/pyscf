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
import pyscf.ao2mo

### Integral transformation ###
def transform_integrals(myadc):
    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]


    occ = occ_a, occ_b
    vir = vir_a, vir_b

    eris = lambda:None

    eris.oovv = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,occ,vir,vir))
    eris.vvvv = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,vir,vir,vir))
    eris.oooo = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,occ,occ,occ))
    eris.voov = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,occ,occ,vir))
    eris.ooov = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,occ,occ,vir))
    eris.vovv = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,occ,vir,vir))
    eris.vvoo = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,vir,occ,occ))
    eris.vvvo = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,vir,vir,occ))
    eris.ovoo = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,vir,occ,occ))
    eris.ovov = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,vir,occ,vir))
    eris.vooo = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,occ,occ,occ))
    eris.oovo = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,occ,vir,occ))
    eris.vovo = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,occ,vir,occ))
    eris.vvov = transform_antisymmetrize_integrals(myadc._scf._eri, (vir,vir,occ,vir))
    eris.ovvo = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,vir,vir,occ))
    eris.ovvv = transform_antisymmetrize_integrals(myadc._scf._eri, (occ,vir,vir,vir))

    return eris

# TODO: disk flag
def transform_antisymmetrize_integrals(v2e_ao, mo, disk = False):

    mo_1, mo_2, mo_3, mo_4 = mo

    mo_1_a, mo_1_b = mo_1
    mo_2_a, mo_2_b = mo_2
    mo_3_a, mo_3_b = mo_3
    mo_4_a, mo_4_b = mo_4

    v2e_a = None
    v2e_a = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_3_a, mo_2_a, mo_4_a), compact=False)
    v2e_a = v2e_a.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_a.shape[1], mo_4_a.shape[1])
    v2e_a = v2e_a.transpose(0,2,1,3).copy()

    if (mo_1_a is mo_2_a):
        v2e_a -= v2e_a.transpose(1,0,2,3).copy()
    elif (mo_3_a is mo_4_a):
        v2e_a -= v2e_a.transpose(0,1,3,2).copy()
    else:
        v2e_temp = None
        v2e_temp = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_4_a, mo_2_a, mo_3_a), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_a.shape[1], mo_4_a.shape[1], mo_2_a.shape[1], mo_3_a.shape[1])
        v2e_a -= v2e_temp.transpose(0,2,3,1).copy()
        del v2e_temp

    v2e_a = disk_helper.dataset(v2e_a) if disk else v2e_a

    v2e_b = None
    v2e_b = pyscf.ao2mo.general(v2e_ao, (mo_1_b, mo_3_b, mo_2_b, mo_4_b), compact=False)
    v2e_b = v2e_b.reshape(mo_1_b.shape[1], mo_3_b.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_b = v2e_b.transpose(0,2,1,3).copy()

    if (mo_1_b is mo_2_b):
        v2e_b -= v2e_b.transpose(1,0,2,3).copy()
    elif (mo_3_b is mo_4_b):
        v2e_b -= v2e_b.transpose(0,1,3,2).copy()
    else:
        v2e_temp = None
        v2e_temp = pyscf.ao2mo.general(v2e_ao, (mo_1_b, mo_4_b, mo_2_b, mo_3_b), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_b.shape[1], mo_4_b.shape[1], mo_2_b.shape[1], mo_3_b.shape[1])
        v2e_b -= v2e_temp.transpose(0,2,3,1).copy()
        del v2e_temp

    v2e_b = disk_helper.dataset(v2e_b) if disk else v2e_b

    v2e_ab = None
    v2e_ab = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_3_a, mo_2_b, mo_4_b), compact=False)
    v2e_ab = v2e_ab.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_ab = v2e_ab.transpose(0,2,1,3).copy()

    v2e_ab = disk_helper.dataset(v2e_ab) if disk else v2e_ab

    return (v2e_a, v2e_ab, v2e_b)
