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
from pyscf import lib
from pyscf.lib import logger
import time
import h5py
import tempfile

### Incore integral transformation for integrals in Chemists' notation###
def transform_integrals_incore(myadc):

     cell = myadc.cell
     kpts = myadc.kpts
     #nkpts = myadc.nkpts
     #nocc = myadc.nocc
     #nmo = myadc.nmo
     #nvir = nmo - nocc

     dtype = myadc.mo_coeff[0].dtype
     print (dtype)




#    cput0 = (time.clock(), time.time())
#    log = logger.Logger(myadc.stdout, myadc.verbose)
#
#    occ = myadc.mo_coeff[:,:myadc._nocc]
#    vir = myadc.mo_coeff[:,myadc._nocc:]
#
#    nocc = occ.shape[1]
#    nvir = vir.shape[1]
#
#    eris = lambda:None
#
#    # TODO: check if myadc._scf._eri is not None
#
#    eris.oooo = ao2mo.general(myadc._scf._eri, (occ, occ, occ, occ), compact=False).reshape(nocc, nocc, nocc, nocc).copy()
#    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ, vir, occ, occ), compact=False).reshape(nocc, nvir, nocc, nocc).copy()
#    eris.oovv = ao2mo.general(myadc._scf._eri, (occ, occ, vir, vir), compact=False).reshape(nocc, nocc, nvir, nvir).copy()
#    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ, vir, vir, occ), compact=False).reshape(nocc, nvir, nvir, nocc).copy()
#    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ, vir, vir, vir), compact=True).reshape(nocc, nvir, -1).copy()
#
#    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
#        eris.vvvv = ao2mo.general(myadc._scf._eri, (vir, vir, vir, vir), compact=False).reshape(nvir, nvir, nvir, nvir)
#        eris.vvvv = np.ascontiguousarray(eris.vvvv.transpose(0,2,1,3)) 
#        eris.vvvv = eris.vvvv.reshape(nvir*nvir, nvir*nvir)
#
#    log.timer('ADC integral transformation', *cput0)
#    return eris
