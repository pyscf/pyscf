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

import time
import numpy as np
import pyscf.ao2mo as ao2mo
import pyscf.adc
import pyscf.adc.radc

from pyscf import lib
from pyscf.pbc import scf
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_ao2mo
from pyscf import __config__

class RADC(pyscf.adc.radc.RADC):

    #max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        assert (isinstance(mf, scf.khf.KSCF))
        #pyscf.adc.radc.RADC.__init__(self, mf, frozen, mo_coeff, mo_occ)
      
        self._scf = mf

        self.kpts = self._scf.kpts
        #self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        #kpts = mf.kpts
        #nkpts = mf.nkpts
        #nocc = mf.nocc
        #nmo = mf.nmo
        #nvir = nmo - nocc
        #self.ip_partition = None
        #self.ea_partition = None
        #self.direct = True  # If possible, use GDF to compute Wvvvv on-the-fly

        ##################################################
        # don't modify the following attributes, unless you know what you are doing
#        self.keep_exxdiv = False
#
#        keys = set(['kpts', 'khelper', 'ip_partition',
#                    'ea_partition', 'max_space', 'direct'])
#        self._keys = self._keys.union(keys)
#        self.__imds__ = None

    transform_integrals = kadc_ao2mo.transform_integrals_incore


    def kernel_gs(self):

        eris = self.transform_integrals()
