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

import time
import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df
from pyscf.adc import radc
from pyscf.adc import uadc
from pyscf import __config__

class RADC(radc.RADC):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        radc.RADC.__init__(self, mf, frozen, mo_coeff, mo_occ)

#        if getattr(mf, 'with_df', None):
#            self.with_df = mf.with_df
#        else:
#            self.with_df = df.DF(mf.mol)
#            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
#        self._keys.update(['with_df'])
#
#    def reset(self, mol=None):
#        self.with_df.reset(mol)
#        return radc.RADC.reset(self, mol)
#
#    def ao2mo(self, mo_coeff=None):
#        return _make_df_eris(self, mo_coeff)

class UADC(uadc.UADC):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        uadc.UADC.__init__(self, mf, frozen, mo_coeff, mo_occ)


