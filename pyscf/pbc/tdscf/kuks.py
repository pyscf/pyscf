#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import lib
from pyscf.pbc import dft
from pyscf.pbc.tdscf import kuhf
from pyscf.pbc.tdscf.krks import _rebuild_df


class TDA(kuhf.TDA):
    def kernel(self, x0=None):
        _rebuild_df(self)
        kuhf.TDA.kernel(self, x0=x0)

KTDA = TDA

class TDDFT(kuhf.TDHF):
    def kernel(self, x0=None):
        _rebuild_df(self)
        kuhf.TDHF.kernel(self, x0=x0)

RPA = KTDDFT = TDDFT

dft.kuks.KUKS.TDA   = lib.class_as_method(KTDA)
dft.kuks.KUKS.TDHF  = None
dft.kuks.KUKS.TDDFT = lib.class_as_method(TDDFT)
