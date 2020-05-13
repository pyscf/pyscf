#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
from pyscf.tdscf import uhf


class TDA(uhf.TDA):
    def gen_vind(self, mf):
        vind, hdiag = uhf.TDA.gen_vind(self, mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

    def nuc_grad_method(self):
        raise NotImplementedError

CIS = TDA


class TDHF(uhf.TDHF):
    def gen_vind(self, mf):
        vind, hdiag = uhf.TDHF.gen_vind(self, mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

    def nuc_grad_method(self):
        raise NotImplementedError

RPA = TDUHF = TDHF


from pyscf.pbc import scf
scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)

