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
from pyscf.pbc.tdscf import rhf as td_rhf
from pyscf.pbc.tdscf.rhf import TDBase


class TDA(TDBase):

    singlet = None

    init_guess = uhf.TDA.init_guess
    kernel = uhf.TDA.kernel
    _gen_vind = uhf.TDA.gen_vind
    gen_vind = td_rhf.TDA.gen_vind

CIS = TDA


class TDHF(TDA):

    singlet = None

    init_guess = uhf.TDHF.init_guess
    kernel = uhf.TDHF.kernel
    _gen_vind = uhf.TDHF.gen_vind
    gen_vind = td_rhf.TDA.gen_vind

RPA = TDUHF = TDHF


from pyscf.pbc import scf
scf.uhf.UHF.TDA = lib.class_as_method(TDA)
scf.uhf.UHF.TDHF = lib.class_as_method(TDHF)
