#!/usr/bin/env python
# Copyright 2021-2022 The PySCF Developers. All Rights Reserved.
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


import numpy
from pyscf import lib
from pyscf.tdscf import dhf
from pyscf.data import nist
from pyscf.dft.rks import KohnShamDFT
from pyscf import __config__


class TDA(dhf.TDA):
    pass

class TDDFT(dhf.TDHF):
    pass

RPA = TDDKS = TDDFT

from pyscf import dft
dft.dks.DKS.TDA   = dft.dks.RDKS.TDA   = lib.class_as_method(TDA)
dft.dks.DKS.TDHF  = dft.dks.RDKS.TDHF  = None
dft.dks.DKS.TDDFT = dft.dks.RDKS.TDDFT = lib.class_as_method(TDDFT)
dft.dks.DKS.CasidaTDDFT = dft.dks.RDKS.CasidaTDDFT = None
