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

from pyscf.pbc.cc import ccsd
from pyscf.pbc.cc import kccsd_rhf as krccsd
from pyscf.pbc.cc import kccsd_uhf as kuccsd
from pyscf.pbc.cc import kccsd     as kgccsd
from pyscf.pbc.cc import eom_kccsd_rhf
from pyscf.pbc.cc import eom_kccsd_uhf
from pyscf.pbc.cc import eom_kccsd_ghf
from pyscf.pbc.cc.kccsd_rhf_ksymm import KsymAdaptedRCCSD

def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_rhf()
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

CCSD = RCCSD

def UCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_uhf()
    return ccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)

def GCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_ghf()
    return ccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KGCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_ghf()
    return kgccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.pbc import scf
    assert isinstance(mf, scf.khf.KSCF)
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return krccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

KCCSD = KRCCSD

def KUCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.pbc import scf
    assert isinstance(mf, scf.khf.KSCF)
    if not isinstance(mf, scf.kuhf.KUHF):
        mf = mf.to_uhf()
    return kuccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)
