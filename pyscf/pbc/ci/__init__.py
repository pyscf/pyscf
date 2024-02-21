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

from pyscf.pbc import scf
from pyscf.pbc.ci import cisd, kcis_rhf

def RCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_rhf()
    return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)

CISD = RCISD

def UCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_uhf()
    return cisd.UCISD(mf, frozen, mo_coeff, mo_occ)

def GCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_ghf()
    return cisd.GCISD(mf, frozen, mo_coeff, mo_occ)

def KCIS(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.to_rhf()
    return kcis_rhf.KCIS(mf, frozen, mo_coeff, mo_occ)

CIS = KCIS
