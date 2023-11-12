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

from pyscf.pbc.cc import ccsd

def KRCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.pbc import scf
    from pyscf.pbc.mpicc import kccsd_rhf
    assert isinstance(mf, scf.khf.KSCF)
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return kccsd_rhf.RCCSD(mf, frozen)

KCCSD = KRCCSD
