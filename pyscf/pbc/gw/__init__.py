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

'''
Periodic G0W0 approximation
'''

from pyscf.pbc.gw import krgw_ac
from pyscf.pbc.gw import kugw_ac
from pyscf.pbc.gw import krgw_cd
from pyscf.pbc import scf

def KRGW(mf, freq_int='ac', frozen=None):
    # mf = mf.to_rhf()
    if freq_int.lower() == 'ac':
        return krgw_ac.KRGWAC(mf, frozen)
    elif freq_int.lower() == 'cd':
        return krgw_cd.KRGWCD(mf, frozen)
    else:
        raise RuntimeError("GW frequency integration method %s not recognized. "
                           "With PBC, options are 'ac' and 'cd'."%(freq_int))

def KUGW(mf, freq_int='ac', frozen=None):
    # mf = mf.to_uhf()
    if freq_int.lower() == 'ac':
        return kugw_ac.KUGWAC(mf, frozen)
    elif freq_int.lower() == 'cd':
        raise RuntimeError('GWCD does not support UHF or UKS methods.')
    else:
        raise RuntimeError("GW frequency integration method %s not recognized. "
                           "With PBC, options are 'ac' and 'cd'."%(freq_int))

KGW = KRGW
