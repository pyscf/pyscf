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
G0W0 approximation
'''

#from pyscf.gw.gw import GW
from pyscf.gw import gw_ac
from pyscf.gw import ugw_ac
from pyscf.gw import gw_cd
from pyscf.gw import gw_exact
from pyscf import scf

def GW(mf, freq_int='ac', frozen=None, tdmf=None):
    if isinstance(mf, scf.ghf.GHF):
        raise RuntimeError('GW does not support GHF or GKS methods.')
    if freq_int.lower() == 'ac':
        if isinstance(mf, scf.uhf.UHF):
            return ugw_ac.GWAC(mf, frozen)
        else:
            return gw_ac.GWAC(mf, frozen)
    elif freq_int.lower() == 'cd':
        if isinstance(mf, scf.uhf.UHF):
            raise RuntimeError('GWCD does not support UHF or UKS methods.')
        else:
            return gw_cd.GWCD(mf, frozen)
    elif freq_int.lower() == 'exact':
        if isinstance(mf, scf.uhf.UHF):
            raise RuntimeError('GWExact does not support UHF or UKS methods.')
        else:
            return gw_exact.GWExact(mf, frozen, tdmf)
    else:
        raise RuntimeError("GW frequency integration method %s not recognized. "
                           "Options are 'ac', 'cd', and 'exact'."%(freq_int))
