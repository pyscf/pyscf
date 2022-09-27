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

'''
Moller-Plesset perturbation theory
'''

from pyscf import scf
from pyscf.mp import mp2
from pyscf.mp import dfmp2
from pyscf.mp import ump2
from pyscf.mp import gmp2
from pyscf.mp import dfgmp2

def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.ghf.GHF):
        return GMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)
MP2.__doc__ = mp2.MP2.__doc__

def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf import lib
    from pyscf.soscf import newton_ah

    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError('RMP2 cannot be used with UHF method.')
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RMP2 method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UMP2 method is called.')
        return UMP2(mf, frozen, mo_coeff, mo_occ)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = mf.to_rhf()

    if getattr(mf, 'with_df', None):
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)
RMP2.__doc__ = mp2.RMP2.__doc__

def UMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = mf.to_uhf()

    if getattr(mf, 'with_df', None):
        #raise NotImplementedError('DF-UMP2')
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)
UMP2.__doc__ = ump2.UMP2.__doc__

def GMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.ghf.GHF):
        mf = mf.to_ghf()

    if getattr(mf, 'with_df', None):
        return dfgmp2.DFGMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return gmp2.GMP2(mf, frozen, mo_coeff, mo_occ)
GMP2.__doc__ = gmp2.GMP2.__doc__
