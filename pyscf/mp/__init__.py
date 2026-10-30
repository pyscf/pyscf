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
from pyscf.mp import dfump2
from pyscf.mp import dfromp2
from pyscf.mp import romp2
from pyscf.mp import gmp2
from pyscf.mp import dfgmp2
from pyscf.mp import cabs

def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('GHF'):
        return GMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)
MP2.__doc__ = mp2.MP2.__doc__

def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf import lib

    if mf.istype('UHF'):
        raise RuntimeError('RMP2 cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        if mo_coeff is None and mo_occ is None:
            if getattr(mf, 'with_df', None):
                return dfromp2.DFROMP2(mf, frozen)
            return romp2.ROMP2(mf, frozen)
        lib.logger.warn(mf, 'RMP2 method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UMP2 method is called.')
        return UMP2(mf, frozen, mo_coeff, mo_occ)

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    if getattr(mf, 'with_df', None):
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)
RMP2.__doc__ = mp2.RMP2.__doc__

def UMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.remove_soscf()
    if not mf.istype('UHF'):
        mf = mf.to_uhf()

    if getattr(mf, 'with_df', None):
        return dfump2.DFUMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)
UMP2.__doc__ = ump2.UMP2.__doc__

def ROMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''Semi-canonical ROMP2 for ROHF/ROKS references.

    With a density-fitting mean-field object (or the ``with_df`` attribute),
    this returns the DF-ROMP2 object; otherwise the 4-center-integral ROMP2
    object is returned.
    '''
    mf = mf.remove_soscf()
    if not mf.istype('ROHF'):
        raise RuntimeError('ROMP2 requires an ROHF (or ROKS) reference.')
    if mo_coeff is not None or mo_occ is not None:
        raise NotImplementedError('ROMP2 does not support custom mo_coeff/'
                                  'mo_occ. The ROHF orbitals are always '
                                  'semi-canonicalized.')
    if getattr(mf, 'with_df', None):
        return dfromp2.DFROMP2(mf, frozen)
    return romp2.ROMP2(mf, frozen)
ROMP2.__doc__ = dfromp2.DFROMP2.__doc__

def GMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    mf = mf.remove_soscf()
    if not mf.istype('GHF'):
        mf = mf.to_ghf()

    if getattr(mf, 'with_df', None):
        return dfgmp2.DFGMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return gmp2.GMP2(mf, frozen, mo_coeff, mo_occ)
GMP2.__doc__ = gmp2.GMP2.__doc__
