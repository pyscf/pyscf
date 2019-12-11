# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

# TODO: Add description of the module
'''
===================================
Algebraic Diagrammatic Construction
===================================
'''

from pyscf import scf
from pyscf.adc import uadc

def ADC(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if (frozen != 0):
        raise NotImplementedError

    if isinstance(mf, scf.uhf.UHF):
        return UADC(mf, frozen, mo_coeff, mo_occ)
    else:
        mf = scf.addons.convert_to_uhf(mf)
        return UADC(mf, frozen, mo_coeff, mo_occ)


def UADC(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = uadc.UADC.__doc__

    if (frozen != 0):
        raise NotImplementedError

    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-UADC')
    else:
        return uadc.UADC(mf, frozen, mo_coeff, mo_occ)
