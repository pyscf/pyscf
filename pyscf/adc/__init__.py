# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib
from pyscf.adc import uadc
from pyscf.adc import radc
from pyscf.adc import radc_amplitudes
from pyscf.adc import uadc_amplitudes
from pyscf.adc import radc_ip
from pyscf.adc import radc_ip_cvs
from pyscf.adc import radc_ea
from pyscf.adc import uadc_ip
from pyscf.adc import uadc_ip_cvs
from pyscf.adc import uadc_ea


def ADC(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    if isinstance(mf, scf.uhf.UHF):
        return UADC(mf, frozen, mo_coeff, mo_occ)
    #elif isinstance(mf, scf.rohf.ROHF):
    #    lib.logger.warn(mf, 'RADC method does not support ROHF reference. ROHF object '
    #                    'is converted to UHF object and UADC method is called.')
    #    mf = scf.addons.convert_to_uhf(mf)
    #    return UADC(mf, frozen, mo_coeff, mo_occ)
    # TODO add ROHF functionality
    elif isinstance(mf, scf.rhf.RHF):
        return RADC(mf, frozen, mo_coeff, mo_occ)
    else :
        raise RuntimeError('ADC code only supports RHF, ROHF, and UHF references')


ADC.__doc__ = uadc.UADC.__doc__


def UADC(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    return uadc.UADC(mf, frozen, mo_coeff, mo_occ)

UADC.__doc__ = uadc.UADC.__doc__

def RADC(mf, frozen=None, mo_coeff=None, mo_occ=None):
    __doc__ = radc.RADC.__doc__

    if not (frozen is None or frozen == 0):
        raise NotImplementedError

    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.rhf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    return radc.RADC(mf, frozen, mo_coeff, mo_occ)

RADC.__doc__ = radc.RADC.__doc__
