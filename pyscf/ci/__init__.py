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

from pyscf import lib
from pyscf import scf
from pyscf.ci import cisd
from pyscf.ci import ucisd
from pyscf.ci import gcisd

def CISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, scf.uhf.UHF):
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCISD method is called.')
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCISD(mf, frozen, mo_coeff, mo_occ)
CISD.__doc__ = cisd.CISD.__doc__

def RCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-RCISD')
    else:
        return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)
RCISD.__doc__ = cisd.RCISD.__doc__

def UCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-UCISD')
    else:
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)
UCISD.__doc__ = ucisd.UCISD.__doc__


def GCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.ghf.GHF):
        mf = scf.addons.convert_to_ghf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-GCISD')
    else:
        return gcisd.GCISD(mf, frozen, mo_coeff, mo_occ)
GCISD.__doc__ = gcisd.GCISD.__doc__
