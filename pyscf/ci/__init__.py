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

from pyscf import scf
from pyscf.ci import cisd
from pyscf.ci import ucisd
from pyscf.ci import gcisd
from pyscf.cc import qcisd

def CISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('ROHF'):
        from pyscf import lib
        lib.logger.warn(mf, 'RCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCISD method is called.')
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCISD(mf, frozen, mo_coeff, mo_occ)
CISD.__doc__ = cisd.CISD.__doc__

def RCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.df.df_jk import _DFHF

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    if isinstance(mf, _DFHF) and mf.with_df:
        from pyscf import lib
        lib.logger.warn(mf, f'DF-RCISD for DFHF method {mf} is not available. '
                        'Normal RCISD method is called.')
        return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)
RCISD.__doc__ = cisd.RCISD.__doc__

def UCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.df.df_jk import _DFHF

    mf = mf.remove_soscf()
    if not mf.istype('UHF'):
        mf = mf.to_uhf()

    if isinstance(mf, _DFHF) and mf.with_df:
        from pyscf import lib
        lib.logger.warn(mf, f'DF-UCISD for DFHF method {mf} is not available. '
                        'Normal UCISD method is called.')
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)
UCISD.__doc__ = ucisd.UCISD.__doc__


def GCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.df.df_jk import _DFHF

    mf = mf.remove_soscf()
    if not mf.istype('GHF'):
        mf = mf.to_ghf()

    if isinstance(mf, _DFHF) and mf.with_df:
        raise NotImplementedError('DF-GCISD')
    else:
        return gcisd.GCISD(mf, frozen, mo_coeff, mo_occ)
GCISD.__doc__ = gcisd.GCISD.__doc__


def QCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        raise NotImplementedError
    elif mf.istype('GHF'):
        raise NotImplementedError
    else:
        return RQCISD(mf, frozen, mo_coeff, mo_occ)
QCISD.__doc__ = qcisd.QCISD.__doc__

scf.hf.SCF.QCISD = QCISD

def RQCISD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib

    if mf.istype('UHF'):
        raise RuntimeError('RQCISD cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        lib.logger.warn(mf, 'RQCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UQCISD method is called.')
        raise NotImplementedError

    mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    elif numpy.iscomplexobj(mo_coeff) or numpy.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError

    else:
        return qcisd.QCISD(mf, frozen, mo_coeff, mo_occ)
RQCISD.__doc__ = qcisd.QCISD.__doc__
