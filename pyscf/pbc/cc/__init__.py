#!/usr/bin/env python
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
from pyscf.pbc.cc import ccsd

def RCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    mf = scf.addons.convert_to_rhf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

CCSD = RCCSD

def UCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    mf = scf.addons.convert_to_uhf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return ccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)

def GCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    mf = scf.addons.convert_to_ghf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return ccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KGCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    from pyscf.pbc.cc import kccsd
    mf = scf.addons.convert_to_ghf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return kccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    from pyscf.pbc.cc import kccsd_rhf
    mf = scf.addons.convert_to_rhf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return kccsd_rhf.RCCSD(mf, frozen, mo_coeff, mo_occ)

KCCSD = KRCCSD

def KUCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.lib import logger
    from pyscf.pbc.cc import kccsd_uhf
    mf = scf.addons.convert_to_uhf(mf)
    if mf.cell.dimension == 3 and mf.exxdiv is not None:
        logger.warn(mf, 'mf.exxdiv is %s. It should be set to None in PBC '
                    'CCSD calculations.', mf.exxdiv)
    return kccsd_uhf.UCCSD(mf, frozen, mo_coeff, mo_occ)
