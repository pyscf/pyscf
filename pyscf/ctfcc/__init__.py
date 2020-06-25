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
Parallel Coupled Cluster with CTF and Symtensor
===============================================

Cyclops Tensor Framework (CTF):
    https://github.com/cyclops-community/ctf

Symtensor:
    https://github.com/yangcal/symtensor
'''

from pyscf.cc import ccsd
from pyscf.cc import addons
from pyscf.ctfcc import rccsd
from pyscf.ctfcc import uccsd
from pyscf.ctfcc import gccsd
from pyscf.ctfcc import eom_rccsd
from pyscf.ctfcc import eom_uccsd
from pyscf.ctfcc import eom_gccsd
from pyscf.ctfcc import kccsd_rhf
from pyscf.ctfcc import kccsd_uhf
from pyscf.ctfcc import kccsd
from pyscf.ctfcc import eom_kccsd_rhf
from pyscf.ctfcc import eom_kccsd_uhf
from pyscf.ctfcc import eom_kccsd_ghf
from pyscf import scf
from pyscf.pbc import scf as pbc_scf

def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UCCSD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.ghf.GHF):
        return GCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCCSD(mf, frozen, mo_coeff, mo_occ)
CCSD.__doc__ = ccsd.CCSD.__doc__

scf.hf.SCF.CCSD = CCSD


def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf.cc import dfccsd

    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError('RCCSD cannot be used with UHF method.')
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RCCSD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCCSD method is called.')
        mf = scf.addons.convert_to_uhf(mf)
        return UCCSD(mf, frozen, mo_coeff, mo_occ)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError("DFCCSD not implemented for CTF")

    elif numpy.iscomplexobj(mo_coeff) or numpy.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError("Complex orbitals for molecules not implemented for CTF")
    else:
        return rccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

RCCSD.__doc__ = ccsd.CCSD.__doc__


def UCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-UCCSD')
    else:
        return uccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)
UCCSD.__doc__ = uccsd.UCCSD.__doc__


def GCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.ghf.GHF):
        mf = scf.addons.convert_to_ghf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-GCCSD')
    else:
        return gccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)
GCCSD.__doc__ = gccsd.GCCSD.__doc__

def KCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, pbc_scf.kuhf.KUHF):
        return KUCCSD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, pbc_scf.kghf.KGHF):
        return KGCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        return KRCCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, pbc_scf.kuhf.KUHF):
        raise RuntimeError('KRCCSD cannot be used with KUHF method.')
    return kccsd_rhf.KRCCSD(mf, frozen, mo_coeff, mo_occ)

def KUCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    return kccsd_rhf.KUCCSD(mf, frozen, mo_coeff, mo_occ)

def KGCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    return kccsd_rhf.KGCCSD(mf, frozen, mo_coeff, mo_occ)
