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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf import scf
from pyscf.tdscf import rhf
from pyscf.tdscf import uhf
from pyscf.tdscf import ghf
from pyscf.tdscf import dhf
from pyscf.tdscf.rhf import TDRHF
from pyscf.tdscf.uhf import TDUHF
from pyscf.tdscf.ghf import TDGHF

try:
    from pyscf.dft import KohnShamDFT
    from pyscf.tdscf import rks
    from pyscf.tdscf import uks
    from pyscf.tdscf import gks
    from pyscf.tdscf import dks
    from pyscf.tdscf.rks import TDRKS
    from pyscf.tdscf.uks import TDUKS
    from pyscf.tdscf.gks import TDGKS
except (ImportError, IOError):
    pass


def TDHF(mf):
    if isinstance(mf, scf.hf.KohnShamDFT):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        # Is it correct to call TDUHF for ROHF?
        mf = mf.to_uhf()
    return mf.TDHF()

def TDA(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA()

def TDDFT(mf):
    if isinstance(mf, KohnShamDFT):
        mf = mf.remove_soscf()
        if isinstance(mf, scf.rohf.ROHF):
            mf = mf.to_uks()
        return mf.TDDFT()
    else:
        return TDHF(mf)

TD = TDDFT


def RPA(mf):
    return TDDFT(mf)

def dRPA(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.dRPA()

def dTDA(mf):
    mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.dTDA()
