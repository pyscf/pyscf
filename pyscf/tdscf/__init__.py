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

from pyscf.tdscf import rhf
from pyscf.tdscf import uhf
from pyscf.tdscf import rks
from pyscf.tdscf import uks
from pyscf.tdscf.rhf import TDRHF
from pyscf.tdscf.rks import TDRKS
from pyscf.tdscf.uhf import TDUHF
from pyscf.tdscf.uks import TDUKS
from pyscf import scf
from pyscf import dft


def TDHF(mf):
    if getattr(mf, 'xc', None):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)  # To remove newton decoration
        return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        return rhf.TDHF(mf)

def TDA(mf):
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        if isinstance(mf, dft.rks.KohnShamDFT):
            return uks.TDA(mf)
        else:
            return uhf.TDA(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if isinstance(mf, dft.rks.KohnShamDFT):
            return rks.TDA(mf)
        else:
            return rhf.TDA(mf)

def TDDFT(mf):
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
        if isinstance(mf, dft.rks.KohnShamDFT):
            return uks.tddft(mf)
        else:
            return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if isinstance(mf, dft.rks.KohnShamDFT):
            return rks.tddft(mf)
        else:
            return rhf.TDHF(mf)

TD = TDDFT


def RPA(mf):
    return TDDFT(mf)

def dRPA(mf):
    if isinstance(mf, scf.uhf.UHF):
        return uks.dRPA(mf)
    else:
        return rks.dRPA(mf)

def dTDA(mf):
    if isinstance(mf, scf.uhf.UHF):
        return uks.dTDA(mf)
    else:
        return rks.dTDA(mf)
