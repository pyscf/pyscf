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

from pyscf.zmcscf import zmc2step
from pyscf.zmcscf import gzcasci
from pyscf.mcscf import casci
from pyscf.mcscf import addons


def ZCASSCF(mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
    from pyscf import gto
    from pyscf import scf
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol

    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_rhf(mf)
#    if getattr(mf, 'with_df', None):
#        return DFCASSCF(mf, ncas, nelecas, ncore, frozen)

#    if mf.mol.symmetry:
#        mc = mc1step_symm.CASSCF(mf, ncas, nelecas, ncore, frozen)
#    else:
    mc = zmc2step.ZCASSCF(mf, ncas, nelecas, ncore, frozen)
    return mc

RCASSCF = CASSCF

