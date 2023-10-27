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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from pyscf import lib
from pyscf.tdscf import rhf
from pyscf.pbc import scf
from pyscf import __config__

class TDBase(rhf.TDBase):
    _keys = {'cell'}

    def __init__(self, mf):
        rhf.TDBase.__init__(self, mf)
        self.cell = mf.cell

    def get_ab(self, mf=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nto = rhf.TDBase.get_nto
    analyze = lib.invalid_method('analyze')
    oscillator_strength = lib.invalid_method('oscillator_strength')
    transition_dipole              = lib.invalid_method('transition_dipole')
    transition_quadrupole          = lib.invalid_method('transition_quadrupole')
    transition_octupole            = lib.invalid_method('transition_octupole')
    transition_velocity_dipole     = lib.invalid_method('transition_velocity_dipole')
    transition_velocity_quadrupole = lib.invalid_method('transition_velocity_quadrupole')
    transition_velocity_octupole   = lib.invalid_method('transition_velocity_octupole')
    transition_magnetic_dipole     = lib.invalid_method('transition_magnetic_dipole')
    transition_magnetic_quadrupole = lib.invalid_method('transition_magnetic_quadrupole')


class TDA(TDBase):

    init_guess = rhf.TDA.init_guess
    kernel = rhf.TDA.kernel
    _gen_vind = rhf.TDA.gen_vind

    def gen_vind(self, mf):
        moe = scf.addons.mo_energy_with_exxdiv_none(mf)
        with lib.temporary_env(mf, mo_energy=moe):
            vind, hdiag = self._gen_vind(mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

CIS = TDA


class TDHF(TDA):

    init_guess = rhf.TDHF.init_guess
    kernel = rhf.TDHF.kernel
    _gen_vind = rhf.TDHF.gen_vind
    gen_vind = TDA.gen_vind

RPA = TDRHF = TDHF


scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
scf.rohf.ROHF.TDA = None
scf.rohf.ROHF.TDHF = None
