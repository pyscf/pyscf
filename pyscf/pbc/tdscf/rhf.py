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
from pyscf import __config__

class TDMixin(rhf.TDMixin):
    def __init__(self, mf):
        rhf.TDMixin.__init__(self, mf)
        self.cell = mf.cell
        self._keys = self._keys.union(['cell'])

    def get_ab(self, mf=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError

    get_nto = rhf.TDMixin.get_nto
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


class TDA(TDMixin):

    init_guess = rhf.TDA.init_guess
    kernel = rhf.TDA.kernel
    _gen_vind = rhf.TDA.gen_vind

    def gen_vind(self, mf):
        # gen_vind calls get_jk functions to compute the contraction between
        # two-electron integrals and X,Y amplitudes. There are two choices for
        # the treatment of exxdiv.
        # 1. Removing exxdiv corrections in both orbital energies (in hdiag) and
        #   get_jk functions (in vind function). This treatment can make TDA the
        #   same to CIS method in which exxdiv was completely excluded when
        #   constructing Hamiltonians.
        # 2. Excluding exxdiv corrections from get_jk only. Keep its correction
        #   to orbital energy. This treatment can make the TDDFT excitation
        #   energies closed to the relevant DFT orbital energy gaps.
        # DFT orbital energy gaps can be used as a good estimation for
        # excitation energies. Current implementation takes the second choice so
        # as to make the TDDFT excitation energies agree to DFT orbital energy gaps.
        #
        # There might be a third treatment: Taking treatment 1 first then adding
        # certain types of corrections to the excitation energy at last.
        # I'm not sure how to do this properly.
        #
        # See also issue https://github.com/pyscf/pyscf/issues/1187

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


from pyscf.pbc import scf
scf.hf.RHF.TDA = lib.class_as_method(TDA)
scf.hf.RHF.TDHF = lib.class_as_method(TDHF)
scf.rohf.ROHF.TDA = None
scf.rohf.ROHF.TDHF = None

