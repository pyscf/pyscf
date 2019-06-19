#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.doci import doci_slow
from pyscf.fci import direct_spin1, cistring
from pyscf.mcscf import casci, mc1step


class CASCI(casci.CASCI):
    '''DOCI-CASCI'''
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        casci.CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.fcisolver = doci_slow.DOCI(mf_or_mol)


class CASSCF(mc1step.CASSCF):
    '''DOCI-CASSCF'''
    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        mc1step.CASSCF.__init__(self, mf_or_mol, ncas, nelecas, ncore, frozen)
        self.fcisolver = doci_slow.DOCI(mf_or_mol)
        self.internal_rotation = True

if __name__ == '__main__':
    from pyscf import gto, scf, mcscf
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = None
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]
    mol.basis = 'sto-3g'
    mol.build()

    mf = scf.RHF(mol).run()

    mc = CASSCF(mf, 6, 6)
    mc.kernel()

    mc = CASCI(mf, 6, 6)
    mc.kernel()
