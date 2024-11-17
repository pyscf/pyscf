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

import unittest
import tempfile
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import fci
from pyscf import mcscf

def setUpModule():
    global mol, m
    b = 1.4
    mol = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
    basis = '631g',
    spin = 2,
    )
    m = scf.UHF(mol)
    m.conv_tol = 1e-10
    m.scf()

def tearDownModule():
    global mol, m
    mol.stdout.close()
    del mol, m


class KnownValues(unittest.TestCase):
    def test_ucasscf(self):
        with tempfile.NamedTemporaryFile() as f:
            mc = mcscf.UCASSCF(m, 4, 4)
            mc.chkfile = f.name
            mc.run()
        self.assertAlmostEqual(mc.e_tot, -75.7460662487894, 6)

    def test_with_x2c_scanner(self):
        mc1 = mcscf.UCASSCF(m, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -75.795316854668201, 6)

        mc1 = mcscf.UCASSCF(m, 4, 4).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -75.795316865791847, 6)

    def test_0core_0virtual(self):
        mol = gto.M(atom='He', basis='321g')
        mf = scf.UHF(mol).run()
        mc1 = mcscf.UCASSCF(mf, 2, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.UCASSCF(mf, 1, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.UCASSCF(mf, 1, 0).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

    def test_external_fcisolver(self):
        fcisolver1 = fci.direct_uhf.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_uhf.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            def approx_kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)

        mc = mcscf.UCASSCF(m, 5, (4,2))
        mc.fcisolver = FCI_as_DMRG(mol)
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -75.755924721041396, 7)

    def test_state_average(self):
        mc = mcscf.UCASSCF(m, 5, (4, 2)).state_average_((0.5, 0.5))
        mc.natorb = True
        mc.kernel()
        self.assertAlmostEqual(mc.e_states[0], -75.745603861816, 5)
        self.assertAlmostEqual(mc.e_states[1], -75.6539185381917, 5)

    def test_state_average_mix(self):
        mc = mcscf.UCASSCF(m, 5, (4, 2))
        cis1 = mc.fcisolver.copy()
        cis1.spin = 0
        mc = mcscf.addons.state_average_mix(mc, [cis1, mc.fcisolver], [0.5, 0.5])
        mc.kernel()
        self.assertAlmostEqual(mc.e_states[0], -76.0207219040825, 4)
        self.assertAlmostEqual(mc.e_states[1], -75.75071008513498, 4)

    def test_state_average_mix_external_fcisolver(self):
        fcisolver1 = fci.direct_uhf.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_uhf.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
            def approx_kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)

        solver1 = FCI_as_DMRG(mol)
        solver1.spin = 0
        solver2 = fci.direct_uhf.FCI(mol)
        solver2.spin = 2
        mc = mcscf.UCASSCF(m, 5, (4, 2))
        mc = mcscf.addons.state_average_mix_(mc, [solver1, solver2])
        mc.kernel()
        self.assertAlmostEqual(mc.e_states[0], -76.02072193565274, 4)
        self.assertAlmostEqual(mc.e_states[1], -75.75071005397444, 4)

    def test_frozen(self):
        mc = mcscf.UCASSCF(m, 5, (4,2))
        mc.frozen = 2
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -75.753628185779561, 7)

    #TODO:
    #def test_grad(self):
    #    mc = mcscf.UCASSCF(m, 5, (4,2))
    #    mc.kernel()
    #    self.assertAlmostEqual(mc.e_tot, -75.755924721041396, 7)
    #    self.assertAlmostEqual(abs(mc.get_grad()).max(), 0, 4)

    def test_casci_in_casscf(self):
        mc1 = mcscf.UCASSCF(m, 5, (4,2))
        e_tot, e_ci, fcivec = mc1.casci(mc1.mo_coeff)
        self.assertAlmostEqual(e_tot, -75.733190125187946, 9)


if __name__ == "__main__":
    print("Full Tests for umc1step")
    unittest.main()
