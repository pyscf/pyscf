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
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import fci
from pyscf import mcscf

b = 1.4
mol = gto.M(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': '631g', },
)
m = scf.RHF(mol)
m.conv_tol = 1e-10
m.scf()

molsym = gto.M(
verbose = 5,
output = '/dev/null',
atom = [
    ['N',(  0.000000,  0.000000, -b/2)],
    ['N',(  0.000000,  0.000000,  b/2)], ],
basis = {'N': '631g', },
symmetry = True
)
msym = scf.RHF(molsym)
msym.conv_tol = 1e-10
msym.scf()

def tearDownModule():
    global mol, molsym, m, msym
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym


class KnownValues(unittest.TestCase):
    def test_with_x2c_scanner(self):
        mc1 = mcscf.CASSCF(m, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1 = mcscf.CASSCF(m, 4, 4).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

    def test_0core_0virtual(self):
        mol = gto.M(atom='He', basis='321g')
        mf = scf.RHF(mol).run()
        mc1 = mcscf.CASSCF(mf, 2, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.CASSCF(mf, 1, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 1, 0).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

    def test_cas_natorb(self):
        mc1 = mcscf.CASSCF(msym, 4, 4, ncore=5)
        mo = mc1.sort_mo([4,5,10,13])
        mc1.sorting_mo_energy = True
        mc1.kernel(mo)
        mo0 = mc1.mo_coeff
        ci0 = mc1.ci
        self.assertAlmostEqual(mc1.e_tot, -108.7288793597413, 8)
        casdm1 = mc1.fcisolver.make_rdm1(mc1.ci, 4, 4)
        mc1.ci = None  # Force cas_natorb_ to recompute CI coefficients

        mc1.cas_natorb_(casdm1=casdm1, eris=mc1.ao2mo())
        mo1 = mc1.mo_coeff
        ci1 = mc1.ci
        s = numpy.einsum('pi,pq,qj->ij', mo0[:,5:9], msym.get_ovlp(), mo1[:,5:9])
        self.assertAlmostEqual(fci.addons.overlap(ci0, ci1, 4, 4, s), 1, 9)

    def test_get_h2eff(self):
        mc1 = mcscf.approx_hessian(mcscf.CASSCF(m, 4, 4))
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertAlmostEqual(abs(eri1-eri2).max(), 0, 12)

        mc1 = mcscf.density_fit(mcscf.CASSCF(m, 4, 4))
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertTrue(abs(eri1-eri2).max() > 1e-5)

    def test_get_veff(self):
        mf = m.view(dft.rks.RKS)
        mc1 = mcscf.CASSCF(mf, 4, 4)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        veff1 = mc1.get_veff(mol, dm)
        veff2 = m.get_veff(mol, dm)
        self.assertAlmostEqual(abs(veff1-veff2).max(), 0, 12)

    def test_state_average(self):
        mc1 = mcscf.CASSCF(m, 4, 4).state_average_((0.5,0.5))
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot, -108.80445340617777, 9)
        dm1 = mc1.analyze()
        self.assertAlmostEqual(lib.finger(dm1[0]), 2.6993157521103779, 5)
        self.assertAlmostEqual(lib.finger(dm1[1]), 2.6993157521103779, 5)


if __name__ == "__main__":
    print("Full Tests for mc1step")
    unittest.main()

