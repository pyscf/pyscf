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

def setUpModule():
    global mol, m, mc, molsym, msym
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
    mc = mcscf.UCASCI(m, 5, (4,2)).run()

    b = 1.4
    molsym = gto.M(
    verbose = 7,
    output = '/dev/null',
    atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
    basis = '631g',
    spin = 2,
    symmetry = True,
    )
    msym = scf.UHF(molsym)
    msym.conv_tol = 1e-10
    msym.scf()

def tearDownModule():
    global mol, m, mc, molsym, msym
    mol.stdout.close()
    molsym.stdout.close()
    del mol, m, mc, molsym, msym


class KnownValues(unittest.TestCase):
    def test_with_x2c_scanner(self):
        mc1 = mcscf.UCASCI(m, 5, (4,2)).x2c()
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot, -75.782441558951504, 8)

        mc1 = mcscf.UCASCI(m, 5, (4,2)).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -75.782441558951504, 8)

#    def test_fix_spin_(self):
#        mc1 = mcscf.CASCI(m, 4, 4)
#        mc1.fix_spin_(ss=2).run()
#        self.assertAlmostEqual(mc1.e_tot, -108.03741684418183, 9)
#
#        mc1.fix_spin_(ss=0).run()
#        self.assertAlmostEqual(mc1.e_tot, -108.83741684447352, 9)
#
#    def test_cas_natorb(self):
#        pass

    def test_get_h2eff(self):
        mc1 = mcscf.UCASCI(m, 5, (4,2), ncore=(2,2))
        mc1.max_memory = 0
        mc2 = mcscf.approx_hessian(mc1)
        eri1 = mc1.get_h2eff((m.mo_coeff[0][:,2:7], m.mo_coeff[1][:,2:7]))
        eri2 = mc1.get_h2eff((m.mo_coeff[0][:,2:7], m.mo_coeff[1][:,2:7]))
        self.assertAlmostEqual(abs(eri1[0]-eri2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(eri1[1]-eri2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(eri1[2]-eri2[2]).max(), 0, 12)

    def test_get_veff(self):
        mc1 = mcscf.UCASCI(m.to_rks(), 5, (4,2))
        nao = mol.nao_nr()
        dm = numpy.random.random((2,nao,nao))
        dm = dm + dm.transpose(0,2,1)
        veff1 = mc1.get_veff(mol, dm)
        veff2 = m.get_veff(mol, dm)
        self.assertAlmostEqual(float(abs(veff1-veff2).max()), 0, 12)

    def test_make_rdm1(self):
        dm1 = mc.make_rdm1()
        self.assertAlmostEqual(lib.fp(dm1), -5.0290089869374492, 5)
        dm1 = mc.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(lib.fp(dm1[0]), -5.7326112327013377, 5)
        self.assertAlmostEqual(lib.fp(dm1[1]), 0.70360224576388797, 5)
        self.assertAlmostEqual(lib.fp(dm1[0]+dm1[1]), -5.0290089869374492, 5)

    def test_multi_roots_spin_square(self):
        mc = mcscf.UCASCI(m, 5, (4,2))
        mc.fcisolver.nroots = 2
        mc.natorb = True
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot[0], -75.73319012518794, 7)
        self.assertAlmostEqual(mc.e_tot[1], -75.63476344994703, 7)

        ss, s1 = mc.spin_square()
        self.assertAlmostEqual(ss[0], 2.005756795092406, 7)
        self.assertAlmostEqual(ss[1], 2.006105024567947, 7)
        self.assertAlmostEqual(s1[0], 3.003835411664498, 7)
        self.assertAlmostEqual(s1[1], 3.004067259278958, 7)

        dm1 = mc.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), -5.7326112327013377, 5)
        self.assertAlmostEqual(lib.fp(dm1[1]), 0.70360224576388797, 5)
        self.assertAlmostEqual(lib.fp(dm1[0]+dm1[1]), -5.0290089869374492, 5)

    #TODO:
    #def test_cas_natorb(self):
    #    mc1 = mcscf.UCASCI(m, 5, (4,2))
    #    mc1.sorting_mo_energy = True
    #    mc1.kernel()
    #    mo0 = mc1.mo_coeff
    #    ci0 = mc1.ci
    #    self.assertAlmostEqual(mc1.e_tot, -75.733190125187946, 9)

    #    mc1.ci = None  # Force cas_natorb_ to recompute CI coefficients
    #    mc1.cas_natorb_()
    #    mo1 = mc1.mo_coeff
    #    ci1 = mc1.ci
    #    s = numpy.einsum('pi,pq,qj->ij', mo0[:,5:9], msym.get_ovlp(), mo1[:,5:9])
    #    self.assertAlmostEqual(fci.addons.overlap(ci0, ci1, 4, 4, s), 1, 9)

    def test_sort_mo(self):
        mc1 = mcscf.UCASCI(msym, 5, (4,2))
        mo = mc1.sort_mo([[3,5,6,4,7],[3,4,5,6,7]])
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, mc.e_tot, 9)


if __name__ == "__main__":
    print("Full Tests for CASCI")
    unittest.main()
