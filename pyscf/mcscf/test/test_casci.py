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

import copy
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
verbose = 7,
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
verbose = 7,
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
        mc1 = mcscf.CASCI(m, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.89255507863562, 9)

        mc1 = mcscf.CASCI(m, 4, 4).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -108.89264146901512, 8)

    def test_fix_spin_(self):
        mc1 = mcscf.CASCI(m, 4, 4)
        mc1.fix_spin_(ss=2).run()
        self.assertAlmostEqual(mc1.e_tot, -108.03741684418183, 9)

        mc1.fix_spin_(ss=0).run()
        self.assertAlmostEqual(mc1.e_tot, -108.83741684447352, 9)

    def test_cas_natorb(self):
        mc1 = mcscf.CASCI(msym, 4, 4, ncore=5)
        mo = mc1.sort_mo([4,5,10,13])
        mc1.sorting_mo_energy = True
        mc1.kernel(mo)
        mo0 = mc1.mo_coeff
        ci0 = mc1.ci
        self.assertAlmostEqual(mc1.e_tot, -108.71841138528966, 9)

        mo2, ci2, mo_e = mc1.canonicalize(sort=False, cas_natorb=True,
                                          with_meta_lowdin=False, verbose=7)

        casdm1 = mc1.fcisolver.make_rdm1(mc1.ci, 4, 4)
        mc1.ci = None  # Force cas_natorb_ to recompute CI coefficients
        mc1.cas_natorb_(casdm1=casdm1, sort=True, with_meta_lowdin=True)
        mo1 = mc1.mo_coeff
        ci1 = mc1.ci
        s = numpy.einsum('pi,pq,qj->ij', mo0[:,5:9], msym.get_ovlp(), mo1[:,5:9])
        self.assertAlmostEqual(fci.addons.overlap(ci0, ci1, 4, 4, s), 1, 9)

        self.assertAlmostEqual(float(abs(mo1-mo2).max()), 0, 9)
        self.assertAlmostEqual(ci1.ravel().dot(ci2.ravel()), 1, 9)

    def test_multi_roots(self):
        mc1 = mcscf.CASCI(m, 4, 4)
        mc1.fcisolver.nroots = 2
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot[0], -108.83741684447352, 9)
        self.assertAlmostEqual(mc1.e_tot[1], -108.72522194135604, 9)
        dm1 = mc1.analyze()
        self.assertAlmostEqual(lib.finger(dm1[0]), 2.6252082970845532, 7)
        self.assertAlmostEqual(lib.finger(dm1[1]), 2.6252082970845532, 7)

    def test_external_fcisolver(self):
        class FCI_as_DMRG(fci.direct_spin1.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
        mc1 = mcscf.CASCI(m, 4, 4)
        mc1.fcisolver = FCI_as_DMRG(mol)
        mc1.fcisolver.nroots = 2
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot[0], -108.83741684447352, 9)
        self.assertAlmostEqual(mc1.e_tot[1], -108.72522194135604, 9)
        dm1 = mc1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(lib.finger(dm1[0]), 2.6252082970845532*2, 7)

    def test_get_h2eff(self):
        mc1 = mcscf.approx_hessian(mcscf.CASCI(m, 4, 4))
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertAlmostEqual(abs(eri1-eri2).max(), 0, 12)

        mc1 = mcscf.density_fit(mcscf.CASCI(m, 4, 4))
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc1.get_h2cas(m.mo_coeff[:,5:9])
        self.assertTrue(abs(eri1-eri2).max() > 1e-5)

    def test_get_veff(self):
        mf = m.view(dft.rks.RKS)
        mc1 = mcscf.CASCI(mf, 4, 4)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        dm = dm + dm.T
        veff1 = mc1.get_veff(mol, dm)
        veff2 = m.get_veff(mol, dm)
        self.assertAlmostEqual(abs(veff1-veff2).max(), 0, 12)

    def test_with_ci_init_guess(self):
        mc1 = mcscf.CASCI(msym, 4, 4)
        ci0 = numpy.zeros((6,6))
        ci0[0,1] = 1
        mc1.kernel(ci0=ci0)

        mc2 = mcscf.CASCI(msym, 4, 4)
        mc2.wfnsym = 'A1u'
        mc2.kernel()
        self.assertAlmostEqual(mc1.e_tot, mc2.e_tot, 9)

    def test_slight_symmetry_broken(self):
        mf = copy.copy(msym)
        mf.mo_coeff = numpy.array(msym.mo_coeff)
        u = numpy.linalg.svd(numpy.random.random((4,4)))[0]
        mf.mo_coeff[:,5:9] = mf.mo_coeff[:,5:9].dot(u)
        mc1 = mcscf.CASCI(mf, 4, 4)
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot, -108.83741684445798, 9)

    def test_sort_mo(self):
        mc1 = mcscf.CASCI(msym, 4, 4)
        mo = mc1.sort_mo_by_irrep({'A1u':3, 'A1g':1})
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82542805259033, 9)

    def test_state_average(self):
        mc = mcscf.CASCI(m, 4, 4)
        mc.state_average_([0.5, 0.25, 0.25])
        mc.fcisolver.spin = 2
        mc.run()
        self.assertAlmostEqual(mc.e_tot[0], -108.72522194135607, 9)
        self.assertAlmostEqual(mc.e_tot[1], -108.67148843338228, 9)
        self.assertAlmostEqual(mc.e_tot[2], -108.67148843338228, 9)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc.kernel(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mc.e_tot[0], -108.72522194135607, 9)
        self.assertAlmostEqual(mc.e_tot[1], -108.67148843338228, 9)
        #FIXME: with the initial guess from mc, FCI solver may converge to
        # another state
        #self.assertAlmostEqual(mc.e_tot[2], -108.67148843338228, 9)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 9)
        # Second and third root are degenerated
        #self.assertAlmostEqual(abs((civec[1]*mc.ci[1]).sum()), 1, 9)

    def test_state_average_mix(self):
        mc = mcscf.CASCI(m, 4, 4)
        cis1 = copy.copy(mc.fcisolver)
        cis1.spin = 2
        cis1.nroots = 3
        mc = mcscf.addons.state_average_mix(mc, [cis1, mc.fcisolver], [.25, .25, .25, .25])
        mc.run()
        self.assertAlmostEqual(mc.e_tot[0], -108.72522194135607, 9)
        self.assertAlmostEqual(mc.e_tot[1], -108.67148843338228, 9)
        self.assertAlmostEqual(mc.e_tot[2], -108.67148843338228, 9)
        self.assertAlmostEqual(mc.e_tot[3], -108.83741684447352, 9)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc.kernel(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mc.e_tot[0], -108.72522194135607, 9)
        self.assertAlmostEqual(mc.e_tot[1], -108.67148843338228, 9)
        #FIXME: with the initial guess from mc, FCI solver may converge to
        # another state
        #self.assertAlmostEqual(mc.e_tot[2], -108.67148843338228, 9)
        self.assertAlmostEqual(mc.e_tot[3], -108.83741684447352, 9)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 9)
        self.assertAlmostEqual(abs((civec[3]*mc.ci[3]).sum()), 1, 9)


if __name__ == "__main__":
    print("Full Tests for CASCI")
    unittest.main()

