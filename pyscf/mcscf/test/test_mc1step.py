#!/usr/bin/env python
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

import unittest
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import fci
from pyscf import mcscf

def setUpModule():
    global mol, molsym, m, msym, mc0
    b = 1.4
    mol = gto.M(
    verbose = 0,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': '631g', },
    )
    m = scf.RHF(mol)
    m.conv_tol = 1e-10
    m.chkfile = tempfile.NamedTemporaryFile().name
    m.scf()
    mc0 = mcscf.CASSCF(m, 4, 4).run()

    molsym = gto.M(
    verbose = 0,
    output = '/dev/null',
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': '631g', },
    symmetry = True
    )
    msym = scf.RHF(molsym)
    msym.chkfile = tempfile.NamedTemporaryFile().name
    msym.conv_tol = 1e-10
    msym.scf()

def tearDownModule():
    global mol, molsym, m, msym, mc0
    mol.stdout.close()
    molsym.stdout.close()
    del mol, molsym, m, msym, mc0


class KnownValues(unittest.TestCase):
    def test_with_x2c_scanner(self):
        mc1 = mcscf.CASSCF(m, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1 = mcscf.CASSCF(m, 4, 4).x2c().as_scanner().as_scanner()
        mc1(mol)
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1('N 0 0 0; N 0 0 1.1')
        self.assertAlmostEqual(mc1.e_tot, -109.02535605303684, 7)

    def test_mc1step_symm_with_x2c_scanner(self):
        mc1 = mcscf.CASSCF(msym, 4, 4).x2c().run()
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1 = mcscf.CASSCF(msym, 4, 4).x2c().as_scanner().as_scanner()
        mc1(molsym)
        self.assertAlmostEqual(mc1.e_tot, -108.91497905985173, 7)

        mc1('N 0 0 0; N 0 0 1.1')
        self.assertAlmostEqual(mc1.e_tot, -109.02535605303684, 7)

    def test_0core_0virtual(self):
        mol = gto.M(atom='He', basis='321g', verbose=0)
        mf = scf.RHF(mol).run()
        mc1 = mcscf.CASSCF(mf, 2, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.CASSCF(mf, 1, 2).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 1, 0).run()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 2, 2)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.850576699649737, 9)

        mc1 = mcscf.CASSCF(mf, 1, 2)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

        mc1 = mcscf.CASSCF(mf, 1, 0)
        mc1.mc2step()
        self.assertAlmostEqual(mc1.e_tot, -2.8356798736405673, 9)

    def test_cas_natorb(self):
        mc1 = mcscf.CASSCF(msym, 4, 4, ncore=5)
        mo = mc1.sort_mo([4,5,10,13])
        mc1.sorting_mo_energy = True
        mc1.kernel(mo)
        mo0 = mc1.mo_coeff
        ci0 = mc1.ci
        self.assertAlmostEqual(mc1.e_tot, -108.7288793597413, 7)
        casdm1 = mc1.fcisolver.make_rdm1(mc1.ci, 4, 4)
        mc1.ci = None  # Force cas_natorb_ to recompute CI coefficients

        mc1.cas_natorb_(casdm1=casdm1, eris=mc1.ao2mo())
        mo1 = mc1.mo_coeff
        ci1 = mc1.ci
        s = numpy.einsum('pi,pq,qj->ij', mo0[:,5:9], msym.get_ovlp(), mo1[:,5:9])
        self.assertAlmostEqual(abs(fci.addons.overlap(ci0, ci1, 4, 4, s)), 1, 9)

    def test_get_h2eff(self):
        mc1 = mcscf.CASSCF(m, 4, 4)
        mc2 = mc1.approx_hessian()
        eri1 = mc1.get_h2eff(m.mo_coeff[:,5:9])
        eri2 = mc2.get_h2eff(m.mo_coeff[:,5:9])
        self.assertAlmostEqual(abs(eri1-eri2).max(), 0, 12)

        mc3 = mcscf.density_fit(mc1)
        eri3 = mc3.get_h2eff(m.mo_coeff[:,5:9])
        self.assertTrue(abs(eri1-eri3).max() > 1e-5)

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
        self.assertAlmostEqual(numpy.dot(mc1.e_states, [.5,.5]), -108.80445340617777, 8)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(lib.fp(mo_occ), 1.8748844779923917, 4)
        dm1 = mc1.analyze()
        self.assertAlmostEqual(lib.fp(dm1[0]), 2.6993157521103779, 4)
        self.assertAlmostEqual(lib.fp(dm1[1]), 2.6993157521103779, 4)

    def test_natorb(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.natorb = True
        mc1.conv_tol = 1e-10
        mc1.kernel(mo)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(mc1.e_tot, -105.83025103050596, 9)
        self.assertAlmostEqual(lib.fp(mo_occ), 2.4188178285392317, 4)

        mc1.mc2step(mo)
        mo_occ = lib.chkfile.load(mc1.chkfile, 'mcscf/mo_occ')[5:9]
        self.assertAlmostEqual(mc1.e_tot, -105.83025103050596, 9)
        self.assertAlmostEqual(lib.fp(mo_occ), 2.418822007439851, 4)

    def test_dep4(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.with_dep4 = True
        mc1.max_cycle = 1
        mc1.max_cycle_micro = 6
        mc1.fcisolver.pspace_size = 0
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82840377848402, 6)

    def test_dep4_df(self):
        mc1 = mcscf.CASSCF(msym, 4, 4).density_fit()
        mo = mc1.sort_mo_by_irrep({'A1u':2, 'A1g':2})
        mc1.with_dep4 = True
        mc1.max_cycle = 1
        mc1.max_cycle_micro = 6
        mc1.fcisolver.pspace_size = 0
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82833497389831, 6)

    # FIXME: How to test ci_response_space? The test below seems numerical instable
    #def test_ci_response_space(self):
    #    mc1 = mcscf.CASSCF(m, 4, 4)
    #    mc1.ci_response_space = 9
    #    mc1.max_cycle = 1
    #    mc1.max_cycle_micro = 2
    #    mc1.kernel()
    #    self.assertAlmostEqual(mc1.e_tot, -108.85920100433893, 8)

    #    mc1 = mcscf.CASSCF(m, 4, 4)
    #    mc1.ci_response_space = 1
    #    mc1.max_cycle = 1
    #    mc1.max_cycle_micro = 2
    #    mc1.kernel()
    #    self.assertAlmostEqual(mc1.e_tot, -108.85920400781617, 8)

    def test_chk(self):
        mc2 = mcscf.CASSCF(m, 4, 4)
        mc2.update(mc0.chkfile)
        mc2.max_cycle = 0
        mc2.kernel()
        self.assertAlmostEqual(mc0.e_tot, mc2.e_tot, 8)

    def test_grad(self):
        self.assertAlmostEqual(abs(mc0.get_grad()).max(), 0, 4)

    def test_external_fcisolver(self):
        fcisolver1 = fci.direct_spin1.FCISolver(mol)
        class FCI_as_DMRG(fci.direct_spin1.FCISolver):
            def __getattribute__(self, attr):
                """Prevent 'private' attribute access"""
                if attr in ('make_rdm1s', 'spin_square', 'contract_2e',
                            'absorb_h1e'):
                    raise AttributeError
                else:
                    return object.__getattribute__(self, attr)
            def kernel(self, *args, **kwargs):
                return fcisolver1.kernel(*args, **kwargs)
        mc1 = mcscf.CASSCF(m, 4, 4)
        mc1.fcisolver = FCI_as_DMRG(mol)
        mc1.natorb = True
        mc1.kernel()
        self.assertAlmostEqual(mc1.e_tot, -108.85974001740854, 8)
        dm1 = mc1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(lib.fp(dm1[0]), 5.33303, 4)

    def test_casci_in_casscf(self):
        mc1 = mcscf.CASSCF(m, 4, 4)
        e_tot, e_ci, fcivec = mc1.casci(mc1.mo_coeff)
        self.assertAlmostEqual(e_tot, -108.83741684447352, 7)

    def test_scanner(self):
        mc_scan = mcscf.CASSCF(scf.RHF(mol), 4, 4).as_scanner().as_scanner()
        mc_scan(mol)
        self.assertAlmostEqual(mc_scan.e_tot, -108.85974001740854, 8)

    def test_trust_region(self):
        mc1 = mcscf.CASSCF(msym, 4, 4)
        mc1.max_stepsize = 0.1
        mo = mc1.sort_mo_by_irrep({'A1u':3, 'A1g':1})
        mc1.ah_grad_trust_region = 0.3
        mc1.conv_tol = 1e-7
        mc1.fcisolver.pspace_size = 0
        tot_jk = []
        def count_jk(envs):
            tot_jk.append(envs.get('njk', 0))
        mc1.callback = count_jk
        mc1.kernel(mo)
        self.assertAlmostEqual(mc1.e_tot, -105.82941031838349, 8)
        self.assertEqual(tot_jk, [3,6,6,4,4,3,6,6,3,6,6,3,4,4,3,3,3,3,4,4])

    def test_with_ci_init_guess(self):
        mc2 = mcscf.CASSCF(msym, 4, 4)
        mc2.wfnsym = 'A1u'
        mc2.kernel()
        self.assertAlmostEqual(mc2.e_tot, -108.75147424827954, 8)

    def test_dump_chk(self):
        mcdic = lib.chkfile.load(mc0.chkfile, 'mcscf')
        with h5py.File(mc0.chkfile, 'r+') as f:
            del f['mcscf']

        mcscf.chkfile.dump_mcscf(mc0, **mcdic)
        with h5py.File(mc0.chkfile, 'r') as f:
            self.assertEqual(
                set(f['mcscf'].keys()),
                {'ncore', 'e_tot', 'mo_energy', 'casdm1', 'mo_occ', 'ncas', 'mo_coeff', 'e_cas'})

    def test_state_average1(self):
        mc = mcscf.CASSCF(m, 4, 4)
        mc.state_average_([0.5, 0.25, 0.25])
        mc.fcisolver.spin = 2
        mc.run()
        self.assertAlmostEqual(mc.e_states[0], -108.7513784239438, 6)
        self.assertAlmostEqual(mc.e_states[1], -108.6919327057737, 6)
        self.assertAlmostEqual(mc.e_states[2], -108.6919327057737, 6)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc = mcscf.CASCI(m, 4, 4)
        mc.state_average_([0.5, 0.25, 0.25])
        mc.fcisolver.spin = 2
        mc.kernel(mo_coeff=mo_coeff)
        self.assertAlmostEqual(mc.e_states[0], -108.7513784239438, 6)
        self.assertAlmostEqual(mc.e_states[1], -108.6919327057737, 6)
        self.assertAlmostEqual(mc.e_states[2], -108.6919327057737, 6)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 7)
        # Second and third root are degenerated
        #self.assertAlmostEqual(abs((civec[1]*mc.ci[1]).sum()), 1, 7)

    def test_state_average_mix(self):
        mc = mcscf.CASSCF(m, 4, 4)
        cis1 = mc.fcisolver.copy()
        cis1.spin = 2
        mc = mcscf.addons.state_average_mix(mc, [cis1, mc.fcisolver], [.5, .5])
        mc.run()
        self.assertAlmostEqual(mc.e_states[0], -108.7506795311190, 5)
        self.assertAlmostEqual(mc.e_states[1], -108.8582272809495, 5)

        mc.analyze()
        mo_coeff, civec, mo_occ = mc.cas_natorb(sort=True)

        mc.kernel(mo_coeff=mo_coeff, ci0=civec)
        self.assertAlmostEqual(mc.e_states[0], -108.7506795311190, 5)
        self.assertAlmostEqual(mc.e_states[1], -108.8582272809495, 5)
        self.assertAlmostEqual(abs((civec[0]*mc.ci[0]).sum()), 1, 7)
        self.assertAlmostEqual(abs((civec[1]*mc.ci[1]).sum()), 1, 7)

    def test_small_system(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 .74', symmetry=True, basis='6-31g', verbose=0)
        mf = scf.RHF(mol).run()
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.max_cycle = 5
        mc.mc2step()
        self.assertAlmostEqual(mc.e_tot, -1.14623442196547, 9)
        self.assertTrue(mc.converged)

    def test_mcscf_without_initializing_scf(self):
        mc = mcscf.CASSCF(mol.RHF(), 4, 4)
        mc.kernel(m.mo_coeff)
        self.assertAlmostEqual(mc.e_tot, -108.85974001740854, 7)
        mc.analyze()


if __name__ == "__main__":
    print("Full Tests for mc1step")
    unittest.main()
