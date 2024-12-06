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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import unittest
import tempfile
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.scf import atom_hf

import sys
try:
    import dftd3
except ImportError:
    pass

try:
    import dftd4
except ImportError:
    pass

def setUpModule():
    global mol, mf, n2sym, n2mf, re_ecp1, re_ecp2
    mol = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
    O     0    0        0
    H     0    -0.757   0.587
    H     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.chkfile = tempfile.NamedTemporaryFile().name
    mf.kernel()

    n2sym = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = '''
            N     0    0    0
            N     0    0    1''',
        symmetry = 1,
        basis = 'cc-pvdz')
    n2mf = scf.RHF(n2sym).set(conv_tol=1e-10).run()

    re_basis = '''
    Re    S
         30.4120000              0.0229870             -0.0072090              0.0008190             -0.0059360
         19.0142000             -0.1976360              0.0645740              0.0437380             -0.0062770
         11.8921000              0.5813760             -0.1963520             -0.1772190              0.0754730
          7.4336100             -0.3840040              0.1354270              0.0197240              0.0512310
          4.1811300             -0.6002800              0.2175540              0.5658650             -0.5040200
          1.1685700              0.8264270             -0.3902690             -1.1960450              1.6252170
          0.5539770              0.5102290             -0.3723360             -0.0926010             -1.6511890
          0.1707810              0.0365510              0.2672200              2.2639340              0.1479060
          0.0792870             -0.0087660              0.6716710             -1.0605870              0.9643860
    Re    S
          1.5774000              1.0000000
    Re    P
         18.3488000             -0.0168770              0.0046080              0.0073690              0.0150340
         11.4877000              0.0929150             -0.0268720             -0.0437140             -0.0837840
          5.4325600             -0.2914010              0.0889710              0.1463200              0.2804860
          1.3785900              0.4982270             -0.1794810             -0.3009630             -0.8342650
          0.6887660              0.4796070             -0.2111050             -0.4000020             -0.4202230
          0.3380830              0.1752920             -0.0043780              0.1900900              1.7104440
          0.1491600              0.0173410              0.3822050              0.7604950             -0.3011560
          0.0643210              0.0013480              0.5438540              0.2562560             -0.7920800
    Re    P
          2.0752000              1.0000000
    Re    D
         13.8248000             -0.0010140              0.0014410             -0.0008920
          8.6358600              0.0200690             -0.0233020              0.0318260
          5.3969500             -0.0709260              0.0813920             -0.1289560
          1.4943200              0.2136340             -0.2826220              0.6891180
          0.7134790              0.3740960             -0.4844350              0.2530110
          0.3249100              0.3691180              0.0772300             -1.1068680
          0.1398360              0.2324990              0.6339580              0.1625610
    Re    D
          2.5797000              1.0000000
    Re    F
          1.6543000              1.0000000
    Re    G
          1.8871000              1.0000000
          '''
    re_ecp1 = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = 'Re',
        spin = None,
        ecp = 'lanl2dz',
        basis = re_basis)
    re_ecp2 = gto.M(
        verbose = 7,
        output = '/dev/null',
        atom = 'Re',
        spin = None,
        ecp = {'Re': gto.basis.load_ecp('lanl2dz', 'Zn')},
        basis = re_basis)

def tearDownModule():
    global mol, mf, n2sym, n2mf, re_ecp1, re_ecp2
    mol.stdout.close()
    re_ecp1.stdout.close()
    re_ecp2.stdout.close()
    n2sym.stdout.close()
    del mol, mf, n2sym, n2mf, re_ecp1, re_ecp2


class KnownValues(unittest.TestCase):
    def test_init_guess_minao(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H1    0    -0.757   0.587
        H2    0    0.757    0.587''',
            basis = 'ccpvdz',
        )
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(lib.fp(dm), 2.5912875957299684, 9)

        mol1 = gto.M(atom='Mo', basis='lanl2dz', ecp='lanl2dz',
                     verbose=7, output='/dev/null')
        dm = scf.hf.get_init_guess(mol1, key='minao')
        self.assertAlmostEqual(lib.fp(dm), 2.0674886928183507, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, mol1.intor('int1e_ovlp')), 14, 9)

        mol1.basis = 'sto3g'
        mol1.build(0, 0)
        dm = scf.hf.get_init_guess(mol1, key='minao')
        self.assertAlmostEqual(lib.fp(dm), 1.3085066548762425, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, mol1.intor('int1e_ovlp')), 13.60436071945, 7)
        mol1.stdout.close()

        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost-H'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.spin = 1
        mol.build(0, 0)
        dm = scf.hf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(lib.fp(dm), 2.9572305128956238, 9)

    def test_init_guess_minao_with_ecp(self):
        s = re_ecp1.intor('int1e_ovlp')
        dm = scf.hf.get_init_guess(re_ecp1, key='minao')
        self.assertAlmostEqual(lib.fp(dm), -8.0310571101329202, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, s), 15, 9)

        dm = scf.hf.get_init_guess(re_ecp2, key='minao')
        self.assertAlmostEqual(lib.fp(dm), -9.0532680910696772, 9)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, s), 31.804465975542513, 9)

        mol = gto.M(atom=[['Mg',(0, 0, 0)], ['!Mg',(0, 1, 0)], ['O',(0, 0, 1)]],
                    basis={'O': 'sto3g', 'Mg': 'sto3g'},
                    ecp='lanl2dz')
        dm = mol.RHF().init_guess_by_minao()
        self.assertAlmostEqual(lib.fp(dm), -0.8255591441786179, 9)

    def test_init_guess_atom(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H1    0    -0.757   0.587
        H2    0    0.757    0.587''',
            basis = 'ccpvdz',
        )
        dm = scf.hf.get_init_guess(mol, key='atom')
        self.assertAlmostEqual(lib.fp(dm), 2.78218274161741, 7)

        dm = scf.ROHF(mol).init_guess_by_atom()
        self.assertAlmostEqual(lib.fp(dm[0]), 2.78218274161741/2, 7)

        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost-H'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.spin = 1
        mol.build(0, 0)
        dm = scf.hf.get_init_guess(mol, key='atom')
        self.assertAlmostEqual(lib.fp(dm), 3.0813279501879838, 7)

        mol.basis = {'h': '3-21g'}
        mol.build(0, 0)
        dm = scf.hf.get_init_guess(mol, key='atom')
        self.assertEqual(dm.shape, (4, 4))
        self.assertEqual(abs(dm[:2,:2]).max(), 0)
        self.assertAlmostEqual(lib.fp(dm), -0.47008362287778827, 7)

    def test_init_guess_atom_with_ecp(self):
        s = re_ecp1.intor('int1e_ovlp')
        dm = scf.hf.get_init_guess(re_ecp1, key='atom')
        self.assertAlmostEqual(lib.fp(dm), -4.822111004225718, 6)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, s), 15, 9)

        dm = scf.hf.get_init_guess(re_ecp2, key='atom')
        self.assertAlmostEqual(lib.fp(dm), -14.083500177270547, 6)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, s), 57, 9)

    def test_atom_hf_with_ecp(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom  = 'Cu 0 0 0; Ba 0 0 2',
            basis = {'Ba': 'def2-svp', 'Cu': 'lanl2dz'},
            ecp   = {'Ba':'def2-svp', 'Cu': 'lanl2dz' }, spin=None)
        scf_result = atom_hf.get_atm_nrhf(mol)
        self.assertAlmostEqual(scf_result['Ba'][0], -25.07089468572715, 9)
        self.assertAlmostEqual(scf_result['Cu'][0], -194.92388639203045, 9)

    def test_init_guess_chk(self):
        dm = mol.HF(chkfile=tempfile.NamedTemporaryFile().name).get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.fp(dm), 2.5912875957299684, 5)

        dm = mf.get_init_guess(mol, key='chkfile')
        self.assertAlmostEqual(lib.fp(dm), 3.2111753674560535, 5)

    def test_init_guess_huckel(self):
        dm = scf.hf.RHF(mol).get_init_guess(mol, key='huckel')
        self.assertAlmostEqual(lib.fp(dm), 3.348165771345748, 5)

        dm = scf.ROHF(mol).init_guess_by_huckel()
        self.assertAlmostEqual(lib.fp(dm[0]), 3.348165771345748/2, 5)

        # Initial guess Huckel is not able to handle open-shell system
        mol1 = gto.M(atom='Mo 0 0 0; C 0 0 1', basis='lanl2dz', ecp='lanl2dz',
                     verbose=7, output='/dev/null')
        dm = scf.hf.get_init_guess(mol1, key='huckel')
        self.assertAlmostEqual(lib.fp(dm), 2.01095497354225, 5)
        self.assertAlmostEqual(numpy.einsum('ij,ji->', dm, mol1.intor('int1e_ovlp')), 20, 9)

    def test_init_guess_huckel(self):
        dm = scf.hf.RHF(mol).get_init_guess(mol, key='mod_huckel')
        self.assertAlmostEqual(lib.fp(dm), 3.233072986208057, 5)

        dm = scf.ROHF(mol).init_guess_by_mod_huckel()
        self.assertAlmostEqual(lib.fp(dm[0]), 3.233072986208057/2, 5)

    def test_init_guess_sap(self):
        mol = gto.M(
            verbose = 7,
            output = '/dev/null',
            atom = '''
        O     0    0        0
        H1    0    -0.757   0.587
        H2    0    0.757    0.587''',
            basis = 'ccpvdz',
        )
        dm = scf.hf.RHF(mol).get_init_guess(mol, key='sap')
        self.assertAlmostEqual(lib.fp(dm), 4.2267871571567195, 5)

        dm = scf.ROHF(mol).get_init_guess(mol, key='sap')
        self.assertAlmostEqual(lib.fp(dm[0]), 4.2267871571567195/2, 7)

    def test_1e(self):
        mf = scf.rohf.HF1e(mol)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

        mf = scf.RHF(gto.M(atom='H', spin=1))
        self.assertAlmostEqual(mf.kernel(), -0.46658184955727555, 9)

    def test_1e_symm(self):
        molsym = gto.M(
            atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            basis = 'cc-pvdz',
            symmetry = 1,
        )
        mf = scf.hf_symm.HF1e(molsym)
        self.assertAlmostEqual(mf.scf(), -23.867818585778764, 9)

    def test_get_grad(self):
        g = mf.get_grad(mf.mo_coeff, mf.mo_occ)
        self.assertAlmostEqual(abs(g).max(), 0, 6)

    def test_input_diis(self):
        adiis = scf.hf.diis.ADIIS(mol)
        mf1 = scf.RHF(mol)
        mf1.DIIS = scf.hf.diis.ADIIS
        mf1.max_cycle = 4
        eref = mf1.kernel()

        mf1 = scf.RHF(mol)
        mf1.diis = adiis
        mf1.max_cycle = 1
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, -75.987815719969291, 9)

        mf1.max_cycle = 3
        e2 = mf1.kernel()
        self.assertAlmostEqual(e2, eref, 9)

    def test_energy_tot(self):
        e = n2mf.energy_tot(n2mf.make_rdm1())
        self.assertAlmostEqual(e, n2mf.e_tot, 9)

        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        e = mf.energy_elec(dm)[0]
        self.assertAlmostEqual(e, -59.332199154299914, 9)

    def test_mulliken_pop(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        pop, chg = mf.mulliken_pop(mol, dm)
        self.assertAlmostEqual(abs(pop).sum(), 22.941032799355845, 7)
        pop, chg = mf.mulliken_pop(mol, [dm*.5]*2)
        self.assertAlmostEqual(abs(pop).sum(), 22.941032799355845, 7)

        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='ano')
        self.assertAlmostEqual(abs(pop).sum(), 22.048073484937646, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='minao')
        self.assertAlmostEqual(abs(pop).sum(), 22.098274261783196, 7)
        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, dm, pre_orth_method='scf')
        self.assertAlmostEqual(abs(pop).sum(), 22.15309316506852, 7)

        pop, chg = mf.mulliken_pop_meta_lowdin_ao(mol, [dm*.5]*2, pre_orth_method='ano')
        self.assertAlmostEqual(abs(pop).sum(), 22.048073484937646, 7)

    def test_analyze(self):
        popandchg, dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 4.0049440587033116, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 2.0584447549532596, 6)
        popandchg, dip = mf.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.2031790129016922, 6)

        mf1 = mf.view(scf.rohf.ROHF)
        popandchg, dip = mf1.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 4.0049440587033116, 6)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.2031790129016922, 6)

        mf1 = n2mf.copy()
        (pop, chg), dip = n2mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.5467414321488357, 6)
        self.assertAlmostEqual(numpy.linalg.norm(dip), 0, 9)
        mf1 = n2mf.copy()
        mf1.mo_coeff = numpy.array(n2mf.mo_coeff)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.8893148995392353, 6)

        mf1 = n2mf.view(scf.hf_symm.ROHF)
        mf1.mo_coeff = numpy.array(n2mf.mo_coeff)
        popandchg, dip = mf1.analyze(with_meta_lowdin=False)
        self.assertAlmostEqual(numpy.linalg.norm(popandchg[0]), 3.8893148995392353, 6)

    def test_scf(self):
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    @unittest.skipIf('dispersion' not in sys.modules, "requires the dftd3 library")
    def test_scf_d3(self):
        mf = scf.RHF(mol)
        mf.disp = 'd3bj'
        mf.conv_tol = 1e-10
        mf.chkfile = None
        e_tot = mf.kernel()
        self.assertAlmostEqual(e_tot, -76.03127458778653, 9)

    @unittest.skipIf('dispersion' not in sys.modules, "requires the dftd4 library")
    def test_scf_d4(self):
        mf = scf.RHF(mol)
        mf.disp = 'd4'
        mf.conv_tol = 1e-10
        mf.chkfile = None
        e_tot = mf.kernel()
        self.assertAlmostEqual(e_tot, -76.0277467547733, 9)

    def test_scf_negative_spin(self):
        mol = gto.M(atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            basis = '6-31g',
            spin = -2,
        )
        mf = scf.ROHF(mol).run(conv_tol=1e-10)
        self.assertAlmostEqual(mf.mo_occ[4], 1, 14)
        self.assertAlmostEqual(mf.e_tot, -75.723654936331599, 9)

        mol = gto.M(atom = '''
        O     0    0        0
        H     0    -0.757   0.587
        H     0    0.757    0.587''',
            symmetry = 1,
            basis = '6-31g',
            spin = -2,
        )
        mf = scf.ROHF(mol).set(conv_tol=1e-10)
        mf.irrep_nelec = {'A1': (2, 3)}
        mf.run()
        self.assertAlmostEqual(mf.mo_occ[4], 1, 14)
        self.assertAlmostEqual(mf.e_tot, -75.561433366671935, 9)

    def test_nr_rhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.RHF(pmol).run()
        self.assertAlmostEqual(mf.e_tot, -76.027107008870573, 9)

    def test_nr_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.rohf.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)

    def test_damping(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        f = scf.hf.get_hcore(mol)
        df  = numpy.random.rand(nao,nao)
        df += df.T
        f_prev = f + df
        damp = 0.3
        f_damp = scf.hf.get_fock(mf, h1e=0, s1e=0, vhf=f, dm=0, cycle=0,
                                 diis_start_cycle=2, damp_factor=damp, fock_last=f_prev)
        self.assertAlmostEqual(abs(f_damp - (f*(1-damp) + f_prev*damp)).max(), 0, 9)

    def test_level_shift(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        s = scf.hf.get_ovlp(mol)
        d = numpy.random.random((nao,nao))
        d = d + d.T
        f = scf.hf.level_shift(s, d, scf.hf.get_hcore(mol), .5)
        self.assertAlmostEqual(numpy.linalg.norm(f), 94.230157719053565, 9)

    def test_get_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        d1 = numpy.random.random((nao,nao))
        d2 = numpy.random.random((nao,nao))
        d = (d1+d1.T, d2+d2.T)
        v = scf.hf.get_veff(mol, d)
        self.assertAlmostEqual(numpy.linalg.norm(v), 199.66041114502335, 9)

        pmol = gto.Mole()
        pmol.atom = '''
O     0    0        0
H     0    -0.757   0.587
H     0    0.757    0.587'''
        pmol.basis = '6-31g'
        pmol.cart = True

        mf1 = scf.hf.SCF(pmol)
        mf1.direct_scf = True
        mf1.max_memory = 0
        nao = pmol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,3,nao,nao)) - .5 + 0j
        vhf2 = mf1.get_veff(pmol, dm[0,0], hermi=0)
        self.assertEqual(vhf2.ndim, 2)

        vhf3 = mf1.get_veff(pmol, dm[0], hermi=0)
        self.assertEqual(vhf3.ndim, 3)
        self.assertAlmostEqual(abs(vhf3[0]-vhf2).max(), 0, 12)

        vhf4 = mf1.get_veff(pmol, dm, hermi=0)
        self.assertEqual(vhf4.ndim, 4)
        self.assertAlmostEqual(lib.fp(vhf4), 4.9026999849223287, 12)
        self.assertAlmostEqual(abs(vhf4[0]-vhf3).max(), 0, 12)

    def test_hf_symm(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        self.assertAlmostEqual(mf.scf(), -76.026765673119627, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 4.0049439389172425, 6)

    def test_hf_symm_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.hf_symm.RHF(pmol)
        mf.irrep_nelec = {'B2':4}
        self.assertAlmostEqual(mf.scf(), -75.074736446470723, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.9779576643902912, 6)

    def test_hf_symm_rohf(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -75.627354109594179, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.6783793407635832, 6)

    def test_hf_symm_rohf_fixnocc(self):
        pmol = mol.copy()
        pmol.symmetry = 1
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.hf_symm.ROHF(pmol)
        mf.irrep_nelec = {'B2':(2,1)}
        self.assertAlmostEqual(mf.scf(), -75.008317646307404, 9)
        (pop, chg), dip = mf.analyze()
        self.assertAlmostEqual(numpy.linalg.norm(pop), 3.7873920690764575, 6)

    def test_n2_symm(self):
        mf = scf.hf_symm.RHF(n2sym)
        self.assertAlmostEqual(mf.scf(), -108.9298383856092, 9)

    def test_n2_symm_rohf(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.hf_symm.ROHF(pmol)
        self.assertAlmostEqual(mf.scf(), -108.33899076078299, 9)

    def test_n2_symm_fixnocc(self):
        mf = scf.hf_symm.RHF(n2sym)
        mf.irrep_nelec = {'A1g':8, 'A1u':2, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -106.52905502298771, 9)

    def test_n2_symm_rohf_fixnocc(self):
        pmol = n2sym.copy()
        pmol.charge = 1
        pmol.spin = 1
        mf = scf.hf_symm.ROHF(pmol)
        mf.irrep_nelec = {'A1g':6, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertAlmostEqual(mf.scf(), -108.21954550790898, 9)

    def test_dot_eri_dm(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        j0, k0 = scf.hf.dot_eri_dm(mf._eri, dm+dm.T, hermi=0)
        j1, k1 = scf.hf.dot_eri_dm(mf._eri, dm+dm.T, hermi=1)
        self.assertTrue(numpy.allclose(j0,j1))
        self.assertTrue(numpy.allclose(k0,k1))
        j1, k1 = scf.hf.dot_eri_dm(mf._eri, dm, hermi=0)
        self.assertAlmostEqual(numpy.linalg.norm(j1), 77.035779188661465, 9)
        self.assertAlmostEqual(numpy.linalg.norm(k1), 46.253491700647963, 9)

    def test_ghost_atm_meta_lowdin(self):
        mol = gto.Mole()
        mol.atom = [["O" , (0. , 0.     , 0.)],
                    ['ghost'   , (0. , -0.757, 0.587)],
                    [1   , (0. , 0.757 , 0.587)] ]
        mol.verbose = 0
        mol.spin = 1
        mol.symmetry = True
        mol.basis = {'O':'ccpvdz', 'H':'ccpvdz',
                     'GHOST': gto.basis.load('ccpvdz','H')}
        mol.build()
        mf = scf.RHF(mol)
        self.assertAlmostEqual(mf.kernel(), -75.393287998638741, 9)

    def test_rhf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=10)
        mf = scf.hf.RHF(mol)
        energy = numpy.array([-10, -1, 1, -2, 0, -3])
        self.assertTrue(numpy.allclose(mf.get_occ(energy), [2, 2, 0, 2, 2, 2]))
        mol.stdout.close()

    def test_rhf_symm_get_occ(self):
        mf = scf.RHF(n2sym).set(verbose = 0)
        orbsym = numpy.array([0 , 5, 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5])
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        mo_coeff = lib.tag_array(numpy.eye(energy.size), orbsym=orbsym)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))

        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [0, 2, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))

    def test_rohf_get_occ(self):
        mol = gto.M(verbose=7, output='/dev/null').set(nelectron=8, spin=2)
        mf = scf.rohf.ROHF(mol)
        energy = numpy.array([-10, -1, 1, -2, 0, -3])
        self.assertTrue(numpy.allclose(mf.get_occ(energy), [2, 1, 0, 2, 1, 2]))
        pmol = n2sym.copy()
        pmol.spin = 2
        pmol.symmetry = False
        mf = scf.rohf.ROHF(pmol).set(verbose = 0)
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [0, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        # 0 virtual
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [2, 2, 1, 2, 2, 2, 2, 1]))
        # 0 core
        mf.nelec = (14, 0)
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51])
        self.assertTrue(numpy.allclose(mf.get_occ(energy),
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        mol.stdout.close()

    def test_rohf_symm_get_occ(self):
        pmol = n2sym.copy()
        pmol.charge = 0
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0)
        orbsym = numpy.array([0 , 5, 0 , 5 , 6 , 7 , 0 , 2 , 3 , 5 , 0 , 6 , 7 , 0 , 2 , 3 , 5 , 10, 11, 5])
        energy = numpy.array([34, 2, 54, 43, 42, 33, 20, 61, 29, 26, 62, 52, 13, 51, 18, 78, 85, 49, 84, 7])
        mo_coeff = lib.tag_array(numpy.eye(energy.size), orbsym=orbsym)
        mf.irrep_nelec = {'A1g':7, 'A1u':3, 'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [2, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1]))
        mf.irrep_nelec = {'E1ux':2, 'E1uy':2}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 2, 0, 2, 0, 1, 1, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))
        mf.irrep_nelec = {}
        self.assertTrue(numpy.allclose(mf.get_occ(energy, mo_coeff),
                [0, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2]))

        mf1 = scf.RHF(mol).set(verbose=0).view(scf.hf_symm.ROHF)
        self.assertRaises(RuntimeError, mf1.get_occ, energy, mo_coeff)

    def test_get_occ_extreme_case(self):
        mol = gto.M(atom='He', verbose=7, output='/dev/null')
        mf = scf.RHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, -2.8077839575399737, 12)

        mol.charge = 2
        mf = scf.RHF(mol).run()
        self.assertAlmostEqual(mf.e_tot, 0, 12)
        mol.stdout.close()

    def test_rohf_symm_dump_flags(self):
        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0)
        mf.irrep_nelec = {'A1g':6, 'A1u':4, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

        mf.irrep_nelec = {'A1g':6, 'A1u':10, 'E1ux':2, 'E1uy':2}
        self.assertRaises(ValueError, mf.build)

        mf.irrep_nelec = {'A1g':(4,2), 'A1u': (2, 4)}
        self.assertRaises(ValueError, mf.build)
        mf.irrep_nelec = {'A1g':(4,2), 'A1u': (3, 2)}
        self.assertRaises(ValueError, mf.build)
        mf.irrep_nelec = {'A1g':(4,6)}
        self.assertRaises(ValueError, mf.build)

        pmol.spin = -2
        mf.irrep_nelec = {'A1g':(4,2), 'A1u': (2, 4)}
        self.assertRaises(ValueError, mf.build)
        mf.irrep_nelec = {'A1g':(2,4), 'A1u': (2, 3)}
        self.assertRaises(ValueError, mf.build)
        mf.irrep_nelec = {'A1g':(6,4)}
        self.assertRaises(ValueError, mf.build)

    def test_rhf_dip_moment(self):
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_rhf_quad_moment(self):
        quad = n2mf.quad_moment(unit='au')
        answer = numpy.array([[ 0.65040837,  0.        ,  0.        ],
                              [ 0.        ,  0.65040837,  0.        ],
                              [ 0.        ,  0.        , -1.30081674]])
        self.assertTrue(numpy.allclose(quad, answer))

    def test_rohf_dip_moment(self):
        mf = scf.ROHF(mol).run()
        dip = mf.dip_moment(unit='au')
        self.assertTrue(numpy.allclose(dip, [0.00000, 0.00000, 0.80985]))

    def test_rohf_quad_moment(self):
        mf = scf.ROHF(n2sym).run()
        quad = mf.quad_moment(unit='au')
        answer = numpy.array([[ 0.65040837,  0.        ,  0.        ],
                              [ 0.        ,  0.65040837,  0.        ],
                              [ 0.        ,  0.        , -1.30081674]])
        self.assertTrue(numpy.allclose(quad, answer))

    def test_get_wfnsym(self):
        self.assertEqual(n2mf.wfnsym, 0)

        pmol = n2sym.copy()
        pmol.spin = 2
        mf = scf.ROHF(pmol).set(verbose = 0).run()
        self.assertTrue(mf.wfnsym in (2, 3))

    def test_complex_orbitals(self):
        nao = mol.nao_nr()
        mf = scf.RHF(mol)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.RHF(mol).set(max_memory=0)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.rohf.ROHF(mol)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

        mf = scf.rohf.ROHF(mol).set(max_memory=0)
        mf.kernel(numpy.zeros((nao,nao))*0j)
        self.assertAlmostEqual(mf.e_tot, -76.026765673119627, 9)

    def test_apply(self):
        from pyscf import mp
        self.assertTrue(isinstance(mf.apply(mp.MP2), mp.mp2.RMP2))
        mf1 = scf.RHF(mol)
        self.assertTrue(isinstance(mf1.apply('MP2'), mp.mp2.RMP2))

    def test_update_from_chk(self):
        mf1 = scf.RHF(mol).update(mf.chkfile)
        self.assertAlmostEqual(mf1.e_tot, mf.e_tot, 12)

    def test_mute_chkfile(self):
        # To ensure "mf.chkfile = None" does not affect post-SCF calculations
        mol = gto.M(atom='he', basis='6-311g', verbose=0)
        mf1 = scf.RHF(mol)
        mf1.chkfile = None
        mf1.newton().kernel()
        #mf1.apply('CISD').run()
        #mf1.apply('CCSD').run()
        mf1.apply('TDHF').run()
        #mf1.apply('CASSCF', 2, 2).run()
        mf1.nuc_grad_method().run()

    def test_as_scanner(self):
        mf_scanner = mf.as_scanner().as_scanner()
        mf_scanner.chkfile = None
        self.assertAlmostEqual(mf_scanner(mol), mf.e_tot, 9)

        mf_scanner = mf.x2c().density_fit().newton().as_scanner()
        mf_scanner.chkfile = None
        self.assertAlmostEqual(mf_scanner(mol.atom), -76.075408156235909, 9)

        mol1 = gto.M(atom='H 0 0 0; H 0 0 .9', basis='cc-pvdz')
        ref = mol1.RHF(chkfile=tempfile.NamedTemporaryFile().name).x2c().density_fit().run()
        e1 = mf_scanner('H 0 0 0; H 0 0 .9')
        self.assertAlmostEqual(e1, -1.116394048204042, 9)
        self.assertAlmostEqual(e1, ref.e_tot, 9)

        mfs = mol1.RHF(chkfile=tempfile.NamedTemporaryFile().name).as_scanner()
        mfs.__dict__.update(scf.chkfile.load(ref.chkfile, 'scf'))
        e = mfs(mol1)
        self.assertAlmostEqual(e, -1.1163913004438035, 9)

    def test_natm_eq_0(self):
        mol = gto.M()
        mol.nelectron = 2
        mf = scf.hf.RHF(mol)
        mf.get_hcore = lambda *args: numpy.diag(numpy.arange(3))
        mf.get_ovlp = lambda *args: numpy.eye(3)
        mf._eri = numpy.zeros((3,3,3,3))
        for i in range(3):
            mf._eri[i,i,i,i] = .2
        dm = mf.get_init_guess(mol, key='minao')
        self.assertAlmostEqual(lib.fp(dm), 2., 9)
        mf.kernel()
        self.assertAlmostEqual(mf.e_tot, 0.2, 9)

    def test_uniq_var(self):
        mo_occ = mf.mo_occ.copy()
        nmo = mo_occ.size
        nocc = numpy.count_nonzero(mo_occ > 0)
        nvir = nmo - nocc
        numpy.random.seed(1)
        f = numpy.random.random((nmo,nmo))
        f_uniq = scf.hf.pack_uniq_var(f, mo_occ)
        self.assertEqual(f_uniq.size, nocc*nvir)
        f1 = scf.hf.unpack_uniq_var(f_uniq, mo_occ)
        self.assertAlmostEqual(abs(f1 + f1.T).max(), 0, 12)

        mo_occ[4:7] = 1
        ndocc = 4
        nocc = 7
        f_uniq = scf.hf.pack_uniq_var(f, mo_occ)
        self.assertEqual(f_uniq.size, nocc*(nmo-ndocc)-(nocc-ndocc)**2)

        f1 = scf.hf.unpack_uniq_var(f_uniq, mo_occ)
        self.assertAlmostEqual(abs(f1 + f1.T).max(), 0, 12)

    def test_check_convergence(self):
        mf1 = scf.RHF(n2sym)
        mf1.diis = False
        count = [0]
        def check_convergence(envs):
            count[0] += 1
            return envs['norm_gorb'] < 0.1
        mf1.check_convergence = check_convergence
        mf1.kernel()
        self.assertAlmostEqual(mf1.e_tot, -108.9297980718255, 9)
        self.assertEqual(count[0], 3)

    def test_canonicalize(self):
        n2_rohf = n2mf.view(scf.hf_symm.ROHF)
        e, c = n2_rohf.canonicalize(n2mf.mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

        mo_coeff = numpy.array(n2mf.mo_coeff)
        e, c = n2mf.canonicalize(mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

        n2_rohf = n2mf.view(scf.rohf.ROHF)
        e, c = n2_rohf.canonicalize(n2mf.mo_coeff, n2mf.mo_occ)
        self.assertAlmostEqual(float(abs(e - n2mf.mo_energy).max()), 0, 7)

    def test_get_irrep_nelec(self):
        fock = n2mf.get_fock()
        s1e = n2mf.get_ovlp()
        e, c = n2mf.eig(fock, s1e)
        mo_occ = n2mf.get_occ(e, c)
        irrep_nelec = n2mf.get_irrep_nelec(n2sym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], 4)
        self.assertEqual(irrep_nelec['A1g'], 6)
        self.assertEqual(irrep_nelec['E1ux'], 2)
        self.assertEqual(irrep_nelec['E1uy'], 2)
        n2_rhf = n2mf.copy()
        n2_rhf.irrep_nelec = irrep_nelec
        n2_rhf.irrep_nelec['A2g'] = 0
        n2_rhf.irrep_nelec['E2gx'] = 2
        self.assertRaises(ValueError, n2_rhf.build)
        n2_rhf.irrep_nelec['A1g'] = 32
        self.assertRaises(ValueError, n2_rhf.build)

        n2_rohf = n2mf.view(scf.hf_symm.ROHF)
        irrep_nelec = n2_rohf.get_irrep_nelec(n2sym, c, mo_occ)
        self.assertEqual(irrep_nelec['A1u'], (2,2))
        self.assertEqual(irrep_nelec['A1g'], (3,3))
        self.assertEqual(irrep_nelec['E1ux'], (1,1))
        self.assertEqual(irrep_nelec['E1uy'], (1,1))

        n2_rohf.irrep_nelec = irrep_nelec
        n2_rohf.irrep_nelec['A2g'] = 0
        n2_rohf.nelec = (8,6)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (2,2)
        n2_rohf.irrep_nelec['E2gx'] = 0
        n2_rohf.irrep_nelec['E2gy'] = 0
        n2_rohf.irrep_nelec['E2ux'] = 0
        n2_rohf.irrep_nelec['E2uy'] = 0
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (2,0)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (0,2)
        self.assertRaises(ValueError, n2_rohf.build)
        n2_rohf.irrep_nelec['A1g'] = (3,2)
        n2_rohf.irrep_nelec['A1u'] = (2,3)
        self.assertRaises(ValueError, n2_rohf.build)

    def test_rohf_spin_square(self):
        mf1 = mf.view(scf.rohf.ROHF)
        ss, s = mf1.spin_square()
        self.assertAlmostEqual(ss, 0, 12)
        self.assertAlmostEqual(s, 1, 12)

        mf1.nelec = (6, 4)
        ss, s = mf1.spin_square()
        self.assertAlmostEqual(ss, 2, 12)
        self.assertAlmostEqual(s, 3, 12)

    def test_get_vj(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dm = numpy.random.random((nao,nao))
        ref = mf.get_j(mol, dm)

        mf1 = scf.RHF(mol)
        mf1.max_memory = 0
        vj1 = mf1.get_j(mol, dm)

        self.assertAlmostEqual(abs(ref-vj1).max(), 0, 12)
        self.assertAlmostEqual(numpy.linalg.norm(vj1), 77.035779188661465, 9)

        orig = mf1.opt.prescreen
        self.assertEqual(orig, scf._vhf._fpointer('CVHFnrs8_prescreen').value)
        mf1.opt.prescreen = orig
        mf1.opt.prescreen = 'CVHFnoscreen'
        self.assertEqual(mf1.opt.prescreen, scf._vhf._fpointer('CVHFnoscreen').value)

        # issue #1114
        dm = numpy.eye(nao, dtype=int)
        vj, vk = mf.get_jk(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), 1.6593323222866125, 9)
        self.assertAlmostEqual(lib.fp(vk), -1.4662135224053987, 9)
        vj, vk = mf1.get_jk(mol, dm)
        self.assertAlmostEqual(lib.fp(vj), 1.6593323222866125, 9)
        self.assertAlmostEqual(lib.fp(vk), -1.4662135224053987, 9)

    def test_get_vk_direct_scf(self):
        numpy.random.seed(1)
        nao = mol.nao
        dm = numpy.random.random((nao,nao))
        vk1 = mf.get_k(mol, dm, hermi=0)

        mf1 = scf.RHF(mol)
        mf1.max_memory = 0
        vk2 = mf1.get_k(mol, dm, hermi=0)
        self.assertAlmostEqual(abs(vk1 - vk2).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vk1), -12.365527167710301, 12)

    def test_get_vj_lr(self):
        numpy.random.seed(1)
        nao = mol.nao
        dm = numpy.random.random((nao,nao))
        vj1 = mf.get_j(mol, dm, omega=1.5)

        mf1 = scf.RHF(mol)
        mf1.max_memory = 0
        vj2 = mf1.get_j(mol, dm, omega=1.5)
        self.assertAlmostEqual(abs(vj1 - vj2).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vj1), -10.015956161068031, 12)

    def test_get_vk_lr(self):
        numpy.random.seed(1)
        nao = mol.nao
        dm = numpy.random.random((nao,nao))
        vk1 = mf.get_k(mol, dm, hermi=0, omega=1.5)

        mf1 = scf.RHF(mol)
        mf1.max_memory = 0
        vk2 = mf1.get_k(mol, dm, hermi=0, omega=1.5)
        self.assertAlmostEqual(abs(vk1 - vk2).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vk1), -11.399103957754445, 12)

    def test_reset(self):
        mf = scf.RHF(mol).density_fit().x2c().newton()
        mf.reset(n2sym)
        self.assertTrue(mf.mol is n2sym)
        self.assertTrue(mf._scf.mol is n2sym)
        self.assertTrue(mf.with_df.mol is n2sym)
        self.assertTrue(mf.with_x2c.mol is n2sym)
        self.assertTrue(mf._scf.with_df.mol is n2sym)
        self.assertTrue(mf._scf.with_x2c.mol is n2sym)

    def test_schwarz_condition(self):
        mol = gto.M(atom='''
                    H    0   0   0
                    H    0   0   4.
                    ''', unit='B',
                    basis = [[0, (2.7, 1)], [0, (1e2, 1)]])
        mf = scf.RHF(mol)
        mf.direct_scf_tol = 1e-18
        opt = mf.init_direct_scf()
        shls = i, j, k, l = 0, 2, 3, 3
        q = opt.q_cond
        self.assertTrue(mol.intor_by_shell('int2e', shls).ravel()[0] < q[i,j] * q[k,l])

    @unittest.skip('Numerical accuracy issue in libcint 5.2')
    def test_schwarz_condition_numerical_error(self):
        mol = gto.M(atom='''
                    H    0   0   0
                    H    0   0   6
                    ''', unit='B',
                    basis = [[0, (.6, 1)], [0, (1e3, 1)]])
        omega = 5.
        with mol.with_short_range_coulomb(omega):
            mf = scf.RHF(mol)
            # sr eri cannot reach the accuracy 1e-18
            mf.direct_scf_tol = 1e-18
            opt = mf.init_direct_scf()
            shls = i, j, k, l = 2, 0, 1, 1
            q = opt.q_cond
            eri = mol.intor('int2e')
            self.assertTrue(eri[shls] < q[i,j] * q[k,l])
            self.assertTrue(eri[shls] < eri[i,i,k,k]**.5 * eri[j,j,l,l]**.5)
            self.assertTrue(eri[shls] < eri[i,i,l,l]**.5 * eri[j,j,k,k]**.5)


if __name__ == "__main__":
    print("Full Tests for rhf")
    unittest.main()
