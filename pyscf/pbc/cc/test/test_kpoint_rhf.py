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
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

from pyscf.lib import finger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf

import pyscf.cc
import pyscf.pbc.cc as pbcc
import make_test_cell
from pyscf.pbc.lib import kpts_helper
#from pyscf.pbc.cc.kccsd_rhf import kconserve_pmatrix
import pyscf.pbc.cc.kccsd_t_rhf as kccsd_t_rhf


cell = pbcgto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
#cell.verbose = 7
cell.output = '/dev/null'
cell.build()

# Helper functions
def kconserve_pmatrix(nkpts, kconserv):
    Ps = np.zeros((nkpts, nkpts, nkpts, nkpts))
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                # Chemist's notation for momentum conserving t2(ki,kj,ka,kb)
                kb = kconserv[ki, ka, kj]
                Ps[ki, kj, ka, kb] = 1
    return Ps

def rand_t1_t2(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    np.random.seed(1)
    t1 = (np.random.random((nkpts, nocc, nvir)) +
          np.random.random((nkpts, nocc, nvir)) * 1j - .5 - .5j)
    t2 = (np.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) +
          np.random.random((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir)) * 1j - .5 - .5j)
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
    Ps = kconserve_pmatrix(nkpts, kconserv)
    t2 = t2 + np.einsum('xyzijab,xyzw->yxwjiba', t2, Ps)
    return t1, t2

def rand_r1_r2_ip(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    np.random.seed(1)
    r1 = (np.random.random((nocc,)) +
          np.random.random((nocc,)) * 1j - .5 - .5j)
    r2 = (np.random.random((nkpts, nkpts, nocc, nocc, nvir)) +
          np.random.random((nkpts, nkpts, nocc, nocc, nvir)) * 1j - .5 - .5j)
    return r1, r2

def rand_r1_r2_ea(kmf, mycc):
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    np.random.seed(1)
    r1 = (np.random.random((nvir,)) +
          np.random.random((nvir,)) * 1j - .5 - .5j)
    r2 = (np.random.random((nkpts, nkpts, nocc, nvir, nvir)) +
          np.random.random((nkpts, nkpts, nocc, nvir, nvir)) * 1j - .5 - .5j)
    return r1, r2

def make_rand_kmf():
    # Not perfect way to generate a random mf.
    # CSC = 1 is not satisfied and the fock matrix is neither
    # diagonal nor sorted.
    np.random.seed(2)
    kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]))
    kmf.exxdiv = None
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((3, nmo))
    kmf.mo_occ[:, :2] = 2
    kmf.mo_energy = np.arange(nmo) + np.random.random((3, nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2
    kmf.mo_coeff = (np.random.random((3, nmo, nmo)) +
                    np.random.random((3, nmo, nmo)) * 1j - .5 - .5j)
    # Round to make this insensitive to small changes between PySCF versions
    mat_veff = kmf.get_veff().round(4)
    mat_hcore = kmf.get_hcore().round(4)
    kmf.get_veff = lambda *x: mat_veff
    kmf.get_hcore = lambda *x: mat_hcore
    return kmf

rand_kmf = make_rand_kmf()

#TODO Delete me; these functions were used to check the changes on
#     master and dev to see whether the answers were the same after
#     changes to the eris.mo_energy
def _run_ip_matvec(cc, r1, r2, kshift):
    try:  # Different naming & calling conventions between master/dev
        vector = cc.ip_amplitudes_to_vector(r1, r2)
    except:
        vector = cc.amplitudes_to_vector_ip(r1, r2)
    try:
        vector = cc.ipccsd_matvec(vector, kshift)
    except:
        cc.kshift = kshift
        vector = cc.ipccsd_matvec(vector)
    try:
        Hr1, Hr2 = cc.ip_vector_to_amplitudes(vector)
    except:
        Hr1, Hr2 = cc.vector_to_amplitudes_ip(vector)
    return Hr1, Hr2

def _run_ea_matvec(cc, r1, r2, kshift):
    try:  # Different naming & calling conventions between master/dev
        vector = cc.ea_amplitudes_to_vector(r1, r2)
    except:
        vector = cc.amplitudes_to_vector_ea(r1, r2)
    try:
        vector = cc.eaccsd_matvec(vector, kshift)
    except:
        cc.kshift = kshift
        vector = cc.eaccsd_matvec(vector)
    try:
        Hr1, Hr2 = cc.ea_vector_to_amplitudes(vector)
    except:
        Hr1, Hr2 = cc.vector_to_amplitudes_ea(vector)
    return Hr1, Hr2

def run_kcell(cell, n, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    abs_kpts = cell.make_kpts(nk, wrap_around=True)

    #############################################
    # Running HF                                #
    #############################################
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    #kmf.verbose = 7
    ekpt = kmf.scf()


    cc = pbcc.kccsd_rhf.RCCSD(kmf)
    cc.conv_tol=1e-8
    #cc.verbose = 7
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc

class KnownValues(unittest.TestCase):
    def test_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 9)
        self.assertAlmostEqual(ecc, cc_311, 6)

    def test_single_kpt(self):
        cell = pbcgto.Cell()
        cell.atom = '''
        H 0 0 0
        H 1 0 0
        H 0 1 0
        H 0 1 1
        '''
        cell.a = np.eye(3)*2
        cell.basis = [[0, [1.2, 1]], [1, [1.0, 1]]]
        cell.verbose = 0
        cell.build()

        kpts = cell.get_abs_kpts([.5,.5,.5]).reshape(1,3)
        mf = pbcscf.KRHF(cell, kpts=kpts).run(conv_tol=1e-9)
        kcc = pbcc.kccsd_rhf.RCCSD(mf)
        kcc.level_shift = .05
        e0 = kcc.kernel()[0]

        mf = pbcscf.RHF(cell, kpt=kpts[0]).run(conv_tol=1e-9)
        mycc = pbcc.RCCSD(mf)
        e1 = mycc.kernel()[0]
        self.assertAlmostEqual(e0, e1, 5)

    def test_frozen_n3(self):
        mesh = 5
        cell = make_test_cell.test_cell_n3([mesh]*3)
        nk = (1, 1, 2)
        ehf_bench = -8.348616843863795
        ecc_bench = -0.037920339437169

        abs_kpts = cell.make_kpts(nk, with_gamma_point=True)

        # RHF calculation
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-9
        ehf = kmf.scf()

        # KRCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[0],[0,1]])
        cc.diis_start_cycle = 1
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 9)
        self.assertAlmostEqual(ecc, ecc_bench, 8)

    def _test_cu_metallic_nonequal_occ(self, kmf, cell, ecc1_bench=-0.9646107739333411):
        assert cell.mesh == [7, 7, 7]
        max_cycle = 5  # Too expensive to do more

        # The following calculation at full convergence gives -0.711071910294612
        # for a cell.mesh = [25, 25, 25].
        mycc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=0)
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc1, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc1, ecc1_bench, 6)

    def _test_cu_metallic_frozen_occ(self, kmf, cell):
        assert cell.mesh == [7, 7, 7]
        ecc2_bench = -0.7651806468801496
        max_cycle = 5

        # The following calculation at full convergence gives -0.6440448716452378
        # for a cell.mesh = [25, 25, 25].  It is equivalent to an RHF supercell [1, 1, 2]
        # calculation with frozen = [0, 3].
        mycc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[2, 3], [0, 1]])
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.05
        mycc.max_cycle = max_cycle
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc2, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc2, ecc2_bench, 6)

    def _test_cu_metallic_frozen_vir(self, kmf, cell):
        assert cell.mesh == [7, 7, 7]
        ecc3_bench = -0.76794053711557086
        max_cycle = 5

        # The following calculation at full convergence gives -0.58688462599474
        # for a cell.mesh = [25, 25, 25].  It is equivalent to a supercell [1, 1, 2]
        # calculation with frozen = [0, 3, 35].
        mycc = pbcc.kccsd_rhf.RCCSD(kmf, frozen=[[1, 17], [0]])
        mycc.diis_start_cycle = 1
        mycc.max_cycle = max_cycle
        mycc.iterative_damping = 0.05
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc3, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc3, ecc3_bench, 6)

        ew, ev = mycc.ipccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(ew[0][0], -3.028339571372944, 6)
        self.assertAlmostEqual(ew[0][1], -2.850636489429295, 6)
        self.assertAlmostEqual(ew[0][2], -2.801491561537961, 6)

        ew, ev = mycc.eaccsd(nroots=3, koopmans=True, kptlist=[1])
        self.assertAlmostEqual(ew[0][0], 3.266064683223669, 6)
        self.assertAlmostEqual(ew[0][1], 3.281390137070985, 6)
        self.assertAlmostEqual(ew[0][2], 3.426297911456726, 6)

        check_gamma = False  # Turn me on to run the supercell calculation!

        if check_gamma:
            from pyscf.pbc.tools.pbc import super_cell
            supcell = super_cell(cell, nk)
            kmf = pbcscf.RHF(supcell, exxdiv=None)
            ehf = kmf.scf()

            mycc = pbcc.RCCSD(kmf, frozen=[0, 3, 35])
            mycc.max_cycle = max_cycle
            mycc.iterative_damping = 0.04
            ecc, t1, t2 = mycc.kernel()

            print('Gamma energy =', ecc/np.prod(nk))
            print('K-point energy =', ecc3_bench)

            ew, ev = mycc.ipccsd(nroots=5)
            # For cell mesh of [25, 25, 25], we get:
            #
            # EOM-CCSD root 0 E = -3.052456841625895
            # EOM-CCSD root 1 E = -2.989798972232893
            # EOM-CCSD root 2 E = -2.839646545189692
            # EOM-CCSD root 3 E = -2.836645046801352
            # EOM-CCSD root 4 E = -2.831020659800223

            ew, ev = mycc.eaccsd(nroots=5)
            # For cell mesh of [25, 25, 25], we get:
            #
            # EOM-CCSD root 0 E = 3.049774979170073
            # EOM-CCSD root 1 E = 3.104127952392612
            # EOM-CCSD root 2 E = 3.109435080273549
            # EOM-CCSD root 3 E = 3.139400145624026
            # EOM-CCSD root 4 E = 3.151896524990866

    def test_cu_metallic_high_cost(self):
        mesh = 7
        cell = make_test_cell.test_cell_cu_metallic([mesh]*3)
        nk = [1,1,2]
        ehf_bench = -52.5393701339723

        # KRHF calculation
        kmf = pbcscf.KRHF(cell, exxdiv=None)
        kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
        kmf.conv_tol_grad = 1e-6  # Stricter tol needed for answer to agree with supercell
        ehf = kmf.scf()

        self.assertAlmostEqual(ehf, ehf_bench, 6)

        # Run CC calculations
        self._test_cu_metallic_nonequal_occ(kmf, cell)
        self._test_cu_metallic_frozen_occ(kmf, cell)
        self._test_cu_metallic_frozen_vir(kmf, cell)

    def test_cu_metallic_smearing(self):
        mesh = 7
        cell = make_test_cell.test_cell_cu_metallic([mesh]*3)
        nk = [1,1,2]
        ehf_bench = -52.539316985400433

        # KRHF calculation
        kmf = pbcscf.addons.smearing_(pbcscf.KRHF(cell, exxdiv=None), 0.001, "fermi")
        kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
        kmf.conv_tol_grad = 1e-6  # Stricter tol needed for answer to agree with supercell
        ehf = kmf.scf()

        self.assertAlmostEqual(ehf, ehf_bench, 6)
        with self.assertRaises(RuntimeError):
            self._test_cu_metallic_nonequal_occ(kmf, cell)
        kmf.smearing_method = False
        kmf.mo_occ = kmf.get_occ()

        # Run CC calculations
        self._test_cu_metallic_nonequal_occ(kmf, cell, -0.96676526820520137)

    def test_ccsd_t_high_cost(self):
        n = 14
        cell = make_test_cell.test_cell_n3([n]*3)

        kpts = cell.make_kpts([1, 1, 2])
        kpts -= kpts[0]
        kmf = pbcscf.KRHF(cell, kpts=kpts)
        ehf = kmf.kernel()

        rand_cc = pbcc.KRCCSD(kmf)
        ecc, t1, t2 = rand_cc.kernel()

        energy_t = kccsd_t_rhf.kernel(rand_cc)
        energy_t_bench = -0.00191443154358
        self.assertAlmostEqual(energy_t, energy_t_bench, 6)

    def test_rand_ccsd(self):
        '''Single (eom-)ccsd iteration with random t1/t2.'''
        rand_cc = pbcc.KRCCSD(rand_kmf)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        t1, t2 = rand_cc.t1, rand_cc.t2
        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (-4.6942326686+9.50185397111j), 6)
        self.assertAlmostEqual(finger(Ht2), (17.1490394799+110.137726574j), 6)

        # Excited state results
        kshift = 0
        r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (-0.456418558025-0.0485067398162j), 6)
        self.assertAlmostEqual(finger(Hr2), (0.616016341219+2.08777776589j), 6)

        r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (-0.234979092885-0.218401823892j), 6)
        self.assertAlmostEqual(finger(Hr2), (-3.56244154449+2.12051064183j), 6)

    def test_rand_ccsd_frozen0(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and lowest lying orbital
        at multiple k-points frozen.'''
        rand_cc = pbcc.KRCCSD(rand_kmf, frozen=1)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]

        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (-8.06918006043+8.2779236131j), 6)
        self.assertAlmostEqual(finger(Ht2), (30.6692903818-14.2701276046j), 6)

        frozen = [[0,],[0,],[0,]]
        rand_cc = pbcc.KRCCSD(rand_kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (-8.06918006043+8.2779236131j), 6)
        self.assertAlmostEqual(finger(Ht2), (30.6692903818-14.2701276046j), 6)

        # Excited state results
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        kshift = 0
        r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.289384011655-0.394002590665j), 6)
        self.assertAlmostEqual(finger(Hr2), (0.056437476036+0.156522915807j), 6)

        r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.298028415374+0.0944020804565j), 6)
        self.assertAlmostEqual(finger(Hr2), (-0.243561845158+0.869173612894j), 6)

    def test_rand_ccsd_frozen1(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and single frozen occupied
        orbital.'''
        frozen = [[0,],[],[]]
        rand_cc = pbcc.KRCCSD(rand_kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        # Manually zero'ing out the frozen elements of the t1/t2
        # N.B. the 0'th element frozen means we are freezing the 1'th
        #      element in the current padding scheme
        t1[0, 1] = 0.0
        t2[0, :, :, 1, :] = 0.0
        t2[:, 0, :, :, 1] = 0.0

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (-9.31532552971+16.3972283898j), 6)
        self.assertAlmostEqual(finger(Ht2), (-4.42939435314+52.147616355j), 6)

        # Excited state results
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        kshift = 0
        r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        r1[1] = 0.0
        r2[0, :, 1] = 0.0
        r2[:, 0, :, 1] = 0.0
        Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (-0.558560718395-0.344470539404j), 6)
        self.assertAlmostEqual(finger(Hr2), (0.882960101238+0.0752022769822j), 6)

        r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        r2[0, :, 1] = 0.0
        Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.010947007472-0.287095461151j), 6)
        self.assertAlmostEqual(finger(Hr2), (-2.58907863831+0.685390702884j), 6)

    def test_rand_ccsd_frozen2(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and full occupied frozen
        at a single k-point.'''
        frozen = [[],[0,1],[]]
        rand_cc = pbcc.KRCCSD(rand_kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        # Manually zero'ing out the frozen elements of the t1/t2
        # N.B. the 0'th element frozen means we are freezing the 1'th
        #      element in the current padding scheme
        t1[1, [0,1]] = 0.0
        t2[1, :, :, [0,1], :] = 0.0
        t2[:, 1, :, :, [0,1]] = 0.0

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (-0.931278705177+2.16347477318j), 6)
        self.assertAlmostEqual(finger(Ht2), (29.0079567454-0.114082762172j), 6)

        # Excited state results
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        kshift = 1
        r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        r1[[0,1]] = 0.0
        r2[1, :, [0,1]] = 0.0
        r2[:, 1, :, [0,1]] = 0.0
        Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.0 + 0.0j), 6)
        self.assertAlmostEqual(finger(Hr2), (-0.336011745573-0.0454220386975j), 6)

        r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        r2[1, :, [0,1]] = 0.0
        Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (-0.00152035195068-0.502318229581j), 6)
        self.assertAlmostEqual(finger(Hr2), (-1.59488320866+0.838903632811j), 6)

    def test_rand_ccsd_frozen3(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and single frozen virtual
        orbital.'''
        kconserv = kpts_helper.get_kconserv(rand_kmf.cell, rand_kmf.kpts)

        frozen = [[],[],[3]]  # freezing one virtual
        rand_cc = pbcc.KRCCSD(rand_kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_kmf.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)
        # Manually zero'ing out the frozen elements of the t1/t2
        t1[2, :, 0] = 0.0
        for ki in range(rand_cc.nkpts):
          for kj in range(rand_cc.nkpts):
            for ka in range(rand_cc.nkpts):
              kb = kconserv[ki, ka, kj]
              if ka == 2:
                  t2[ki, kj, ka, :, :, 0] = 0.0
              if kb == 2:
                  t2[ki, kj, ka, :, :, :, 0] = 0.0

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(finger(Ht1), (5.3320153970710118-7.9402122992688602j), 6)
        self.assertAlmostEqual(finger(Ht2), (-236.46389414847206-360.1605297160217j), 6)

        # Excited state results
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        kshift = 2
        r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        r1[0] = 0.0
        for ki in range(rand_cc.nkpts):
          for kj in range(rand_cc.nkpts):
            ka = kconserv[ki, kshift, kj]
            if ka == 2:
                r2[ki, kj, :, :, 0] = 0.0

        Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.4067595510145880 +  0.0770280877446436j), 6)
        self.assertAlmostEqual(finger(Hr2), (0.0926714318228812 + -1.0702702421619084j), 6)

        r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        r1[0] = 0.0
        for kj in range(rand_cc.nkpts):
          for ka in range(rand_cc.nkpts):
            kb = kconserv[kshift, ka, kj]
            if ka == 2:
                r2[kj, ka, :, 0, :] = 0.0
            if kb == 2:
                r2[kj, ka, :, :, 0] = 0.0

        Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        self.assertAlmostEqual(finger(Hr1), (0.0070404498167285 + -0.1646809321907418j), 6)
        self.assertAlmostEqual(finger(Hr2), (0.4518315588945250 + -0.5508323185152750j), 6)

    def test_h4_fcc_k2(self):
        '''Metallic hydrogen fcc lattice.  Checks versus a corresponding
        supercell calculation.

        NOTE: different versions of the davidson may converge to a different
        solution for the k-point IP/EA eom.  If you're getting the wrong
        root, check to see if it's contained in the supercell set of
        eigenvalues.'''
        cell = pbcgto.Cell()
        cell.atom = [['H', (0.000000000, 0.000000000, 0.000000000)],
                     ['H', (0.000000000, 0.500000000, 0.250000000)],
                     ['H', (0.500000000, 0.500000000, 0.500000000)],
                     ['H', (0.500000000, 0.000000000, 0.750000000)]]
        cell.unit = 'Bohr'
        cell.a = [[1.,0.,0.],[0.,1.,0],[0,0,2.2]]
        cell.verbose = 7
        cell.spin = 0
        cell.charge = 0
        cell.basis = [[0, [1.0, 1]],]
        cell.pseudo = 'gth-pade'
        cell.output = '/dev/null'
        #cell.max_memory = 1000
        for i in range(len(cell.atom)):
            cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
        cell.build()

        nmp = [2, 1, 1]

        kmf = pbcscf.KRHF(cell)
        kmf.kpts = cell.make_kpts(nmp, scaled_center=[0.0,0.0,0.0])
        e = kmf.kernel()

        mycc = pbcc.KCCSD(kmf)
        ekccsd, _, _ = mycc.kernel()
        self.assertAlmostEqual(ekccsd, -0.06146759560406628, 6)

        # Getting more roots than 1 is difficult
        e = mycc.eaccsd(nroots=1, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], 5.079427283440857, 6)
        e = mycc.eaccsd(nroots=1, kptlist=(1,))[0]
        self.assertAlmostEqual(e[0][0], 4.183328878177331, 6)

        e = mycc.ipccsd(nroots=1, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], -3.471710821544506, 6)
        e = mycc.ipccsd(nroots=1, kptlist=(1,))[0]
        self.assertAlmostEqual(e[0][0], -4.272015727359054, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        myscc = pbcc.KCCSD(mf)
        eccsd, _, _ = myscc.kernel()
        eccsd /= np.prod(nmp)
        self.assertAlmostEqual(eccsd, -0.06146759560406628, 6)

        e = myscc.eaccsd(nroots=4, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], 4.183328873793568, 6)
        self.assertAlmostEqual(e[0][1], 4.225034294249784, 6)
        self.assertAlmostEqual(e[0][2], 5.068962665511664, 6)
        self.assertAlmostEqual(e[0][3], 5.07942727935064 , 6)

        e = myscc.ipccsd(nroots=4, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], -4.272015724869052, 6)
        self.assertAlmostEqual(e[0][1], -4.254298274388934, 6)
        self.assertAlmostEqual(e[0][2], -3.471710821688812, 6)
        self.assertAlmostEqual(e[0][3], -3.462817764320668, 6)


    def test_h4_fcc_k2_frozen(self):
        '''Metallic hydrogen fcc lattice with frozen lowest lying occupied
        and highest lying virtual orbitals.  Checks versus a corresponding
        supercell calculation.

        NOTE: different versions of the davidson may converge to a different
        solution for the k-point IP/EA eom.  If you're getting the wrong
        root, check to see if it's contained in the supercell set of
        eigenvalues.'''
        cell = pbcgto.Cell()
        cell.atom = [['H', (0.000000000, 0.000000000, 0.000000000)],
                     ['H', (0.000000000, 0.500000000, 0.250000000)],
                     ['H', (0.500000000, 0.500000000, 0.500000000)],
                     ['H', (0.500000000, 0.000000000, 0.750000000)]]
        cell.unit = 'Bohr'
        cell.a = [[1.,0.,0.],[0.,1.,0],[0,0,2.2]]
        cell.verbose = 7
        cell.spin = 0
        cell.charge = 0
        cell.basis = [[0, [1.0, 1]],]
        cell.pseudo = 'gth-pade'
        cell.output = '/dev/null'
        #cell.max_memory = 1000
        for i in range(len(cell.atom)):
            cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
        cell.build()

        nmp = [2, 1, 1]

        kmf = pbcscf.KRHF(cell)
        kmf.kpts = cell.make_kpts(nmp, scaled_center=[0.0,0.0,0.0])
        e = kmf.kernel()

        frozen = [[0, 3], []]
        mycc = pbcc.KCCSD(kmf, frozen=frozen)
        ekccsd, _, _ = mycc.kernel()
        self.assertAlmostEqual(ekccsd, -0.04683399814247455, 6)

        # Getting more roots than 1 is difficult
        e = mycc.eaccsd(nroots=1, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], 5.060562738181741, 6)
        e = mycc.eaccsd(nroots=1, kptlist=(1,))[0]
        self.assertAlmostEqual(e[0][0], 4.188511644938458, 6)

        e = mycc.ipccsd(nroots=1, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], -3.477663551987023, 6)
        e = mycc.ipccsd(nroots=1, kptlist=(1,))[0]
        self.assertAlmostEqual(e[0][0], -4.23523412155825, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        myscc = pbcc.KCCSD(mf, frozen=[0, 7])
        eccsd, _, _ = myscc.kernel()
        eccsd /= np.prod(nmp)
        self.assertAlmostEqual(eccsd, -0.04683401678904569, 6)

        e = myscc.eaccsd(nroots=4, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], 4.188511680212755, 6)
        self.assertAlmostEqual(e[0][1], 4.205924087610756, 6)
        self.assertAlmostEqual(e[0][2], 5.060562771978923, 6)
        self.assertAlmostEqual(e[0][3], 5.077249823137741, 6)

        e = myscc.ipccsd(nroots=4, kptlist=(0,))[0]
        self.assertAlmostEqual(e[0][0], -4.261818242746091, 6)
        self.assertAlmostEqual(e[0][1], -4.235233956876479, 6)
        self.assertAlmostEqual(e[0][2], -3.477663568390151, 6)
        self.assertAlmostEqual(e[0][3], -3.459133332687474, 6)

if __name__ == '__main__':
    print("Full kpoint_rhf test")
    unittest.main()
