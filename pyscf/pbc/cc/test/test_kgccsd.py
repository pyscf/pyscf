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
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import unittest
import numpy as np

from pyscf import gto, scf

from pyscf.lib import fp
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
import pyscf.pbc.tools

import pyscf.pbc.cc as pbcc
import pyscf.pbc.cc.kccsd_t as kccsd_t
import pyscf.pbc.cc.kccsd

from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.tools.make_test_cell as make_test_cell


def setUpModule():
    global cell, rand_kmf
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
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.mesh = [15] * 3
    cell.build()

    rand_kmf = make_rand_kmf()

def tearDownModule():
    global cell, rand_kmf
    cell.stdout.close()
    del cell, rand_kmf


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
    '''Antisymmetrized t1/t2 for spin-orbitals.'''
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
    t2 = t2 - np.einsum('xyzijab,xyzw->yxzjiab', t2, Ps)
    t2 = t2 + np.einsum('xyzijab,xyzw->yxwjiba', t2, Ps)
    return t1, t2


def rand_r1_r2_ip(kmf, mycc, kshift):
    '''Antisymmetrized 1p/2p1h operators for spin-orbitals.'''
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    np.random.seed(1)
    r1 = (np.random.random((nocc,)) +
          np.random.random((nocc,)) * 1j - .5 - .5j)
    r2 = (np.random.random((nkpts, nkpts, nocc, nocc, nvir)) +
          np.random.random((nkpts, nkpts, nocc, nocc, nvir)) * 1j - .5 - .5j)
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
    Ps = kconserve_pmatrix(nkpts, kconserv)
    r2 = r2 - np.einsum('xyija,xyz->yxjia', r2, Ps[:, :, kshift, :])
    return r1, r2


def rand_r1_r2_ea(kmf, mycc, kshift):
    '''Antisymmetrized 1h/2h1p operators for spin-orbitals.'''
    nkpts = mycc.nkpts
    nocc = mycc.nocc
    nmo = mycc.nmo
    nvir = nmo - nocc
    np.random.seed(1)
    r1 = (np.random.random((nvir,)) +
          np.random.random((nvir,)) * 1j - .5 - .5j)
    r2 = (np.random.random((nkpts, nkpts, nocc, nvir, nvir)) +
          np.random.random((nkpts, nkpts, nocc, nvir, nvir)) * 1j - .5 - .5j)
    kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
    Ps = kconserve_pmatrix(nkpts, kconserv)
    r2 = r2 - np.einsum('xyjab,xyz->xzjba', r2, Ps[kshift, :, :, :])
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
    return kmf


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
    abs_kpts = cell.make_kpts(nk, wrap_around=True)

    # RHF calculation
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    ekpt = kmf.scf()

    # RCCSD calculation
    cc = pbcc.kccsd.CCSD(pbcscf.addons.convert_to_ghf(kmf))
    cc.conv_tol = 1e-8
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc


class KnownValues(unittest.TestCase):
    def test_111_n0(self):
        L = 10.0
        n = 11
        cell = make_test_cell.test_cell_n0(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73491491306419987
        cc_111 = -1.1580008204825658e-05
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,7)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n1(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (1, 1, 1)
        hf_111 = -0.73506011036963814
        cc_111 = -0.023265431169472835
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,7)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_111_n3(self):
        n = 11
        cell = make_test_cell.test_cell_n3([n]*3)
        nk = (1, 1, 1)
        hf_111 = -7.4117951240232118
        cc_111 = -0.19468901057053406
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_111,7)
        self.assertAlmostEqual(ecc, cc_111,6)

    def test_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.042702177586414237
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 7)
        self.assertAlmostEqual(ecc, cc_311, 6)

    def test_211_n3(self):
        cell = make_test_cell.test_cell_n3_diffuse(precision=1e-9)
        nk = (2, 1, 1)

        abs_kpts = cell.make_kpts(nk, wrap_around=True)

        # GHF calculation
        kmf = pbcscf.KGHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-14
        escf = kmf.scf()
        self.assertAlmostEqual(escf, -6.1870676561726574, 6)

        # GCCSD calculation
        cc = pbcc.kccsd.CCSD(kmf)
        cc.conv_tol = 1e-8
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ecc, -0.067648363303697334, 6)

    def test_frozen_n3_high_cost(self):
        mesh = 5
        cell = make_test_cell.test_cell_n3([mesh]*3)
        nk = (1, 1, 3)
        ehf_bench = -9.15349763559837
        ecc_bench = -0.06713556649654

        abs_kpts = cell.make_kpts(nk, with_gamma_point=True)

        # RHF calculation
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-9
        ehf = kmf.scf()

        # KGCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pbcc.kccsd.CCSD(kmf, frozen=[[0,1],[],[0]])
        cc.diis_start_cycle = 1
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 7)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

    def _test_cu_metallic_nonequal_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc1_bench = -1.1633910051553982
        max_cycle = 2  # Too expensive to do more!

        # The following calculation at full convergence gives -0.711071910294612
        # for a cell.mesh = [25, 25, 25].
        mycc = pbcc.KGCCSD(kmf, frozen=None)
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.04
        mycc.max_cycle = max_cycle
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc1, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc1, ecc1_bench, 5)

    def _test_cu_metallic_frozen_occ(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc2_bench = -1.0430822430909346
        max_cycle = 2

        # The following calculation at full convergence gives -0.6440448716452378
        # for a cell.mesh = [25, 25, 25].  It is equivalent to a supercell [1, 1, 2]
        # calculation with frozen = [0, 3].
        mycc = pbcc.KGCCSD(kmf, frozen=[[2, 3], [0, 1]])
        mycc.diis_start_cycle = 1
        mycc.iterative_damping = 0.04
        mycc.max_cycle = max_cycle
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc2, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc2, ecc2_bench, 6)

    def _test_cu_metallic_frozen_vir(self, kmf, cell, nk=[1,1,1]):
        assert cell.mesh == [7, 7, 7]
        ecc3_bench = -0.94610600274627665
        max_cycle = 2

        # The following calculation at full convergence gives -0.58688462599474
        # for a cell.mesh = [25, 25, 25].  It is equivalent to a supercell [1, 1, 2]
        # calculation with frozen = [0, 3, 35].
        mycc = pbcc.KGCCSD(kmf, frozen=[[2, 3, 34, 35], [0, 1]])
        mycc.max_cycle = max_cycle
        mycc.iterative_damping = 0.05
        eris = mycc.ao2mo()
        eris.mo_energy = [f.diagonal() for f in eris.fock]
        ecc3, t1, t2 = mycc.kernel(eris=eris)

        self.assertAlmostEqual(ecc3, ecc3_bench, 6)

        check_gamma = False  # Turn me on to run the supercell calculation!

        if check_gamma:
            from pyscf.pbc.tools.pbc import super_cell
            supcell = super_cell(cell, nk)
            kmf = pbcscf.RHF(supcell, exxdiv=None)
            ehf = kmf.scf()

            mycc = pbcc.RCCSD(kmf, frozen=[0, 3, 35])
            mycc.max_cycle = max_cycle
            mycc.iterative_damping = 0.05
            ecc, t1, t2 = mycc.kernel()

            print('Gamma energy =', ecc/np.prod(nk))
            print('K-point energy =', ecc3)

    def test_cu_metallic_high_cost(self):
        mesh = 7
        cell = make_test_cell.test_cell_cu_metallic([mesh]*3, precision=1e-9)
        nk = [1,1,2]

        ehf_bench = -52.57191968827088

        # KRHF calculation
        kmf = pbcscf.KRHF(cell, exxdiv=None)
        kmf.kpts = cell.make_kpts(nk, scaled_center=[0.0, 0.0, 0.0], wrap_around=True)
        kmf.conv_tol_grad = 1e-6  # Stricter tol needed for answer to agree with supercell
        ehf = kmf.scf()

        self.assertAlmostEqual(ehf, ehf_bench, 6)

        # Run CC calculations
        self._test_cu_metallic_nonequal_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_occ(kmf, cell, nk=nk)
        self._test_cu_metallic_frozen_vir(kmf, cell, nk=nk)

    def test_ccsd_t_high_cost(self):
        n = 14
        cell = make_test_cell.test_cell_n3([n]*3)

        kpts = cell.make_kpts([1, 1, 2])
        kpts -= kpts[0]
        kmf = pbcscf.KRHF(cell, kpts=kpts, exxdiv=None)
        ehf = kmf.kernel()

        mycc = pbcc.KGCCSD(kmf)
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)

        eris.mo_energy = [eris.fock[i].diagonal() for i in range(len(kpts))]
        energy_t = kccsd_t.kernel(mycc, eris=eris)
        energy_t_bench = -0.00191440345386
        self.assertAlmostEqual(energy_t, energy_t_bench, 6)

        mycc = pbcc.KGCCSD(kmf, frozen=2)
        eris = mycc.ao2mo()
        ecc, t1, t2 = mycc.kernel(eris=eris)

        #eris.mo_energy = [eris.fock[i].diagonal() for i in range(len(kpts))]
        energy_t = kccsd_t.kernel(mycc, eris=eris)
        energy_t_bench = -0.0006758542603695721
        self.assertAlmostEqual(energy_t, energy_t_bench, 6)

    def test_rand_ccsd(self):
        '''Single (eom-)ccsd iteration with random t1/t2.'''
        kmf = pbcscf.addons.convert_to_ghf(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        rand_cc = pbcc.KGCCSD(kmf)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(fp(Ht1), (-10.5746290133+4.22371219606j), 6)
        self.assertAlmostEqual(fp(Ht2), (-250.696532783+706.190346877j), 6)

        # Excited state results
        rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        # TODO: fill in as the eom-kgccsd completed...
        #
        #kshift = 0
        #r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc, kshift)
        #Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        #print fp(Hr1)
        #print fp(Hr2)
        ##self.assertAlmostEqual(fp(Hr1), (-0.717532270254-0.440422108955j), 6)
        ##self.assertAlmostEqual(fp(Hr2), (0.681025691811+6.11804388788j), 6)

        #r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc, kshift)
        #Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        #print fp(Hr1)
        #print fp(Hr2)
        ##self.assertAlmostEqual(fp(Hr1), (-0.780441346305-0.72371724202j), 6)
        ##self.assertAlmostEqual(fp(Hr2), (-5.64822780257+5.40589051948j) , 6)

    def test_rand_ccsd_frozen0(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and lowest lying orbital
        at multiple k-points frozen.'''
        kmf = pbcscf.addons.convert_to_ghf(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        # frozen = 1
        rand_cc = pbcc.KGCCSD(kmf, frozen=1)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(fp(Ht1), (12.6235870016-0.263432509044j), 6)
        self.assertAlmostEqual(fp(Ht2), (94.8802678168+910.369938369j), 6)

        # frozen = [[0,],[0,],[0,]], should be same as above
        frozen = [[0,],[0,],[0,]]
        rand_cc = pbcc.KGCCSD(kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(rand_kmf, rand_cc)

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(fp(Ht1), (12.6235870016-0.263432509044j), 6)
        self.assertAlmostEqual(fp(Ht2), (94.8802678168+910.369938369j), 6)

        ## Excited state results
        #rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        # TODO: fill in as the eom-kgccsd completed...
        #
        #kshift = 0
        #r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc, kshift)
        #Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        #print fp(Hr1)
        #print fp(Hr2)
        ##self.assertAlmostEqual(fp(Hr1), (0.24563231525-0.513480233538j)  , 6)
        ##self.assertAlmostEqual(fp(Hr2), (-0.168078406643-0.259459463532j), 6)

        #r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc, kshift)
        #Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        #print fp(Hr1)
        #print fp(Hr2)
        ##self.assertAlmostEqual(fp(Hr1), (0.408247189016+0.329977757156j), 6)
        ##self.assertAlmostEqual(fp(Hr2), (0.906733733269-2.40456195366j) , 6)

    def test_rand_ccsd_frozen1(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and single frozen occupied
        orbital.'''
        kmf = pbcscf.addons.convert_to_ghf(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        frozen = [[0,],[],[]]
        rand_cc = pbcc.KGCCSD(kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        # Manually zero'ing out the frozen elements of the t1/t2
        # N.B. the 0'th element frozen means we are freezing the 1'th
        #      element in the current padding scheme
        t1[0, 1] = 0.0
        t2[0, :, :, 1, :] = 0.0
        t2[:, 0, :, :, 1] = 0.0

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(fp(Ht1), (-15.7950827762+31.0483053388j), 6)
        self.assertAlmostEqual(fp(Ht2), (263.884192539+96.7615664563j), 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        ## Excited state results
        #rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        #kshift = 0
        #r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        #r1[1] = 0.0
        #r2[0, :, 1] = 0.0
        #r2[:, 0, :, 1] = 0.0
        #Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (-0.558560718395-0.344470539404j), 6)
        #self.assertAlmostEqual(fp(Hr2), (0.882960101238+0.0752022769822j), 6)

        #r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        #r2[0, :, 1] = 0.0
        #Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (0.010947007472-0.287095461151j), 6)
        #self.assertAlmostEqual(fp(Hr2), (-2.58907863831+0.685390702884j), 6)

    def test_rand_ccsd_frozen2(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and full occupied frozen
        at a single k-point.'''
        kmf = pbcscf.addons.convert_to_ghf(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        frozen = [[],[0,1,2,3],[]]
        rand_cc = pbcc.KGCCSD(kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
        # Manually zero'ing out the frozen elements of the t1/t2
        # N.B. the 0'th element frozen means we are freezing the 1'th
        #      element in the current padding scheme
        t1[1, [0,1,2,3]] = 0.0
        t2[1, :, :, [0,1,2,3], :] = 0.0
        t2[:, 1, :, :, [0,1,2,3]] = 0.0

        Ht1, Ht2 = rand_cc.update_amps(t1, t2, eris)
        self.assertAlmostEqual(fp(Ht1), (-19.2772171332-10.5977304455j), 6)
        self.assertAlmostEqual(fp(Ht2), (227.434582141+298.826965082j), 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        ## Excited state results
        #rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        #kshift = 1
        #r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        #r1[[0,1]] = 0.0
        #r2[1, :, [0,1]] = 0.0
        #r2[:, 1, :, [0,1]] = 0.0
        #Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (0.0 + 0.0j), 6)
        #self.assertAlmostEqual(fp(Hr2), (-0.336011745573-0.0454220386975j), 6)

        #r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        #r2[1, :, [0,1]] = 0.0
        #Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (-0.00152035195068-0.502318229581j), 6)
        #self.assertAlmostEqual(fp(Hr2), (-1.59488320866+0.838903632811j), 6)

    def test_rand_ccsd_frozen3(self):
        '''Single (eom-)ccsd iteration with random t1/t2 and single frozen virtual
        orbital.'''
        kconserv = kpts_helper.get_kconserv(rand_kmf.cell, rand_kmf.kpts)

        kmf = pbcscf.addons.convert_to_ghf(rand_kmf)
        # Round to make this insensitive to small changes between PySCF versions
        mat_veff = kmf.get_veff().round(4)
        mat_hcore = kmf.get_hcore().round(4)
        kmf.get_veff = lambda *x: mat_veff
        kmf.get_hcore = lambda *x: mat_hcore

        frozen = [[],[],[5]]  # freezing one virtual
        rand_cc = pbcc.KGCCSD(kmf, frozen=frozen)
        eris = rand_cc.ao2mo(rand_cc.mo_coeff)
        eris.mo_energy = [eris.fock[k].diagonal() for k in range(rand_cc.nkpts)]
        t1, t2 = rand_t1_t2(kmf, rand_cc)
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
        self.assertAlmostEqual(fp(Ht1), (-19.6637196882-16.2773841431j), 6)
        self.assertAlmostEqual(fp(Ht2), (881.655146297+1283.71020059j), 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        ## Excited state results
        #rand_cc.t1, rand_cc.t2, rand_cc.eris = t1, t2, eris

        #kshift = 2
        #r1, r2 = rand_r1_r2_ip(rand_kmf, rand_cc)
        #r1[0] = 0.0
        #for ki in range(rand_cc.nkpts):
        #  for kj in range(rand_cc.nkpts):
        #    ka = kconserv[ki, kshift, kj]
        #    if ka == 2:
        #        r2[ki, kj, :, :, 0] = 0.0

        #Hr1, Hr2 = _run_ip_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (0.4067595510145880 +  0.0770280877446436j), 6)
        #self.assertAlmostEqual(fp(Hr2), (0.0926714318228812 + -1.0702702421619084j), 6)

        #r1, r2 = rand_r1_r2_ea(rand_kmf, rand_cc)
        #r1[0] = 0.0
        #for kj in range(rand_cc.nkpts):
        #  for ka in range(rand_cc.nkpts):
        #    kb = kconserv[kshift, ka, kj]
        #    if ka == 2:
        #        r2[kj, ka, :, 0, :] = 0.0
        #    if kb == 2:
        #        r2[kj, ka, :, :, 0] = 0.0

        #Hr1, Hr2 = _run_ea_matvec(rand_cc, r1, r2, kshift)
        #self.assertAlmostEqual(fp(Hr1), (0.0070404498167285 + -0.1646809321907418j), 6)
        #self.assertAlmostEqual(fp(Hr2), (0.4518315588945250 + -0.5508323185152750j), 6)

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
        cell.max_memory = 1000
        for i in range(len(cell.atom)):
            cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
        cell.build()

        nmp = [2, 1, 1]

        kmf = pbcscf.KRHF(cell)
        kmf.kpts = cell.make_kpts(nmp, scaled_center=[0.0,0.0,0.0])
        e = kmf.kernel()

        #mymp = pbmp.KMP2(kmf)
        #ekmp2, _ = mymp.kernel()
        #print("KMP2 corr energy (per cell) = ", ekmp2)

        mycc = pbcc.KGCCSD(kmf)
        ekccsd, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ekccsd, -0.06146759560406628, 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        ## Getting more roots than 1 is difficult
        #e = mycc.eaccsd(nroots=1, kptlist=(0,))[0]
        #self.assertAlmostEqual(e, 5.079427283440857, 6)
        #e = mycc.eaccsd(nroots=1, kptlist=(1,))[0]
        #self.assertAlmostEqual(e, 4.183328878177331, 6)

        #e = mycc.ipccsd(nroots=1, kptlist=(0,))[0]
        #self.assertAlmostEqual(e, -3.471710821544506, 6)
        #e = mycc.ipccsd(nroots=1, kptlist=(1,))[0]
        #self.assertAlmostEqual(e, -4.272015727359054, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        ##mysmp = pbmp.KMP2(mf)
        ##emp2, _ = mysmp.kernel()
        ##print("MP2 corr energy (per cell) = ", emp2 / np.prod(nmp))

        myscc = pbcc.KGCCSD(mf)
        eccsd, _, _ = myscc.kernel()
        eccsd /= np.prod(nmp)
        self.assertAlmostEqual(eccsd, -0.06146759560406628, 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        #e = myscc.eaccsd(nroots=4, kptlist=(0,))[0]
        #self.assertAlmostEqual(e[0][0], 4.183328873793568, 6)
        #self.assertAlmostEqual(e[0][1], 4.225034294249784, 6)
        #self.assertAlmostEqual(e[0][2], 5.068962665511664, 6)
        #self.assertAlmostEqual(e[0][3], 5.07942727935064 , 6)

        #e = myscc.ipccsd(nroots=4, kptlist=(0,))[0]
        #self.assertAlmostEqual(e[0][0], -4.272015724869052, 6)
        #self.assertAlmostEqual(e[0][1], -4.254298274388934, 6)
        #self.assertAlmostEqual(e[0][2], -3.471710821688812, 6)
        #self.assertAlmostEqual(e[0][3], -3.462817764320668, 6)

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
        cell.max_memory = 1000
        for i in range(len(cell.atom)):
            cell.atom[i][1] = tuple(np.dot(np.array(cell.atom[i][1]),np.array(cell.a)))
        cell.build()

        nmp = [2, 1, 1]

        kmf = pbcscf.KRHF(cell)
        kmf.kpts = cell.make_kpts(nmp, scaled_center=[0.0,0.0,0.0])
        e = kmf.kernel()

        #mymp = pbmp.KMP2(kmf)
        #ekmp2, _ = mymp.kernel()
        #print("KMP2 corr energy (per cell) = ", ekmp2)

        # By not applying a level-shift, one gets a different initial CCSD answer.
        # One can check however that the t1/t2 from level-shifting are a solution
        # of the CCSD equations done without level-shifting.
        frozen = [[0, 1, 6, 7], []]
        mycc = pbcc.KGCCSD(kmf, frozen=frozen)
        ekccsd, t1, t2 = mycc.kernel()
        self.assertAlmostEqual(ekccsd, -0.04683399814247455, 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        ## Getting more roots than 1 is difficult
        #e = mycc.eaccsd(nroots=1, kptlist=(0,))[0]
        #self.assertAlmostEqual(e, 5.060562738181741, 6)
        #e = mycc.eaccsd(nroots=1, kptlist=(1,))[0]
        #self.assertAlmostEqual(e, 4.188511644938458, 6)

        #e = mycc.ipccsd(nroots=1, kptlist=(0,))[0]
        #self.assertAlmostEqual(e, -3.477663551987023, 6)
        #e = mycc.ipccsd(nroots=1, kptlist=(1,))[0]
        #self.assertAlmostEqual(e, -4.23523412155825, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        #mysmp = pbmp.KMP2(mf)
        #emp2, _ = mysmp.kernel()
        #print("MP2 corr energy (per cell) = ", emp2 / np.prod(nmp))

        myscc = pbcc.KGCCSD(mf, frozen=[0, 1, 14, 15])
        eccsd, _, _ = myscc.kernel()
        eccsd /= np.prod(nmp)
        self.assertAlmostEqual(eccsd, -0.04683401678904569, 6)

        # TODO: fill in as the eom-kgccsd completed...
        #
        #e = myscc.eaccsd(nroots=4, kptlist=(0,))[0]
        #self.assertAlmostEqual(e[0][0], 4.188511680212755, 6)
        #self.assertAlmostEqual(e[0][1], 4.205924087610756, 6)
        #self.assertAlmostEqual(e[0][2], 5.060562771978923, 6)
        #self.assertAlmostEqual(e[0][3], 5.077249823137741, 6)

        #e = myscc.ipccsd(nroots=4, kptlist=(0,))[0]
        #self.assertAlmostEqual(e[0][0], -4.261818242746091, 6)
        #self.assertAlmostEqual(e[0][1], -4.235233956876479, 6)
        #self.assertAlmostEqual(e[0][2], -3.477663568390151, 6)
        #self.assertAlmostEqual(e[0][3], -3.459133332687474, 6)

if __name__ == '__main__':
    print("Full kpoint test")
    unittest.main()
