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
#          Verena Neufeld

import unittest
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import dft as pbcdft

import pyscf.pbc.cc as pbcc
import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.lib import kpts_helper


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
    #cell.verbose = 7
    cell.output = '/dev/null'
    cell.mesh = [15] * 3
    cell.build()

    rand_kmf = make_rand_kmf()

def tearDownModule():
    global cell, rand_kmf
    cell.stdout.close()
    del cell, rand_kmf


def make_rand_kmf():
    # Not perfect way to generate a random mf.
    # CSC = 1 is not satisfied and the fock matrix is neither
    # diagonal nor sorted.
    np.random.seed(2)
    nkpts = 3
    kmf = pbcscf.KRHF(cell, kpts=cell.make_kpts([1, 1, nkpts]))
    kmf.exxdiv = None
    nmo = cell.nao_nr()
    kmf.mo_occ = np.zeros((nkpts, nmo))
    kmf.mo_occ[:, :2] = 2
    kmf.mo_energy = np.arange(nmo) + np.random.random((nkpts, nmo)) * .3
    kmf.mo_energy[kmf.mo_occ == 0] += 2
    kmf.mo_coeff = (np.random.random((nkpts, nmo, nmo)) +
                    np.random.random((nkpts, nmo, nmo)) * 1j - .5 - .5j)
    ## Round to make this insensitive to small changes between PySCF versions
    #mat_veff = kmf.get_veff().round(4)
    #mat_hcore = kmf.get_hcore().round(4)
    #kmf.get_veff = lambda *x: mat_veff
    #kmf.get_hcore = lambda *x: mat_hcore
    return kmf


def run_kcell(cell, n, nk):
    #############################################
    # Do a k-point calculation                  #
    #############################################
    abs_kpts = cell.make_kpts(nk, wrap_around=True)
    #cell.verbose = 7

    # HF
    kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
    kmf.conv_tol = 1e-14
    ekpt = kmf.scf()

    # DCSD
    cc = pbcc.kdcsd_rhf.RDCSD(kmf)
    cc.conv_tol = 1e-8
    ecc, t1, t2 = cc.kernel()
    return ekpt, ecc


class KnownValues(unittest.TestCase):
    def test_311_n1_high_cost(self):
        L = 7.0
        n = 9
        cell = make_test_cell.test_cell_n1(L,[n]*3)
        nk = (3, 1, 1)
        hf_311 = -0.92687629918229486
        cc_311 = -0.04336682332846571
        escf, ecc = run_kcell(cell,n,nk)
        self.assertAlmostEqual(escf,hf_311, 8)
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
        kcc = pbcc.kdcsd_rhf.RDCSD(mf)
        kcc.level_shift = .05
        e0 = kcc.kernel()[0]

        mf = pbcscf.RHF(cell, kpt=kpts[0]).run(conv_tol=1e-9)
        mycc = pbcc.RDCSD(mf)
        e1 = mycc.kernel()[0]
        self.assertAlmostEqual(e0, e1, 5)

    def test_frozen_n3(self):
        mesh = 5
        cell = make_test_cell.test_cell_n3([mesh]*3)
        nk = (1, 1, 2)
        ehf_bench = -8.348616843863795
        ecc_bench = -0.038375755632854835

        abs_kpts = cell.make_kpts(nk, with_gamma_point=True)

        # RHF calculation
        kmf = pbcscf.KRHF(cell, abs_kpts, exxdiv=None)
        kmf.conv_tol = 1e-9
        ehf = kmf.scf()

        # KRCCSD calculation, equivalent to running supercell
        # calculation with frozen=[0,1,2] (if done with larger mesh)
        cc = pbcc.kdcsd_rhf.RDCSD(kmf, frozen=[[0],[0,1]])
        cc.diis_start_cycle = 1
        ecc, t1, t2 = cc.kernel()
        self.assertAlmostEqual(ehf, ehf_bench, 8)
        self.assertAlmostEqual(ecc, ecc_bench, 6)

    def test_dcsd_non_hf(self):
        '''Tests dcsd for non-Hartree-Fock references
        using supercell vs k-point calculation.'''
        n = 14
        cell = make_test_cell.test_cell_n3([n]*3)

        nk = [2, 1, 1]
        kpts = cell.make_kpts(nk)
        kpts -= kpts[0]
        kks = pbcdft.KRKS(cell, kpts=kpts)
        ekks = kks.kernel()

        khf = pbcscf.KRHF(cell)
        khf.__dict__.update(kks.__dict__)

        mycc = pbcc.KRDCSD(khf)
        eris = mycc.ao2mo()
        ekcc, t1, t2 = mycc.kernel(eris=eris)

        # Run supercell
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nk)
        rks = pbcdft.RKS(supcell)
        erks = rks.kernel()

        rhf = pbcscf.RHF(supcell)
        rhf.__dict__.update(rks.__dict__)

        mycc = pbcc.RDCSD(rhf)
        eris = mycc.ao2mo()
        ercc, t1, t2 = mycc.kernel(eris=eris)
        self.assertAlmostEqual(ercc/np.prod(nk), -0.16235728620565748, 6)
        self.assertAlmostEqual(ercc/np.prod(nk), ekcc, 6)


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

        mycc = pbcc.KDCSD(kmf)
        ekccsd, _, _ = mycc.kernel()
        self.assertAlmostEqual(ekccsd, -0.06351536244501263, 6)

        # Start of supercell calculations
        from pyscf.pbc.tools.pbc import super_cell
        supcell = super_cell(cell, nmp)
        supcell.build()
        mf = pbcscf.KRHF(supcell)
        e = mf.kernel()

        myscc = pbcc.KDCSD(mf)
        eccsd, _, _ = myscc.kernel()
        eccsd /= np.prod(nmp)
        self.assertAlmostEqual(eccsd, -0.06351536244501263, 6)

if __name__ == '__main__':
    unittest.main()
