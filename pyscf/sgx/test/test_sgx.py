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

import unittest
from pyscf import gto
from pyscf import scf, dft
from pyscf.sgx import sgx


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_reset(self):
        mol = gto.M(atom='He')
        mol1 = gto.M(atom='C')
        mf = scf.RHF(mol).COSX()
        mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertEqual(mf.undo_sgx().__class__.__name__, 'RHF')

    def test_sgx_scf(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )
        mf = sgx.sgx_fit(scf.RHF(mol), 'weigend')
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df._symm_ovlp_fit = False
        energy = mf.kernel()
        self.assertAlmostEqual(energy, -76.02686422219752, 9)

        mf = sgx.sgx_fit(scf.RHF(mol))
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.optk = False
        mf.with_df.dfj = False
        energy = mf.kernel()
        self.assertAlmostEqual(energy, -76.02673747035047, 8)

    def test_sgx_pjs(self):
        mol = gto.Mole()
        atom0 = [["O" , (0. , 0.     , 0.)],
                 [1   , (0. , -0.757 , 0.587)],
                 [1   , (0. , 0.757  , 0.587)],]
        atom = []
        for i in range(3):
            atom = atom + [[z, (c[0] + i * 5, c[1], c[2])] for z, c in atom0]
        mol.build(
            atom=atom,
            basis='sto-3g',
            verbose=0,
        )
        mf = sgx.sgx_fit(mol.RHF())
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.sgx_tol_energy = None
        mf.with_df.sgx_tol_potential = None
        energy = mf.kernel()
        mf = sgx.sgx_fit(mol.RHF())
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.sgx_tol_energy = None
        mf.with_df.sgx_tol_potential = None
        mf.with_df.sgx_tol_energy = 1e-9
        mf.with_df.sgx_tol_potential = "auto"
        energy2 = mf.kernel()
        self.assertAlmostEqual(energy, energy2, 9)

    def test_sgx_tda(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )

        def run_tda(mf, singlet=True):
            td = mf.TDA()
            td.singlet = singlet
            td.nstates = 3
            td.max_cycle = 100
            e = td.kernel()[0]
            self.assertTrue(all(td.converged))
            return e

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-10
        mf.kernel()
        e_singlet = run_tda(mf)
        e_triplet = run_tda(mf, singlet=False)

        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.conv_tol = 1e-10
        mf.kernel()
        e_pbe = run_tda(mf)

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        mf_sgx.kernel()
        e_sgx_singlet = run_tda(mf_sgx)
        e_sgx_triplet = run_tda(mf_sgx, singlet=False)

        # SGX screening errors largely cancel in the response: the TDA
        # excitation energies of the SGX ground state agree with the
        # exact-K result within the screening tolerance.
        self.assertLess(abs(e_sgx_singlet - e_singlet).max(), 2e-6)
        self.assertLess(abs(e_sgx_triplet - e_triplet).max(), 2e-6)

        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        mf_sgx.kernel()
        e_sgx_pbe = run_tda(mf_sgx)
        self.assertLess(abs(e_sgx_pbe - e_pbe).max(), 2e-6)

    def test_sgx_tddft(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-10
        mf.kernel()
        td = mf.TDDFT()
        td.nstates = 3
        td.max_cycle = 100
        e_ref = td.kernel()[0]
        self.assertTrue(all(td.converged))

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        mf_sgx.kernel()
        td = mf_sgx.TDDFT()
        td.nstates = 3
        td.max_cycle = 100
        e_sgx = td.kernel()[0]
        self.assertTrue(all(td.converged))
        self.assertLess(abs(e_sgx - e_ref).max(), 2e-6)

    def test_sgx_tda_rsh(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = 'cam-b3lyp'
        mf.conv_tol = 1e-10
        mf.kernel()
        td = mf.TDA()
        td.nstates = 3
        td.max_cycle = 100
        e_ref = td.kernel()[0]
        self.assertTrue(all(td.converged))

        mf = dft.RKS(mol)
        mf.xc = 'cam-b3lyp'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        mf_sgx.kernel()
        td = mf_sgx.TDA()
        td.nstates = 3
        td.max_cycle = 100
        e_sgx = td.kernel()[0]
        self.assertTrue(all(td.converged))
        self.assertLess(abs(e_sgx - e_ref).max(), 2e-6)

    def test_sgx_soscf(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )
        # newton-first pattern: the second-order solver applied to an
        # SGX object that has not run a plain kernel yet
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        nt = mf_sgx.newton()
        nt.conv_tol = 1e-10
        e_soscf = nt.kernel()
        self.assertTrue(nt.converged)

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        mf_sgx = sgx.sgx_fit(mf)
        mf_sgx.conv_tol = 1e-10
        e_scf = mf_sgx.kernel()
        # The second-order solver converges to the same SGX SCF solution
        self.assertAlmostEqual(e_soscf, e_scf, 8)

if __name__ == "__main__":
    print("Full Tests for SGX")
    unittest.main()
