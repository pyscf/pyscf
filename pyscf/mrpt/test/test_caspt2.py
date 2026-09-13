#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Huanchen Zhai <hczhai.ok@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto, scf, mcscf, mp
from pyscf.mrpt import caspt2

def setUpModule():
    global mol, mf, mc, mfdf, mcdf, mcsa
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom = [['H', (0, 0, 2 * i + j)] for i in range(7) for j in [0, 0.8]]
    mol.basis = 'sto3g'
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-16
    mf.kernel()

    mc = mcscf.CASCI(mf, 6, 8)
    mc.verbose = 0
    mc.fcisolver.conv_tol = 1e-15
    mc.kernel()

    mfdf = scf.RHF(mol).density_fit()
    mfdf.conv_tol = 1e-16
    mfdf.kernel()

    mcdf = mcscf.CASCI(mfdf, 6, 8).density_fit()
    mcdf.verbose = 0
    mcdf.fcisolver.conv_tol = 1e-15
    mcdf.kernel()

    mcsa = mcscf.CASSCF(mf, 6, 8)
    mcsa.verbose = 0
    mcsa.fcisolver.conv_tol = 1e-15
    mcsa.conv_tol = 1e-12
    mcsa.fcisolver.nroots = 4
    mcsa = mcscf.state_average_(mcsa, [1.0 / 4] * 4)
    mcsa.kernel()

def tearDownModule():
    global mol, mf, mc, mcsa
    mol.stdout.close()
    del mol, mf, mc, mcsa

class KnownValues(unittest.TestCase):
    def test_h14_energy(self):
        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738839196242218, 6)
        self.assertAlmostEqual(e_corr, -0.101052262721276, 6)

        # test no preconditioner
        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.precond = False
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 50
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738839196242218, 6)
        self.assertAlmostEqual(e_corr, -0.101052262721276, 6)

        # test density fitting
        mr = caspt2.CASPT2(mcdf, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738889000482914, 6)
        self.assertAlmostEqual(e_corr, -0.100967334903654, 6)

    def test_h14_energy_outcore(self):
        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.io_method = 'outcore'
        mr.io_blksize = 2
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738839196242218, 6)
        self.assertAlmostEqual(e_corr, -0.101052262721276, 6)

        mr = caspt2.CASPT2(mcdf, frozen=0, ipea_shift=0.0)
        mr.io_method = 'outcore'
        mr.io_blksize = 2
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738889000482914, 6)
        self.assertAlmostEqual(e_corr, -0.100967334903654, 6)

    def test_h14_frozen(self):
        mr = caspt2.CASPT2(mc, frozen=2, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696530433363129, 6)
        self.assertAlmostEqual(e_corr, -0.058743499842188, 6)

        mr = caspt2.CASPT2(mcdf, frozen=2, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696594446201059, 6)
        self.assertAlmostEqual(e_corr, -0.058672780621799, 6)

    def test_h14_ipea(self):
        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738652255819474, 6)
        self.assertAlmostEqual(e_corr, -0.100865322298533, 6)

        mr = caspt2.CASPT2(mcdf, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.738702188755190, 6)
        self.assertAlmostEqual(e_corr, -0.100780523175931, 6)

    def test_h14_frozen_ipea(self):
        mr = caspt2.CASPT2(mc, frozen=2, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696398497645751, 6)
        self.assertAlmostEqual(e_corr, -0.058611564124809, 6)

        mr = caspt2.CASPT2(mcdf, frozen=2, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696462625943729, 6)
        self.assertAlmostEqual(e_corr, -0.058540960364469, 6)

    def test_h14_frozen_ipea_outcore(self):
        mr = caspt2.CASPT2(mc, frozen=2, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.io_method = 'outcore'
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696398497645751, 6)
        self.assertAlmostEqual(e_corr, -0.058611564124809, 6)

        mr = caspt2.CASPT2(mcdf, frozen=2, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.io_method = 'outcore'
        mr.max_cycle = 20
        e_tot, e_corr = mr.kernel()
        self.assertAlmostEqual(e_tot, -7.696462625943729, 6)
        self.assertAlmostEqual(e_corr, -0.058540960364469, 6)

    def test_h14_state_averaged(self):
        self.assertAlmostEqual(mcsa.e_tot, -7.31040024150554, 5)
        e_tots = [-7.705310980116393, -7.331345990369886, -7.276966640063838, -7.201366333952514]
        e_corrs = [-0.055354234346437, -0.072478006575832, -0.083993983140532, -0.061562754412204]
        for root, (e_tot_ref, e_corr_ref) in enumerate(zip(e_tots, e_corrs)):
            mr = caspt2.CASPT2(mcsa, frozen=2, ipea_shift=0.0, root=root)
            mr.conv_tol_normt = 1e-12
            mr.max_cycle = 20
            e_tot, e_corr = mr.kernel()
            self.assertAlmostEqual(e_tot, e_tot_ref, 5)
            self.assertAlmostEqual(e_corr, e_corr_ref, 5)

    def test_o2_triplet(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='3-21g', spin=2, symmetry='d2h', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1E-16)

        mc = mcscf.CASSCF(mf, 6, 8)
        mc.fcisolver.conv_tol = 1e-14
        mc.conv_tol = 1e-12
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -148.860532892526000, 8) # openmolcas -148.8605328927

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -149.003124164605765, 6) # openmolcas -149.0031241781

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -148.999916301282497, 6) # openmolcas -148.9999122653

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.io_method = 'outcore'
        mr.io_blksize = 3
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -149.003124164605765, 6) # openmolcas -149.0031241781

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.io_method = 'outcore'
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -148.999916301282497, 6) # openmolcas -148.9999122653

    def test_h2o_pbc(self):
        from pyscf.pbc import gto, scf

        cell = gto.Cell()
        cell.basis = 'gth-dzvp'
        cell.pseudo = 'gth-hf-rev'
        cell.atom = """
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        """
        cell.a = np.identity(3) * 20
        cell.verbose = 0
        cell.output = '/dev/null'
        cell.precision = 1e-12
        cell.incore_anyway = True
        cell.build()

        mf = scf.RHF(cell, exxdiv=None).density_fit()
        mf.max_cycle = 20
        mf.kernel()

        mc = mcscf.CASCI(mf, 5, 6).density_fit()
        mc.fcisolver.conv_tol = 1e-14
        mc.verbose = 0
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -16.599331298138100, 8)

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -16.800730004379748, 6)

    def test_n2_ncas_zero(self):
        mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto3g', spin=0, symmetry='d2h', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1E-16)

        mc = mcscf.CASCI(mf, 0, 0)
        mc.mo_coeff = mf.mo_coeff

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()

        mx = mp.MP2(mf).run()
        self.assertAlmostEqual(mr.e_tot, mx.e_tot, 8)

        mf = scf.RHF(mol).density_fit().run(conv_tol=1E-16)

        mc = mcscf.CASCI(mf, 0, 0).density_fit()
        mc.mo_coeff = mf.mo_coeff

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()

        mx = mp.MP2(mf).run()
        self.assertAlmostEqual(mr.e_tot, mx.e_tot, 8)

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.0)
        mr.io_method = 'outcore'
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, mx.e_tot, 8)

    def test_o2_nvir_zero(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='sto3g', spin=2, symmetry='d2h', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1E-16)

        mc = mcscf.CASSCF(mf, 6, 8)
        mc.fcisolver.conv_tol = 1e-14
        mc.conv_tol = 1e-12
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -147.737311636923000, 8)

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -147.741885949329969, 6)

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.io_method = 'outcore'
        mr.io_blksize = 2
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -147.741885949329969, 6)

        mf = scf.RHF(mol).density_fit().run(conv_tol=1E-16)

        mc = mcscf.CASSCF(mf, 6, 8).density_fit()
        mc.fcisolver.conv_tol = 1e-14
        mc.conv_tol = 1e-12
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -147.737383841650000, 8)

        mr = caspt2.CASPT2(mc, frozen=0, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -147.741956779797476, 6)

    def test_o2_ncore_zero(self):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.207', basis='3-21g', spin=2, symmetry='d2h', verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1E-16)

        mc = mcscf.CASSCF(mf, 6, 8)
        mc.fcisolver.conv_tol = 1e-14
        mc.conv_tol = 1e-12
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -148.860532892526000, 8)

        mr = caspt2.CASPT2(mc, frozen=4, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -148.934693321749251, 6)

        mr = caspt2.CASPT2(mc, frozen=4, ipea_shift=0.25)
        mr.io_method = 'outcore'
        mr.io_blksize = 1
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -148.934693321749251, 6)

        mf = scf.RHF(mol).density_fit().run(conv_tol=1E-16)

        mc = mcscf.CASSCF(mf, 6, 8).density_fit()
        mc.fcisolver.conv_tol = 1e-14
        mc.conv_tol = 1e-12
        mc.kernel()
        self.assertAlmostEqual(mc.e_tot, -148.860195506983000, 8)

        mr = caspt2.CASPT2(mc, frozen=4, ipea_shift=0.25)
        mr.conv_tol_normt = 1e-12
        mr.max_cycle = 20
        mr.kernel()
        self.assertAlmostEqual(mr.e_tot, -148.934354332369566, 6)
