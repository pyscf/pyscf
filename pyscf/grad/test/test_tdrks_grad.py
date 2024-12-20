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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import numpy
from pyscf import lib
from pyscf import gto, dft
from pyscf import tdscf
from pyscf.grad import tdrks as tdrks_grad

def setUpModule():
    global mol, mf_lda, mf_gga, nstates
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        ['H' , (0. , 0. , 1.804)],
        ['F' , (0. , 0. , 0.)], ]
    mol.unit = 'B'
    mol.basis = '631g'
    mol.build()
    with lib.temporary_env(dft.radi, ATOM_SPECIFIC_TREUTLER_GRIDS=False):
        mf_lda = dft.RKS(mol).set(xc='LDA,')
        mf_lda.grids.prune = False
        mf_lda.conv_tol = 1e-10
        mf_lda.kernel()
        mf_gga = dft.RKS(mol).set(xc='b88,')
        mf_gga.grids.prune = False
        mf_gga.conv_tol = 1e-10
        mf_gga.kernel()
    nstates = 5 # to ensure the first 3 TDSCF states are converged

def tearDownModule():
    global mol, mf_lda, mf_gga
    del mol, mf_lda, mf_gga

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_tda_singlet_lda(self):
        td = tdscf.TDA(mf_lda).run(conv_tol=1e-6, nstates=nstates)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(td.xy[2])
        self.assertAlmostEqual(g1[0,2], -9.23916667e-02, 6)

    def test_tda_triplet_lda(self):
        td = tdscf.TDA(mf_lda).run(singlet=False, nstates=nstates)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.3311324654, 6)

        td_solver = td.as_scanner()
        pmol = mol.copy()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual(abs((e1[2]-e2[2])/.002 - g1[0,2]).max(), 0, 4)

    def test_tda_singlet_b88(self):
        td = tdscf.TDA(mf_gga).run(conv_tol=1e-6, nstates=nstates)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -9.32506535e-02, 6)

    @unittest.skipIf(not hasattr(dft, 'xcfun'), 'xcfun not available')
    def test_tda_singlet_b3lyp_xcfun(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp5'
        mf._numint.libxc = dft.xcfun
        mf.conv_tol = 1e-14
        mf.scf()

        td = mf.TDA()
        td.nstates = nstates
        e, z = td.kernel()
        tdg = td.Gradients()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -1.21504524e-01, 6)
# [[ 0  0  -1.21504524e-01]
#  [ 0  0   1.21505341e-01]]
        td_solver = td.as_scanner()
        pmol = mol.copy()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual(abs((e1[2]-e2[2])/.002 - g1[0,2]).max(), 0, 4)

    def test_tda_triplet_b3lyp(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp5'
        mf.conv_tol = 1e-12
        mf.kernel()
        td = tdscf.TDA(mf).run(singlet=False, nstates=nstates)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.3633375, 5)

        td_solver = td.as_scanner()
        pmol = mol.copy()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual(abs((e1[2]-e2[2])/.002 - g1[0,2]).max(), 0, 4)

    def test_tda_singlet_mgga(self):
        mf = dft.RKS(mol)
        mf.xc = 'm06l'
        mf.conv_tol = 1e-14
        mf.kernel()
        td = mf.TDA().run(nstates=nstates)
        tdg = td.Gradients()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -0.1461843283, 6)

        td_solver = td.as_scanner()
        pmol = mol.copy()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        # FIXME: why the error is larger than 1e-4? Issue of grids response?
        self.assertAlmostEqual(abs((e1[2]-e2[2])/.002 - g1[0,2]).max(), 0, 3)

    def test_tddft_lda(self):
        td = tdscf.TDDFT(mf_lda).run(nstates=nstates, conv_tol=1e-8)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -1.31315477e-01, 6)

        td_solver = td.as_scanner()
        pmol = mol.copy()
        e1 = td_solver(pmol.set_geom_('H 0 0 1.805; F 0 0 0', unit='B'))
        e2 = td_solver(pmol.set_geom_('H 0 0 1.803; F 0 0 0', unit='B'))
        self.assertAlmostEqual(abs((e1[2]-e2[2])/.002 - g1[0,2]).max(), 0, 3)

    def test_tddft_b3lyp_high_cost(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp5'
        mf._numint.libxc = dft.xcfun
        mf.grids.prune = False
        mf.scf()
        td = tdscf.TDDFT(mf).run(nstates=nstates)
        tdg = td.nuc_grad_method()
        g1 = tdg.kernel(state=3)
        self.assertAlmostEqual(g1[0,2], -1.55778110e-01, 6)

    def test_range_separated_high_cost(self):
        mol = gto.M(atom="H; H 1 1.", basis='631g', verbose=0)
        mf = dft.RKS(mol).set(xc='CAMB3LYP')
        mf._numint.libxc = dft.xcfun
        td = mf.apply(tdscf.TDA).set(nstates=nstates)
        tdg_scanner = td.nuc_grad_method().as_scanner().as_scanner()
        g = tdg_scanner(mol, state=3)[1]
        self.assertAlmostEqual(lib.fp(g), 0.60109310253094916, 6)
        smf = td.as_scanner()
        e1 = smf(mol.set_geom_("H; H 1 1.001"))[2]
        e2 = smf(mol.set_geom_("H; H 1 0.999"))[2]
        self.assertAlmostEqual((e1-e2)/0.002*lib.param.BOHR, g[1,0], 4)


if __name__ == "__main__":
    print("Full Tests for TD-RKS gradients")
    unittest.main()
