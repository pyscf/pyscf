#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import grad
from pyscf.qmmm import itrf

def setUpModule():
    global mol, mm_coords, mm_charges, mm_radii
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''O       -1.464   0.099   0.300
                  H       -1.956   0.624  -0.340
                  H       -1.797  -0.799   0.206''',
        basis = '631G')

    mm_coords = [(1.369, 0.146,-0.395),
                  (1.894, 0.486, 0.335),
                  (0.451, 0.165,-0.083)]
    mm_charges = [-1.040, 0.520, 0.520]
    mm_radii = [6.3, 0.32, 0.32]

def tearDownModule():
    global mol, mm_coords, mm_charges, mm_radii


class KnowValues(unittest.TestCase):
    def test_grad_mm(self):
        mf = itrf.mm_charge(scf.RHF(mol), mm_coords, mm_charges, mm_radii)
        e_hf = mf.kernel()
        self.assertAlmostEqual(e_hf, -76.02861628942037, 8)

        # qm grad
        g_hf = itrf.mm_charge_grad(grad.RHF(mf), mm_coords, mm_charges, mm_radii)
        g_hf_qm = g_hf.kernel()
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_qm), 0.04561356204333569, 6)

        # mm grad
        g_hf_mm_h1 = g_hf.grad_hcore_mm(mf.make_rdm1())
        g_hf_mm_nuc = g_hf.grad_nuc_mm()
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_mm_h1), 0.38906742696919, 6)
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_mm_nuc), 0.37076081680889555, 6)

        # finite difference for MM atoms
        mm_coords1 = [(1.369, 0.147,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mf1 = itrf.mm_charge(scf.RHF(mol), mm_coords1, mm_charges, mm_radii)
        e1 = mf1.kernel()

        mm_coords2 = [(1.369, 0.145,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mf2 = itrf.mm_charge(scf.RHF(mol), mm_coords2, mm_charges, mm_radii)
        e2 = mf2.kernel()
        self.assertAlmostEqual((e1 - e2) / 0.002*lib.param.BOHR,
                               (g_hf_mm_h1+g_hf_mm_nuc)[0,1], 6)

    def test_grad_mm_gaussian_model(self):
        mm_r20 = numpy.array(mm_radii)*20
        mf = itrf.mm_charge(scf.RHF(mol), mm_coords, mm_charges, mm_r20)
        g_hf = itrf.mm_charge_grad(grad.RHF(mf), mm_coords, mm_charges, mm_r20)
        g_hf_qm_nuc = g_hf.grad_nuc()
        g_hf_mm_nuc = g_hf.grad_nuc_mm()

        # finite difference for QM atoms
        mol1 = mol.set_geom_('''O       -1.464   0.099   0.300
                                H       -1.956   0.624  -0.3395
                                H       -1.797  -0.799   0.206''', inplace=False)
        e1 = itrf.mm_charge(scf.RHF(mol1), mm_coords, mm_charges, mm_r20).energy_nuc()
        mol1 = mol.set_geom_('''O       -1.464   0.099   0.300
                                H       -1.956   0.624  -0.3405
                                H       -1.797  -0.799   0.206''', inplace=False)
        e2 = itrf.mm_charge(scf.RHF(mol1), mm_coords, mm_charges, mm_r20).energy_nuc()
        self.assertAlmostEqual((e1 - e2) / 0.001*lib.param.BOHR, g_hf_qm_nuc[1,2], 6)

        # finite difference for MM atoms
        mm_coords1 = [(1.369, 0.147,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        e1 = itrf.mm_charge(scf.RHF(mol), mm_coords1, mm_charges, mm_r20).energy_nuc()
        mm_coords2 = [(1.369, 0.145,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        e2 = itrf.mm_charge(scf.RHF(mol), mm_coords2, mm_charges, mm_r20).energy_nuc()
        self.assertAlmostEqual((e1 - e2) / 0.002*lib.param.BOHR, g_hf_mm_nuc[0,1], 6)

    def test_grad_mm_point_charge(self):
        mf = itrf.mm_charge(scf.RHF(mol), mm_coords, mm_charges)
        e_hf = mf.kernel()
        self.assertAlmostEqual(e_hf, -76.00057498193152, 8)

        # qm grad
        g_hf = itrf.mm_charge_grad(grad.RHF(mf), mm_coords, mm_charges)
        g_hf_qm = g_hf.kernel()
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_qm), 0.030903934128232773, 6)

        # mm grad
        g_hf_mm_h1 = g_hf.grad_hcore_mm(mf.make_rdm1())
        g_hf_mm_nuc = g_hf.grad_nuc_mm()
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_mm_h1), 0.511663689758269, 6)
        self.assertAlmostEqual(numpy.linalg.norm(g_hf_mm_nuc), 0.4915404602273757, 6)

        # finite difference for MM atoms
        mm_coords1 = [(1.369, 0.147,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mf1 = itrf.mm_charge(scf.RHF(mol), mm_coords1, mm_charges)
        e1 = mf1.kernel()

        mm_coords2 = [(1.369, 0.145,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mf2 = itrf.mm_charge(scf.RHF(mol), mm_coords2, mm_charges)
        e2 = mf2.kernel()
        self.assertAlmostEqual((e1 - e2) / 0.002*lib.param.BOHR,
                               (g_hf_mm_h1+g_hf_mm_nuc)[0,1], 6)

if __name__ == "__main__":
    print("Full Tests for qmmm MM force.")
    unittest.main()
