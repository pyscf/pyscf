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
# Authors: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np
from pyscf.lib.misc import finger
from pyscf.pbc import gto
from pyscf.pbc import scf
from pyscf.pbc.scf import khf, kuhf
from pyscf.pbc.lib import kpts as libkpts

def setUpModule():
    global cell, cell1, kpts0, kmf
    cell = gto.Cell()
    cell.atom = """
        Si  0.0 0.0 0.0
        Si  1.3467560987 1.3467560987 1.3467560987
    """
    cell.a = [[0.0, 2.6935121974, 2.6935121974],
              [2.6935121974, 0.0, 2.6935121974],
              [2.6935121974, 2.6935121974, 0.0]]
    cell.basis = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.mesh = [20,] * 3
    cell.space_group_symmetry = True
    cell.build()

    cell1 = cell.copy()
    cell1.build(symmorphic=True)

    kpts0 = cell.make_kpts([3,3,3])
    kmf = scf.KRKS(cell, kpts0)
    kmf.max_cycle=1
    kmf.kernel()

def tearDownModule():
    global cell, cell1, kpts0, kmf
    del cell, cell1, kpts0, kmf

class KnownValues(unittest.TestCase):
    def test_make_kpts_ibz(self):
        kmesh = [16,16,16]
        kpts = cell.make_kpts(kmesh, space_group_symmetry=True)
        self.assertEqual(kpts.nkpts_ibz, 145)
        error = False
        for star, star_op in zip(kpts.stars, kpts.stars_ops):
            for i, k in enumerate(star):
                if star_op[i] != kpts.stars_ops_bz[k]:
                    error = True
                    break
        self.assertEqual(error, False)
        self.assertAlmostEqual(finger(kpts.kpts_ibz), 2.211640884021115, 9)
        #self.assertAlmostEqual(finger(kpts.stars_ops_bz), 61.98395458751813, 9)

        kpts1 = cell1.make_kpts(kmesh, space_group_symmetry=True, time_reversal_symmetry=True)
        self.assertAlmostEqual(abs(kpts1.kpts_ibz - kpts.kpts_ibz).max(), 0, 9)

        kpts2 = cell1.make_kpts(kmesh, space_group_symmetry=True, time_reversal_symmetry=False)
        self.assertEqual(kpts2.nkpts_ibz, 245)
        self.assertAlmostEqual(finger(kpts2.kpts_ibz), -2.0196383066365353, 9)
        #self.assertAlmostEqual(finger(kpts2.stars_ops_bz), 177.9781708308629, 9)

        kpts3 = cell.make_kpts(kmesh, with_gamma_point=False, space_group_symmetry=True)
        self.assertEqual(kpts3.nkpts_ibz, 408)
        self.assertAlmostEqual(finger(kpts3.kpts_ibz), -2.581114561328012, 9)
        #self.assertAlmostEqual(finger(kpts3.stars_ops_bz), -9.484769880571442, 9)

        kpts4 = cell1.make_kpts(kmesh, with_gamma_point=False, space_group_symmetry=True)
        self.assertEqual(kpts4.nkpts_ibz, 816)
        self.assertAlmostEqual(finger(kpts4.kpts_ibz), -1.124492399508386, 9)
        #self.assertAlmostEqual(finger(kpts4.stars_ops_bz), -16.75874526830733, 9)

        kpts5 = cell.make_kpts(kmesh, time_reversal_symmetry=True)
        self.assertEqual(kpts5.nkpts_ibz, 2052)

    def test_transform(self):
        kpts = libkpts.make_kpts(cell, kpts0, space_group_symmetry=True, time_reversal_symmetry=True)
        dms_ibz = kmf.make_rdm1()[kpts.ibz2bz]
        dms_bz = kpts.transform_dm(dms_ibz)
        self.assertAlmostEqual(abs(dms_bz - kmf.make_rdm1()).max(), 0, 7)

        mo_coeff_ibz = np.asarray(kmf.mo_coeff)[kpts.ibz2bz]
        mo_coeff_bz = kpts.transform_mo_coeff(mo_coeff_ibz)
        dms_bz = khf.make_rdm1(mo_coeff_bz, kmf.mo_occ)
        self.assertAlmostEqual(abs(dms_bz - kmf.make_rdm1()).max(), 0, 7)

        mo_occ_ibz = kpts.check_mo_occ_symmetry(kmf.mo_occ)
        mo_occ_bz = kpts.transform_mo_occ(mo_occ_ibz)
        self.assertAlmostEqual(abs(mo_occ_bz - np.asarray(kmf.mo_occ)).max(), 0, 8)

        mo_energy_ibz = np.asarray(kmf.mo_energy)[kpts.ibz2bz]
        mo_energy_bz = kpts.transform_mo_energy(mo_energy_ibz)
        self.assertAlmostEqual(abs(mo_energy_bz - np.asarray(kmf.mo_energy)).max(), 0 , 7)

        fock_ibz = kmf.get_fock()[kpts.ibz2bz]
        fock_bz = kpts.transform_fock(fock_ibz)
        self.assertAlmostEqual(abs(fock_bz - kmf.get_fock()).max(), 0, 7)

        kumf = kmf.to_uhf()
        mo_coeff_ibz = np.asarray(kumf.mo_coeff)[:,kpts.ibz2bz]
        mo_coeff_bz = kpts.transform_mo_coeff(mo_coeff_ibz)
        dms_bz = kuhf.make_rdm1(mo_coeff_bz, kumf.mo_occ)
        self.assertAlmostEqual(abs(dms_bz - kumf.make_rdm1()).max(), 0, 7)

        mo_occ_ibz = np.asarray(kumf.mo_occ)[:,kpts.ibz2bz]
        mo_occ_bz = kpts.transform_mo_occ(mo_occ_ibz)
        self.assertAlmostEqual(abs(mo_occ_bz - np.asarray(kumf.mo_occ)).max(), 0, 8)

        mo_energy_ibz = np.asarray(kumf.mo_energy)[:,kpts.ibz2bz]
        mo_energy_bz = kpts.transform_mo_energy(mo_energy_ibz)
        self.assertAlmostEqual(abs(mo_energy_bz - np.asarray(kumf.mo_energy)).max(), 0 , 7)

        fock_ibz = kumf.get_fock()[:,kpts.ibz2bz]
        fock_bz = kpts.transform_fock(fock_ibz)
        self.assertAlmostEqual(abs(fock_bz - kumf.get_fock()).max(), 0, 7)

    def test_symmetrize_density(self):
        rho0 = kmf.get_rho()

        kpts = libkpts.make_kpts(cell, kpts0, space_group_symmetry=True, time_reversal_symmetry=True)
        dms_ibz = kmf.make_rdm1()[kpts.ibz2bz]
        nao = dms_ibz.shape[-1]
        rho = 0.
        for k in range(kpts.nkpts_ibz):
            rho_k = khf.get_rho(kmf, dms_ibz[k].reshape((-1,nao,nao)), kpts=kpts.kpts_ibz[k].reshape((-1,3)))
            rho += kpts.symmetrize_density(rho_k, k, cell.mesh)
        rho *= 1.0 / kpts.nkpts
        self.assertAlmostEqual(abs(rho - rho0).max(), 0, 8)


if __name__ == "__main__":
    print("Tests for kpts")
    unittest.main()
