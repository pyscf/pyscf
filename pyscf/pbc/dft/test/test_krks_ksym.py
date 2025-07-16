#!/usr/bin/env python
# Copyright 2020-2023 The PySCF Developers. All Rights Reserved.
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

from pyscf.lib import finger
from pyscf.dft import radi
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.scf import chkfile
from pyscf.pbc.dft import krks, kuks, multigrid

def make_primitive_cell(mesh):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = 'Si 0.,  0.,  0.; Si 1.3467560987,  1.3467560987,  1.3467560987'
    cell.a = '''0.            2.6935121974    2.6935121974
                2.6935121974  0.              2.6935121974
                2.6935121974  2.6935121974    0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = mesh
    cell.spin = 0
    cell.verbose = 5
    cell.output = '/dev/null'
    cell.space_group_symmetry = True
    cell.build()
    return cell

def setUpModule():
    global cell, He, nk
    cell = make_primitive_cell([16]*3)
    nk = [1,2,2]

    L = 2.
    He = pbcgto.Cell()
    He.verbose = 0
    He.a = np.eye(3)*L
    He.atom =[['He' , ( L/2+0., L/2+0., L/2+0.)],]
    He.basis = {'He': [[0, (4.0, 1.0)], [0, (1.0, 1.0)]]}
    He.space_group_symmetry = True
    He.build()

def tearDownModule():
    global cell, He, nk
    del cell, He, nk

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_krks_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kmf0 = krks.KRKS(cell, kpts=kpts0)
        kmf0.xc = 'lda'
        kmf0.kernel()
        rho0 = kmf0.get_rho()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)
        rho = kmf.get_rho()
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

    def test_krks_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(cell, kpts=kpts0)
        kmf0.xc = 'lda'
        kmf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 6)

    def test_kuks_gamma_center(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=True)
        kumf0 = kuks.KUKS(cell, kpts=kpts0)
        kumf0.xc = 'lda'
        kumf0.kernel()
        rho0 = kumf0.get_rho()

        kpts = cell.make_kpts(nk, with_gamma_point=True,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf.kernel()
        rho = kumf.get_rho()
        self.assertAlmostEqual(kumf.e_tot, kumf0.e_tot, 7)
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

    def test_kuks_monkhorst(self):
        kpts0 = cell.make_kpts(nk, with_gamma_point=False)
        kumf0 = kuks.KUKS(cell, kpts=kpts0)
        kumf0.xc = 'lda'
        kumf0.kernel()

        kpts = cell.make_kpts(nk, with_gamma_point=False,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf.kernel()
        self.assertAlmostEqual(kumf.e_tot, kumf0.e_tot, 6)

    def test_krks_symorb(self):
        cell1 = cell.copy()
        cell1.build(symmorphic=True)
        kpts = cell1.make_kpts([2,2,2], with_gamma_point=True,space_group_symmetry=True)
        kmf = pscf.KRKS(cell1, kpts=kpts).run()
        kmf1 = pscf.KRKS(cell1, kpts=kpts, use_ao_symmetry=False).run()
        self.assertAlmostEqual(kmf.e_tot, kmf1.e_tot, 7)
        assert abs(kmf.mo_coeff[0].orbsym - np.asarray([0, 4, 4, 4, 4, 4, 4, 0])).sum() == 0
        assert abs(kmf.mo_coeff[1].orbsym - np.asarray([0, 3, 4, 4, 0, 3, 4, 4])).sum() == 0
        assert abs(kmf.mo_coeff[2].orbsym - np.asarray([0, 0, 2, 2, 0, 2, 2, 0])).sum() == 0
        assert getattr(kmf1.mo_coeff[0], 'orbsym', None) is None

    def test_rsh(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0)
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts)
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 9)

        kmf.xc = 'wb97'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, -2.504752684826571, 7)

    def test_lda_df(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).density_fit()
        kmf0.xc = 'lda'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts).density_fit()
        kmf.xc = 'lda'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

    def test_gga_df(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).density_fit()
        kmf0.xc = 'pbe'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts).density_fit()
        kmf.xc = 'pbe'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

    def test_gga_df_newton(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).density_fit()
        kmf0.xc = 'pbe'
        kmf0 = kmf0.newton()
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts).density_fit()
        kmf.xc = 'pbe'
        kmf = kmf.newton()
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 9)

    def test_rsh_df(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).density_fit()
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts).density_fit()
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

        kmf.xc = 'hse06'
        kmf.kernel()
        # Different to pyscf-2.7 since SR part was computed as DF_FR - DF_LR. A
        # direct DF_SR would lead to smaller HF exchange. The total energy is
        # slightly lower.
        self.assertAlmostEqual(kmf.e_tot, -2.4973700806518035, 5)

    def test_rsh_mdf_high_cost(self):
        kpts0 = He.make_kpts(nk, with_gamma_point=False)
        kmf0 = krks.KRKS(He, kpts=kpts0).mix_density_fit()
        kmf0.xc = 'camb3lyp'
        kmf0.kernel()

        kpts = He.make_kpts(nk, with_gamma_point=False, space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(He, kpts=kpts).mix_density_fit()
        kmf.xc = 'camb3lyp'
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

    def test_multigrid(self):
        kmf0 = krks.KRKS(cell, kpts=cell.make_kpts(nk))
        kmf0.xc = 'lda'
        kmf0._numint = multigrid.MultiGridNumInt(cell)
        kmf0.kernel()
        rho0 = kmf0.get_rho()

        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf._numint = multigrid.MultiGridNumInt(cell)
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)
        rho = kmf.get_rho()
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

        kmf._numint = multigrid.MultiGridNumInt(cell)
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)
        rho = kmf.get_rho()
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

    def test_multigrid_kuks(self):
        kmf0 = pscf.KUKS(cell, kpts=cell.make_kpts(nk))
        kmf0.xc = 'lda'
        kmf0._numint = multigrid.MultiGridNumInt(cell)
        kmf0.kernel()
        rho0 = kmf0.get_rho()

        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KUKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf._numint = multigrid.MultiGridNumInt(cell)
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)
        rho = kmf.get_rho()
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

        kmf._numint = multigrid.MultiGridNumInt(cell)
        kmf.kernel()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 7)
        rho = kmf.get_rho()
        error = np.amax(np.absolute(rho - rho0))
        self.assertAlmostEqual(error, 0., 7)

    def test_to_uhf(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf.xc = 'lda'
        kmf.kernel()
        dm = kmf.make_rdm1()
        dm = np.asarray([dm,dm]) / 2.

        kumf = kmf.to_uks()
        kumf.max_cycle = 1
        kumf.kernel(dm)
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 8)

    def test_to_rhf(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf.kernel()
        dm = kumf.make_rdm1()

        kmf = kumf.to_rks()
        kmf.max_cycle = 1
        kmf.kernel(dm[0]+dm[1])
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 8)

    def test_convert_from(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf.kernel()
        dm = kumf.make_rdm1()

        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf = kmf.convert_from_(kumf)
        kmf.max_cycle = 1
        kmf.kernel(dm[0]+dm[1])
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 8)

        dm = kmf.make_rdm1()
        kumf = kumf.convert_from_(kmf)
        kumf.max_cycle = 1
        kumf.kernel(np.asarray([dm,dm]) / 2.)
        self.assertAlmostEqual(kmf.e_tot, kumf.e_tot, 7)

    def test_get_bands(self):
        kpts = cell.make_kpts(nk,
                              space_group_symmetry=True,
                              time_reversal_symmetry=True)
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.xc = 'lda'
        kumf.chkfile = 'test_get_bands_ksymm.chk'
        kumf.kernel()

        band_kpts = np.array([[-0.205736, 0.308604, 0.308604],
                              [-0.25717,  0.308604, 0.308604]])
        kumf = pscf.KUKS(cell, kpts=kpts)
        kumf.__dict__.update(chkfile.load('test_get_bands_ksymm.chk', 'scf'))
        E_nk = kumf.get_bands(band_kpts, kpts=kpts)[0]
        E_F = kumf.get_fermi()
        self.assertAlmostEqual(finger(np.asarray(E_nk[0])), 0.5575755379561839, 6)
        self.assertAlmostEqual(finger(np.asarray(E_nk[1])), 0.5575755379561839, 6)
        self.assertAlmostEqual(E_F[0], 0.3093399745201863, 6)
        self.assertAlmostEqual(E_F[1], 0.3093399745201863, 6)

        kmf = pscf.KRKS(cell, kpts=kpts)
        kmf = kmf.convert_from_(kumf)
        E_nk = kmf.get_bands(band_kpts, kpts=kpts)[0]
        E_F = kmf.get_fermi()
        self.assertAlmostEqual(finger(np.asarray(E_nk)), 0.5575755379561839, 6)
        self.assertAlmostEqual(E_F, 0.3093399745201863, 6)

    def test_krks_multigrid_newton(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KRKS(cell, kpts=kpts).multigrid_numint().newton().run()
        kmf0 = krks.KRKS(cell, kpts=kpts.kpts).multigrid_numint().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

    def test_kuks_multigrid_newton(self):
        kpts = cell.make_kpts(nk,space_group_symmetry=True,time_reversal_symmetry=True)
        kmf = pscf.KUKS(cell, kpts=kpts).multigrid_numint().newton().run()
        kmf0 = kuks.KUKS(cell, kpts=kpts.kpts).multigrid_numint().run()
        self.assertAlmostEqual(kmf.e_tot, kmf0.e_tot, 8)

if __name__ == '__main__':
    print("Full Tests for DFT with k-point symmetry")
    unittest.main()
