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

import unittest
import warnings
import numpy as np
import pyscf
from pyscf import gto, scf, lib, dft
from pyscf.scf import dispersion


class KnownKS(dft.KohnShamDFT):
    def __init__(self, xc='b3lyp'):
        self.xc = xc
        self.disp = None


class KnownHF(scf.hf.SCF):
    def __init__(self):
        self.disp = None


class TestDispersionLogic(unittest.TestCase):
    def test_d3cso(self):
        self.assertEqual(dispersion.parse_disp('pbe-d3cso'), ('pbe', 'd3cso', False))
        # simple-dftd3 test/unit/test_param.f90::test_d3cso_mb01 (MB16-43/01)
        mol = gto.M(atom='''
            Na -1.85528263484662 +3.58670515364616 -2.41763729306344
            H  +4.40178023537845 +0.02338844412653 -4.95457749372945
            O  -2.98706033463438 +4.76252065456814 +1.27043301573532
            H  +0.79980886075526 +1.41103455609189 -5.04655321620119
            F  -4.20647469409936 +1.84275767548460 +4.55038084858449
            H  -3.54356121843970 -3.18835665176557 +1.46240021785588
            H  +2.70032160109941 +1.06818452504054 -1.73234650374438
            O  +3.73114088824361 -2.07001543363453 +2.23160937604731
            N  -1.75306819230397 +0.35951417150421 +1.05323406177129
            H  +5.41755788583825 -1.57881830078929 +1.75394002750038
            H  -2.23462868255966 -2.13856505054269 +4.10922285746451
            Cl +1.01565866207568 -3.21952154552768 -3.36050963020778
            B  +2.42119255723593 +0.26626435093114 -3.91862474360560
            B  -3.02526098819107 +2.53667889095925 +2.31664984740423
            N  -2.00438948664892 -2.29235136977220 +2.19782807357059
            Al +1.12226554109716 -1.36942007032045 +0.48455055461782
        ''', unit='Bohr', verbose=0)
        mf = mol.RKS(xc='pbe-d3cso')
        ref = -0.025368289609215090
        self.assertAlmostEqual(mf.get_dispersion(), ref, 12)

    def test_dispersion_ghost_atoms(self):
        real_atoms = 'O 0 0 0; H 0 0 1; H 0 1 0'
        ghost_atoms = 'ghost-H 2 0 0; O 0 0 0; X-H 0 2 0; H 0 0 1; H 0 1 0'
        for charge in [0, 1]:
            mol = gto.M(atom=ghost_atoms, charge=charge, spin=charge, verbose=0)
            real_mol = gto.M(atom=real_atoms, charge=charge, spin=charge, verbose=0)
            mf = mol.UKS(xc='pbe')
            real_mf = real_mol.UKS(xc='pbe')
            coords = mol.atom_coords().copy()
            nao = mol.nao
            for version in ['d3bj', 'd3cso', 'd3bjatm', 'd4']:
                with self.subTest(charge=charge, version=version):
                    mf.disp = real_mf.disp = version
                    self.assertAlmostEqual(mf.get_dispersion(), real_mf.get_dispersion(), 12)
            self.assertEqual(mol.natm, 5)
            self.assertEqual(mol.nao, nao)
            self.assertEqual(mol.charge, charge)
            np.testing.assert_array_equal(mol.atom_coords(), coords)

    def test_dispersion_all_ghost_atoms(self):
        mol = gto.M(atom='ghost-H 0 0 0; X-H 0 0 1', verbose=0)
        mf = mol.RKS(xc='pbe')
        for version in ['d3bj', 'd4']:
            mf.disp = version
            self.assertEqual(mf.get_dispersion(), 0.)
            self.assertEqual(mf.scf_summary['dispersion'], 0.)

    def test_dispersion_mol(self):
        from pyscf.pbc import gto as pbcgto
        mol = gto.M(atom='Na 0 0 0; H 0 0 1.9; ghost-H 0 1 0',
                    basis='lanl2dz', ecp={'Na': 'lanl2dz'}, verbose=0)
        real_mol, atom_idx = dispersion._get_dispersion_mol(mol)
        np.testing.assert_array_equal(atom_idx, [0,1])
        np.testing.assert_array_equal(real_mol.atom_charges(), mol.atom_charges()[:2])
        self.assertTrue(real_mol.has_ecp())
        self.assertEqual(mol.natm, 3)
        self.assertIs(dispersion._get_dispersion_mol(real_mol)[0], real_mol)

        for fractional in [False, True]:
            cell = pbcgto.M(atom='H 0 0 0; ghost-H 0 2 0; H 0 0 1',
                           a=np.eye(3)*8, dimension=1, low_dim_ft_type='inf_vacuum', fractional=fractional, verbose=0)
            real_cell, atom_idx = dispersion._get_dispersion_mol(cell)
            np.testing.assert_array_equal(atom_idx, [0,2])
            np.testing.assert_allclose(real_cell.lattice_vectors(), cell.lattice_vectors())
            np.testing.assert_allclose(real_cell.atom_coords(), cell.atom_coords()[atom_idx])
            self.assertEqual(real_cell.dimension, cell.dimension)
            self.assertEqual(cell.natm, 3)

    def test_parse_disp_none(self):
        # Case 1: All None
        self.assertEqual(dispersion.parse_disp(None, None), (None, None, False))

    def test_parse_disp_explicit(self):
        # Case 2: Explicit disp takes precedence
        # b3lyp normally has no disp.
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj'), ('b3lyp', 'd3bj', False))

        # disp with colon override method
        self.assertEqual(dispersion.parse_disp(None, 'd4:wb97x-3c'), ('wb97x-3c', 'd4', True))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj:pbe'), ('pbe', 'd3bj', False))

        # disp with suffix
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bj2b'), ('b3lyp', 'd3bj', False))
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd3bjatm'), ('b3lyp', 'd3bj', True))

        # d4 always implies 3body
        self.assertEqual(dispersion.parse_disp('b3lyp', 'd4'), ('b3lyp', 'd4', True))

    def test_parse_disp_from_method(self):
        # Case 3: Infer from method
        # b3lyp -> no disp
        self.assertEqual(dispersion.parse_disp('b3lyp'), (None, None, False))

        # wb97x-d3bj -> d3bj
        self.assertEqual(dispersion.parse_disp('wb97x-d3bj'), ('wb97x', 'd3bj', False))

        # wb97x-d4s -> d4s
        self.assertEqual(dispersion.parse_disp('wb97x-d4s'), ('wb97x', 'd4s', True))

        # wb97x-3c -> d4, 3body=True (from whitelist)
        self.assertEqual(dispersion.parse_disp('wb97x-3c'), ('wb97x-3c', 'd4', True))

        # b97-3c -> d3bj with ATM 3-body (from whitelist; per the B97-3c paper)
        self.assertEqual(dispersion.parse_disp('b97-3c'), ('b97-3c', 'd3bj', True))

        # r2scan-3c -> d4, 3body=True (from whitelist)
        self.assertEqual(dispersion.parse_disp('r2scan-3c'), ('r2scan-3c', 'd4', True))

        # underscore spellings are equivalent to the dash spellings
        self.assertEqual(dispersion.parse_disp('b97_3c'), ('b97-3c', 'd3bj', True))
        self.assertEqual(dispersion.parse_disp('r2scan_3c'), ('r2scan-3c', 'd4', True))

    def test_parse_dft_3c_spellings(self):
        # Issue #3399: all spellings of a supported 3c method must resolve to
        # the same (xc, nlc, disp) triple, and unsupported 3c methods must
        # raise for every spelling.
        self.assertEqual(dispersion.parse_dft('b97-3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('b97_3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('gga_xc_b97_3c'), ('b97-3c', False, 'd3bjatm:b97-3c'))
        self.assertEqual(dispersion.parse_dft('r2scan-3c'), ('r2scan', False, 'd4:r2scan-3c'))
        self.assertEqual(dispersion.parse_dft('r2scan_3c'), ('r2scan', False, 'd4:r2scan-3c'))
        self.assertEqual(dispersion.parse_dft('wb97x-3c'), ('wb97x-v', False, 'd4:wb97x-3c'))
        self.assertEqual(dispersion.parse_dft('wb97x_3c'), ('wb97x-v', False, 'd4:wb97x-3c'))

        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('pbeh-3c')
        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('pbeh_3c')
        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('hf-3c')
        with self.assertRaises(NotImplementedError):
            dispersion.parse_dft('hf_3c')

    def test_parse_disp_errors(self):
        # Unknown disp version
        with self.assertRaises(ValueError):
            dispersion.parse_disp('b3lyp', 'unknown_ver')

        # Disp specified but method unknown/missing (if disp string doesn't contain colon)
        # Actually parse_disp(None, 'd3bj') -> raises ValueError "the method used in dispersion d3bj is not specified."
        with self.assertRaises(ValueError):
            dispersion.parse_disp(None, 'd3bj')

    def test_check_disp(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')

        # 1. RHF object (no .xc)
        mf_hf = scf.RHF(mol)
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If mf.disp = None
        mf_hf.disp = None
        # parse_disp('hf', None) -> ('hf', None, False) -> check_disp returns False
        self.assertFalse(dispersion.check_disp(mf_hf))

        # If we set mf.disp = 'd3bj'
        mf_hf.disp = 'd3bj'
        self.assertTrue(dispersion.check_disp(mf_hf))

        # 2. KohnShamDFT object (has .xc)
        mf_dft = KnownKS()
        mf_dft.xc = 'b3lyp'
        mf_dft.disp = None

        # b3lyp -> no disp -> False
        self.assertFalse(dispersion.check_disp(mf_dft))

        # Explicit disp
        self.assertTrue(dispersion.check_disp(mf_dft, disp='d3bj'))

        # Explicit disp=False
        self.assertFalse(dispersion.check_disp(mf_dft, disp=False))

        # Implicit disp from method
        mf_dft.xc = 'wb97x-d3bj'
        self.assertTrue(dispersion.check_disp(mf_dft))

        # Unsupported disp version
        with self.assertRaises(ValueError):
            dispersion.check_disp(mf_dft, disp='unsupported')

    @unittest.skipIf(dispersion.dftd4 is None, 'DFTD4 not installed')
    def test_wb97x_d4(self):
        mol = gto.M(
            verbose = 5,
            output = '/dev/null',
            atom = '''
              O     0.   0.       0.
              H     0.   -0.757   0.587
              H     0.   0.757    0.587''',
            basis = 'def2-svp')
        mf = mol.RKS(xc='wb97x-d4').run()
        if int(pyscf.__version__.split('.')[1]) > 15:
            # corresponding to
            # mf.xc = 'wb97x-v'
            # mf.nlc = False
            # mf.disp = 'd4:wb97x'
            self.assertAlmostEqual(mf.e_tot, -76.37197333535842, 8)
        else:
            # legacy value, corresponding to
            # mf.xc = 'wb97x'
            # mf.nlc = False
            # mf.disp = 'd4:wb97x-2008'
            self.assertAlmostEqual(mf.e_tot, -76.3377143469286, 8)

    def test_parse_coach(self):
        xc = 'COACH'
        self.assertEqual(dispersion.parse_dft(xc), ('coach', '', 'd4:coach'))
        self.assertEqual(dispersion.parse_disp(xc), ('coach', 'd4', True))
        self.assertTrue(dispersion.check_disp(KnownKS(xc)))
        self.assertEqual(dispersion.parse_disp(xc, 'd4:pbe'), ('pbe', 'd4', True))

        with lib.temporary_env(dispersion, dftd4=None):
            with self.assertRaisesRegex(RuntimeError, 'pip install pyscf-dispersion'):
                dispersion._make_d4_model(None, 'coach', True)

if __name__ == "__main__":
    unittest.main()
