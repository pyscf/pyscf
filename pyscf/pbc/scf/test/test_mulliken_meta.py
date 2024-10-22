#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
from pyscf.pbc import gto, scf, tools


class KPTvsSUPCELL_noshift(unittest.TestCase):

    scaled_center = None

    def make_cell_supcell_kpts(self, kmesh, **kwargs):
        cell = gto.Cell()
        cell.verbose = 4
        cell.output = '/dev/null'
        cell.atom = '''
        O   -1.4771410554   -0.0001473417   -0.0820628725
        H   -2.1020842744   -0.6186960400    0.3275021172
        H   -1.4531946703   -0.1515066183   -1.0398792446
        '''
        cell.a = np.eye(3) * 4
        cell.basis = 'sto-3g'
        cell.set(**kwargs)
        cell.build()
        scell = tools.super_cell(cell, kmesh)
        kpts = cell.make_kpts(kmesh, scaled_center=self.scaled_center)
        return cell, scell, kpts

    def test_krhf(self):
        kmesh = [3,1,1]
        cell, scell, kpts = self.make_cell_supcell_kpts(kmesh)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kdm = kmf.init_guess_by_1e()
        kchg = kmf.mulliken_meta(cell, kdm, kpts, verbose=0)[1]

        smf = scf.RHF(scell, kpt=kpts[0]).density_fit()
        sdm = smf.init_guess_by_1e()
        schg = smf.mulliken_meta(scell, sdm, verbose=0)[1][:cell.natm]

        self.assertAlmostEqual(abs(kchg-schg).max(), 0, 3)

    def test_kuhf(self):
        spin0 = 2
        kmesh = [3,1,1]
        spin = int(np.prod(kmesh)) * spin0
        cell, scell, kpts = self.make_cell_supcell_kpts(kmesh)

        kmf = scf.KUHF(cell, kpts=kpts).density_fit()
        kdm = kmf.init_guess_by_1e()
        kchg = kmf.mulliken_meta(cell, kdm, kpts, verbose=0)[1]
        kchg_spin = kmf.mulliken_meta_spin(cell, kdm, kpts, verbose=0)[1]

        smf = scf.UHF(scell, kpt=kpts[0]).density_fit()
        sdm = smf.init_guess_by_1e()
        schg = smf.mulliken_meta(scell, sdm, verbose=0)[1][:cell.natm]
        schg_spin = smf.mulliken_meta_spin(scell, sdm, verbose=0)[1][:cell.natm]

        self.assertAlmostEqual(abs(kchg-schg).max(), 0, 3)
        self.assertAlmostEqual(abs(kchg_spin-schg_spin).max(), 0, 3)

    def test_kghf(self):
        spin0 = 2
        kmesh = [3,1,1]
        spin = int(np.prod(kmesh)) * spin0
        cell, scell, kpts = self.make_cell_supcell_kpts(kmesh)

        kmf = scf.KGHF(cell, kpts=kpts).density_fit()
        kdm = kmf.init_guess_by_1e()
        kchg = kmf.mulliken_meta(cell, kdm, kpts, verbose=0)[1]

        smf = scf.GHF(scell, kpt=kpts[0]).density_fit()
        sdm = smf.init_guess_by_1e()
        schg = smf.mulliken_meta(scell, sdm, verbose=0)[1][:cell.natm]

        self.assertAlmostEqual(abs(kchg-schg).max(), 0, 3)

    def test_krohf(self):
        spin0 = 2
        kmesh = [3,1,1]
        spin = int(np.prod(kmesh)) * spin0
        cell, scell, kpts = self.make_cell_supcell_kpts(kmesh)

        kmf = scf.KROHF(cell, kpts=kpts).density_fit()
        kdm = kmf.init_guess_by_1e()
        kchg = kmf.mulliken_meta(cell, kdm, kpts, verbose=0)[1]

        smf = scf.ROHF(scell, kpt=kpts[0]).density_fit()
        sdm = smf.init_guess_by_1e()
        schg = smf.mulliken_meta(scell, sdm, verbose=0)[1][:cell.natm]

        self.assertAlmostEqual(abs(kchg-schg).max(), 0, 3)

class KPTvsSUPCELL_shift(KPTvsSUPCELL_noshift):

    scaled_center = [0.3765, 0.7729, 0.1692]

if __name__ == "__main__":
    print("Full Tests for PBC SCF mulliken_meta charges")
    unittest.main()
