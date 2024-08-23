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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

import unittest
import numpy as np
from pyscf.agf2 import aux, _agf2


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.nmo = 100
        self.nocc = 20
        self.nvir = 80
        self.naux = 400
        np.random.seed(1)

    @classmethod
    def tearDownClass(self):
        del self.nmo, self.nocc, self.nvir, self.naux
        np.random.seed()

    def test_c_ragf2(self):
        xija = np.random.random((self.nmo, self.nocc, self.nocc, self.nvir))
        gf_occ = aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc))
        gf_vir = aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir))
        vv1, vev1 = _agf2.build_mats_ragf2_outcore(xija, gf_occ.energy, gf_vir.energy)
        vv2, vev2 = _agf2.build_mats_ragf2_incore(xija, gf_occ.energy, gf_vir.energy)
        self.assertAlmostEqual(np.max(np.absolute(vv1-vv2)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.absolute(vev1-vev2)), 0.0, 8)

    def test_c_dfragf2(self):
        qxi = np.random.random((self.naux, self.nmo*self.nocc)) / self.naux
        qja = np.random.random((self.naux, self.nocc*self.nvir)) / self.naux
        gf_occ = aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc))
        gf_vir = aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir))
        vv1, vev1 = _agf2.build_mats_dfragf2_outcore(qxi, qja, gf_occ.energy, gf_vir.energy)
        vv2, vev2 = _agf2.build_mats_dfragf2_incore(qxi, qja, gf_occ.energy, gf_vir.energy)
        self.assertAlmostEqual(np.max(np.absolute(vv1-vv2)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.absolute(vev1-vev2)), 0.0, 8)

    def test_c_uagf2(self):
        xija = np.random.random((2, self.nmo, self.nocc, self.nocc, self.nvir))
        gf_occ = (aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc)),
                  aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc)))
        gf_vir = (aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir)),
                  aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir)))
        vv1, vev1 = _agf2.build_mats_uagf2_outcore(xija, (gf_occ[0].energy, gf_occ[1].energy), (gf_vir[0].energy, gf_vir[1].energy))
        vv2, vev2 = _agf2.build_mats_uagf2_incore(xija, (gf_occ[0].energy, gf_occ[1].energy), (gf_vir[0].energy, gf_vir[1].energy))
        self.assertAlmostEqual(np.max(np.absolute(vv1-vv2)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.absolute(vev1-vev2)), 0.0, 8)

    def test_c_dfuagf2(self):
        qxi = np.random.random((2, self.naux, self.nmo*self.nocc)) / self.naux
        qja = np.random.random((2, self.naux, self.nocc*self.nvir)) / self.naux
        gf_occ = (aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc)),
                  aux.GreensFunction(np.random.random(self.nocc), np.eye(self.nmo, self.nocc)))
        gf_vir = (aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir)),
                  aux.GreensFunction(np.random.random(self.nvir), np.eye(self.nmo, self.nvir)))
        vv1, vev1 = _agf2.build_mats_dfuagf2_outcore(qxi, qja, (gf_occ[0].energy, gf_occ[1].energy), (gf_vir[0].energy, gf_vir[1].energy))
        vv2, vev2 = _agf2.build_mats_dfuagf2_incore(qxi, qja, (gf_occ[0].energy, gf_occ[1].energy), (gf_vir[0].energy, gf_vir[1].energy))
        self.assertAlmostEqual(np.max(np.absolute(vv1-vv2)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.absolute(vev1-vev2)), 0.0, 8)


if __name__ == '__main__':
    print('AGF2 C implementations')
    unittest.main()
