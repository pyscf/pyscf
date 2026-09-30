# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#
from pyscf.pbc import gto, scf, adc
import numpy as np
import unittest
import warnings
warnings.simplefilter('ignore', ResourceWarning)


nroots = 3


def setUpModule():
    global cell, kmf
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.build()
    kpts = cell.make_kpts([2, 2, 1])
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.verbose = 0
    kmf.kernel()


def tearDownModule():
    global cell, kmf
    del cell, kmf


def _fno(pct_occ=None, thresh=None, nvir_act=None, mode=None):
    """Ground-state FNO generator and its additively-corrected ADC(3) energy."""
    fg = adc.KRADC2FNO(kmf)
    fg.approx_trans_moments = True
    fg.verbose = 0
    if mode is not None:
        fg.mode = mode
    fg.kernel_gs(pct_occ=pct_occ, thresh=thresh, nvir_act=nvir_act)
    return fg


def _fno_es(method_type, ref_state, pct_occ):
    """Excited-state SS/SA-FNO generator."""
    fg = adc.KRADC2FNO(kmf)
    fg.approx_trans_moments = True
    fg.verbose = 0
    fg.method_type = method_type
    fg.ref_state = ref_state
    fg.kernel(nroots, pct_occ=pct_occ, kptlist=[0])
    return fg


def _gs_corr(fg, i=None):
    """Corrected ADC(3) ground-state correlation energy in the FNO space."""
    frozen = fg.frozen[i] if i is not None else fg.frozen
    mo_coeff = fg.mo_coeff[i] if i is not None else fg.mo_coeff
    mo_energy = fg.mo_energy[i] if i is not None else fg.mo_energy
    mo_occ = fg.mo_occ[i] if (i is not None and isinstance(fg.mo_coeff, list)) else fg.mo_occ
    k = adc.KRADC(kmf, frozen, mo_coeff, mo_occ, mo_energy)
    k.verbose = 0
    k.method = 'adc(3)'
    return fg.correct_corr(k.kernel_gs()[0], i)


def _es_corr(fg, method_type, i=None):
    """Corrected ADC(3) root-0 excitation energy in the FNO space."""
    frozen = fg.frozen[i] if i is not None else fg.frozen
    mo_coeff = fg.mo_coeff[i] if i is not None else fg.mo_coeff
    mo_occ = fg.mo_occ[i] if (i is not None and isinstance(fg.mo_coeff, list)) else fg.mo_occ
    mo_energy = fg.mo_energy[i] if i is not None else fg.mo_energy
    guess = fg.v_ssfno[i] if i is not None else fg.v_ssfno
    k = adc.KRADC(kmf, frozen, mo_coeff, mo_occ, mo_energy)
    k.verbose = 0
    k.method = 'adc(3)'
    k.method_type = method_type
    e, v, p, _ = k.kernel(nroots, guess=guess, kptlist=[0])
    return fg.correct(e, i)[0][0]


class KnownValues(unittest.TestCase):

    def test_canonical(self):
        # Canonical references (no FNO truncation).
        k = adc.KRADC(kmf)
        k.verbose = 0
        e_mp2 = k.kernel_gs()[0]
        k.method = 'adc(3)'
        e_mp3 = k.kernel_gs()[0]
        self.assertAlmostEqual(e_mp2, -0.1490009409, 4)
        self.assertAlmostEqual(e_mp3, -0.1191959948, 4)

    def test_gs_fno_pct_occ(self):
        self.assertAlmostEqual(_gs_corr(_fno(pct_occ=0.5)), -0.1502090671, 4)

    def test_gs_fno_thresh(self):
        self.assertAlmostEqual(_gs_corr(_fno(thresh=0.05)), -0.1259622004, 4)

    def test_gs_fno_nvir_act(self):
        self.assertAlmostEqual(_gs_corr(_fno(nvir_act=2)), -0.1392170203, 4)

    def test_gs_fno_per_kpt(self):
        self.assertAlmostEqual(_gs_corr(_fno(thresh=0.05, mode='per_kpt')), -0.1360134673, 4)

    def test_gs_fno_multithreshold(self):
        fg = _fno(pct_occ=[0.5, 0.9])
        self.assertAlmostEqual(_gs_corr(fg, 0), -0.1502090671, 4)
        self.assertAlmostEqual(_gs_corr(fg, 1), -0.1259622004, 4)

    def test_ss_fno_ea(self):
        self.assertAlmostEqual(_es_corr(_fno_es('ea', [[0], [0]], 0.9), 'ea'), 1.0069869404, 4)

    def test_sa_fno_ea(self):
        self.assertAlmostEqual(_es_corr(_fno_es('ea', [[0, 1], [0]], 0.9), 'ea'), 0.9951643793, 4)

    def test_ss_fno_ip(self):
        self.assertAlmostEqual(_es_corr(_fno_es('ip', [[0], [0]], 0.9), 'ip'), -0.7491959409, 4)


if __name__ == '__main__':
    print("k-point FNO (conventional / SS / SA) tests")
    unittest.main(warnings='ignore')
