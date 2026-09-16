# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Ivan Chernyshov <ivan.chernyshov@gmail.com>
#

import unittest
import io, re
import numpy
from pyscf import gto, scf, dft
from pyscf.solvent import pcm, cosmors
from pyscf.data.nist import BOHR as _BOHR

def setUpModule():
    global mol, cm0, cm1, mf0
    mol = gto.M(atom='''
           6        0.000000    0.000000   -0.542500
           8        0.000000    0.000000    0.677500
           1        0.000000    0.935307   -1.082500
           1        0.000000   -0.935307   -1.082500
                ''', basis='sto3g', verbose=0,
                output='/dev/null')
    # ideal conductor
    cm0 = pcm.PCM(mol)
    cm0.eps = float('inf')
    cm0.method = 'C-PCM'
    cm0.lebedev_order = 29
    cm0.verbose = 0
    # computation
    mf0 = dft.RKS(mol, xc='b3lyp').PCM(cm0)
    mf0.kernel()
    # water
    cm1 = pcm.PCM(mol)
    cm1.eps = 78.4
    cm1.method = 'C-PCM'
    cm1.lebedev_order = 29
    cm1.verbose = 0

def tearDownModule():
    global mol, cm0, cm1, mf0
    mol.stdout.close()
    del mol, cm0, cm1, mf0

class TestCosmoRS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_finite_epsilon(self):
        mf1 = dft.RKS(mol, xc='b3lyp').PCM(cm1)
        mf1.kernel()
        def save_cosmo_file(mf):
            with io.StringIO() as outp:
                cosmors.write_cosmo_file(outp, mf)
        self.assertRaises(ValueError, save_cosmo_file, mf1)

    def test_cosmo_file(self):
        with io.StringIO() as outp:
            cosmors.write_cosmo_file(outp, mf0)
            text = outp.getvalue()
        E_diel = float(re.search(r'Dielectric energy \[a.u.\] += +(-*\d+\.\d+)', text).group(1))
        self.assertAlmostEqual(E_diel, -0.0023256022, 5)

    def test_pcm_parameters(self):
        ps = cosmors.get_pcm_parameters(mf0)
        self.assertAlmostEqual(ps['energies']['e_tot'], -112.953044138, 5)
        self.assertAlmostEqual(ps['energies']['e_diel'], -0.0023256022, 5)
        self.assertAlmostEqual(ps['pcm_data']['area'] * _BOHR**2, 64.848604, 2)

    def test_occ_charge_split_is_exact(self):
        # The COSMO equations are linear in the surface potential, so the
        # nuclear and electronic charges must add back up to the full solution.
        smod = mf0.with_solvent
        K = smod._intermediates['K']
        R = smod._intermediates['R']
        v = smod._intermediates['v_grids']
        v_nuc = smod.v_grids_n
        q_nuc = numpy.linalg.solve(K, R.dot(v_nuc))
        q_elec = numpy.linalg.solve(K, R.dot(v - v_nuc))
        self.assertAlmostEqual(
            abs(q_nuc + q_elec - smod._intermediates['q']).max(), 0, 10)

    def test_occ_sum_rule(self):
        # After the correction the total screening charge must equal the exact
        # Gauss value -f_eps * Q, for neutral and charged solutes alike.
        for charge, spin, atom in ((0, 0, 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'),
                                   (-1, 0, 'O 0 0 0; H 0 0 0.97')):
            m = gto.M(atom=atom, basis='6-31g', charge=charge, spin=spin,
                      verbose=0, output='/dev/null')
            cm = pcm.PCM(m)
            cm.eps = float('inf')
            cm.method = 'C-PCM'
            cm.lebedev_order = 29
            cm.verbose = 0
            mf = scf.RHF(m).PCM(cm)
            mf.kernel()
            ps = cosmors.get_pcm_parameters(mf)
            f_eps = ps['pcm_data']['f_eps']
            self.assertAlmostEqual(ps['screening_charge']['total'],
                                   -f_eps * charge, 9)
            # the uncorrected sum misses it, so the test is not vacuous
            self.assertTrue(abs(ps['screening_charge']['correction']) > 1e-4)
            m.stdout.close()

    def test_occ_values(self):
        occ = cosmors.get_outlying_charge_correction(mf0)
        self.assertAlmostEqual(occ['f_nuc'], 1.0016847551, 6)
        self.assertAlmostEqual(occ['f_elec'], 1.0019332410, 6)
        ps = cosmors.get_pcm_parameters(mf0)
        self.assertAlmostEqual(ps['energies']['e_diel_corr'], -0.0023399469, 7)
        self.assertAlmostEqual(ps['energies']['e_tot_corr'], -112.9530584825, 5)
        # the correction shifts the segment charges without being a no-op
        seg = ps['segments']
        self.assertTrue(
            abs(numpy.array(seg['charge_corr']) - numpy.array(seg['charge'])).max() > 1e-8)
        self.assertAlmostEqual(
            abs(numpy.array(seg['sigma_corr']) * numpy.array(seg['area'])
                - numpy.array(seg['charge_corr'])).max(), 0, 12)

    def test_occ_unsupported_model(self):
        # Only C-PCM and COSMO have R proportional to the identity. The others
        # must refuse rather than report a wrong correction.
        for method in ('IEF-PCM', 'SS(V)PE'):
            cm = pcm.PCM(mol)
            cm.eps = float('inf')
            cm.method = method
            cm.lebedev_order = 29
            cm.verbose = 0
            mf = dft.RKS(mol, xc='b3lyp').PCM(cm)
            mf.kernel()
            self.assertRaises(NotImplementedError,
                              cosmors.get_outlying_charge_correction, mf)

    def test_sas_volume(self):
        V1 = cosmors.get_sas_volume(mf0.with_solvent.surface, step = 0.2) * _BOHR**3
        self.assertAlmostEqual(V1, 46.391962, 3)
        V2 = cosmors.get_sas_volume(mf0.with_solvent.surface, step = 0.05) * _BOHR**3
        self.assertAlmostEqual(V2, 46.497054, 3)


if __name__ == "__main__":
    print("Full Tests for COSMO-RS")
    unittest.main()
