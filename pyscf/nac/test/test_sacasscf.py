#!/usr/bin/env python
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
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.nac.sacasscf import NonAdiabaticCouplings
import unittest


def diatomic(atom1, atom2, r, basis, ncas, nelecas, nstates,
             charge=None, spin=None, symmetry=False, cas_irrep=None):
    global mols
    xyz = '{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0'.format(atom1, atom2, r)
    mol = gto.M(atom=xyz, basis=basis, charge=charge, spin=spin,
                symmetry=symmetry, verbose=0, output='/dev/null')

    mols.append(mol)

    mf = scf.RHF(mol)

    mc = mcscf.CASSCF(mf.run(), ncas, nelecas).set(natorb=True)

    if spin is not None:
        s = spin*0.5

    else:
        s = (mol.nelectron % 2)*0.5

    mc.fix_spin_(ss=s*(s+1), shift=1)
    mc = mc.state_average([1.0/float(nstates), ]*nstates)
    mc.conv_tol = mc.conv_tol_diabatize = 1e-12
    mo = None

    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc.kernel(mo)

    return mc.nac_method()

def setUpModule():
    global mols 
    mols = []

def tearDownModule():
    global mols, diatomic
    [m.stdout.close() for m in mols]
    del mols, diatomic


class KnownValues(unittest.TestCase):

    def test_nac_h2_sa2casscf22_sto3g(self):
        # z_orb:    no
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 2)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.24611972496341E-01, 2.24611972496341E-01],
                           [3.91518173397213E-18, -3.91518173397213E-18]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


    def test_nac_h2_sa3casscf22_sto3g(self):
        # z_orb:    no
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, 'STO-3G', 2, 2, 3)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.24611972496341E-01,2.24611972496341E-01 ],
                           [3.91518173397213E-18, -3.91518173397213E-18]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_h2_sa2caasf22_631g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 2)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.63335709207419E-01,2.63335709207420E-01],
                           [-4.13635186565710E-16,4.47060252146777E-16 ]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


    def test_nac_h2_sa3casscf22_631g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     no
        mc_grad = diatomic('H', 'H', 1.3, '6-31G', 2, 2, 3)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[-2.61263051047980E-01,-2.61263051047980E-01],
                           [-5.77124316768522E-17,2.47338992900795E-17 ]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_lih_sa2casscf22_sto3g(self):
        # z_orb:    yes
        # z_ci:     yes
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 1.5, 'STO-3G', 2, 2, 2)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[1.83701729060390E-01, -6.91462064586138E-02],
                           [9.14842536971979E-02, -9.14842536971979E-02]])
        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)

    def test_nac_lih_cms3ftlda22_sto3g(self):
        # z_orb:    yes
        # z_ci:     no
        # z_is:     yes
        mc_grad = diatomic('Li', 'H', 2.5, 'STO-3G', 2, 2, 3)

        # OpenMolcas v23.02 - PC
        de_ref = np.array([[2.68015835251472E-01, -6.48474666167559E-02],
                           [1.24870721811750E-01, -1.24870721811750E-01]])

        for i in range(2):
            with self.subTest(use_etfs=bool(i)):
                de = mc_grad.kernel(state=(0, 1), use_etfs=bool(i))[:, 0]
                de *= np.sign(de[0]) * np.sign(de_ref[i, 0])
                # TODO: somehow confirm sign convention
                self.assertAlmostEqual(de[0], de_ref[i, 0], 5)
                self.assertAlmostEqual(de[1], de_ref[i, 1], 5)


if __name__ == "__main__":
    print("Full Tests for SA-CASSCF non-adiabatic couplings of diatomic molecules")
    unittest.main()
