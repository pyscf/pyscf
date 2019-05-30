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

import unittest
import numpy

from pyscf import __config__
__config__.shci_SHCIEXE = 'shci_emulator'
__config__.shci_SHCISCRATCHDIR = '.'
from pyscf import gto, scf, mcscf
from pyscf.shciscf import shci

# Test whether SHCI executable can be found. If it can, trigger tests that
# require it.
NO_SHCI = True
if shci.settings.SHCIEXE != 'shci_emulator' and shci.settings.SHCIEXE != None:
    print("Found SHCI =>", shci.settings.SHCIEXE)
    NO_SHCI = False
else:
    print("No SHCI found")


def make_o2():
    b = 1.208
    mol = gto.Mole()
    mol.build(
        verbose=0,
        output=None,
        atom="O 0 0 %f; O 0 0 %f" % (-b / 2, b / 2),
        basis='ccpvdz',
        symmetry=True)

    # Create HF molecule
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    return mf.run()


def D2htoDinfh(SHCI, norb, nelec):
    from pyscf import symm

    coeffs = numpy.zeros(shape=(norb, norb)).astype(complex)
    nRows = numpy.zeros(shape=(norb, ), dtype=int)
    rowIndex = numpy.zeros(shape=(2 * norb, ), dtype=int)
    rowCoeffs = numpy.zeros(shape=(2 * norb, ), dtype=float)

    i, orbsym1, ncore = 0, [0] * len(SHCI.orbsym), len(SHCI.orbsym) - norb

    while i < norb:
        symbol = symm.basis.linearmole_irrep_id2symb(SHCI.groupname,
                                                     SHCI.orbsym[i + ncore])
        if (symbol[0] == 'A'):
            coeffs[i, i] = 1.0
            orbsym1[i] = 1
            nRows[i] = 1
            rowIndex[2 * i] = i
            rowCoeffs[2 * i] = 1.
            if len(symbol) == 3 and symbol[2] == 'u':
                orbsym1[i] = 2
        else:
            if (i == norb - 1):
                print("the orbitals dont have dinfh symmetry")
                exit(0)
            l = int(symbol[1])
            orbsym1[i], orbsym1[i + 1] = 2 * l + 3, -(2 * l + 3)
            if (len(symbol) == 4 and symbol[2] == 'u'):
                orbsym1[i], orbsym1[i + 1] = orbsym1[i] + 1, orbsym1[i + 1] - 1
            if (symbol[3] == 'x'):
                m1, m2 = 1., -1.
            else:
                m1, m2 = -1., 1.

            nRows[i] = 2
            if m1 > 0:
                coeffs[i, i], coeffs[i, i + 1] = (
                    (-1)**l) * 1.0 / (2.0**0.5), ((-1)**l) * 1.0j / (2.0**0.5)
                rowIndex[2 * i], rowIndex[2 * i + 1] = i, i + 1
            else:
                coeffs[i, i + 1], coeffs[i, i] = (
                    (-1)**l) * 1.0 / (2.0**0.5), ((-1)**l) * 1.0j / (2.0**0.5)
                rowIndex[2 * i], rowIndex[2 * i + 1] = i + 1, i

            rowCoeffs[2 * i], rowCoeffs[2 * i + 1] = (
                (-1)**l) * 1.0 / (2.0**0.5), ((-1)**l) * 1.0 / (2.0**0.5)
            i = i + 1

            nRows[i] = 2
            if (m1 > 0):  #m2 is the complex number
                rowIndex[2 * i] = i - 1
                rowIndex[2 * i + 1] = i
                rowCoeffs[2 * i], rowCoeffs[
                    2 * i + 1] = 1.0 / (2.0**0.5), -1.0 / (2.0**0.5)
                coeffs[i, i -
                       1], coeffs[i, i] = 1.0 / (2.0**0.5), -1.0j / (2.0**0.5)
            else:
                rowIndex[2 * i] = i
                rowIndex[2 * i + 1] = i - 1
                rowCoeffs[2 * i], rowCoeffs[
                    2 * i + 1] = 1.0 / (2.0**0.5), -1.0 / (2.0**0.5)
                coeffs[i, i], coeffs[i, i -
                                     1] = 1.0 / (2.0**0.5), -1.0j / (2.0**0.5)

        i = i + 1

    return coeffs, nRows, rowIndex, rowCoeffs, orbsym1


def DinfhtoD2h(SHCI, norb, nelec):
    from pyscf import symm

    nRows = numpy.zeros(shape=(norb, ), dtype=int)
    rowIndex = numpy.zeros(shape=(2 * norb, ), dtype=int)
    rowCoeffs = numpy.zeros(shape=(4 * norb, ), dtype=float)

    i, ncore = 0, len(SHCI.orbsym) - norb

    while i < norb:
        symbol = symm.basis.linearmole_irrep_id2symb(SHCI.groupname,
                                                     SHCI.orbsym[i + ncore])
        if (symbol[0] == 'A'):
            nRows[i] = 1
            rowIndex[2 * i] = i
            rowCoeffs[4 * i] = 1.
        else:
            l = int(symbol[1])

            if (symbol[3] == 'x'):
                m1, m2 = 1., -1.
            else:
                m1, m2 = -1., 1.

            nRows[i] = 2
            rowIndex[2 * i], rowIndex[2 * i + 1] = i, i + 1
            if m1 > 0:
                rowCoeffs[4 * i], rowCoeffs[4 * i + 2] = (
                    (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)
            else:
                rowCoeffs[4 * i + 1], rowCoeffs[4 * i + 3] = -(
                    (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)

            i = i + 1

            nRows[i] = 2
            rowIndex[2 * i], rowIndex[2 * i + 1] = i - 1, i
            if (m1 > 0):  #m2 is the complex number
                rowCoeffs[4 * i + 1], rowCoeffs[4 * i + 3] = -(
                    (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)
            else:
                rowCoeffs[4 * i], rowCoeffs[4 * i + 2] = (
                    (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)

        i = i + 1

    return nRows, rowIndex, rowCoeffs


class KnownValues(unittest.TestCase):
    @unittest.skipIf(NO_SHCI, "No SHCI Settings Found")
    def test_SHCI_CASCI(self):
        """
        Compare SHCI-CASCI calculation to CASCI calculation.
        """
        mf = make_o2()
        # Number of orbital and electrons
        ncas = 8
        nelecas = 12
        dimer_atom = 'O'

        mc = mcscf.CASCI(mf, ncas, nelecas)
        e_casscf = mc.kernel()[0]
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.fcisolver = shci.SHCI(mf.mol)
        mc.fcisolver.stochastic = True
        mc.fcisolver.nPTiter = 0  # Turn off perturbative calc.
        mc.fcisolver.sweep_iter = [0]
        mc.fcisolver.sweep_epsilon = [1e-12]
        e_shciscf = mc.kernel()[0]
        self.assertAlmostEqual(e_shciscf, e_casscf, places=6)
        mc.fcisolver.cleanup_dice_files()

    @unittest.skipIf(NO_SHCI, "No SHCI Settings Found")
    def test_SHCISCF_CASSCF(self):
        """
        Compare SHCI-CASSCF calculation to CASSCF calculation.
        """
        mf = make_o2()
        # Number of orbital and electrons
        ncas = 8
        nelecas = 12
        dimer_atom = 'O'

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        e_casscf = mc.kernel()[0]

        mc = shci.SHCISCF(mf, ncas, nelecas)
        mc.fcisolver.stochastic = True
        mc.fcisolver.nPTiter = 0  # Turn off perturbative calc.
        mc.fcisolver.sweep_iter = [0]
        mc.fcisolver.sweep_epsilon = [1e-12]
        e_shciscf = mc.kernel()[0]
        self.assertAlmostEqual(e_shciscf, e_casscf, places=6)
        mc.fcisolver.cleanup_dice_files()

    def test_D2htoDinfh(self):
        SHCI = lambda: None
        SHCI.groupname = 'Dooh'
        #SHCI.orbsym = numpy.array([15,14,0,6,7,2,3,10,11,15,14,17,16,5,13,12,16,17,12,13])
        SHCI.orbsym = numpy.array([
            15, 14, 0, 7, 6, 2, 3, 10, 11, 15, 14, 17, 16, 5, 12, 13, 17, 16,
            12, 13
        ])

        coeffs, nRows, rowIndex, rowCoeffs, orbsym = D2htoDinfh(SHCI, 20, 20)
        coeffs1, nRows1, rowIndex1, rowCoeffs1, orbsym1 = shci.D2htoDinfh(
            SHCI, 20, 20)
        self.assertTrue(numpy.array_equal(coeffs1, coeffs))
        self.assertTrue(numpy.array_equal(nRows1, nRows))
        self.assertTrue(numpy.array_equal(rowIndex1, rowIndex))
        self.assertTrue(numpy.array_equal(rowCoeffs1, rowCoeffs))
        self.assertTrue(numpy.array_equal(orbsym1, orbsym))

    def test_DinfhtoD2h(self):
        SHCI = lambda: None
        SHCI.groupname = 'Dooh'
        #SHCI.orbsym = numpy.array([15,14,0,6,7,2,3,10,11,15,14,17,16,5,13,12,16,17,12,13])
        SHCI.orbsym = numpy.array([
            15, 14, 0, 7, 6, 2, 3, 10, 11, 15, 14, 17, 16, 5, 12, 13, 17, 16,
            12, 13
        ])

        nRows, rowIndex, rowCoeffs = DinfhtoD2h(SHCI, 20, 20)
        nRows1, rowIndex1, rowCoeffs1 = shci.DinfhtoD2h(SHCI, 20, 20)
        self.assertTrue(numpy.array_equal(nRows1, nRows))
        self.assertTrue(numpy.array_equal(rowIndex1, rowIndex))
        self.assertTrue(numpy.array_equal(rowCoeffs1, rowCoeffs))


if __name__ == "__main__":
    print("Tests for shciscf interface")
    unittest.main()
