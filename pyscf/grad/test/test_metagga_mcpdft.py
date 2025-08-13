#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.edu>

import unittest

from pyscf import scf, gto, df, dft
from pyscf.data.nist import BOHR
from pyscf import mcpdft

def diatomic(
    atom1,
    atom2,
    r,
    fnal,
    basis,
    ncas,
    nelecas,
    nstates,
    charge=None,
    spin=None,
    symmetry=False,
    cas_irrep=None,
    density_fit=False,
    grids_level=9,
):
    """Used for checking diatomic systems to see if the Lagrange Multipliers are working properly."""
    global mols
    xyz = "{:s} 0.0 0.0 0.0; {:s} {:.3f} 0.0 0.0".format(atom1, atom2, r)
    mol = gto.M(
        atom=xyz,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=0,
        output="/dev/null",
    )
    mols.append(mol)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis=df.aug_etb(mol))

    mc = mcpdft.CASSCF(mf.run(), fnal, ncas, nelecas, grids_level=grids_level)
    if spin is None:
        spin = mol.nelectron % 2

    ss = spin * (spin + 2) * 0.25
    mc.fix_spin_(ss=ss, shift=2)

    if nstates > 1:
        mc = mc.state_average(
            [
                1.0 / float(nstates),
            ]
            * nstates,
        )

    mc.conv_tol = 1e-12
    mc.conv_grad_tol = 1e-6
    mo = None
    if symmetry and (cas_irrep is not None):
        mo = mc.sort_mo_by_irrep(cas_irrep)

    mc_grad = mc.run(mo).nuc_grad_method()
    mc_grad.conv_rtol = 1e-12
    return mc_grad


def setUpModule():
    global mols, original_grids
    mols = []
    original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False


def tearDownModule():
    global mols, diatomic, original_grids
    [m.stdout.close() for m in mols]
    dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = original_grids
    del mols, diatomic, original_grids


class KnownValues(unittest.TestCase):

    def test_grad_lih_sstm06l22_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "tM06L", "STO-3G", 2, 2, 1, grids_level=1)
        de = mc.kernel()[1, 0] / BOHR

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   4015363355dc691a80bc94d4b2b094318b213e36
        DE_REF = -1.0546009263404388

        self.assertAlmostEqual(de, DE_REF, 5)

    def test_grad_lih_sa2tm06l22_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "tM06L", "STO-3G", 2, 2, 2, grids_level=1)

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   4015363355dc691a80bc94d4b2b094318b213e36
        DE_REF = [-1.0351271000, -0.8919881992]

        for state in range(2):
            with self.subTest(state=state):
                de = mc.kernel(state=state)[1, 0] / BOHR
                self.assertAlmostEqual(de, DE_REF[state], 5)

    def test_grad_lih_ssmc2322_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "MC23", "STO-3G", 2, 2, 1, grids_level=1)
        de = mc.kernel()[1, 0] / BOHR

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   e82ba940654cd0b91f799e889136a316fda34b10
        DE_REF = -1.0641645070

        self.assertAlmostEqual(de, DE_REF, 5)

    def test_grad_lih_sa2mc2322_sto3g(self):
        mc = diatomic("Li", "H", 0.8, "MC23", "STO-3G", 2, 2, 2, grids_level=1)

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   e82ba940654cd0b91f799e889136a316fda34b10
        DE_REF = [-1.0510225010, -0.8963063432]

        for state in range(2):
            with self.subTest(state=state):
                de = mc.kernel(state=state)[1, 0] / BOHR
                self.assertAlmostEqual(de, DE_REF[state], 5)

    def test_grad_lih_sa2mc2322_sto3g_df(self):
        mc = diatomic(
            "Li", "H", 0.8, "MC23", "STO-3G", 2, 2, 2, grids_level=1, density_fit=df
        )

        # Numerical from this software
        # PySCF commit:         f2c2d3f963916fb64ae77241f1b44f24fa484d96
        # PySCF-forge commit:   ee6ac742fbc79d170bc4b63ef2b2c4b49478c53a
        DE_REF = [-1.0510303416, -0.8963992331]

        for state in range(2):
            with self.subTest(state=state):
                de = mc.kernel(state=state)[1, 0] / BOHR
                self.assertAlmostEqual(de, DE_REF[state], 5)


if __name__ == "__main__":
    print("Full Tests for MC-PDFT gradients with meta-GGA functionals")
    unittest.main()
