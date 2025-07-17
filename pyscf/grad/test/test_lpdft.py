#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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

# The following tests are broken down into a couple of different categories.
#   1. Check accuracy of analytical gradients to numerical gradients for different Lagrange multiplier situations. Some
#      tests are redundant since as long as the tests with both MO and CI Lagrange multipliers pass, then everything
#      should be good. All other tests are marked as slow and used for thorough debugging.
#   2. Check API as scanner object.
#   3. L-PDFT gradients for multi_state_mix type objects.

import unittest

from pyscf import scf, gto, df, dft, lib
from pyscf import mcpdft
from pyscf.fci.addons import _unpack_nelec

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
    neleca, nelecb = _unpack_nelec (nelecas, spin=spin)
    spin = neleca-nelecb

    ss = spin * (spin + 2) * 0.25
    mc = mc.multi_state(
        [
            1.0 / float(nstates),
        ]
        * nstates,
        "lin",
    )
    mc.fix_spin_(ss=ss, shift=2)
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

    def test_grad_hhe_lin3ftlda22_631g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     no
        """
        n_states = 3
        mc_grad = diatomic(
            "He", "H", 1.4, "ftLDA,VWN3", "6-31G", 2, 2, n_states, charge=1
        )

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [-0.0744181053, -0.0840211222, -0.0936241392]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_hhe_lin2ftlda24_631g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    no
        ci:     yes
        """
        n_states = 2
        mc_grad = diatomic(
            "He", "H", 1.4, "ftLDA,VWN3", "6-31G", 4, 2, n_states, charge=1
        )

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [0.0025153073, -0.1444551635]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_hhe_lin2ftlda22_631g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     yes
        """
        n_states = 2
        # The L-PDFT ground state is flat at 1.4, so shift it slightly
        mc_grad = diatomic(
            "He", "H", 1.2, "ftLDA,VWN3", "6-31G", 2, 2, n_states, charge=1
        )

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [0.012903562, -0.239149778]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 5)

    def test_grad_lih_lin3ftlda22_sto3g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     no
        """
        n_states = 3
        mc_grad = diatomic("Li", "H", 1.4, "ftLDA,VWN3", "STO-3G", 2, 2, n_states)

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [-0.0269959347, -0.052808735, -0.0785029927]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 6)

    def test_grad_lih_lin2ftlda46_sto3g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    no
        ci:     yes
        """
        n_states = 2
        mc_grad = diatomic("Li", "H", 1.4, "ftLDA,VWN3", "STO-3G", 6, 4, n_states)

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [-0.0289711885, -0.0525535764]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 7)

    def test_grad_lih_lin2ftlda22_sto3g_slow(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     yes
        """
        n_states = 2
        mc_grad = diatomic("Li", "H", 1.4, "ftLDA,VWN3", "STO-3G", 2, 2, n_states)

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [-0.0302731558, -0.0528615182]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 5)

    def test_grad_lih_lin2ftpbe22_sto3g(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     yes
        """
        n_states = 2
        mc_grad = diatomic("Li", "H", 1.4, "ftpbe", "STO-3G", 2, 2, n_states)

        # Numerical from this software
        # PySCF commit:         6c1ea86eb60b9527d6731efa65ef99a66b8f84d2
        # PySCF-forge commit:   ea0a4c164de21e84eeb30007afcb45344cfc04ff
        NUM_REF = [-0.0318512447, -0.0544779213]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 5)

    def test_grad_scanner(self):
        # Tests API and Scanner capabilities
        n_states = 2
        mc_grad1 = diatomic(
            "Li", "H", 1.5, "ftLDA,VWN3", "STO-3G", 2, 2, n_states, grids_level=1
        )
        mol1 = mc_grad1.base.mol
        mc_grad2 = diatomic(
            "Li", "H", 1.6, "ftLDA,VWN3", "STO-3G", 2, 2, n_states, grids_level=1
        ).as_scanner()

        for state in range(n_states):
            with self.subTest(state=state):
                de1 = mc_grad1.kernel(state=state)
                e1 = mc_grad1.base.e_states[state]
                e2, de2 = mc_grad2(mol1, state=state)
                self.assertTrue(mc_grad1.converged)
                self.assertTrue(mc_grad2.converged)
                self.assertAlmostEqual(e1, e2, 6)
                self.assertAlmostEqual(lib.fp(de1), lib.fp(de2), 6)

    def test_grad_lih_lin2ftpbe22_sto3g_df(self):
        """System has the following Lagrange multiplier sectors:
        orb:    yes
        ci:     yes
        """
        n_states = 2
        mc_grad = diatomic(
            "Li", "H", 1.4, "ftpbe", "STO-3G", 2, 2, n_states, density_fit=True
        )

        # Numerical from this software
        # PySCF commit:         eafc3575234aca3832d270f4e1193bec2119d2b4
        # PySCF-forge commit:   2e1596dc919783c751c3cf24bb776e4efcb51a34
        NUM_REF = [-0.03209560095886714, -0.05453488593608611]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 5)

    def test_rohf_sanity (self):
        n_states = 3
        mc_grad = diatomic(
            "Li", "H", 1.4, "ftpbe", "6-31g", 4, 2, n_states, density_fit=False, spin=2
        )
        mc_grad_ref = diatomic(
            "Li", "H", 1.4, "ftpbe", "6-31g", 4, (2,0), n_states, density_fit=False
        )

        # Numerical from this software
        # PySCF commit:         bee0ce288a655105e27fcb0293b203939b7aecc9
        # PySCF-forge commit:   50bc1da117ced9613948bee14a99a02c7b2c5769
        NUM_REF = [-0.062533, -0.058472, -0.058472]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 4)
                de_ref = mc_grad_ref.kernel(state=i)[1, 0]
                self.assertAlmostEqual (de, de_ref, 6)

    def test_dfrohf_sanity (self):
        n_states = 3
        mc_grad = diatomic(
            "Li", "H", 1.4, "ftpbe", "6-31g", 4, 2, n_states, density_fit=True, spin=2
        )
        mc_grad_ref = diatomic(
            "Li", "H", 1.4, "ftpbe", "6-31g", 4, (2,0), n_states, density_fit=True,
        )

        # Numerical from this software
        # PySCF commit:         bee0ce288a655105e27fcb0293b203939b7aecc9
        # PySCF-forge commit:   50bc1da117ced9613948bee14a99a02c7b2c5769
        NUM_REF = [-0.062548, -0.058501, -0.058501]
        for i in range(n_states):
            with self.subTest(state=i):
                de = mc_grad.kernel(state=i)[1, 0]
                self.assertAlmostEqual(de, NUM_REF[i], 4)
                de_ref = mc_grad_ref.kernel(state=i)[1, 0]
                self.assertAlmostEqual (de, de_ref, 6)



if __name__ == "__main__":
    print("Full Tests for L-PDFT gradients API")
    unittest.main()
