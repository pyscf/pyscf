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

import unittest
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf.cc import ccsd_t, ccsd_t_lambda
from pyscf.df.grad import ccsd as dfccsd_grad
from pyscf.df.grad import ccsd_t as dfccsd_t_grad

AUXBASIS = 'ccpvdz-jkfit'


def make_mol(coords=None):
    if coords is None:
        atom = 'H 0 0 0; F 0 0 1.1'
        unit = 'Angstrom'
    else:
        atom = [('H', coords[0]), ('F', coords[1])]
        unit = 'Bohr'
    return gto.M(atom=atom, basis='6-31g', unit=unit, verbose=0)

def make_cc(mol, frozen=None):
    mf = scf.RHF(mol).density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = 1e-13
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-11
    mycc.conv_tol_normt = 1e-9
    return mycc

def numeric_grad(efunc, mol, step=1e-3):
    coords = mol.atom_coords()
    de = numpy.zeros_like(coords)
    for i in range(mol.natm):
        for x in range(3):
            cp = coords.copy()
            cp[i,x] += step
            cm = coords.copy()
            cm[i,x] -= step
            de[i,x] = (efunc(cp) - efunc(cm)) / (2 * step)
    return de


class KnownValues(unittest.TestCase):
    def test_nuc_grad_method_is_df_aware(self):
        mol = make_mol()
        mycc = cc.CCSD(scf.RHF(mol).density_fit(auxbasis=AUXBASIS).run())
        self.assertTrue(isinstance(mycc.nuc_grad_method(), dfccsd_grad.Gradients))

    def test_df_ccsd_grad(self):
        mol = make_mol()
        mycc = make_cc(mol)
        mycc.kernel()
        g1 = mycc.nuc_grad_method().kernel()
        ref = numeric_grad(lambda c: make_cc(make_mol(c)).run().e_tot, mol)
        self.assertAlmostEqual(abs(g1 - ref).max(), 0, 6)

    def test_df_ccsd_grad_frozen(self):
        mol = make_mol()
        mycc = make_cc(mol, frozen=1)
        mycc.kernel()
        g1 = mycc.nuc_grad_method().kernel()
        ref = numeric_grad(lambda c: make_cc(make_mol(c), frozen=1).run().e_tot, mol)
        self.assertAlmostEqual(abs(g1 - ref).max(), 0, 6)

    def test_df_ccsd_t_grad(self):
        def e_tot(coords=None):
            mycc = make_cc(make_mol(coords))
            eris = mycc.ao2mo()
            mycc.kernel(eris=eris)
            return mycc, eris, mycc.e_tot + ccsd_t.kernel(mycc, eris)

        mol = make_mol()
        mycc, eris, _ = e_tot()
        l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)[1:]
        g1 = dfccsd_t_grad.Gradients(mycc).kernel(mycc.t1, mycc.t2, l1, l2,
                                                  eris=eris)
        ref = numeric_grad(lambda c: e_tot(c)[2], mol)
        self.assertAlmostEqual(abs(g1 - ref).max(), 0, 6)

    def test_df_ccsd_grad_differs_from_conventional(self):
        # The conventional assembly differentiates the exact ERIs and is not a
        # derivative of the density-fitted energy.
        from pyscf.grad import ccsd as ccsd_grad
        mol = make_mol()
        mycc = make_cc(mol)
        mycc.kernel()
        g_df = mycc.nuc_grad_method().kernel()
        g_conv = ccsd_grad.Gradients(mycc).kernel()
        self.assertTrue(abs(g_df - g_conv).max() > 1e-6)

    def test_requires_df_scf_reference(self):
        # dfccsd.RCCSD accepts a conventional-ERI mean field and builds its own
        # with_df.  The HF half of the gradient would then differentiate the
        # exact ERIs while the correlation half differentiates the fitted ones.
        from pyscf.cc import dfccsd
        mol = make_mol()
        mycc = dfccsd.RCCSD(scf.RHF(mol).run())
        self.assertRaises(AssertionError, mycc.nuc_grad_method)

    def test_df_uccsd_grad_not_implemented(self):
        mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g', verbose=0)
        mf = scf.UHF(mol).density_fit(auxbasis=AUXBASIS).run()
        self.assertRaises(NotImplementedError, cc.UCCSD(mf).nuc_grad_method)


if __name__ == '__main__':
    print('Full Tests for DF-CCSD gradients')
    unittest.main()
