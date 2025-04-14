#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
import tempfile
from functools import reduce
import numpy
import pyscf
from pyscf.tools import finite_diff
from pyscf.scf.hf import SCF

class KnownValues(unittest.TestCase):
    def test_grad_scanner(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        geom_ref = mol.atom_coords()
        mf = mol.RHF().run()
        ref = mf.Gradients().kernel()
        dat = finite_diff.kernel(mol.RHF(), .5e-2)
        assert abs(dat - ref).max() < 1e-4
        # Ensure geometry is restored
        assert abs(mol.atom_coords() - geom_ref).max() < 1e-9

    def test_hessian_scanner(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        geom_ref = mol.atom_coords()
        ref = mol.RHF().run().Hessian().kernel()
        mf_g = mol.RHF().Gradients()
        dat = finite_diff.kernel(mf_g, .5e-2)
        assert abs(dat - ref).max() < 1e-4
        # Ensure geometry is restored
        assert abs(mol.atom_coords() - geom_ref).max() < 1e-9

    def test_no_scanner(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        geom_ref = mol.atom_coords()
        mf = mol.RHF().run()
        ref = mf.Gradients().kernel()

        attrs = {**mf.__class__.__dict__, **SCF.__dict__}
        attrs.pop('as_scanner')
        FakeRHF = type('RHF', (object,), attrs)
        fake_mf = mf.view(FakeRHF)

        dat = finite_diff.kernel(mol.RHF(), .5e-2)
        assert abs(dat - ref).max() < 1e-4
        # Ensure geometry is restored
        assert abs(mol.atom_coords() - geom_ref).max() < 1e-9

    def test_convergence_failed(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1')
        mol.verbose = 4
        geom_ref = mol.atom_coords()
        mf = mol.RHF().run()
        ref = mf.Gradients().kernel()

        class AlwaysFailed(mf.__class__):
            def kernel(self, *args, **kw):
                res = super().kernel(*args, **kw)
                self.converged = False
                return res

        fake_mf = mf.view(AlwaysFailed).set(conv_tol=1e-3)
        with self.assertRaises(RuntimeError):
            dat = finite_diff.kernel(fake_mf, .5e-2)
        # Ensure geometry is restored
        assert abs(mol.atom_coords() - geom_ref).max() < 1e-9

if __name__ == "__main__":
    print("Full Tests for finite_diff")
    unittest.main()
