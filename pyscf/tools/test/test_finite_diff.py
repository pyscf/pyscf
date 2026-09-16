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

import io
import unittest
import numpy
import pyscf
from pyscf import gto, scf, dft, tdscf
from pyscf.hessian import thermo
from pyscf.tools import finite_diff
from pyscf.scf.hf import SCF

H2O = 'O 0 0 0.117; H 0 0.757 -0.470; H 0 -0.757 -0.470'

def h2o(**kwargs):
    return pyscf.M(atom=H2O, basis='sto-3g', unit='Angstrom', verbose=0, **kwargs)

def rks_fine(mol):
    mf = dft.RKS(mol, xc='pbe0')
    # A Hessian needs a much finer grid than the default; the grid error is
    # independent of the displacement and otherwise swamps everything else.
    mf.grids.level = 5
    mf.grids.prune = None
    return mf.run(conv_tol=1e-13)

class KnownValues(unittest.TestCase):
    def test_grad_scanner(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        geom_ref = mol.atom_coords()
        mf = mol.RHF().run()
        e_ref = mf.e_tot
        ref = mf.Gradients().kernel()
        dat = finite_diff.kernel(mol.RHF(), .5e-2)
        assert abs(dat - ref).max() < 1e-4
        # Ensure geometry is restored
        assert abs(mol.atom_coords() - geom_ref).max() < 1e-9
        assert mf.e_tot == e_ref

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

    def test_Gradients_class(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        mf = mol.RHF().run()
        ref = mf.Gradients().kernel()
        grad_obj = finite_diff.Gradients(mf)
        grad_obj.displacement = .5e-2
        dat = grad_obj.kernel()
        assert abs(dat - ref).max() < 1e-4

    def test_Hessian_class(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        mf = mol.RHF().run()
        ref = mf.Hessian().kernel()
        mf_g = mol.RHF().Gradients()
        hess_obj = finite_diff.Hessian(mf_g)
        hess_obj.displacement = .5e-2
        dat = hess_obj.kernel()
        assert abs(dat - ref).max() < 1e-4

    def test_grad_as_scanner(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1', verbose=0)
        mf = mol.RHF().run()
        e_ref = mf.e_tot
        ref = mf.Gradients().kernel()
        g_scan = finite_diff.Gradients(mf).as_scanner()
        g_scan.displacement = .5e-2
        e, g = g_scan(mol)
        assert abs(g - ref).max() < 1e-4
        assert abs(e - e_ref) < 1e-9

    def test_hessian_against_analytic(self):
        ref = scf.RHF(h2o()).run(conv_tol=1e-14).Hessian().kernel()
        dat = finite_diff.kernel(scf.RHF(h2o()).set(conv_tol=1e-14).Gradients())
        self.assertLess(abs(dat - ref).max(), 2e-6)

    def test_hessian_is_symmetric(self):
        dat = finite_diff.kernel(scf.RHF(h2o()).set(conv_tol=1e-14).Gradients())
        self.assertAlmostEqual(abs(dat - dat.transpose(1,0,3,2)).max(), 0, 12)

    def test_symmetry_is_pinned_under_displacement(self):
        # Point group detection has a finite tolerance that the displacement
        # falls inside, so without pinning the displaced geometry keeps the
        # full group and the wavefunction is symmetry-adapted to a group it
        # no longer has.
        ref = scf.RHF(h2o()).run(conv_tol=1e-14).Hessian().kernel()
        mf = scf.RHF(h2o(symmetry=True)).set(conv_tol=1e-14)
        dat = finite_diff.kernel(mf.Gradients())
        self.assertLess(abs(dat - ref).max(), 2e-6)

    def test_rks_hessian_against_analytic(self):
        mol = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g', verbose=0)
        ref = rks_fine(mol).Hessian().kernel()
        dat = finite_diff.kernel(rks_fine(mol).Gradients())
        self.assertLess(abs(dat - ref).max(), 5e-5)

    def test_tddft_hessian(self):
        mol = h2o()
        mf = dft.RKS(mol, xc='pbe0').run(conv_tol=1e-13)
        td = tdscf.TDA(mf)
        td.conv_tol = 1e-9
        td.max_cycle = 500
        td.run(nstates=5)

        h1 = finite_diff.kernel(td.nuc_grad_method().as_scanner(state=1))
        self.assertAlmostEqual(h1[0,0,0,0], -0.31248087, 5)
        freq = thermo.harmonic_analysis(mol, h1, imaginary_freq=False)
        ref = [-8060.26, -2462.18, 5261.28]
        self.assertLess(abs(freq['freq_wavenumber'] - ref).max(), 1.0)

        # The scanner really does pin the root it was built for
        h2 = finite_diff.kernel(td.nuc_grad_method().as_scanner(state=2))
        self.assertGreater(abs(h1 - h2).max(), 1e-3)

    def test_warns_on_loose_settings(self):
        out = io.StringIO()
        mol = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g', verbose=4)
        mol.stdout = out
        mf = dft.RKS(mol, xc='pbe0')   # default conv_tol and default grid
        mf.stdout = out
        mf.run()
        g = mf.Gradients()
        g.stdout = out
        finite_diff.kernel(g, 1e-3)
        log = out.getvalue()
        self.assertIn('conv_tol', log)
        self.assertIn('DFT grid', log)


if __name__ == "__main__":
    print("Full Tests for finite_diff")
    unittest.main()
