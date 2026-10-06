#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
from pyscf import gto, scf, dft


def setUpModule():
    global mol_cs, mol_os
    mol_cs = gto.M(atom='O 0 0 0.20; H 0 1.43 -0.90; H 0 -1.43 -0.90',
                   basis='sto-3g', unit='Bohr', verbose=0)
    mol_os = gto.M(atom='O 0 0 0.15; H 0 0 -1.75',
                   basis='sto-3g', spin=1, unit='Bohr', verbose=0)

def tearDownModule():
    global mol_cs, mol_os
    del mol_cs, mol_os


def _spin_rotate(mo, theta, phi, nao):
    '''Global rotation of the spin quantization axis. It maps a GHF solution
    onto an exactly degenerate one with complex off-diagonal spin blocks.'''
    c, s = numpy.cos(theta/2), numpy.sin(theta/2)
    u = numpy.array([[c*numpy.exp(-1j*phi/2), -s*numpy.exp(-1j*phi/2)],
                     [s*numpy.exp( 1j*phi/2),  c*numpy.exp( 1j*phi/2)]])
    r = numpy.zeros((2*nao, 2*nao), dtype=complex)
    eye = numpy.eye(nao)
    for i in range(2):
        for j in range(2):
            r[i*nao:(i+1)*nao, j*nao:(j+1)*nao] = u[i,j] * eye
    return r.dot(mo)

def _finite_diff(build, solve, coords, dm0=None, step=2e-4):
    '''Central finite difference of the total energy wrt the coordinates'''
    de = numpy.zeros_like(coords)
    for ia in range(coords.shape[0]):
        for x in range(3):
            cp = coords.copy(); cp[ia,x] += step
            cm = coords.copy(); cm[ia,x] -= step
            mp = solve(build(cp), dm0)
            mm = solve(build(cm), dm0)
            assert mp.converged and mm.converged
            de[ia,x] = (mp.e_tot - mm.e_tot) / (2*step)
    return de


class KnownValues(unittest.TestCase):
    def test_ghf_grad_matches_rhf(self):
        # A closed-shell GHF solution is the RHF solution
        mr = scf.RHF(mol_cs).run(conv_tol=1e-12)
        mg = scf.GHF(mol_cs).run(conv_tol=1e-12)
        self.assertAlmostEqual(mg.e_tot, mr.e_tot, 9)
        g_r = mr.nuc_grad_method().kernel()
        g_g = mg.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_g - g_r).max(), 0, 7)

    def test_ghf_grad_matches_uhf(self):
        # A collinear GHF solution is the UHF solution
        mu = scf.UHF(mol_os).run(conv_tol=1e-12)
        mg = scf.GHF(mol_os).run(conv_tol=1e-12)
        self.assertAlmostEqual(mg.e_tot, mu.e_tot, 9)
        g_u = mu.nuc_grad_method().kernel()
        g_g = mg.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g_g - g_u).max(), 0, 7)

    def test_grad_spin_rotation_invariance(self):
        # Rotating the spin quantization axis produces an exactly degenerate
        # solution with complex spin blocks. The gradient must not change.
        mf = scf.GHF(mol_os).run(conv_tol=1e-12)
        nao = mol_os.nao_nr()
        ref = mf.nuc_grad_method().kernel()
        for theta, phi in ((0.7, 1.1), (1.9, 2.7)):
            mo = _spin_rotate(mf.mo_coeff.astype(complex), theta, phi, nao)
            mf1 = scf.GHF(mol_os)
            mf1.mo_coeff, mf1.mo_occ, mf1.mo_energy = mo, mf.mo_occ, mf.mo_energy
            mf1.converged = True
            dm = mf1.make_rdm1()
            # the rotated density really does exercise the complex blocks
            self.assertTrue(abs(dm.imag).max() > 1e-3)
            self.assertTrue(abs(dm[:nao,nao:]).max() > 1e-3)
            self.assertAlmostEqual(mf1.energy_tot(dm), mf.e_tot, 9)
            g = mf1.nuc_grad_method().kernel()
            self.assertAlmostEqual(abs(g - ref).max(), 0, 9)

    def test_ghf_grad_finite_diff(self):
        # Spin-frustrated H3 converges to a complex non-collinear solution
        atom = [['H', (0.0, 1.2, 0.0)],
                ['H', (1.0392, -0.6, 0.0)],
                ['H', (-1.0392, -0.6, 0.1)]]
        coords = numpy.array([a[1] for a in atom])

        def build(c):
            return gto.M(atom=[[a[0], tuple(x)] for a, x in zip(atom, c)],
                         basis='sto-3g', spin=1, unit='Bohr', verbose=0)

        def solve(mol, dm0=None):
            mf = scf.GHF(mol)
            mf.conv_tol = 1e-12
            mf.max_cycle = 200
            if dm0 is None:
                numpy.random.seed(7)
                nso = mol.nao_nr() * 2
                z = numpy.random.rand(nso, nso) + 1j*numpy.random.rand(nso, nso)
                dm0 = z + z.conj().T
                dm0 *= mol.nelectron / numpy.trace(dm0).real
            mf.kernel(dm0=dm0)
            return mf

        mf = solve(build(coords))
        self.assertTrue(mf.converged)
        dm = mf.make_rdm1()
        nao = mf.mol.nao_nr()
        self.assertTrue(abs(dm.imag).max() > 1e-3)
        self.assertTrue(abs(dm[:nao,nao:]).max() > 1e-3)

        ana = mf.nuc_grad_method().kernel()
        num = _finite_diff(build, solve, coords, dm0=dm)
        self.assertAlmostEqual(abs(ana - num).max(), 0, 6)
        # translational invariance
        self.assertAlmostEqual(abs(ana.sum(axis=0)).max(), 0, 9)

    def test_grad_atmlst(self):
        mf = scf.GHF(mol_cs).run(conv_tol=1e-12)
        g = mf.nuc_grad_method()
        ref = g.kernel()
        sub = g.kernel(atmlst=[0, 2])
        self.assertAlmostEqual(abs(sub - ref[[0, 2]]).max(), 0, 9)

    def test_x2c_refused(self):
        # X2C1eGHF is a mixin over GHF, so it inherits GHF.Gradients. Its
        # one-electron Hamiltonian is not the spin-free one this module
        # differentiates, so it has to refuse rather than drop a term.
        mf = scf.GHF(mol_cs).x2c1e()
        mf.conv_tol = 1e-10
        mf.kernel()
        self.assertRaises(NotImplementedError, mf.nuc_grad_method().kernel)

    def test_soc_refused(self):
        # A spin-orbit ECP adds a complex, spin-mixing term to hcore whose
        # derivative is not available here.
        m = gto.M(atom='I 0 0 0; H 0 0 1.61', basis='sto-3g',
                  ecp={'I': 'crenbl'}, verbose=0)
        if not m.has_ecp_soc():
            self.skipTest('basis set does not provide a spin-orbit ECP')
        mf = scf.GHF(m)
        mf.with_soc = True
        mf.conv_tol = 1e-9
        mf.kernel()
        self.assertRaises(NotImplementedError, mf.nuc_grad_method().kernel)

    def test_soc_flag_without_soc_ecp_is_allowed(self):
        # with_soc alone changes nothing when the molecule has no spin-orbit
        # ECP, so the gradient must still work.
        mf = scf.GHF(mol_cs)
        mf.with_soc = True
        mf.conv_tol = 1e-11
        mf.kernel()
        ref = scf.RHF(mol_cs).run(conv_tol=1e-11).nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(mf.nuc_grad_method().kernel() - ref).max(), 0, 7)

    def test_gks_grad_matches_rks(self):
        for xc in ('lda,vwn', 'pbe', 'b3lyp'):
            mr = dft.RKS(mol_cs, xc=xc)
            mr.conv_tol = 1e-12
            mr.grids.prune = None
            mr.kernel()
            mg = dft.GKS(mol_cs, xc=xc)
            mg.conv_tol = 1e-12
            mg.grids.prune = None
            mg.kernel()
            self.assertAlmostEqual(mg.e_tot, mr.e_tot, 9)
            g_r = mr.nuc_grad_method().kernel()
            g_g = mg.nuc_grad_method().kernel()
            self.assertAlmostEqual(abs(g_g - g_r).max(), 0, 7)

    def test_gks_grad_finite_diff(self):
        atom = [['O', (0.0, 0.0, 0.20)],
                ['H', (0.0, 1.43, -0.90)],
                ['H', (0.0, -1.43, -0.90)]]
        coords = numpy.array([a[1] for a in atom])

        def build(c):
            return gto.M(atom=[[a[0], tuple(x)] for a, x in zip(atom, c)],
                         basis='sto-3g', unit='Bohr', verbose=0)

        def solve(mol, dm0=None):
            mf = dft.GKS(mol, xc='pbe')
            mf.conv_tol = 1e-12
            mf.grids.level = 5
            mf.grids.prune = None
            mf.kernel(dm0=dm0)
            return mf

        mf = solve(build(coords))
        num = _finite_diff(build, solve, coords, dm0=mf.make_rdm1())

        g = mf.nuc_grad_method()
        g.grid_response = True
        self.assertAlmostEqual(abs(g.kernel() - num).max(), 0, 6)

    def test_gks_noncollinear_not_implemented(self):
        # The XC nuclear derivatives are only available for the collinear
        # scheme; the others must refuse rather than return a wrong number.
        ref = dft.GKS(mol_cs, xc='lda,vwn')
        ref.conv_tol = 1e-10
        ref.kernel()
        for scheme in ('ncol', 'mcol'):
            # Reuse the converged orbitals: the gradient has to refuse before
            # it reaches the XC evaluation, so no SCF (and no mcfun) is needed.
            mf = dft.GKS(mol_cs, xc='lda,vwn')
            mf.collinear = scheme
            mf.mo_coeff, mf.mo_occ, mf.mo_energy = (
                ref.mo_coeff, ref.mo_occ, ref.mo_energy)
            mf.converged = True
            self.assertRaises(NotImplementedError,
                              mf.nuc_grad_method().kernel)


if __name__ == '__main__':
    print('Full Tests for GHF and GKS gradients')
    unittest.main()
