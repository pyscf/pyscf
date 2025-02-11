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

import unittest
import numpy as np
from pyscf import gto
from pyscf import dft
from pyscf.solvent import pcm
from pyscf.solvent.hessian.pcm import analytical_grad_vmat, analytical_hess_nuc, analytical_hess_solver, analytical_hess_qv

def setUpModule():
    global mol, epsilon, lebedev_order, eps, xc, tol
    mol = gto.Mole()
    mol.atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
    '''
    mol.basis = 'sto3g'
    mol.output = '/dev/null'
    mol.build(verbose=0)
    epsilon = 78.3553
    lebedev_order = 17
    eps = 1e-3
    xc = 'B3LYP'
    tol = 1e-3

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _make_mf(method='C-PCM', restricted=True):
    if restricted:
        mf = dft.rks.RKS(mol, xc=xc).density_fit().PCM()
    else:
        mf = dft.uks.UKS(mol, xc=xc).density_fit().PCM()
    mf.with_solvent.method = method
    mf.with_solvent.eps = epsilon
    mf.with_solvent.lebedev_order = lebedev_order
    mf.conv_tol = 1e-12
    mf.conv_tol_cpscf = 1e-7
    mf.grids.atom_grid = (99,590)
    mf.verbose = 0
    mf.kernel()
    return mf

def _check_hessian(mf, h, ix=0, iy=0):
    pmol = mf.mol.copy()
    pmol.build()

    g = mf.nuc_grad_method()
    g.auxbasis_response = True
    g.kernel()
    g_scanner = g.as_scanner()

    coords = pmol.atom_coords()
    v = np.zeros_like(coords)
    v[ix,iy] = eps
    pmol.set_geom_(coords + v, unit='Bohr')
    pmol.build()
    _, g0 = g_scanner(pmol)

    pmol.set_geom_(coords - v, unit='Bohr')
    pmol.build()
    _, g1 = g_scanner(pmol)

    h_fd = (g0 - g1)/2.0/eps

    print(f'Norm of H({ix},{iy}) diff, {np.linalg.norm(h[ix,:,iy,:] - h_fd)}')
    assert(np.linalg.norm(h[ix,:,iy,:] - h_fd) < tol)

def _fd_grad_vmat(pcmobj, dm, atmlst=None):
    '''
    dv_solv / da
    slow version with finite difference
    '''
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    if atmlst is None:
        atmlst = range(mol.natm)
    nao = mol.nao
    coords = mol.atom_coords(unit='Bohr')
    def pcm_vmat_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        return v

    mol.verbose = 0
    vmat = np.empty([len(atmlst), 3, nao, nao])
    eps = 1e-5
    for i0, ia in enumerate(atmlst):
        for ix in range(3):
            dv = np.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            vmat0 = pcm_vmat_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            vmat1 = pcm_vmat_scanner(mol)

            grad_vmat = (vmat0 - vmat1)/2.0/eps
            vmat[i0,ix] = grad_vmat
    pcmobj.reset(pmol)
    return vmat

def _fd_hess_contribution(pcmobj, dm, gradient_function):
    pmol = pcmobj.mol.copy()
    mol = pmol.copy()
    coords = mol.atom_coords(unit='Bohr')

    def pcm_grad_scanner(mol):
        pcmobj.reset(mol)
        e, v = pcmobj._get_vind(dm)
        pcm_grad = gradient_function(pcmobj, dm)
        return pcm_grad

    mol.verbose = 0
    de = np.zeros([mol.natm, mol.natm, 3, 3])
    eps = 1e-5
    for ia in range(mol.natm):
        for ix in range(3):
            dv = np.zeros_like(coords)
            dv[ia,ix] = eps
            mol.set_geom_(coords + dv, unit='Bohr')
            g0 = pcm_grad_scanner(mol)

            mol.set_geom_(coords - dv, unit='Bohr')
            g1 = pcm_grad_scanner(mol)

            de[ia,:,ix,:] = (g0 - g1)/2.0/eps
    pcmobj.reset(pmol)
    return de

class KnownValues(unittest.TestCase):
    def test_hess_cpcm(self):
        print('testing C-PCM Hessian with DF-RKS')
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-RKS")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_uhf_hess_iefpcm(self):
        print("testing IEF-PCM hessian with DF-UKS")
        mf = _make_mf(method='IEF-PCM', restricted=False)
        hobj = mf.Hessian()
        hobj.set(auxbasis_response=2)
        h = hobj.kernel()
        _check_hessian(mf, h, ix=0, iy=0)
        _check_hessian(mf, h, ix=0, iy=1)

    def test_grad_vmat_cpcm(self):
        print("testing C-PCM dV_solv/dx")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_grad_vmat_iefpcm(self):
        print("testing IEF-PCM dV_solv/dx")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_grad_vmat_ssvpe(self):
        print("testing SS(V)PE dV_solv/dx")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()

        dm = mf.make_rdm1()

        test_grad_vmat = analytical_grad_vmat(hobj.base.with_solvent, dm)
        ref_grad_vmat = _fd_grad_vmat(hobj.base.with_solvent, dm)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_nuc_iefpcm(self):
        print("testing IEF-PCM d2E_nuc/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_nuc(hobj.base.with_solvent, dm)
        from pyscf.solvent.grad.pcm import grad_nuc
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_nuc)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_qv_iefpcm(self):
        print("testing IEF-PCM d2E_elec/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_qv(hobj.base.with_solvent, dm)
        from pyscf.solvent.grad.pcm import grad_qv
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_qv)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_cpcm(self):
        print("testing C-PCM d2E_KR/dx2")
        mf = _make_mf(method='C-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_iefpcm(self):
        print("testing IEF-PCM d2E_KR/dx2")
        mf = _make_mf(method='IEF-PCM')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

    def test_hess_solver_ssvpe(self):
        print("testing SS(V)PE d2E_KR/dx2")
        mf = _make_mf(method='SS(V)PE')
        hobj = mf.Hessian()
        dm = mf.make_rdm1()

        test_grad_vmat = analytical_hess_solver(hobj.base.with_solvent, dm)
        from pyscf.solvent.grad.pcm import grad_solver
        ref_grad_vmat = _fd_hess_contribution(hobj.base.with_solvent, dm, grad_solver)

        np.testing.assert_allclose(ref_grad_vmat, test_grad_vmat, atol = 1e-10)

if __name__ == "__main__":
    print("Full Tests for Hessian of PCMs")
    unittest.main()