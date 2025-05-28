#!/usr/bin/env python
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import unittest
import numpy as np
from pyscf.pbc import gto, dft, df
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as rks_grad
from pyscf.pbc.grad import uks as uks_grad
from pyscf.pbc.grad import krks as krks_grad

def setUpModule():
    global He_orth, He_nonorth, cell_orth, cell_nonorth, dm, dm1

    He_basis = [[0, ( 1, 1, .1), (.5, .1, 1)],
                [1, (.8, 1)]]
    He_orth = gto.M(atom='He 0 0 0; He 0 0 2',
                    basis=He_basis,
                    unit='B',
                    precision=1e-8,
                    ke_cutoff=150,
                    pseudo = 'gth-pbe',
                    a=np.eye(3)*5)

    L = 2.7
    He_nonorth = gto.M(atom='He 0.01 0 0; He 1.35 1.35 1.35',
                       basis=He_basis,
                       unit='B',
                       precision = 1e-8,
                       ke_cutoff=100,
                       pseudo='gth-pbe',
                       a=[[0.0, L, L], [L, 0.0, L], [L, L, 0.0]])

    cell_orth = gto.M(
        a = np.eye(3)*3.5668,
        atom = '''C     0.      0.      0.
                  C     1.8     1.8     1.8   ''',
        basis = 'gth-dzv',
        pseudo = 'gth-pbe',
        precision = 1e-8,
        ke_cutoff = 200,
    )

    np.random.seed(2)
    cell_nonorth = gto.M(
        a = np.eye(3)*3.5668 + np.random.random((3,3)),
        atom = '''C     0.      0.      0.
                  C     0.8917  0.8917  0.8917''',
        basis = 'gth-dzv',
        pseudo = 'gth-pade',
        precision = 1e-8,
        ke_cutoff = 200,
    )

    nao = cell_orth.nao
    dm = np.random.random((nao,nao)) * .2
    dm = (dm + dm.T) * .5
    dm1 = np.asarray([dm,dm])*.5

def tearDownModule():
    global He_orth, He_nonorth, cell_orth, cell_nonorth, dm, dm1
    del He_orth, He_nonorth, cell_orth, cell_nonorth, dm, dm1

def _fftdf_energy_grad(cell, xc):
    mf = dft.KRKS(cell, kpts=np.zeros((1,3)))
    mf.xc = xc
    e = mf.kernel()
    grad = krks_grad.Gradients(mf)
    g = grad.kernel()
    return e, g

def _multigrid2_energy_grad(cell, xc, spin=0):
    if spin == 0:
        mf = dft.RKS(cell)
    elif spin == 1:
        mf = dft.UKS(cell)
    mf.xc =  xc
    mf.with_df = multigrid.MultiGridFFTDF2(cell)
    mf.with_df.ngrids = 2
    e = mf.kernel()
    if spin == 0:
        g = rks_grad.Gradients(mf).kernel()
    elif spin == 1:
        g = uks_grad.Gradients(mf).kernel()
    return e, g

def _test_veff(cell, xc, dm, spin=0, tol=1e-7):
    if spin == 0:
        mf = dft.RKS(cell)
    elif spin == 1:
        mf = dft.UKS(cell)
    mf.xc = xc
    ref = mf.get_veff(dm=dm)

    mf.with_df = multigrid.MultiGridFFTDF2(cell)
    vxc = mf.get_veff(dm=dm)
    assert vxc.shape == ref.shape
    assert abs(ref-vxc).max() < tol

def with_grid_level_method(method):
    from contextlib import contextmanager
    from pyscf.pbc.dft.multigrid import _backend_c as backend

    @contextmanager
    def _ContextManager():
        try:
            backend.TaskList.grid_level_method = method
            yield
        finally:
            backend.TaskList.grid_level_method = "pyscf"
    return _ContextManager()

class KnownValues(unittest.TestCase):
    def test_ntasks(self):
        from pyscf.pbc.dft.multigrid.multigrid_pair import multi_grids_tasks
        with with_grid_level_method("cp2k"):
            task_list = multi_grids_tasks(He_orth, hermi=1, ngrids=2)
            assert task_list.ntasks == [1150, 0]
            task_list = multi_grids_tasks(He_nonorth, hermi=1, ngrids=2)
            assert task_list.ntasks == [2771, 732]
        with with_grid_level_method("pyscf"):
            task_list = multi_grids_tasks(He_orth, hermi=1, ngrids=2)
            assert task_list.ntasks == [1150, 0]
            task_list = multi_grids_tasks(He_nonorth, hermi=1, ngrids=2)
            assert task_list.ntasks == [3249, 254]

    def test_orth_get_pp(self):
        ref = df.FFTDF(cell_orth).get_pp()
        out = multigrid.MultiGridFFTDF2(cell_orth).get_pp(return_full=True)
        assert out.shape == ref.shape
        assert abs(ref-out).max() < 1e-7

    def test_nonorth_get_pp(self):
        ref = df.FFTDF(cell_nonorth).get_pp()
        out = multigrid.MultiGridFFTDF2(cell_nonorth).get_pp(return_full=True)
        assert out.shape == ref.shape
        assert abs(ref-out).max() < 1e-7

    def test_orth_lda_veff(self):
        xc = 'lda, vwn'
        _test_veff(cell_orth, xc, dm, spin=0)
        _test_veff(cell_orth, xc, dm1, spin=1)

    def test_orth_gga_veff(self):
        xc = 'pbe, pbe'
        _test_veff(cell_orth, xc, dm, spin=0)
        _test_veff(cell_orth, xc, dm1, spin=1)

    def test_nonorth_lda_veff(self):
        xc = 'lda, vwn'
        _test_veff(cell_nonorth, xc, dm, spin=0)
        _test_veff(cell_nonorth, xc, dm1, spin=1)

    def test_nonorth_gga_veff(self):
        xc = 'pbe, pbe'
        _test_veff(cell_nonorth, xc, dm, spin=0)
        _test_veff(cell_nonorth, xc, dm1, spin=1)

    def test_orth_lda_dft(self):
        xc = 'lda, vwn'
        e0, g0 = _fftdf_energy_grad(He_orth, xc)
        e,  g  = _multigrid2_energy_grad(He_orth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_orth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-6
        assert abs(g1-g0).max() < 1e-6

    def test_orth_gga_dft(self):
        xc = 'pbe, pbe'
        e0, g0 = _fftdf_energy_grad(He_orth, xc)
        e,  g  = _multigrid2_energy_grad(He_orth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_orth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-6
        assert abs(g1-g0).max() < 1e-6

    def test_nonorth_lda_dft(self):
        xc = 'lda, vwn'
        e0, g0 = _fftdf_energy_grad(He_nonorth, xc)
        e,  g  = _multigrid2_energy_grad(He_nonorth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_nonorth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-7
        assert abs(g1-g0).max() < 1e-7

    def test_nonorth_gga_dft(self):
        xc = 'pbe, pbe'
        e0, g0 = _fftdf_energy_grad(He_nonorth, xc)
        e,  g  = _multigrid2_energy_grad(He_nonorth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_nonorth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-7
        assert abs(g1-g0).max() < 1e-7

    def test_orth_j_dft(self):
        xc = ''
        e0, g0 = _fftdf_energy_grad(He_orth, xc)
        e,  g  = _multigrid2_energy_grad(He_orth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_orth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-7
        assert abs(g1-g0).max() < 1e-7

    def test_nonorth_j_dft(self):
        xc = ''
        e0, g0 = _fftdf_energy_grad(He_nonorth, xc)
        e,  g  = _multigrid2_energy_grad(He_nonorth, xc, 0)
        e1, g1 = _multigrid2_energy_grad(He_nonorth, xc, 1)
        assert abs(e-e0) < 1e-7
        assert abs(e1-e0) < 1e-7
        assert abs(g-g0).max() < 1e-7
        assert abs(g1-g0).max() < 1e-7

if __name__ == '__main__':
    print("Full Tests for multigrid2")
    unittest.main()
