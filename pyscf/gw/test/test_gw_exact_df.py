#!/usr/bin/env python

import pytest
import numpy as np

from pyscf import dft, gto, scf
from pyscf.gw.gw_exact_df import GWExactDF
from pyscf.gw.ugw_exact_df import UGWExactDF


@pytest.fixture
def h2o_mol():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()
    return mol


def test_gw_exact_df_pbe0(h2o_mol):
    mf = dft.RKS(h2o_mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = GWExactDF(mf)
    gw.eta = 1.0e-5
    gw.qpe_linearized = False
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.42657296, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16495549, abs=1e-5)

    e_tot, e_hf, e_c = gw.energy_tot()
    assert e_c == pytest.approx(-0.49425105, abs=1e-7)


def test_gw_exact_df_rpae(h2o_mol):
    mf = scf.RHF(h2o_mol)
    mf.kernel()

    gw = GWExactDF(mf)
    gw.eta = 1.0e-5
    gw.qpe_linearized = False
    gw.RPAE = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.42572262, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16089239, abs=1e-5)


@pytest.fixture(scope='module')
def lih_exact_gw():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    mf = dft.RKS(mol, xc='pbe')
    mf.grids.level = 0
    mf.run(conv_tol=1e-12)
    rsolver = GWExactDF(mf)
    usolver = UGWExactDF(mf.to_uks())
    rsolver.kernel()
    usolver.kernel()

    return rsolver, usolver


def _check_green_function(lih_exact_gw, mode, fullsigma):
    rsolver, usolver = lih_exact_gw
    omega = np.array([-.5, .1, .8])
    ref = rsolver.make_gf(omega, eta=.05, mode=mode, fullsigma=fullsigma)
    actual = usolver.make_gf(omega, eta=.05, mode=mode, fullsigma=fullsigma)
    for name, a, b in zip(('G', 'G0', 'Sigma'), actual, ref):
        np.testing.assert_allclose(a[0], b, atol=1e-9)
        np.testing.assert_allclose(a[1], b, atol=1e-9)


def test_linear_full(lih_exact_gw):
    _check_green_function(lih_exact_gw, 'linear', True)


def test_linear_diagonal(lih_exact_gw):
    _check_green_function(lih_exact_gw, 'linear', False)


def test_dyson_full(lih_exact_gw):
    _check_green_function(lih_exact_gw, 'dyson', True)


def test_dyson_diagonal(lih_exact_gw):
    _check_green_function(lih_exact_gw, 'dyson', False)


def test_density_closed_shell(lih_exact_gw):
    rsolver, usolver = lih_exact_gw
    for mode in ('linear', 'dyson'):
        ref = rsolver.make_rdm1(nw=120, mode=mode)
        actual = usolver.make_rdm1(nw=120, mode=mode)
        np.testing.assert_allclose(actual[0]*2, ref, atol=1e-9)
        np.testing.assert_allclose(actual[1]*2, ref, atol=1e-9)
