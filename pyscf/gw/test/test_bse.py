#!/usr/bin/env python

import tempfile
import numpy as np
from scipy import linalg
import pytest
from pyscf import dft, gto, scf
from pyscf.gw import bse
from pyscf.gw.bse import BSE
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.ugw_ac import UGWAC


@pytest.fixture(scope='module')
def h2o_pbe_gw():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    gw = GWAC(mf)
    gw.kernel()
    return gw


@pytest.fixture(scope='module')
def h2o_cation_pbe_ugw():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    gw = UGWAC(mf)
    gw.kernel()
    return gw


def test_bse_singlet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('s')[0]
    assert exci[0] == pytest.approx(0.25749397, abs=1e-5)


def test_bse_triplet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('t')[0]
    assert exci[0] == pytest.approx(0.22299263, abs=1e-5)


def test_bse_unrestricted(h2o_cation_pbe_ugw):
    bse = BSE(h2o_cation_pbe_ugw)
    exci = bse.kernel('u')[0]
    assert exci[0] == pytest.approx(0.02114003, abs=1e-5)


def test_bse_energy_specific_singlet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('s', e_min=0.4)[0]
    assert exci[0] == pytest.approx(0.42691789, abs=1e-5)


def test_bse_energy_specific_triplet(h2o_pbe_gw):
    bse = BSE(h2o_pbe_gw)
    exci = bse.kernel('t', e_min=0.4)[0]
    assert exci[0] == pytest.approx(0.45195324, abs=1e-5)


@pytest.fixture(scope="module")
def lih_gw():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    gw = GWAC(mf)
    gw.nw = 40
    gw.kernel()

    return gw


def _check_iterative(lih_gw, multi, tda=False, trial='identity'):
    dense = bse.BSE(lih_gw)
    dense.TDA = tda
    ref = dense.full_diagonalization(multi)[0]
    iterative = bse.BSE(lih_gw)
    iterative.TDA = tda
    iterative.nroot = 2
    iterative.init_ntri = 2
    iterative.trial = trial
    iterative.nocc_sub = 2
    iterative.nvir_sub = 3
    iterative.residue_thresh = 1e-10
    energies, x, y = iterative.kernel(multi, precond_exact_diag=tda)
    np.testing.assert_allclose(energies, ref[:2], atol=1e-8)
    # BSE eigenvectors have the symplectic normalization X^T X - Y^T Y = I.
    x = np.concatenate([a.reshape(2, -1) for a in x], axis=1)
    y = np.concatenate([a.reshape(2, -1) for a in y], axis=1)
    np.testing.assert_allclose(x @ x.T - y @ y.T, np.eye(2), atol=1e-8)


def test_singlet_davidson(lih_gw):
    _check_iterative(lih_gw, 's')


def test_triplet_davidson(lih_gw):
    _check_iterative(lih_gw, 't')


def test_tda_subspace_guess(lih_gw):
    _check_iterative(lih_gw, 's', tda=True, trial='subspace')


def test_unrestricted_closed_shell_spectrum(lih_gw):
    ugw = UGWAC(lih_gw._scf.to_uhf())
    # Compare BSE spin representations with identical quasiparticle energies.
    # Independent analytic continuations can differ beyond this test's tolerance.
    ugw.mo_energy = np.repeat(lih_gw.mo_energy[None, :], 2, axis=0)
    restricted = bse.BSE(lih_gw)
    ugw.Lpq = np.repeat(restricted.Lpq, 2, axis=0)
    singlet = restricted.full_diagonalization('s')[0]
    triplet = restricted.full_diagonalization('t')[0]
    unrestricted = bse.BSE(ugw)
    energies = unrestricted.full_diagonalization('u')[0]
    np.testing.assert_allclose(energies, np.sort(np.r_[singlet, triplet]), atol=1e-7)
    spin = bse._get_spin_square(unrestricted.nocc, unrestricted.X_vec,
                               unrestricted.Y_vec, unrestricted.mo_coeff,
                               unrestricted.mf.get_ovlp())
    np.testing.assert_allclose(np.minimum(abs(spin), abs(spin-2)), 0., atol=1e-7)
    unrestricted.analyze(thresh=.1)


def test_checkpoint_growth_and_restart():
    rng = np.random.default_rng(8)
    arrays = [rng.standard_normal((6, 8)) for _ in range(3)]
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        bse.davidson_save_checkpoint(f.name, *arrays, 2)
        loaded = [np.zeros_like(a) for a in arrays]
        assert bse.davidson_load_from_checkpoint(f.name, *loaded) == 2
        bse.davidson_save_checkpoint(f.name, *arrays, 6)
        bse.davidson_save_checkpoint(f.name, *arrays, 6)
        loaded = [np.zeros_like(a) for a in arrays]
        assert bse.davidson_load_from_checkpoint(f.name, *loaded) == 6
        for actual, expected in zip(loaded, arrays):
            np.testing.assert_array_equal(actual, expected)
        # A restart overwrites a larger previous Davidson space.
        bse.davidson_save_checkpoint(f.name, *arrays, 3)
        assert bse.davidson_load_from_checkpoint(f.name, *loaded, nload=2) == 2
        for actual, expected in zip(loaded, arrays):
            np.testing.assert_array_equal(actual[:2], expected[:2])


def _check_lanczos_moments(lih_gw, tda):
    obj = bse.BSE(lih_gw)
    obj.TDA = tda
    energy, xs, ys = obj.full_diagonalization('s')
    x, y = xs[0].reshape(len(energy), -1), ys[0].reshape(len(energy), -1)
    plus, minus = x+y, x-y
    apb = (minus.T*energy) @ minus
    amb = (plus.T*energy) @ plus
    trial = np.random.default_rng(10).normal(size=len(energy))
    trial /= np.linalg.norm(trial)
    alpha, beta = bse.bse_lanczos(obj, 's', u1=trial, nsteps=3)
    tri = np.diag(alpha)+np.diag(beta[:-1], 1)+np.diag(beta[:-1], -1)
    matrix = apb if tda else amb @ apb
    left = trial if tda else trial @ apb / (trial @ apb @ trial)
    power = np.eye(len(energy))
    small_power = np.eye(3)
    # Three Lanczos steps preserve the first five spectral moments.
    for order in range(1, 6):
        power = power @ matrix
        small_power = small_power @ tri
        assert left @ power @ trial == pytest.approx(small_power[0, 0], abs=5e-9, rel=0), f"moment {order}"


def test_tda_lanczos_moments(lih_gw):
    _check_lanczos_moments(lih_gw, True)


def test_full_bse_lanczos_moments(lih_gw):
    _check_lanczos_moments(lih_gw, False)


@pytest.fixture
def restart_space():
    rng = np.random.default_rng(22)
    trial = np.linalg.qr(rng.normal(size=(48, 28)))[0].T.copy()
    a, b = rng.normal(size=(2, 28, 28))
    mp = a @ a.T + np.eye(28)*3
    mm = b @ b.T + np.eye(28)*2

    return trial, mp, mm


def test_tda_restart(restart_space):
    trial, mp, mm = restart_space
    energies, vectors = np.linalg.eigh(mp)
    cutoff = (energies[1]+energies[2])/2
    expected = vectors[:, 2:14].T @ trial
    saved = trial.copy()
    n = bse.davidson_restart(mp, None, saved, 12, e_min=cutoff)
    assert n == 12
    np.testing.assert_allclose(saved[:n] @ saved[:n].T, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(saved[:n].T @ saved[:n], expected.T @ expected, atol=1e-12)


def test_full_bse_restart(restart_space):
    trial, mp, mm = restart_space
    squared, vectors = linalg.eigh(mp, mm, type=3)
    energies = np.sqrt(squared[:12])
    xpy = vectors[:, :12] / np.sqrt(energies)
    xmy = mp @ xpy / energies
    expected = np.concatenate([xpy.T @ trial, xmy.T @ trial])
    saved = trial.copy()
    n = bse.davidson_restart(mp, mm, saved, 12)
    assert n == 24
    np.testing.assert_allclose(saved[:n] @ saved[:n].T, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(expected @ saved[:n].T @ saved[:n], expected, atol=1e-11)


def test_restart_when_full_space_is_already_small(restart_space):
    trial, mp, mm = restart_space
    trial = trial[:8, :10].copy()
    projector = np.linalg.qr(trial.T)[0]
    n = bse.davidson_restart(mp[:8, :8], mm[:8, :8], trial, 6)
    assert n == 8
    np.testing.assert_allclose(trial @ trial.T, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(trial.T @ trial, projector @ projector.T, atol=1e-12)
