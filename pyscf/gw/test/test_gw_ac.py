#!/usr/bin/env python

import tempfile
import pytest
import numpy as np
from pyscf import gto, dft, scf
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.ugw_ac import UGWAC
from pyscf.gw.utils import ac_grid

@pytest.fixture
def h2o_pbe0():
    mol = gto.Mole()
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf

def test_gwac_pade_incore_diag(h2o_pbe0):
    # diag self-energy, incore
    mf = h2o_pbe0
    gw = GWAC(mf)
    gw.orbs=range(4, 6)
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.42657296, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16495549, abs=1e-5)

def test_gwac_pade_incore_full(h2o_pbe0):
    # full self-energy, incore
    mf = h2o_pbe0
    gw = GWAC(mf)
    gw.fullsigma = True
    gw.orbs=range(4, 6)
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.42657296, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16495549, abs=1e-5)

def test_gwac_pade_outcore_diag(h2o_pbe0):
    mf = h2o_pbe0
    # diag self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.outcore = True
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.42657296, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16495549, abs=1e-5)

def test_gwac_pade_outcore_full(h2o_pbe0):
    mf = h2o_pbe0
    # full self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.fullsigma = True
    gw.outcore = True
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.42657296, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16495549, abs=1e-5)

def test_gwac_pade_frozen_core(h2o_pbe0):
    mf = h2o_pbe0
    # frozen core
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.frozen = 1
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.42667346, abs=1e-5)
    assert gw.mo_energy[5] == pytest.approx(0.16490656, abs=1e-5)

def test_gwac_pade_frozen_list(h2o_pbe0):
    mf = h2o_pbe0
    # frozen list
    gw = GWAC(mf)
    gw.orbs = [4, 7]
    gw.frozen = [0, 5]
    gw.kernel()
    assert gw.mo_energy[4] == pytest.approx(-0.43309464, abs=1e-5)
    assert gw.mo_energy[7] == pytest.approx(0.73675504, abs=2e-5)


def test_loop_ao2mo(h2o_pbe0):
    mf = h2o_pbe0
    gw = GWAC(mf)
    gw.initialize_df()
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    nocc = mf.mol.nelectron // 2
    ijslice = (0, nocc, 0, nmo)
    # small blocksize to force multiple blocks
    Lpq_loop = gw.loop_ao2mo(mo_coeff=mo_coeff, ijslice=ijslice, blksize=4)
    Lpq_full = gw.ao2mo(mo_coeff=mo_coeff)
    assert np.allclose(Lpq_loop, Lpq_full[:, :nocc, :], atol=1e-15)


def _check_continuation(obj):
    omega = 1j*np.linspace(.05, 4, 40)
    scale = np.array([[1., .3], [.3, .8]])
    def exact(z):
        return scale[..., None] * (1/(z+1.2)+.4/(z-2.3))
    obj.ac_fit(np.moveaxis(exact(omega), -1, 0), omega, axis=0)
    target = np.array([-.6, .2, .7])+.1j
    expected = exact(target)
    np.testing.assert_allclose(obj.ac_eval(target), expected, atol=1e-7)
    np.testing.assert_allclose(obj.ac_eval(target, axis=0), expected.transpose(2, 0, 1), atol=1e-7)
    np.testing.assert_allclose(obj.ac_eval(target[0]), expected[..., 0], atol=1e-7)
    np.testing.assert_allclose(obj[0, 1].ac_eval(target), expected[0, 1], atol=1e-7)
    np.testing.assert_allclose(obj.diagonal().ac_eval(target), expected[[0, 1], [0, 1]], atol=1e-7)
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        obj.save(f.name)
        restored = ac_grid.load_ac(f.name)
        np.testing.assert_array_equal(restored.ac_eval(target), obj.ac_eval(target))


def test_ac_two_pole_pade():
    _check_continuation(ac_grid.PadeAC(npts=6))


def test_ac_two_pole_fit():
    _check_continuation(ac_grid.TwoPoleAC(orbs=[0, 1], nocc=1))


def test_imaginary_frequency_quadrature():
    freq, weight = ac_grid._get_scaled_legendre_roots(64)
    assert np.all(freq > 0)
    assert np.all(weight > 0)
    assert np.dot(weight, 1 / (1 + freq ** 2)) == pytest.approx(np.pi / 2, abs=5e-13, rel=0)


@pytest.fixture(scope='module')
def lih_hf():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run(conv_tol=1e-12)


@pytest.fixture(scope='module')
def lih_density_gw(lih_hf):
    rgw = GWAC(lih_hf)
    ugw = UGWAC(lih_hf.to_uhf())
    for obj in (rgw, ugw):
        # Resolve the high-frequency tail for the particle-number sum rule.
        obj.nw = 2000
        obj.fullsigma = True
        obj.rdm = True
        obj.kernel()
    return rgw, ugw


def _check_density(lih_density_gw, mode):
    rgw, ugw = lih_density_gw
    mf = rgw._scf
    dm = rgw.make_rdm1(mode=mode)
    udm = ugw.make_rdm1(mode=mode)
    np.testing.assert_allclose(udm[0]*2, dm, atol=1e-9)
    np.testing.assert_allclose(udm[1]*2, dm, atol=1e-9)
    ao = rgw.make_rdm1(ao_repr=True, mode=mode)
    uao = ugw.make_rdm1(ao_repr=True, mode=mode)
    c = mf.mo_coeff
    np.testing.assert_allclose(ao, c @ dm @ c.T, atol=1e-12)
    np.testing.assert_allclose(uao.sum(axis=0), ao, atol=1e-9)
    np.testing.assert_allclose(dm, dm.T, atol=1e-12)
    if mode == 'linear':
        assert np.trace(dm) == pytest.approx(mf.mol.nelectron, abs=5e-07, rel=0)


def test_linear_density(lih_density_gw):
    _check_density(lih_density_gw, 'linear')


def test_dyson_density(lih_density_gw):
    _check_density(lih_density_gw, 'dyson')


def _check_green_function(lih_density_gw, mode):
    rgw, ugw = lih_density_gw
    omega = np.array([-.7, -.1, .6])
    rgf = rgw.make_gf(omega, eta=.05, mode=mode)
    ugf = ugw.make_gf(omega, eta=.05, mode=mode)
    for r, u in zip(rgf, ugf):
        np.testing.assert_allclose(u[0], r, atol=1e-7)
        np.testing.assert_allclose(u[1], r, atol=1e-7)


def test_linear_green_function(lih_density_gw):
    _check_green_function(lih_density_gw, 'linear')


def test_dyson_green_function(lih_density_gw):
    _check_green_function(lih_density_gw, 'dyson')


def _check_outcore(lih_hf, fullsigma, restricted=False):
    cls = GWAC if restricted else UGWAC
    mf = lih_hf if restricted else lih_hf.to_uhf()
    ref, out = cls(mf), cls(mf)
    for obj in (ref, out):
        obj.nw = 40
        obj.ac_pade_npts = 8
        obj.qpe_tol = 1e-10
        obj.fullsigma = fullsigma
        obj.rdm = fullsigma
    out.outcore = True
    out.segsize = 1
    ref.kernel()
    out.kernel()
    np.testing.assert_allclose(out.mo_energy, ref.mo_energy, atol=1e-8)
    omega = np.array([-.6, .1, .7])
    for actual, expected in zip(out.make_gf(omega, eta=.05), ref.make_gf(omega, eta=.05)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    if fullsigma:
        np.testing.assert_allclose(out.make_rdm1(), ref.make_rdm1(), atol=1e-9)
    else:
        sigma = out.make_gf(omega, eta=.05)[2]
        diagonal = np.diagonal(sigma, axis1=-3, axis2=-2).swapaxes(-1, -2)
        np.testing.assert_allclose(diagonal, out.acobj.ac_eval(omega+.05j), atol=1e-12)


def test_outcore_diagonal_self_energy(lih_hf):
    _check_outcore(lih_hf, False)


def test_outcore_full_self_energy(lih_hf):
    _check_outcore(lih_hf, True)


def test_restricted_outcore_diagonal_self_energy(lih_hf):
    _check_outcore(lih_hf, False, restricted=True)


def test_restricted_outcore_full_self_energy(lih_hf):
    _check_outcore(lih_hf, True, restricted=True)
