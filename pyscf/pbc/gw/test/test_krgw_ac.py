#!/usr/bin/env python

import numpy as np
import pytest

from pyscf.pbc import df, dft, gto, scf
from pyscf.pbc.gw import krgw_ac, kugw_ac
from pyscf.pbc.gw.krgw_ac import KRGWAC


@pytest.fixture(scope="module")
def diamond_pbe():
    cell = gto.Cell()
    cell.build(
        unit="angstrom",
        a="""
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        """,
        atom="C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375",
        dimension=3,
        verbose=0,
        output="/dev/null",
        pseudo="gth-pade",
        basis="gth-szv",
        precision=1e-10,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf.build()

    kmf = dft.KRKS(cell, kpts).rs_density_fit()
    kmf.xc = "pbe"
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_krgwac_pade_no_fc_high_cost(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = False
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.62044205, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.96572609, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.52637438, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.07517363, abs=1e-4)


def test_krgwac_pade_no_fc_outcore_high_cost(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = False
    gw.outcore = True
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.62044205, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.96572609, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.52637438, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(1.07517363, abs=1e-4)


def test_krgwac_pade_with_fc_high_cost(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = True
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 7))

    assert gw.mo_energy[0][3] == pytest.approx(0.44025061, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.80148565, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(0.35193483, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(0.92909525, abs=1e-4)


def test_krgwac_pade_with_fc_frozen_core_high_cost(diamond_pbe):
    gw = KRGWAC(diamond_pbe)
    gw.ac = "pade"
    gw.qpe_linearized = False
    gw.fc = True
    gw.frozen = 1
    gw.kernel()

    assert gw.mo_energy[0][3] == pytest.approx(0.44092615, abs=1e-4)
    assert gw.mo_energy[0][4] == pytest.approx(0.79820946, abs=1e-4)


@pytest.fixture(scope="module")
def h2_hf():
    cell = gto.M(atom='H 0 0 0; H 0 0 .74', a=np.eye(3)*4,
                 basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                 precision=1e-6, verbose=0)
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
    mf = mf.density_fit(auxbasis='weigend').run(conv_tol=1e-10)
    return mf


def run_small_gw(mf, method, unrestricted=False, **options):
    obj = method(mf.to_uhf() if unrestricted else mf)
    obj.fc = False
    obj.nw = 32
    obj.ac_pade_npts = 12
    for name, value in options.items():
        setattr(obj, name, value)
    obj.kernel()
    assert np.isfinite(obj.mo_energy).all()
    return obj


@pytest.fixture(scope="module")
def lih_hf():
    cell = gto.M(atom='Li 0 0 0; H 0 0 1.6', a=np.eye(3)*4,
                 basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                 precision=1e-7, verbose=0)
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None)
    mf = mf.density_fit(auxbasis='weigend').run(conv_tol=1e-11)
    return mf


def configure_frozen_gw(obj):
    obj.fc = False
    obj.nw = 40
    obj.ac_pade_npts = 8
    obj.qpe_tol = 1e-10
    return obj


def test_frozen_index_equivalence(lih_hf):
    ref = configure_frozen_gw(krgw_ac.KRGWAC(lih_hf, frozen=1))
    ref.kernel()
    selected = configure_frozen_gw(krgw_ac.KRGWAC(lih_hf, frozen=[0]))
    selected.kernel(orbs=[1, 2])
    np.testing.assert_allclose(selected.mo_energy[:, 1:3], ref.mo_energy[:, 1:3], atol=1e-8)
    assert list(selected.orbs_frz) == [0, 1]


def test_metallic_outcore_closed_shell(lih_hf):
    mf = scf.addons.smearing_(lih_hf.copy(), sigma=.15).run(conv_tol=1e-11)
    ref = configure_frozen_gw(krgw_ac.KRGWAC(mf))
    ref.fullsigma = True
    ref.rdm = True
    ref.kernel()
    out = configure_frozen_gw(krgw_ac.KRGWAC(mf))
    out.fullsigma = True
    out.rdm = True
    out.outcore = True
    out.kernel()
    np.testing.assert_allclose(out.mo_energy, ref.mo_energy, atol=1e-8)
    spin = configure_frozen_gw(kugw_ac.KUGWAC(mf.to_uhf()))
    spin.fullsigma = True
    spin.rdm = True
    spin.kernel()
    np.testing.assert_allclose(spin.sigmaI[0], ref.sigmaI, atol=1e-10)
    # Analytic continuation is intended for valence quasiparticle energies.
    np.testing.assert_allclose(spin.mo_energy[0, :, 1:], ref.mo_energy[:, 1:], atol=1e-8)
    np.testing.assert_allclose(spin.sigmaI[1], ref.sigmaI, atol=1e-10)
    np.testing.assert_allclose(spin.mo_energy[1, :, 1:], ref.mo_energy[:, 1:], atol=1e-8)


def test_outcore_self_energy(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC)
    out = run_small_gw(h2_hf, krgw_ac.KRGWAC, outcore=True)
    np.testing.assert_allclose(out.mo_energy, ref.mo_energy, atol=1e-8)


def test_finite_size_corrected_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC, fc=True)
    out = run_small_gw(h2_hf, kugw_ac.KUGWAC, unrestricted=True, fc=True)
    np.testing.assert_allclose(out.mo_energy[0], ref.mo_energy, atol=2e-7)
    np.testing.assert_allclose(out.mo_energy[1], ref.mo_energy, atol=2e-7)


def test_unrestricted_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC)
    out = run_small_gw(h2_hf, kugw_ac.KUGWAC, unrestricted=True)
    np.testing.assert_allclose(out.mo_energy[0], ref.mo_energy, atol=1e-7)
    np.testing.assert_allclose(out.mo_energy[1], ref.mo_energy, atol=1e-7)


def test_green_function_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC, fullsigma=True)
    out = run_small_gw(h2_hf, kugw_ac.KUGWAC, unrestricted=True, fullsigma=True)
    omega = np.array([-.6, .0, .8])
    gf, gf0, sigma = ref.make_gf(omega, eta=.05)
    ugf, ugf0, usigma = out.make_gf(omega, eta=.05)
    np.testing.assert_allclose(ugf[0], gf, atol=1e-7)
    np.testing.assert_allclose(ugf[1], gf, atol=1e-7)
    np.testing.assert_allclose(ugf0[0], gf0, atol=1e-9)
    np.testing.assert_allclose(usigma[0], sigma, atol=1e-8)
    for iw in range(len(omega)):
        np.testing.assert_allclose(
            (np.linalg.inv(gf0[0, :, :, iw]) - sigma[0, :, :, iw]) @ gf[0, :, :, iw],
            np.eye(2), atol=1e-10)


def test_density_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC, fullsigma=True, rdm=True, nw=80)
    out = run_small_gw(h2_hf, kugw_ac.KUGWAC, unrestricted=True, fullsigma=True, rdm=True, nw=80)
    saved = ref.sigmaI.copy(), out.sigmaI.copy()
    dm = ref.make_rdm1()
    udm = out.make_rdm1()
    np.testing.assert_allclose(udm[0]*2, dm, atol=1e-8)
    np.testing.assert_allclose(udm[1]*2, dm, atol=1e-8)
    ao = ref.make_rdm1(ao_repr=True)
    uao = out.make_rdm1(ao_repr=True)
    c = np.asarray(h2_hf.mo_coeff)
    np.testing.assert_allclose(ao, c @ dm @ c.conj().transpose(0, 2, 1), atol=1e-10)
    np.testing.assert_allclose(uao.sum(axis=0), ao, atol=1e-8)
    np.testing.assert_array_equal(ref.sigmaI, saved[0])
    np.testing.assert_array_equal(out.sigmaI, saved[1])
    assert dm[0, 0, 0] > 1.5
    assert dm[0, 1, 1] < .5


def test_finite_size_grid_closed_shell_limit(h2_hf):
    ref = run_small_gw(h2_hf, krgw_ac.KRGWAC, fc=True, fc_grid=True)
    spin = run_small_gw(h2_hf, kugw_ac.KUGWAC, unrestricted=True, fc=True, fc_grid=True)
    np.testing.assert_allclose(spin.mo_energy[0], ref.mo_energy, atol=2e-7)
    np.testing.assert_allclose(spin.mo_energy[1], ref.mo_energy, atol=2e-7)
