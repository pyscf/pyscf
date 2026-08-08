#!/usr/bin/env python

import numpy as np
import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.krpa import KRPA


@pytest.fixture(scope="module")
def diamond_krhf():
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
        pseudo="gth-pbe",
        basis="gth-dzv",
        precision=1e-12,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSGDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_krpa_no_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = False
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.1852772037535004, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.694392044197565, abs=1e-6)


def test_krpa_no_fc_outcore(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.outcore = True
    rpa.segsize = 2
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.1852772037535004, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.694392044197565, abs=1e-6)


def test_krpa_acfd_exx_high_cost(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = False
    rpa.acfd_exx = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.18527720400362488, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.694392045437178, abs=1e-6)


def test_krpa_with_fc(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.20723389722097715, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.716348738655793, abs=1e-6)


def test_krpa_with_fc_outcore(diamond_krhf):
    rpa = KRPA(diamond_krhf)
    rpa.fc = True
    rpa.outcore = True
    rpa.segsize = 2
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.20723389722097715, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-10.716348738655793, abs=1e-6)


def test_krpa_get_idx_metal():
    from pyscf.pbc.gw.krpa import get_idx_metal
    cases = [
        ([2.0, 1.5, 0.5, 0.0], ([0], [1, 2], [3])),
        ([1.9, 0.7, 0.0], ([], [0, 1], [2])),
        ([2.0, 1.2, 0.1], ([0], [1, 2], [])),
        ([1.9, 1.0, 0.1], ([], [0, 1, 2], [])),
    ]
    for mo_occ, expected in cases:
        result = tuple(list(idx) for idx in get_idx_metal(np.asarray(mo_occ)))
        assert result == expected


def test_krpa_get_rho_response_metal_all_fractional():
    from pyscf.pbc.gw.krpa import get_rho_response_metal
    omega = 0.7
    mo_energy = np.array([[-1.0, -0.2, 0.8]])
    mo_occ = np.array([[1.8, 1.0, 0.2]])
    Lpq = [np.arange(18).reshape(2, 3, 3).astype(np.complex128) / 20]

    eia = mo_energy[0, :, None] - mo_energy[0, None, :]
    fia = mo_occ[0, :, None] - mo_occ[0, None, :]
    weight = eia * fia / (omega**2 + eia**2)
    expected = np.einsum("Pia,ia,Qia->PQ", Lpq[0], weight, Lpq[0].conj())

    result = get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, [0])
    np.testing.assert_allclose(result, expected)


def test_krpa_kconserv_shifted_kmesh():
    ''' This test checks if the kconserv table constructed by `get_kconserv_ria_efficient`
        remains invariant to a rigid shift of a given k-mesh.
    '''
    from pyscf.pbc.gw.krpa import get_kconserv_ria_efficient

    cell = gto.Cell()
    cell.build(
        a=np.eye(3) * 3,
        atom="H 0 0 0",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, scaled_center=[0, 0, 0])
    shifted_kpts = cell.make_kpts(
        kmesh, scaled_center=[0.6223 / 2, 0.2953 / 2, 0]
    )

    reference = get_kconserv_ria_efficient(cell, kpts)
    result = get_kconserv_ria_efficient(cell, shifted_kpts)
    np.testing.assert_array_equal(result, reference)


@pytest.fixture(scope="module")
def water_krhf():
    cell = gto.Cell()
    cell.build(
        unit="angstrom",
        atom="""
        O          0.00000        0.00000        0.11779
        H          0.00000        0.75545       -0.47116
        H          0.00000       -0.75545       -0.47116
        """,
        a=np.eye(3) * 5,
        verbose=0,
        output="/dev/null",
        pseudo=None,
        basis="cc-pvdz",
        precision=1e-12,
    )

    kpts = cell.make_kpts([1, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSGDF(cell, kpts)
    gdf.build()

    kmf = scf.KRHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12

    yield kmf

    cell.stdout.close()


def test_krpa_exx_with_frozen(water_krhf):
    ''' Check that HF exchange energy calculated inside KRPA agrees with that from
        `mf.get_jk` for both non-smeared and smeared cases and with or without frozen.
    '''
    kmf = water_krhf

    for sigma_ev in [0., 1.]:
        if sigma_ev > 1e-4:
            scf.addons.smearing_(kmf, sigma=sigma_ev/27.211399, method='fermi')

        kmf.kernel()

        from pyscf.pbc.gw.krpa import get_rpa_exx
        rpa = KRPA(kmf, frozen=0)
        mf = rpa._scf
        dm = mf.make_rdm1()
        vk = mf.get_k(dm_kpts=dm)
        e_x_ref = np.einsum('kij,kji->', vk, dm).real * -0.25 / len(mf.kpts)
        e_x = get_rpa_exx(rpa)

        assert e_x == pytest.approx(e_x_ref, abs=1e-6)

        rpa = KRPA(kmf, frozen=2)
        e_x = get_rpa_exx(rpa)

        assert e_x == pytest.approx(e_x_ref, abs=1e-6)
