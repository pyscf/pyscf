#!/usr/bin/env python

import numpy as np
import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.kurpa import KURPA


@pytest.fixture(scope="module")
def hydrogen_kuhf():
    cell = gto.Cell()
    cell.build(
        unit="B",
        a=[[0.0, 6.74027466, 6.74027466], [6.74027466, 0.0, 6.74027466], [6.74027466, 6.74027466, 0.0]],
        atom="""H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733""",
        basis="gth-dzvp",
        pseudo="gth-pade",
        verbose=0,
        output="/dev/null",
        charge=0,
        spin=3,
    )

    kpts = cell.make_kpts([3, 1, 1], scaled_center=[0, 0, 0])
    gdf = df.RSDF(cell, kpts)
    gdf.build()

    kmf = scf.KUHF(cell, kpts, exxdiv="ewald")
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12
    kmf.kernel()

    yield kmf

    cell.stdout.close()


def test_kurpa_no_fc(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = False
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04288352903004621, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-1.584806462873674, abs=1e-6)


def test_kurpa_no_fc_outcore(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = False
    rpa.outcore = True
    rpa.segsize = 3
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04288352903004621, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-1.584806462873674, abs=1e-6)


def test_kurpa_acfd_exx_high_cost(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = False
    rpa.acfd_exx = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.042883522669012034, abs=1e-6)
    assert rpa.e_tot == pytest.approx(-1.5848064557082748, abs=1e-6)


def test_kurpa_with_fc(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = True
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04295466718074476, abs=1e-6)


def test_kurpa_with_fc_outcore(hydrogen_kuhf):
    rpa = KURPA(hydrogen_kuhf)
    rpa.fc = True
    rpa.outcore = True
    rpa.segsize = 3
    rpa.kernel()

    assert rpa.e_corr == pytest.approx(-0.04295466718074476, abs=1e-6)


def test_kurpa_get_idx_metal():
    from pyscf.pbc.gw.kurpa import get_idx_metal
    cases = [
        ([1.0, 0.75, 0.25, 0.0], ([0], [1, 2], [3])),
        ([0.9, 0.35, 0.0], ([], [0, 1], [2])),
        ([1.0, 0.6, 0.05], ([0], [1, 2], [])),
        ([0.9, 0.5, 0.1], ([], [0, 1, 2], [])),
    ]
    for mo_occ, expected in cases:
        result = tuple(list(idx) for idx in get_idx_metal(np.asarray(mo_occ)))
        assert result == expected


def test_kurpa_get_rho_response_metal_all_fractional():
    from pyscf.pbc.gw.kurpa import get_rho_response_metal
    omega = 0.7
    mo_energy = np.array([[[-1.0, -0.2, 0.8]], [[-0.9, 0.1, 1.0]]])
    mo_occ = np.array([[[0.9, 0.5, 0.1]], [[0.8, 0.4, 0.2]]])
    Lpq = np.arange(36).reshape(1, 2, 2, 3, 3).astype(np.complex128) / 20

    expected = np.zeros((2, 2), dtype=np.complex128)
    for spin in range(2):
        eia = mo_energy[spin, 0, :, None] - mo_energy[spin, 0, None, :]
        fia = mo_occ[spin, 0, :, None] - mo_occ[spin, 0, None, :]
        weight = eia * fia / (omega**2 + eia**2)
        expected += np.einsum(
            "Pia,ia,Qia->PQ", Lpq[0, spin], weight, Lpq[0, spin].conj()
        )

    result = get_rho_response_metal(omega, mo_energy, mo_occ, Lpq, [0])
    np.testing.assert_allclose(result, expected)


@pytest.fixture(scope="module")
def water_kuhf():
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

    kmf = scf.KUHF(cell, kpts).rs_density_fit()
    kmf.with_df = gdf
    kmf.conv_tol = 1e-12

    yield kmf

    cell.stdout.close()


def test_kurpa_exx_with_frozen(water_kuhf):
    ''' Check that HF exchange energy calculated inside KURPA agrees with that from
        `mf.get_jk` for both non-smeared and smeared cases and with or without frozen.

        NOTE: KURPA currently ignores the `frozen` attribute, so this test does not
        test frozen-orbital behavior at the moment. Any future implementation of
        frozen-orbital support in KURPA should ensure that this test continues to pass.
    '''
    kmf = water_kuhf

    for sigma_ev in [0., 1.]:
        if sigma_ev > 1e-4:
            scf.addons.smearing_(kmf, sigma=sigma_ev/27.211399, method='fermi')

        kmf.kernel()

        from pyscf.pbc.gw.kurpa import get_rpa_exx
        rpa = KURPA(kmf, frozen=0)
        mf = rpa._scf
        dm = mf.make_rdm1()
        vk = mf.get_k(dm_kpts=dm)
        e_x_ref = np.einsum('skij,skji->', vk, dm).real * -0.5 / len(mf.kpts)
        e_x = get_rpa_exx(rpa)

        assert e_x == pytest.approx(e_x_ref, abs=1e-6)

        rpa = KURPA(kmf, frozen=2)
        e_x = get_rpa_exx(rpa)

        assert e_x == pytest.approx(e_x_ref, abs=1e-6)
