#!/usr/bin/env python

import pytest

from pyscf.pbc import df, gto, scf
from pyscf.pbc.gw.kugw_ac import KUGWAC


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


def test_kugwac_pade_no_fc(hydrogen_kuhf):
    gw = KUGWAC(hydrogen_kuhf)
    gw.qpe_linearized = False
    gw.fc = False
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 5))

    assert gw.mo_energy[0][0][1] == pytest.approx(-0.28661016, abs=1e-5)
    assert gw.mo_energy[0][0][2] == pytest.approx(0.13952572, abs=1e-5)
    assert gw.mo_energy[1][1][0] == pytest.approx(-0.34174199, abs=1e-5)
    assert gw.mo_energy[1][1][1] == pytest.approx(0.08296260, abs=1e-5)


def test_kugwac_pade_with_fc(hydrogen_kuhf):
    gw = KUGWAC(hydrogen_kuhf)
    gw.qpe_linearized = False
    gw.fc = True
    gw.kernel(kptlist=[0, 1, 2], orbs=range(0, 5))

    assert gw.mo_energy[0][0][1] == pytest.approx(-0.48063839, abs=1e-5)
    assert gw.mo_energy[0][0][2] == pytest.approx(0.13870787, abs=1e-5)
    assert gw.mo_energy[1][1][0] == pytest.approx(-0.53502818, abs=1e-5)
    assert gw.mo_energy[1][1][1] == pytest.approx(0.08214267, abs=1e-5)


def test_kugwac_pade_with_fc_frozen_orbitals(hydrogen_kuhf):
    gw = KUGWAC(hydrogen_kuhf)
    gw.qpe_linearized = False
    gw.fc = True
    gw.frozen = [12, 13, 14]
    gw.kernel()

    assert gw.mo_energy[0][0][1] == pytest.approx(-0.47649992, abs=1e-5)
    assert gw.mo_energy[0][0][2] == pytest.approx(0.14513332, abs=1e-5)
