#!/usr/bin/env python

import pytest

from pyscf.pbc import df, dft, gto, tools
from pyscf.pbc.gw.gw_ac import GWAC


@pytest.fixture(scope="module")
def diamond_supercell_pbe():
    ucell = gto.Cell()
    ucell.build(
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
        precision=1e-12,
    )

    cell = tools.super_cell(ucell, [3, 1, 1])
    cell.verbose = 0
    cell.output = "/dev/null"

    gdf = df.RSDF(cell)
    gdf.build()

    mf = dft.RKS(cell).rs_density_fit()
    mf.xc = "pbe"
    mf.exxdiv = None
    mf.with_df = gdf
    mf.conv_tol = 1e-12
    mf.kernel()

    yield mf


def test_gwac_pade_diamond_supercell_high_cost(diamond_supercell_pbe):
    gw = GWAC(diamond_supercell_pbe)
    gw.kernel()

    assert gw.mo_energy[5] == pytest.approx(0.52637379, abs=1e-4)
    assert gw.mo_energy[10] == pytest.approx(0.62044176, abs=1e-4)
    assert gw.mo_energy[12] == pytest.approx(0.96572544, abs=1e-4)
    assert gw.mo_energy[15] == pytest.approx(1.0751724, abs=1e-4)
