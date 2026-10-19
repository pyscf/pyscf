#!/usr/bin/env python

import numpy as np
import pytest

from pyscf.pbc import df, dft, gto, scf, tools
from pyscf.pbc.gw import gw_ac, krgw_ac
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


def test_gamma_and_kpoint_gw():
    cell = gto.M(atom='H 0 0 0; H 0 0 .74', a=np.eye(3)*4,
                 basis='gth-szv', pseudo='gth-pade', mesh=[11]*3,
                 precision=1e-7, verbose=0)
    mf = scf.RHF(cell, exxdiv=None).density_fit(auxbasis='weigend').run(conv_tol=1e-11)
    ref = krgw_ac.KRGWAC(scf.addons.convert_to_kscf(mf))
    actual = gw_ac.GWAC(mf)
    for obj in (ref, actual):
        obj.fc = False
        obj.nw = 32
        obj.ac_pade_npts = 8
        obj.kernel()
    np.testing.assert_allclose(actual.mo_energy, ref.mo_energy[0], atol=1e-8)
    assert actual.loop_ao2mo(ijslice=(0, 0, 0, 1)).shape == (actual.with_df.get_naoaux(), 0, 1)
    incore = actual.mo_energy.copy()
    out = gw_ac.GWAC(mf)
    out.nw = 32
    out.ac_pade_npts = 8
    out.outcore = True
    out.segsize = 1
    out.kernel()
    np.testing.assert_allclose(out.mo_energy, incore, atol=1e-8)
    np.testing.assert_allclose(actual.loop_ao2mo(ijslice=(0, 1, 0, 2)),
                               actual.ao2mo()[:, :1, :], atol=1e-12)
