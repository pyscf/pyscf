#!/usr/bin/env python

import pytest

from pyscf import dft, gto, scf
from pyscf.gw.gw_exact_df import GWExactDF


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
