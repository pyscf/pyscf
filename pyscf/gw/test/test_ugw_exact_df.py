#!/usr/bin/env python

import numpy as np
import pytest

from pyscf import dft, gto, scf
from pyscf.gw.ugw_exact_df import UGWExactDF


@pytest.fixture
def h2o_cation_mol():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.charge = 1
    mol.spin = 1
    mol.basis = 'def2-svp'
    mol.build()
    return mol


def test_ugw_exact_df_pbe0(h2o_cation_mol):
    mf = dft.UKS(h2o_cation_mol)
    mf.xc = 'pbe0'
    mf.kernel()

    gw = UGWExactDF(mf)
    gw.eta = 1.0e-5
    gw.qpe_linearized = False
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.02679348, abs=1e-5)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15525785, abs=1e-5)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99401046, abs=1e-5)
    assert gw.mo_energy[1][4] == pytest.approx(-0.42543723, abs=1e-5)

    e_tot, e_hf, e_c = gw.energy_tot()
    assert e_c == pytest.approx(-0.51678119, abs=1e-7)


def test_ugw_exact_df_rpae_smoke(h2o_cation_mol):
    mf = scf.UHF(h2o_cation_mol)
    mf.kernel()

    gw = UGWExactDF(mf)
    gw.eta = 1.0e-5
    gw.qpe_linearized = False
    gw.RPAE = True
    gw.kernel()

    assert gw.mo_energy.shape == mf.mo_energy.shape
    assert np.all(np.isfinite(gw.mo_energy[:, 3:5]))
