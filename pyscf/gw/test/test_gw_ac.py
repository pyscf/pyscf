#!/usr/bin/env python

import pytest
from pyscf import gto, scf, dft
from pyscf.gw.gw_ac import GWAC

@pytest.fixture
def h2o_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
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
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

def test_gwac_pade_incore_full(h2o_pbe0):
    # full self-energy, incore
    mf = h2o_pbe0
    gw = GWAC(mf)
    gw.fullsigma = True
    gw.orbs=range(4, 6)
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

def test_gwac_pade_outcore_diag(h2o_pbe0):
    mf = h2o_pbe0
    # diag self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.outcore = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

def test_gwac_pade_outcore_full(h2o_pbe0):
    mf = h2o_pbe0
    # full self-energy, outcore
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.fullsigma = True
    gw.outcore = True
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42657296) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16495549) < 1e-5

def test_gwac_pade_frozen_core(h2o_pbe0):
    mf = h2o_pbe0
    # frozen core
    gw = GWAC(mf)
    gw.orbs = range(4, 6)
    gw.frozen = 1
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.42667346) < 1e-5
    assert abs(gw.mo_energy[5] - 0.16490656) < 1e-5

def test_gwac_pade_frozen_list(h2o_pbe0):
    mf = h2o_pbe0
    # frozen list
    gw = GWAC(mf)
    gw.orbs = [4, 7]
    gw.frozen = [0, 5]
    gw.kernel()
    assert abs(gw.mo_energy[4] - -0.43309464) < 1e-5
    assert abs(gw.mo_energy[7] - 0.73675504) < 1e-5
