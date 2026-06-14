#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.evgw_exact import EVGWExact


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


def test_evgw_exact(h2o_pbe0):
    gw = EVGWExact(h2o_pbe0)
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.44293498, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16822268, abs=1e-4)


def test_evgw_exact_w0(h2o_pbe0):
    gw = EVGWExact(h2o_pbe0)
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.43324186, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16626025, abs=1e-4)
