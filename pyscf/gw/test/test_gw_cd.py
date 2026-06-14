#!/usr/bin/env python

import pytest
from pyscf import dft, gto
from pyscf.gw.gw_cd import GWCD


@pytest.fixture
def h2o_pbe():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()
    return mf


def test_gwcd(h2o_pbe):
    mf = h2o_pbe
    nocc = mf.mol.nelectron // 2

    gw = GWCD(mf)
    gw.orbs = range(0, nocc + 3)
    gw.kernel()

    assert gw.mo_energy[nocc - 1] == pytest.approx(-0.41284735, abs=1e-5)
    assert gw.mo_energy[nocc] == pytest.approx(0.16574524, abs=1e-5)
    assert gw.mo_energy[0] == pytest.approx(-19.53387986, abs=1e-5)
