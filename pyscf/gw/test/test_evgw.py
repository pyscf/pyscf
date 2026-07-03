#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.evgw import EVGW


@pytest.fixture
def hf_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.basis = 'sto-3g'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_evgw(hf_pbe0):
    gw = EVGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.37042505, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.49661519, abs=1e-4)


def test_evgw0(hf_pbe0):
    gw = EVGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.37042505, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.49661519, abs=1e-4)
