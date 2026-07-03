#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.uevgw import UEVGW


@pytest.fixture
def hf_cation_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.basis = 'sto-3g'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_uevgw(hf_cation_pbe0):
    gw = UEVGW(hf_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.08974359, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.10536617, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.04968434, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.36255845, abs=1e-4)


def test_uevgw0(hf_cation_pbe0):
    gw = UEVGW(hf_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.08974359, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.10536617, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.04968434, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.36255845, abs=1e-4)
