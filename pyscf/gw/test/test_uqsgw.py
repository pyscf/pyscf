#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.uqsgw import UQSGW


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


def test_uqsgw(hf_cation_pbe0):
    gw = UQSGW(hf_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.11080314, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.10047463, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-1.06570687, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.33427964, abs=1e-4)
