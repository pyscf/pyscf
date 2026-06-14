#!/usr/bin/env python

import pytest

from pyscf import dft, gto
from pyscf.gw.qsgw_exact import QSGWExact


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


def test_qsgw_exact(h2o_pbe0):
    gw = QSGWExact(h2o_pbe0)
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.45766166, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16255341, abs=1e-4)
