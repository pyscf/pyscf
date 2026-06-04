#!/usr/bin/env python

import numpy as np
import pytest

from pyscf import dft, gto
from pyscf.gw.uqsgw_exact import UQSGWExact


@pytest.fixture
def h2o_cation_pbe0():
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
    mol.basis = 'def2-svp'
    mol.charge = 1
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()
    return mf


def test_uqsgw_exact(h2o_cation_pbe0):
    mf = h2o_cation_pbe0
    gw = UQSGWExact(mf)
    gw.kernel()

    assert gw.mo_energy.shape == mf.mo_energy.shape
    assert np.all(np.isfinite(gw.mo_energy[:, 3:6]))
    assert not np.allclose(gw.mo_energy[:, 3:6], mf.mo_energy[:, 3:6])
