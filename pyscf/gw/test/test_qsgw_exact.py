#!/usr/bin/env python

import pytest

from pyscf import dft, gto, lib
from pyscf.gw.qsgw_exact import QSGWExact


@pytest.fixture
def h2o_pbe0():
    # Pin threads to keep the reference energies reproducible across BLAS builds.
    current_nthreads = lib.num_threads()
    lib.num_threads(1)

    try:
        mol = gto.Mole()
        mol.verbose = 5
        mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.0, -0.7571, 0.5861)], [1, (0.0, 0.7571, 0.5861)]]
        mol.basis = 'def2-svp'
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.kernel()
        yield mf
    finally:
        lib.num_threads(current_nthreads)


def test_qsgw_exact(h2o_pbe0):
    gw = QSGWExact(h2o_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.43819477, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.16301207, abs=1e-4)
