#!/usr/bin/env python

import pytest

from pyscf import dft, gto, lib
from pyscf.gw.uevgw_exact import UEVGWExact


@pytest.fixture
def h2o_cation_pbe0():
    # Pin threads to keep the reference energies reproducible across BLAS builds.
    current_nthreads = lib.num_threads()
    lib.num_threads(1)

    try:
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
        yield mf
    finally:
        lib.num_threads(current_nthreads)


def test_uevgw_exact(h2o_cation_pbe0):
    gw = UEVGWExact(h2o_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.02685505, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15525548, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99402032, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.42542699, abs=1e-4)


def test_uevgw_exact_w0(h2o_cation_pbe0):
    gw = UEVGWExact(h2o_cation_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.W0 = True
    gw.kernel()

    assert gw.mo_energy[0][4] == pytest.approx(-1.02685505, abs=1e-4)
    assert gw.mo_energy[0][5] == pytest.approx(-0.15525548, abs=1e-4)
    assert gw.mo_energy[1][3] == pytest.approx(-0.99402032, abs=1e-4)
    assert gw.mo_energy[1][4] == pytest.approx(-0.42542699, abs=1e-4)
