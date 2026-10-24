#!/usr/bin/env python

import os
import tempfile
import numpy as np
import pytest

from pyscf import dft, gto, scf
from pyscf.gw.qsgw import QSGW
from pyscf.gw.uqsgw import UQSGW


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


def test_qsgw(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439850, abs=1e-4)


def test_qsgw_low_nw2(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.nw2 = 30
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439851, abs=1e-4)


def test_qsgw_outcore(hf_pbe0):
    gw = QSGW(hf_pbe0)
    gw.verbose = 0
    gw.max_cycle = 1
    gw.outcore = True
    gw.kernel()

    assert gw.mo_energy[4] == pytest.approx(-0.38781745, abs=1e-4)
    assert gw.mo_energy[5] == pytest.approx(0.50439852, abs=1e-4)


def _check_restart(mode):
    mol = gto.M(atom='He 0 0 0; H 0 0 1', charge=1, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    energies = []
    for cls, reference in ((QSGW, mf), (UQSGW, mf.to_uhf())):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, 'checkpoint.h5')
            obj = cls(reference)
            obj.mode = mode
            obj.nw = 32
            obj.ac_pade_npts = 8
            obj.max_cycle = 50
            obj.conv_tol = 1e-9
            obj.chkfile = checkpoint
            obj.kernel()
            saved = obj.mo_energy.copy()
            restart = cls(reference)
            restart.mode = mode
            restart.nw = 32
            restart.ac_pade_npts = 8
            restart.max_cycle = 50
            restart.conv_tol = 1e-9
            restart.chkfile = checkpoint
            restart.load_chkfile = True
            restart.outcore = True
            restart.kernel()
            np.testing.assert_allclose(restart.mo_energy, saved, atol=1e-8)
            energies.append(saved)
    np.testing.assert_allclose(energies[1], np.array([energies[0]]*2), atol=1e-8)


def test_restart_mode_a():
    _check_restart('a')


def test_restart_mode_c():
    _check_restart('c')
