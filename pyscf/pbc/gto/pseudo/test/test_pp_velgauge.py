import pathlib

import numpy as np
from pyscf.pbc import gto as pgto
from pyscf.pbc.gto.pseudo.ppnl_velgauge import get_gth_pp_nl_velgauge, get_gth_pp_nl_velgauge_commutator


# Reference values obtained from https://github.com/pyscf/pyscf-forge/pull/136
# (K. Hanasaki),
# functions pyscf.rttddft.gto_ps_pp_int01.get_pp_nl01,
#           pyscf.rttddft.gto_ps_pp_int01.get_pp_nl01xx

def test_pp_velgauge1():
    cell = pgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell.a = np.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [20]*3
    cell.build()

    A_over_c = np.array([1.0,0.3,0.2])
    kpts = np.array([[0.0,0.0,0.0]])
    fspath = pathlib.Path(__file__).parent / "ppnl_refvals1.txt"
    ref_vals = np.loadtxt(fspath, dtype=np.complex128)

    new_vals = get_gth_pp_nl_velgauge(cell, q=A_over_c, kpts=kpts)

    assert np.allclose(ref_vals, new_vals)

def test_pp_velgauge2():
    cell = pgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; S 0 2 2; C 2 0 2'
    cell.a = np.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [20]*3
    cell.build()

    A_over_c = np.array([1.0,0.3,0.2])
    kpts = np.array([[0.0,0.0,0.0]])
    fspath = pathlib.Path(__file__).parent / "ppnl_refvals2.txt"
    ref_vals = np.loadtxt(fspath, dtype=np.complex128)

    new_vals = get_gth_pp_nl_velgauge(cell, q=A_over_c, kpts=kpts)

    assert np.allclose(ref_vals, new_vals)


def test_pp_velgauge3():
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 2.5
    cell.mesh = [30] * 3
    cell.build()

    A_over_c = np.array([3.0,-0.3,0.1])
    kpts = np.array([[-0.1,0.0,0.3]])

    fspath = pathlib.Path(__file__).parent / "ppnl_refvals3.txt"
    ref_vals = np.loadtxt(fspath, dtype=np.complex128)

    new_vals = get_gth_pp_nl_velgauge(cell, q=A_over_c, kpts=kpts)
    new_vals = new_vals[0]

    print(new_vals)

    assert new_vals.shape == ref_vals.shape
    assert np.allclose(ref_vals, new_vals)


def test_ppxx_velgauge1():
    cell = pgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell.a = np.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [20]*3
    cell.build()

    A_over_c = np.array([3.0,-0.3,0.1])
    kpts = np.array([[0, 0, 0]])
    fspath = pathlib.Path(__file__).parent / "ppxx_refvals1.npy"
    ref_vals = np.load(fspath).transpose(1,0,2,3)

    new_vals = get_gth_pp_nl_velgauge_commutator(cell, q=A_over_c, kpts=kpts)

    assert np.allclose(ref_vals, new_vals)

def test_ppxx_velgauge2():
    cell = pgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell.a = np.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [20]*3
    cell.build()

    A_over_c = np.array([3.0,-0.3,0.1])
    kpts = np.array([[-0.1,0.0,0.3]])
    fspath = pathlib.Path(__file__).parent / "ppxx_refvals2.npy"
    ref_vals = np.load(fspath).transpose(1,0,2,3)

    new_vals = get_gth_pp_nl_velgauge_commutator(cell, q=A_over_c, kpts=kpts)

    assert np.allclose(ref_vals, new_vals)

