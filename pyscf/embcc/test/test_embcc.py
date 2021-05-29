# TODO
# Make this a unittest module

from timeit import default_timer as timer
import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.pbc
import pyscf.pbc.cc
import pyscf.pbc.tools

from pyscf.pbc.df.df_incore import IncoreGDF

import pyscf.embcc
import pyscf.embcc.k2gamma_gdf


def test_dimer(atoms=['H', 'H'], d=1.3, basis='cc-pvdz', bno_threshold=-1):

    t0 = timer()
    mol = pyscf.gto.Mole(atom='%s 0 0 0 ; %s 0 0 %f' % (*atoms, d), basis=basis)
    mol.build()
    mf = pyscf.scf.RHF(mol)
    #mf = mf.density_fit()
    mf.kernel()
    t_hf = timer()-t0

    t0 = timer()
    ecc = pyscf.embcc.EmbCC(mf, bno_threshold=bno_threshold)
    if atoms[0] == atoms[1]:
        ecc.make_atom_cluster(0, symmetry_factor=2)
    else:
        ecc.make_atom_cluster(0)
        ecc.make_atom_cluster(1)
    ecc.kernel()
    t_ecc = timer()-t0
    print("E(Emb-CCSD)= %+16.8f Ha" % ecc.e_tot)

    print("T(HF)= %.2f s  T(Emb-CCSD)= %.2f s" % (t_hf, t_ecc))

    if bno_threshold <= 0:
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
        assert np.allclose(cc.e_tot, ecc.e_tot)

def make_cubic(a, atom="He", basis="gth-dzv", supercell=False):
    amat = a * np.eye(3)
    atom = "%s %f %f %f" % (atom, a/2, a/2, a/2)
    #atom = "%s %f %f %f ; %s %f %f %f" % (atom, 0, 0, 0, atom, a/2, a/2, a/2)
    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = basis
    cell.pseudo = "gth-pade"
    #cell.precision = 1e-5
    cell.verbose = 10
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def make_tetragonal(a, c, atoms=["H", "H"], basis="gth-dzv", supercell=False, output=None):
    amat = a * np.eye(3)
    amat[2,2] = c
    atom = "%s %f %f %f ; %s %f %f %f" % (atoms[0], 0, 0, 0, atoms[1], a/2, a/2, c/2)
    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = basis
    cell.pseudo = "gth-pade"
    #cell.precision = 1e-5
    cell.verbose = 10
    if output is not None:
        cell.output = output
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def make_diamond_cell(a, atoms=['C', 'C'], basis='gth-dzv', supercell=False):
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.verbose = 10
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def make_perovskite(a, atoms=["Sr", "Ti", "O"], basis="gth-dzvp-molopt-sr", supercell=False):
    amat = a * np.eye(3)
    coords = np.asarray([
                [0,     0,      0],
                [a/2,   a/2,    a/2],
                [0,     a/2,    a/2],
                [a/2,   0,      a/2],
                [a/2,   a/2,    0]
                ])
    atom = [
        (atoms[0], coords[0]),
        (atoms[1], coords[1]),
        (atoms[2], coords[2]),
        (atoms[2], coords[3]),
        (atoms[2], coords[4]),
        ]

    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.verbose = 10
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)
    return cell

def test_fci_solver(a=2.0, kmesh=[2,2,2], bno_threshold=1e-4):

    t0 = timer()
    cell = make_cubic(a, "He")
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    #kmf.with_df.linear_dep_threshold = 1e-7
    #kmf.with_df.linear_dep_method = 'canonical-orth'
    #kmf.with_df.linear_dep_always = True
    kmf.kernel()
    t_hf = timer()-t0

    t0 = timer()
    ecc = pyscf.embcc.EmbCC(kmf, solver='FCI', bno_threshold=bno_threshold)
    ecc.make_atom_cluster(0)
    ecc.kernel()
    t_ecc = timer()-t0
    print("E(Emb-FCI)= %+16.8f Ha" % ecc.e_tot)

    print("T(HF)= %.2f s  T(Emb-FCI)= %.2f s" % (t_hf, t_ecc))

    kcc = pyscf.pbc.cc.KCCSD(kmf)
    kcc.kernel()
    print("E(k-CCSD)=   %+16.8f Ha" % kcc.e_tot)


def test_cubic(atom='He', a=2.0, basis='gth-dzv', kmesh=[2,2,2], bno_threshold=1e-5):

    t0 = timer()
    cell = make_cubic(a, atom, basis=basis)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    #kmf.with_df.linear_dep_threshold = 1e-7
    #kmf.with_df.linear_dep_method = 'canonical-orth'
    #kmf.with_df.linear_dep_always = True
    kmf.kernel()
    t_hf = timer()-t0

    t0 = timer()
    ecc = pyscf.embcc.EmbCC(kmf, bno_threshold=bno_threshold)
    ecc.make_atom_cluster(0)
    ecc.kernel()
    t_ecc = timer()-t0
    print("E(Emb-CCSD)= %+16.8f Ha" % ecc.e_tot)

    print("T(HF)= %.2f s  T(Emb-CCSD)= %.2f s" % (t_hf, t_ecc))

    if bno_threshold <= 0:
        kcc = pyscf.pbc.cc.KCCSD(kmf)
        kcc.kernel()
        print("E(k-CCSD)=   %+16.8f Ha" % kcc.e_tot)
        #assert np.allclose(kcc.e_tot, ecc.e_tot)

    #if bno_threshold <= 0:
    #    cc = pyscf.cc.CCSD(kmf)
    #    cc.kernel()
    #    print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
    #    #assert np.allclose(kcc.e_tot, ecc.e_tot)


def test_perovskite(a=3.9, kmesh=[1,1,2], bno_threshold=1e-6):

    t0 = timer()
    cell = make_perovskite(a, ["He", "C", "O"], "gth-dzv")
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    #kmf.with_df.linear_dep_threshold = 1e-7
    #kmf.with_df.linear_dep_method = 'canonical-orth'
    #kmf.with_df.linear_dep_always = True
    kmf.kernel()
    t_hf = timer()-t0

    t0 = timer()
    ecc = pyscf.embcc.EmbCC(kmf, bno_threshold=bno_threshold)
    ecc.make_atom_cluster(0)
    ecc.make_atom_cluster(1)
    ecc.make_atom_cluster(2, symmetry_factor=3)
    ecc.kernel()
    t_ecc = timer()-t0
    print("E(Emb-CCSD)= %+16.8f Ha" % ecc.e_tot)

    print("T(HF)= %.2f s  T(Emb-CCSD)= %.2f s" % (t_hf, t_ecc))


def test_canonical_orth(c=1.2, lindep_threshold=1e-8, kmesh=[1,1,2], output=None):

    a = 4.0
    basis = "gth-aug-dzvp"
    cell = make_tetragonal(a, c, basis=basis, output=output)
    kpts = cell.make_kpts(kmesh)

    # Canonical Orth
    cell.lindep_threshold = lindep_threshold
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    try:
        kmf.kernel()
    except Exception as e:
        print("Exception: %r" % e)
        return np.nan, np.nan

    # Canonical CCSD
    kcc = pyscf.pbc.cc.KCCSD(kmf)
    try:
        kcc.kernel()
    except Exception as e:
        print("Exception: %r" % e)
        return kmf.e_tot, np.nan

    print("CCSD E= %16.8g" % kcc.e_tot)

    #kcc = pyscf.embcc.EmbCC(kmf, bno_threshold=-1)
    #kcc.make_atom_cluster(0)
    #kcc.kernel()
    #print("Emb-CCSD E= %16.8g" % kcc.e_tot)

    return kmf.e_tot, kcc.e_tot

def sample_canonical_orth():
    #cs = [4.0, 3.0, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
    cs = [2.0, 1.9, 1.8, 1.7, 1.6]
    threshs = [None, 1e-10, 1e-8, 1e-6, 1e-4]

    for i, thresh in enumerate(threshs):
        e_hfs = []
        e_ccs = []
        for j, c in enumerate(cs):
            out = "t-%r-c-%.1f.txt" % (thresh, c)
            e_hf, e_cc = test_canonical_orth(c, thresh, output=out)
            e_hfs.append(e_hf)
            e_ccs.append(e_cc)

        data = np.stack((cs, e_hfs, e_ccs), axis=1)
        np.savetxt(("threshold-%r.txt" % thresh), data)

def test_diamond(a=3.56, basis='gth-dzv', kmesh=[2, 2, 2], bno_threshold=1e-6):

    ncells = np.product(kmesh)

    # k-point calculation
    cell = make_diamond_cell(a, basis=basis)
    cell.lindep_threshold = 1e-7
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    kcc = pyscf.embcc.EmbCC(kmf, bno_threshold=bno_threshold)
    kcc.make_atom_cluster(0, symmetry_factor=2)
    t0 = timer()
    kcc.kernel()
    print("E(EmbCC)= %16.8f Ha" % kcc.e_tot)
    print("T(EmbCC)= %.2f s" % (timer()-t0))


def test_diamond_bno_threshold(bno_threshold=[1e-3, 1e-4, 1e-5, 1e-6], kmesh=[2, 2, 2]):

    a = 3.5
    ncells = np.product(kmesh)

    # k-point calculation
    cell = make_diamond_cell(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.with_df.linear_dep_method = "regularize"
    kmf.kernel()

    kcc = pyscf.embcc.EmbCC(kmf, bno_threshold=bno_threshold)
    kcc.make_atom_cluster(0, symmetry_factor=2)
    t0 = timer()
    kcc.kernel()
    print("Time for k-EmbCC= %.3f" % (timer()-t0))
    print("N(EmbCC) = %r" % kcc.get_cluster_sizes())
    print("E(EmbCC) = %r" % kcc.get_energies())

    # For [2,2,2], gth-dzv, a=3.5
    if kmesh == [2,2,2]:
        N_EXPECTED = np.array([[77, 52, 39, 14]])
        E_EXPECTED = np.array([-11.16455488, -11.15595256, -11.1383086 , -11.09207628])
        e = kcc.get_energies()
        if not np.allclose(e, E_EXPECTED):
            print("Got:      %r" % e)
            print("Expected: %r" % E_EXPECTED)

            print("Cluster sizes: %r" % kcc.get_cluster_sizes())
            print("Expected:      %r" % N_EXPECTED)
            raise RuntimeError()

        #assert np.allclose(kcc.get_energies(), E_EXPECTED)
        #assert np.all(kcc.get_cluster_sizes() == N_EXPECTED)
    # For [3,3,3], gth-dzv, a=3.5
    elif kmesh == [3,3,3]:
        N_EXPECTED = np.array([[89, 61, 39, 14]])
        E_EXPECTED = np.array([-11.22353507, -11.21390107, -11.19618965, -11.15312301])
        assert np.all(kcc.get_cluster_sizes() == N_EXPECTED)
        assert np.allclose(kcc.get_energies(), E_EXPECTED)

#def test_diamond(EXPECTED=None, kmesh=[2, 2, 2], bath_tol=1e-4, bno_threshold=1e-4):
#
#    a = 3.5
#    ncells = np.product(kmesh)
#
#    # k-point calculation
#    cell = make_diamond(a)
#    kpts = cell.make_kpts(kmesh)
#    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
#    kmf = kmf.density_fit()
#    kmf.kernel()
#
#    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=bath_tol, bno_threshold=bno_threshold)
#    kcc.opts.popfile = None
#    kcc.opts.orbfile = None
#    kcc.make_atom_cluster(0, symmetry_factor=2)
#    t0 = timer()
#    kcc.kernel()
#    print("Time for k-EmbCC= %.3f" % (timer()-t0))
#    print("k-EmbCC E= %16.8g" % kcc.e_tot)
#    if EXPECTED:
#        assert np.isclose(kcc.e_tot, EXPECTED)
#    else:
#        EXPECTED = kcc.e_tot
#
#    # Supercell calculations
#    scell = make_diamond(a, supercell=kmesh)
#    smf = pyscf.pbc.scf.RHF(scell)
#    smf = smf.density_fit()
#    smf.kernel()
#
#    scc = pyscf.embcc.EmbCC(smf, bath_tol=bath_tol, bno_threshold=bno_threshold)
#    scc.opts.popfile = None
#    scc.opts.orbfile = None
#    scc.make_atom_cluster(0, symmetry_factor=2*ncells)
#    t0 = timer()
#    scc.kernel()
#    print("Time for supercell-EmbCC= %.3f" % (timer()-t0))
#    print("SC-CCSD E= %16.8g" % (scc.e_tot/ncells))
#    if EXPECTED:
#        assert np.isclose(scc.e_tot/ncells, EXPECTED)
#
#    # k-point calculation + incore GDF
#    cell = make_diamond(a)
#    kpts = cell.make_kpts(kmesh)
#    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
#    kmf.with_df = IncoreGDF(cell, kpts)
#    kmf.kernel()
#
#    kcc = pyscf.embcc.EmbCC(kmf, bath_tol=bath_tol, bno_threshold=bno_threshold)
#    kcc.opts.popfile = None
#    kcc.opts.orbfile = None
#    kcc.make_atom_cluster(0, symmetry_factor=2)
#    t0 = timer()
#    kcc.kernel()
#    print("Time for k-EmbCC (incore GDF)= %.3f" % (timer()-t0))
#    print("k-EmbCC (incore GDF) E= %16.8g" % kcc.e_tot)
#    if EXPECTED:
#        assert np.isclose(kcc.e_tot, EXPECTED)

def test_full_ccsd_limit(EXPECTED, kmesh=[2, 2, 2]):

    a = 3.5
    ncells = np.product(kmesh)

    # k-point calculation
    cell = make_diamond_cell(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    # canonical CCSD
    ccsd = pyscf.pbc.cc.KCCSD(kmf)
    ccsd.kernel()
    print("Exact CCSD= %16.8g" % ccsd.e_tot)
    assert np.allclose(ccsd.e_tot, EXPECTED)

    # bath_tol=-1 -> complete environment as bath
    #kcc = pyscf.embcc.EmbCC(kmf, bath_tol=-1)
    kcc = pyscf.embcc.EmbCC(kmf, bno_threshold=[-1])
    kcc.opts.popfile = None
    kcc.opts.orbfile = None
    kcc.make_atom_cluster(0, symmetry_factor=2)
    kcc.kernel()
    print("k-EmbCC E= %16.8g" % kcc.e_tot)
    assert np.allclose(ccsd.e_tot, EXPECTED)

def run_test():

    #test_dimer(['Li', 'H'], d=1.4)
    #test_dimer(['Li', 'Li'], d=2.0)
    #test_cubic()
    #test_cubic('C', basis='gth-tzvp')
    test_diamond(basis='gth-tzvp', bno_threshold=1e-8)
    #test_fci_solver()
    #test_perovskite()
    #test_diamond_bno_threshold(kmesh=[2,2,2])
    #test_diamond_bno_threshold(bno_threshold=-1, kmesh=[2,2,2])
    #test_diamond_bno_threshold(kmesh=[4,4,4])
    #test_canonical_orth()
    #sample_canonical_orth()
    #test_full_ccsd_limit(-11.170842)

if __name__ == "__main__":
    run_test()
