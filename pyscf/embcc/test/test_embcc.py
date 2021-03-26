# TODO
# Make this a unittest module

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.tools

import pyscf.embcc

def make_diamond(a, atoms=["C1", "C2"], supercell=False):
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]

    cell = pyscf.pbc.gto.Cell()
    cell.a = amat
    cell.atom = atom
    cell.basis = "gth-dzv"
    cell.pseudo = "gth-pade"
    cell.precision = 1e-5
    cell.exp_to_discard = 0.1
    cell.verbose = 10
    cell.build()
    if supercell:
        cell = pyscf.pbc.tools.super_cell(cell, supercell)

    return cell


def run_test():

    a = 3.5
    kmesh = [2, 2, 2]
    cell = make_diamond(a)
    kpts = cell.make_kpts(kmesh)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    # Unfold
    kmf = pyscf.embcc.k2gamma_gdf.k2gamma_gdf(kmf, kmesh)

    kcc = pyscf.embcc.EmbCC(kmf, minao="gth-szv", bath_tol=1e-4)
    kcc.make_atom_cluster(0, symmetry_factor=2)
    print("K-CCSD E= %16.8g" % kcc.e_tot)


    scell = make_diamond(a, kmesh)
    smf = pyscf.pbc.scf.RHF(scell)
    smf = smf.density_fit()
    smf.kernel()

    scc = pyscf.embcc.EmbCC(smf, minao="gth-szv", bath_tol=1e-4)
    scc.make_atom_cluster(0, symmetry_factor=2)
    print("SC-CCSD E= %16.8g" % scc.e_tot)

if __name__ == "__main__":
    run_test()
