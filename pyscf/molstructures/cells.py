import logging
import numpy as np

import pyscf
import pyscf.pbc

if __name__ == "__main__":
    from util import *
else:
    from .util import *

__all__ = [
        "build_H2O_16LiH",
        ]

BOHR2ANG = 0.529177210903
ANG2BOHR = 1.0/BOHR2ANG

def build_H2O_16LiH(distance, vacuum=30.0, layers=2, **kwargs):

    #a_matrix = BOHR2ANG = np.asarray([
    #[5.7756481171, 0.0000000000, 0.0000000000],
    #[0.0000000000, 5.7756481171, 0.0000000000],
    #[0.0000000000, 0.0000000000, 20.4200000763]])

    if layers not in (1, 2):
        raise ValueError()

    a_matrix = np.asarray([
        [8.168000000, 0.000000000, 0.000000000],
        [0.000000000, 8.168000000, 0.000000000],
        [0.000000000, 0.000000000, vacuum]])

    atoms, coords = load_datafile("H2O-16LiH.dat")

    if layers == 1:
        mask = coords[:,2] > 0.0
        atoms = atoms[mask]
        coords = coords[mask]

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]

    #print_distances(atom, "Li1")
    print_distances(atom, "Li1", a_matrix)

    # Move water substrate
    deq = 2.1531
    for i in range(3):
        coords[i][2] += (distance - deq)

    # Shift first layer into center of vacuum
    coords[:,2] += (vacuum/2 - 2.042)

    #print_distances(atom, "Li1")

    cell = pyscf.pbc.gto.M(atom=atom, a=a_matrix, **kwargs)
    #cell.build()

    return cell

if __name__ == "__main__":

    #build_H2O_16LiH(3.0)
    cell = build_H2O_16LiH(2.0, layers=2)
    #visualize_atoms(cell.atom)
