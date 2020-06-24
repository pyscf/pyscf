"""Module to setup unit cell of 2D hexagonal boron nitride."""

import logging
import copy

import numpy as np

try:
    from .general import *
# If scipts are ran from within package
except (ImportError, SystemError):
    from general import *

__all__ = [
        "make",
        "make_supercell",
        "add_substrate",
        "get_shell_indices",
        "add_atom_indices",
        ]

log = logging.getLogger(__name__)

def make(a, c, atoms, supercell=(1, 1, 1), distance=np.inf, add_shell_numbers=True, shift_z=True):

    # unit cell
    amat_uc, atoms_uc = make_unit_cell(a, c, atoms)

    # supercell
    a_matrix, atoms = make_supercell(amat_uc, atoms_uc, supercell=supercell)

    # Find most central nitrogen
    n_atom = get_center_atom(a_matrix, atoms, atomtype="N")

    if add_shell_numbers:
        shell_indices = get_shell_indices(a_matrix, a, atoms, n_atom)
        atoms = add_atom_indices(atoms, shell_indices)

    if distance != np.inf:
        atoms = add_substrate(atoms, distance=distance, n_atom_idx=n_atom)

        if shift_z:
            maxz = np.amax([a[1][-1] for a in atoms])
            minz = np.amin([a[1][-1] for a in atoms])
            midz = (maxz+minz)/2.0
            dz = midz - c/2.0
            log.debug("maxz=%f minz=%f midz=%f dz=%f", maxz, minz, midz, dz)
            for atom in atoms:
                atom[1][-1] += dz

    return a_matrix, atoms

def make_unit_cell(a, c, atoms):
    """Make hexagonal unit cell."""

    assert (len(atoms) == 2)

    # Normalized lattice vectors
    a0 = np.asarray([1.0, 0.0, 0.0])
    a1 = np.asarray([0.5, np.sqrt(3.0)/2, 0.0])
    a2 = np.asarray([0.0, 0.0, 1.0])

    a_matrix = np.zeros((3, 3))
    a_matrix[0] = a*a0
    a_matrix[1] = a*a1
    a_matrix[2] = c*a2

    # Internal coordinates
    icoords = np.asarray([
        [1.0/3, 1.0/3, 1.0/2],
        [2.0/3, 2.0/3, 1.0/2],
        ])
    # External coordinates
    ecoords = np.dot(icoords, a_matrix)

    atoms_uc = [
            [atoms[0], ecoords[0]],
            [atoms[1], ecoords[1]],
            ]

    return a_matrix, atoms_uc

def add_substrate(atoms, distance=3.2, n_atom_idx=None, prefix="*"):
    """Add water substrate.

    Parameters
    ----------
    atoms : list
        BN atoms as returned from `make_hex_unit_cell`.
    distance : float
        Distance between oxygen and surface.
    n_atom_idx : int, optional
        Index of surface atom, which the substrate coordinates to. Default: first nitrogen atom.
    """

    # Find first nitrogen atom
    if n_atom_idx is None:
        for idx, atom in enumerate(atoms):
            if atom[0] == "N":
                n_atom_idx = idx
                break
    atom_n = atoms[n_atom_idx]

    # xyz Vector from N -> O
    shift_o = np.asarray([0.34627917, 0.13581116, distance])
    # xyz Vector from O -> H1
    shift_h2 = np.asarray([-0.77956068, -0.2684398, 0.517])
    # xyz Vector from O -> H2
    shift_h0 = np.asarray([-0.27045789, -0.10511938, -0.933])

    coords_o = atom_n[1] + shift_o
    coords_h2 = atom_n[1] + shift_o + shift_h2
    coords_h0 = atom_n[1] + shift_o + shift_h0

    atoms_sub = [
            ["O" + prefix + "1", coords_o],
            ["H" + prefix + "2", coords_h2],
            ["H" + prefix + "0", coords_h0],
            ]
    n_atom_idx += len(atoms_sub)
    atoms_sub += copy.deepcopy(atoms)

    return atoms_sub

def get_shell_indices(a_matrix, a, atoms, center_atom):
    """
    Parameters
    ----------
    a_matrix : ndarray
        Lattice vectors.
    a : float
        Unit cell lattice constant.
    atoms : list
        Atoms
    center_atom : int
        Index of central atom.
    """

    # bond length and tolerance
    b = a / np.sqrt(3.0)
    tol = 1e-10*a

    # distances of nearest neighbours, next-nearest, etc around a central atom on the hexagonal surface
    shells_dist = b*np.asarray([
        0.0,                                # central atom
        1.0,                                # nearest-neighbors (ortho)
        np.sqrt(3.0),                       # meta
        2.0,                                # para
        np.sqrt(7.0),                       # across ortho-meta bond
        3.0,                                # across para atom (or otho atom)
        2*np.sqrt(3.0),                     # across meta atom
        np.sqrt(13.0),                      # across meta-para bond
        4.0,                                # next across ortho atom
        ])

    # Translation into neighbor cells (this yields [-1,-1,0], [-1,0,0], [-1,1,0],...,[1,1,0])
    tin = np.array(np.meshgrid([-1,0,1], [-1,0,1], [0])).T.reshape((9, 3))
    # transform translations from internal to external coordinates
    tex = np.dot(tin, a_matrix)
    # All lattice translations of the center atom position
    center = atoms[center_atom][1]
    centers = center[np.newaxis] + tex

    shell_indices = len(atoms)*[None]
    for atom_idx, atom in enumerate(atoms):
        dists = [np.linalg.norm(atom[1] - c) for c in centers]
        dist = np.amin(dists)
        for shell_idx, shell_dist in enumerate(shells_dist):
            if dist > shell_dist-tol and dist < shell_dist+tol:
                assert (shell_indices[atom_idx] is None)
                shell_indices[atom_idx] = shell_idx

    return shell_indices

def add_atom_indices(atoms, indices, prefix="#"):
    atoms = copy.deepcopy(atoms)
    for i, idx in enumerate(indices):
        if isinstance(idx, int):
            atoms[i][0] += prefix + str(idx)
    return atoms

if __name__ == "__main__":

    a, c = 2.51, 30.0
    sc = (4, 4, 1)
    atoms = ["B", "N"]
    distance = 4.0

    a_matrix, atoms = make(a, c, atoms=atoms, supercell=sc, distance=distance)
    visualize_atoms(atoms, a_matrix, indices=True)
