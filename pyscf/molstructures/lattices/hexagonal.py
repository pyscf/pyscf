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

    a, c = 2.51, 10.0
    sc = (4, 4, 1)
    atoms = ["B", "N"]
    distance = 3.2

    a_matrix, atoms = make(a, c, atoms=atoms, supercell=sc, distance=distance)
    visualize_atoms(atoms, a_matrix, indices=True)

    #ref = """H 5.88 3.46 5.72
    #H 6.39 3.63 4.27
    #O 6.66 3.73 5.20
    #B 1.29 0.70 2.00
    #B 2.55 2.87 2.00
    #B 3.81 5.05 2.00
    #B 5.06 7.22 2.00
    #B 3.81 0.70 2.00
    #B 5.06 2.87 2.00
    #B 6.32 5.05 2.00
    #B 7.57 7.22 2.00
    #B 6.32 0.70 2.00
    #B 7.57 2.87 2.00
    #B 8.83 5.05 2.00
    #B 10.09 7.22 2.00
    #B 8.83 0.70 2.00
    #B 10.09 2.87 2.00
    #B 11.34 5.05 2.00
    #B 12.60 7.22 2.00
    #N 2.55 1.42 2.00
    #N 3.81 3.60 2.00
    #N 5.06 5.77 2.00
    #N 6.32 7.95 2.00
    #N 5.06 1.42 2.00
    #N 6.32 3.60 2.00
    #N 7.57 5.77 2.00
    #N 8.83 7.95 2.00
    #N 7.57 1.42 2.00
    #N 8.83 3.60 2.00
    #N 10.09 5.77 2.00
    #N 11.34 7.95 2.00
    #N 10.09 1.42 2.00
    #N 11.34 3.60 2.00
    #N 12.60 5.77 2.00
    #N 13.85 7.95 2.00"""

    #ref="""5.8848699130 3.4640408839 5.71700000000
    #    6.3939727004 3.6273612958 4.26700000000
    #    6.6644305907 3.7324806797 5.20000000000
    #    1.2944720000 0.6967070000 1.9996680000
    #    2.5504565567 2.8716975549 1.9995382176
    #    3.8064175128 5.0469969904 2.0003546472
    #    5.0624350751 7.2223780438 2.0014071730
    #    3.8062926218 0.6962529135 2.0001958196
    #    5.0622540727 2.8715620758 2.0014348620
    #    6.3182004626 5.0469377116 2.0028089602
    #    7.5740702489 7.2224434846 2.0017055725
    #    6.3181019094 0.6961900093 2.0015271752
    #    7.5740099987 2.8715106653 2.0028458212
    #    8.8298994553 5.0470284660 2.0023236716
    #    10.0859240046 7.2223339056 2.0008677755
    #    8.8298671810 0.6961228629 2.0013495380
    #    10.0857545005 2.8716831002 2.0005057810
    #    11.3417709433 5.0469911404 1.9996170630
    #    12.5977645849 7.2222727337 2.0006252347
    #    2.5504528649 1.4215297352 2.0001858400
    #    3.8063819878 3.5967701297 2.0012485940
    #    5.0623413269 5.7721038675 2.0026259662
    #    6.3182617717 7.9477122224 2.0016795990
    #    5.0622048194 1.4213476926 2.0020270554
    #    6.3181514198 3.5966695226 2.0043775345  
    #    7.5740485050 5.7721773001 2.0043577256
    #    8.8300307416 7.9474750657 2.0017316786
    #    7.5740075331 1.4212914889 2.0034872407
    #    8.8298678785 3.5967699709 2.0035198381
    #    10.0858659396 5.7720851029 2.0016183961
    #    11.3418803526 7.9474067513 2.0017782839
    #    10.0857404189 1.4214876504 2.0009212127
    #    11.3417145387 3.5967752907 2.0001414713
    #    12.5977078237 5.7720493736 2.0001252461
    #    13.8537300000 7.9472590000 2.0023270000"""

    #for atom in atoms:
    #    print("%s %r" % (atom[0], atom[1]))
    #ref = ref.split("\n")
    #ref = [r.split() for r in ref]

    ##ref = [(r[0], np.asarray([float(r[1]), float(r[2]), float(r[3])])) for r in ref]
    #ref = [(None, np.asarray([float(r[0]), float(r[1]), float(r[2])])) for r in ref]
    #print(ref)

    #for atom in range(4, len(ref)):
    #    mind = np.inf
    #    for atom2 in range(4, len(ref)):
    #        if atom == atom2:
    #            continue
    #        d = np.linalg.norm(ref[atom][1] - ref[atom2][1])
    #        if d < mind:
    #            mind = d
    #    print(mind)

    ##for i in range(4, 20):
    #    #print(np.linalg.norm(ref[3][1] - ref[i][1]))
    #print(np.linalg.norm(ref[2][1] - ref[-11][1]))

    #for i in atoms:
    #    if i[0] != "N#0":
    #        continue
    #    print(np.linalg.norm(atoms[0][1] - i[1]))
    ##print(np.linalg.norm(atoms[3][1] - atoms[2][1]))







