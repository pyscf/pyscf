import copy
import numpy as np

try:
    from .general import *
# If scipts are ran from within package
except (ImportError, SystemError):
    from general import *

__all__ = [
        "make_cubic_unit_cell",
        ]

def make_cubic_unit_cell(a, c, atomtypes, supercell=None):

    #a_matrix = a * np.eye(3)
    #a_matrix[2] = np.asarray((0, 0, c))
    a_matrix = np.diag([a, a, c])

    # Internal coordinates
    icoords = np.asarray([
        [1.0/4, 1.0/4, 1.0/2],
        [1.0/4, 3.0/4, 1.0/2],
        [3.0/4, 3.0/4, 1.0/2],
        [3.0/4, 1.0/4, 1.0/2],
        ])
    # External coordinates
    ecoords = np.dot(icoords, a_matrix)

    atoms = [
            [atomtypes[0], ecoords[0]],
            [atomtypes[1], ecoords[1]],
            [atomtypes[0], ecoords[2]],
            [atomtypes[1], ecoords[3]],
            ]

    if supercell is not None:
        a_matrix, atoms = make_supercell(a_matrix, atoms, supercell)

    return a_matrix, atoms

def add_water_substrate(a_matrix, atoms, distance=1.795, theta=23.21, relpos=0.625, d_owh1=1.12, d_owh2=0.97, a_h2o=104.45, prefix="*"):

    gamma = 180.0 - (a_h2o - theta)

    theta = np.deg2rad(theta)
    gamma = np.deg2rad(gamma)

    idx_m1 = get_center_atom(a_matrix, atoms, atomtype="Mg")
    idx_o1 = get_center_atom(a_matrix, atoms, atomtype="O")
    assert atoms[idx_m1][0] == "Mg"
    assert atoms[idx_o1][0] == "O"
    # Rename
    atoms[idx_m1][0] = atoms[idx_m1][0] + "#0"
    atoms[idx_o1][0] = atoms[idx_o1][0] + "#0"

    pos_o1 = atoms[idx_o1][1]
    pos_m1 = atoms[idx_m1][1]

    p = (1-relpos)*pos_o1 + relpos*pos_m1
    pos_ow = p + np.asarray((0, 0, distance))

    v_m1o1 = (pos_o1 - pos_m1)
    u_m1o1 = v_m1o1 / np.linalg.norm(v_m1o1)

    assert np.isclose(u_m1o1[2], 0)

    shift_h1 = np.cos(theta)*d_owh1*u_m1o1 + np.asarray((0, 0, -np.sin(theta)*d_owh1))
    pos_h1 = pos_ow + shift_h1

    shift_h2 = np.cos(gamma)*d_owh2*(-u_m1o1) + np.asarray((0, 0, np.sin(gamma)*d_owh2))
    pos_h2 = pos_ow + shift_h2

    atoms_sub = [
            ["O" + prefix + "1", pos_ow],
            ["H" + prefix + "2", pos_h2],
            ["H" + prefix + "0", pos_h1],
            ]
    atoms_sub += copy.deepcopy(atoms)

    return atoms_sub


def _do_some_geometry(o1mg1, owmg1, o1h1, owh1, theta):
    b = o1mg1
    e = owmg1
    c = o1h1
    d = owh1

    theta = np.deg2rad(theta)

    h = e * np.sin(theta)
    b2 = e * np.cos(theta)
    b1 = b - b2
    x = np.sqrt(h**2 + b1**1)

    alpha1 = np.arccos(h/x)
    alpha2 = np.arccos((x**2 + d**2 - c**2)/(2*x*d))
    alpha = alpha1 + alpha2
    beta = np.pi/2 - alpha

    print("h = %f" % h)
    print("Beta = %.2f" % np.rad2deg(beta))
    print("b1/b = %f" % (b1/b))
    print("b2/b = %f" % (b2/b))


if __name__ == "__main__":

    ## Numbers for 0% stretch [a=4.21 A]
    #_do_some_geometry(o1mg1=2.25, owmg1=2.2, o1h1=1.67, owh1=1.02, theta=71.1)

    ## Numbers for 5.1% stretch [a=1.051*4.21 A = 4.42471 A]
    #_do_some_geometry(o1mg1=2.7, owmg1=2.06, o1h1=1.38, owh1=1.12, theta=60.6)

    a = 4.21
    sc = (1, 1, 1)
    atoms = ["Mg", "O"]

    a_matrix, atoms = make_cubic_unit_cell(a, atoms)
    #a_matrix, atoms = make_supercell(a_matrix, atoms, sc)

    atoms = add_water_substrate(atoms)

    print(atoms)

    visualize_atoms(atoms, a_matrix, indices=True)


