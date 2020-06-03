import re
import string
import numpy as np

__all__ = [
        "get_closest_atom",
        "get_center_atom",
        "make_supercell",
        "visualize_atoms",
        ]

def get_closest_atom(atoms, point, atomtype=None):
    if atomtype is None:
        dists = [np.linalg.norm(a[1] - point) for a in atoms]
    else:
        dists = [(np.linalg.norm(a[1] - point) if a[0].startswith(atomtype) else np.inf) for a in atoms]
    atom = np.argmin(dists)
    return atom

def get_center_atom(a_matrix, atoms, atomtype=None):
    center = np.sum(a_matrix, axis=0)/2
    atom = get_closest_atom(atoms, center, atomtype=atomtype)
    return atom

def make_supercell(a_matrix, atoms, supercell, unit_cell=(0,0,0)):

    sc = np.asarray(supercell)
    uc = np.asarray(unit_cell)

    assert np.all(sc > 0)
    assert np.all(uc >= 0)
    assert np.all(uc < sc)

    a_matrix_sc = sc[:,np.newaxis] * a_matrix

    a = -uc
    b = sc - uc
    atoms_sc = []
    for i in range(a[0], b[0]):
        for j in range(a[1], b[1]):
            for k in range(a[2], b[2]):
                for atom in atoms:
                    coords = atom[1] + np.dot([i,j,k], a_matrix)
                    atoms_sc.append([atom[0], coords])

    assert (np.product(sc)*len(atoms) == len(atoms_sc))

    return a_matrix_sc, atoms_sc

atom_colors = {
        "H" : "white",
        "B" : "pink",
        "C" : "black",
        "N" : "blue",
        "O" : "red",
        "F" : "cyan",
        "Na" : "violet",
        "Cl" : "green",
        "Mg" : "darkgreen",
        }

def visualize_atoms(atoms, a_matrix=None, size=20, indices=False):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if a_matrix is not None:
        x = np.zeros((1,))
        y = np.zeros((1,))
        z = np.zeros((1,))
        axcolors = ["red", "blue", "green"]
        for d in range(3):
            dx=a_matrix[d,0]
            dy=a_matrix[d,1]
            dz=a_matrix[d,2]
            ax.quiver(x, y, z, dx, dy, dz, color=axcolors[d])

        ax.set_xlim(1.1*a_matrix[0,0])
        ax.set_ylim(1.1*a_matrix[1,1])
        ax.set_zlim(1.1*a_matrix[2,2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    x = [a[1][0] for a in atoms]
    y = [a[1][1] for a in atoms]
    z = [a[1][2] for a in atoms]

    #sym = [a[0] for a in atoms]
    sym = []
    for a in atoms:
        s = ""
        for l in a[0]:
            if l in string.ascii_letters:
                s += l
        sym.append(s)
    print(sym)

    #sym = [re.sub("\d", "", a[0])  for a in atoms]
    color = [atom_colors.get(s, "black") for s in sym]

    ax.scatter(x, y, z, s=size, color=color, depthshade=False, edgecolor="black")

    if indices:
        offset = [0, 0, 1]
        for i, atom in enumerate(atoms):
            idx = atom[0][-1]
            idx = idx if idx.isdigit() else None
            if idx is not None:
                ax.text(x[i]+offset[0], y[i]+offset[1], z[i]+offset[2], idx)

    plt.show()


