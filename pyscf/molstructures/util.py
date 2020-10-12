#import re
import string
import itertools

import numpy as np
import os

__all__ = [
        "Ry",
        "Rz",
        "load_datafile",
        "move_atom",
        "print_distances",
        "visualize_atoms",
        ]

def Ry(alpha, radians=False):
    if not radians:
        alpha = np.deg2rad(alpha)
    r = np.asarray([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)],
        ])
    return r

def Rz(alpha, radians=False):
    """Rotate around z axis."""
    if not radians:
        alpha = np.deg2rad(alpha)
    r = np.asarray([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1],
        ])
    return r

def load_datafile(filename):
    datafile = os.path.join(os.path.dirname(__file__), os.path.join("data", filename))
    data = np.loadtxt(datafile, dtype=[("atoms", object), ("coords", np.float64, (3,))])
    #print(data["atoms"])
    #print(data["coords"])

    return data["atoms"], data["coords"]

def move_atom(coords, origin, distance):
    v = coords - origin
    v /= np.linalg.norm(v)
    coords_out = origin + distance*v
    return coords_out

def get_min_distance_pbc(point1, point2, a_matrix):
    d_min = np.inf
    for d0, d1, d2 in itertools.product([-1, 0, 1], repeat=3):
        p2 = point2 + d0*a_matrix[:,0] + d1*a_matrix[:,1] + d2*a_matrix[:,2]
        d = np.linalg.norm(point1 - p2)
        #print(d0, d1, d2, d)
        if d < d_min:
            d_min = d
    return d_min

def print_distances(atom, origin, a_matrix=None):
    if isinstance(origin, int):
        origin = atom[origin][1]
    elif isinstance(origin, str):
        for at in atom:
            sym, atcoords = at
            if sym == origin:
                origin = atcoords
                break

    #for symbol, coords in atom:
    #    print("Distance to %3s: %.6g" % (symbol, np.linalg.norm(coords - origin)))

    symbols = np.asarray([a[0] for a in atom])
    if a_matrix is not None:
        distances = np.asarray([get_min_distance_pbc(a[1], origin, a_matrix) for a in atom])
    else:
        distances = np.asarray([np.linalg.norm(a[1] - origin) for a in atom])

    sort = np.argsort(distances)
    for i, symbol in enumerate(symbols[sort]):
        print("Distance to %3s: %.6g" % (symbol, distances[sort][i]))


def visualize_atoms(atoms, a_matrix=None, size=20, indices=False, atom_colors=None):

    if atom_colors is None:
        atom_colors = {
            "H" : "white",
            "Li" : "darkgreen",
            "B" : "pink",
            "C" : "black",
            "N" : "blue",
            "O" : "red",
            "F" : "cyan",
            "Na" : "violet",
            "Cl" : "green",
            "Mg" : "darkgreen",
        }


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
    #print(sym)

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


