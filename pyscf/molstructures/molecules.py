"""Molecular systems for testing."""

import logging
#import os
import numpy as np

from pyscf import gto

if __name__ == "__main__":
    from util import *
else:
    from .util import *

__all__ = [
        "build_dimer",
        "build_two_dimers",
        "build_ring",
        "build_methane",
        "build_alkane",
        "build_ethanol",
        "build_ethenol",
        "build_ethanol_old",
        "build_chloroethanol",
        "build_ketene",
        "build_biphenyl",
        "build_H2O_NH3",
        "build_H2O_CH6",
        "build_water_dimer",
        "build_water_borazine",
        "build_water_boronene",
        "build_mn_oxo_porphyrin",
        "build_azomethane",
        "build_propane",
        ]

log = logging.getLogger(__name__)

#def Ry(alpha, radians=False):
#    if not radians:
#        alpha = np.deg2rad(alpha)
#    r = np.asarray([
#        [1, 0, 0],
#        [0, np.cos(alpha), np.sin(alpha)],
#        [0, -np.sin(alpha), np.cos(alpha)],
#        ])
#    return r
#
#def Rz(alpha, radians=False):
#    """Rotate around z axis."""
#    if not radians:
#        alpha = np.deg2rad(alpha)
#    r = np.asarray([
#        [np.cos(alpha), np.sin(alpha), 0],
#        [-np.sin(alpha), np.cos(alpha), 0],
#        [0, 0, 1],
#        ])
#    return r

def build_dimer(d, atoms, add_labels=False, **kwargs):
    if add_labels:
        l0, l1 = "1", "2"
    else:
        l0 = l1 = ""

    atom = "{}{} 0 0 {}; {}{} 0 0 {}".format(atoms[0], l0, -d/2, atoms[1], l1, d/2)
    log.debug("atom = %s", atom)

    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

def build_two_dimers(d, atoms, separation, add_labels=False, **kwargs):
    if add_labels:
        l0, l1, l2, l3 = "1", "2", "3", "4"
    else:
        l0 = l1 = l2 = l3 = ""

    sep = separation / 2

    atom = "{}{} 0 {} {}; {}{} 0 {} {};".format(atoms[0], l0, -sep, -d/2, atoms[1], l1, -sep, d/2)
    atom += " {}{} 0 {} {}; {}{} 0 {} {}".format(atoms[0], l2, sep, -d/2, atoms[1], l3, sep, d/2)
    log.debug("atom = %s", atom)

    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol


def build_ring(d, atoms, **kwargs):
    natom = len(atoms)
    angle = np.linspace(0, 2*np.pi, num=natom, endpoint=False)
    theta = 2*np.pi/natom
    r = d / (np.sqrt(2.0 * (1-np.cos(theta))))
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    z = np.zeros_like(x)
    atom = [("%s%d" % (atoms[i], i+1), x[i], y[i], z[i]) for i in range(natom)]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

def build_methane(dCH=1.087, **kwargs):
    x = 1/np.sqrt(3) * dCH
    atom = [
    ("C1", (0.0, 0.0, 0.0)),
    ("H2", (x, x, x)),
    ("H3", (x, -x, -x)),
    ("H4", (-x, x, -x)),
    ("H5", (-x, -x, x))]

    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

def build_alkane(n, dCH=1.09, dCC=1.54, **kwargs):

    atom = []
    phi = np.arccos(-1.0/3)
    cph = 1/np.sqrt(3.0)
    sph = np.sin(phi/2.0)

    dcy = dCC * cph
    dcz = dCC * sph
    dchs = dCH * sph
    dchc = dCH * cph

    k = 0
    for i in range(n):
        # Carbon atoms
        sign = (-1)**i
        x = 0.0
        y = sign * dcy/2
        z = i*dcz
        atom.append(["C%d" % k, [x, y, z]])
        k += 1
        # Hydrogen atoms on side
        dy = sign * dchc
        atom.append(["H%d" % k, [x+dchs, y+dy, z]])
        k += 1
        atom.append(["H%d" % k, [x-dchs, y+dy, z]])
        k += 1
        # Terminal Hydrogen atoms
        if (i == 0):
            atom.append(["H%d" % k, [0.0, y-dchc, z-dchs]])
            k += 1
        # Not elif, if n == 1 (Methane)
        if (i == n-1):
            atom.append(["H%d" % k, [0.0, y-sign*dchc, z+dchs]])
            k += 1

    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol






def build_ethanol_old(dOH, **kwargs):
    atoms = ["C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6", "O1"]
    coords = np.asarray([
       ( 0.01247000,  0.02254000,  1.08262000),
       (-0.00894000, -0.01624000, -0.43421000),
       (-0.49334000,  0.93505000,  1.44716000),
       ( 1.05522000,  0.04512000,  1.44808000),
       (-0.64695000, -1.12346000,  2.54219000),
       ( 0.50112000, -0.91640000, -0.80440000),
       ( 0.49999000,  0.86726000, -0.84481000),
       (-1.04310000, -0.02739000, -0.80544000),
       (-0.66442000, -1.15471000,  1.56909000),
    ])
    natom = len(atoms)
    oidx = -1
    d = [np.linalg.norm(coords[oidx] - coords[i]) for i in range(natom-1)]
    hidx = np.argmin(d)
    v_oh = (coords[hidx] - coords[oidx])
    dOHeq = np.linalg.norm(v_oh)
    u_oh = v_oh / dOHeq
    coords[hidx] += (dOH-dOHeq)*u_oh
    atom = [(atoms[idx], coords[idx]) for idx in range(natom)]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

def build_chloroethanol(dOH, **kwargs):
    atoms = ["C1", "C2", "Cl3", "O4", "H5", "H6", "H7", "H8", "H9"]
    coords = np.asarray([
        (0.9623650,     -0.5518170,     0.0000000),
        (0.0000000,     0.6114150 ,     0.0000000),
        (-1.682237,     0.0056080 ,     0.0000000),
        (2.2607680,     0.0168280 ,     0.0000000),
        (0.7859910,     -1.1642170,     0.8849280),
        (0.7859910,     -1.1642170,     -0.8849280),
        (0.1338630,     1.2200610 ,     0.8854980),
        (0.1338630,     1.2200610 ,    -0.8854980),
        (2.8979910,     -0.6992380,     0.0000000),
        ])
    natom = len(atoms)
    oidx = 3
    hidx = -1
    #d = [np.linalg.norm(coords[oidx] - coords[i]) for i in range(natom-1)]
    #hidx = np.argmin(d)
    #log.debug("H index: %d", hidx)
    v_oh = (coords[hidx] - coords[oidx])
    dOHeq = np.linalg.norm(v_oh)
    u_oh = v_oh / dOHeq
    coords[hidx] += (dOH-dOHeq)*u_oh
    atom = [(atoms[idx], coords[idx]) for idx in range(natom)]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

#def load_datafile(filename):
#    datafile = os.path.join(os.path.dirname(__file__), os.path.join("data", filename))
#    data = np.loadtxt(datafile, dtype=[("atoms", object), ("coords", np.float64, (3,))])
#    #print(data["atoms"])
#    #print(data["coords"])
#
#    return data["atoms"], data["coords"]
#
#def move_atom(coords, origin, distance):
#    v = coords - origin
#    v /= np.linalg.norm(v)
#    coords_out = origin + distance*v
#    return coords_out

def build_ethanol(distance, **kwargs):
    """Oxygen: O3, Hydrogen: H4, nearest C: C2"""
    atoms, coords = load_datafile("ethanol.dat")

    coords[3] = move_atom(coords[3], coords[2], distance)

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol

def build_ethenol(distance, **kwargs):
    """Oxygen: O3, Hydrogen: H7, nearest C: C2"""
    atoms, coords = load_datafile("ethenol.dat")

    coords[-1] = move_atom(coords[-1], coords[2], distance)

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol

def build_azomethane(distance, **kwargs):
    atoms, coords = load_datafile("azomethane.dat")

    # Rotate molecule
    theta = np.arctan(coords[0][0] / coords[0][1])
    R = Rz(theta, radians=True)
    coords = np.dot(coords, R)

    # Move fragment
    v_nn = coords[1] - coords[0]
    deq = np.linalg.norm(v_nn)
    v_nn /= deq
    distance -= deq
    frag = np.where(np.isin(atoms, ["N2", "C4", "H6", "H9", "H10"]))
    for i in frag:
        coords[i] += distance*v_nn

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]

    #for a in atom:
    #    print("%3s  %.3f  %.3f  %.3f" % (a[0], *a[1]))

    #print_distances(atom, origin=0)
    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol


def build_ketene(dCC, **kwargs):
    atoms = ["C1", "C2", "O3", "H4", "H5"]
    coords = np.asarray([
        (0.0000, 0.0000, 0.0000),
        (0.0000, 0.0000, 1.3150),
        (0.0000, 0.0000, 2.4750),
        (0.0000, 0.9451, -0.5206),
        (0.0000, -0.9451, -0.5206),
    ])

    dCCeq = 1.315
    coords[1:3,2] += (dCC-dCCeq)
    atom = [(atoms[atomidx], coords[atomidx]) for atomidx in range(len(atoms))]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol


def build_biphenyl(angle, **kwargs):
    atoms = [
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12",
        "H13","H14","H15","H16","H17","H18","H19","H20","H21","H22"]
    coords = np.asarray([
        ( 0.0000,  0.0000,  0.7420),
        ( 0.0000,  0.0000, -0.7420),
        (-0.4436,  1.1336,  1.4615),
        ( 0.4436, -1.1336,  1.4615),
        (-0.4436, -1.1336, -1.4615),
        ( 0.4436,  1.1336, -1.4615),
        (-0.4418,  1.1343,  2.8675),
        ( 0.4418, -1.1343,  2.8675),
        (-0.4418, -1.1343, -2.8675),
        ( 0.4418,  1.1343, -2.8675),
        ( 0.0000,  0.0000,  3.5750),
        ( 0.0000,  0.0000, -3.5750),
        (-0.8124,  2.0067,  0.9135),
        ( 0.8124, -2.0067,  0.9135),
        (-0.8124, -2.0067, -0.9135),
        ( 0.8124,  2.0067, -0.9135),
        (-0.7949,  2.0156,  3.4116),
        ( 0.7949, -2.0156,  3.4116),
        (-0.7949, -2.0156, -3.4116),
        ( 0.7949,  2.0156, -3.4116),
        ( 0.0000,  0.0000,  4.6690),
        ( 0.0000,  0.0000, -4.6690),
        ])
    rot_atoms = [2, 5, 6, 9, 10, 12, 15, 16, 19, 20, 22]
    rot_mask = [(int(a[1:]) in rot_atoms) for a in atoms]
    pos[rot_mask] = np.dot(coords[rot_mask], Rz(angle))
    atom = [(atoms[atomidx], coords[atomidx]) for atomidx in range(len(atoms))]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol

def build_H2O_NH3(dNH, **kwargs):
    atoms = ["O1", "H1", "H2", "N1", "H3", "H4", "H5"]
    coords_h2o = np.asarray([
    (0.0000, 0.0000, 0.1173),
    (0.0000, 0.7572, -0.4692),
    (0.0000, -0.7572, -0.4692),
    ])
    angle_h2o = 104.4776
    # rotation along y axis
    coords_h2o = np.dot(coords_h2o, Ry(angle_h2o/2))
    # shift H2 to origin
    coords_h2o -= coords_h2o[2]
    coods_nh3 = np.asarray([
    (0.0000, 0.0000, 0.0000),
    (0.0000, -0.9377, -0.3816),
    (0.8121, 0.4689, -0.3816),
    (-0.8121, 0.4689, -0.3816),
    ])
    coords = np.vstack((coods_h2o, coods_nh3))
    coords[:3,2] += dNH
    atom = [(atoms[idx], coords[idx]) for idx in range(len(atoms))]
    mol = gto.M(
            atom=atom,
            **kwargs)
    return mol


def build_H2O_CH6(dOC, config="2-leg", **kwargs):

    if config != "2-leg":
        raise NotImplementedError()

    water = [
            ["O1", [0.0000, 0.0000, 0.0000]],
            ["H2", [0.0000, 0.7572, -0.5865]],
            ["H3", [0.0000, -0.7572, -0.5865]],
            ]

    # Binding Cs in 2-leg configuration are C4 and C7
    benzene = [
            ["C4" , [0.0000, 1.3970, 0.0000]],
            ["C5" , [1.2098, 0.6985, 0.0000]],
            ["C6" , [1.2098, -0.6985, 0.0000]],
            ["C7" , [0.0000, -1.3970, 0.0000]],
            ["C8" , [-1.2098, -0.6985, 0.0000]],
            ["C9" , [-1.2098, 0.6985, 0.0000]],
            ["H10", [0.0000, 2.4810, 0.0000]],
            ["H11", [2.1486, 1.2405, 0.0000]],
            ["H12", [2.1486, -1.2405, 0.0000]],
            ["H13", [0.0000, -2.4810, 0.0000]],
            ["H14", [-2.1486, -1.2405, 0.0000]],
            ["H15", [-2.1486, 1.2405, 0.0000]],
            ]

    atom = water + benzene

    # Shift water molecule
    atom[0][1][2] += dOC
    atom[1][1][2] += dOC
    atom[2][1][2] += dOC

    mol = gto.M(
            atom=atom,
            **kwargs)

    #import lattices
    #lattices.visualize_atoms(atom, indices=True)

    return mol

def build_water_dimer(dOH, **kwargs):
    """Hydrogen bond is between H1 and O3.
    Molecula A is H1, O2, H4."""
    atoms, coords = load_datafile("water-dimer.dat")

    # New position of H1
    coords_h1 = move_atom(coords[0], coords[2], dOH)
    shift = coords_h1 - coords[0]
    # Move H1, O2, H4
    coords[0] += shift
    coords[1] += shift
    coords[3] += shift

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    #print_distances(atom, coords[2])
    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol


def build_water_borazine(distance, **kwargs):
    """From suppl. mat. of J. Chem. Phys. 147, 044710 (2017)

    Water molecule are O1, H1, H2.
    Nearest nitrogen is N1.
    Nearest hydrogen is H1.
    Next nearest borons are B1, B3.

    H on N1: H4

    Distance measured from O to molecular plane?
    """

    # At 3.32 A
    eq = [
        ["O1", [6.07642824900005, 5.45929539300001, 3.88230000000000]],
        ["H1", [6.08425933200003, 5.49983635649998, 2.90580000000000]],
        ["H2", [5.29230349199999, 4.92565706249998, 4.09910000000000]],
        ["B1", [5.00435273549996, 6.44422098149999, 0.55828330799997]],
        ["B2", [3.72565904700004, 4.27443283650000, 0.41692162499995]],
        ["B3", [6.24820422149998, 4.24146091050005, 0.45255167849998]],
        ["N1", [6.21955164299997, 5.67465859050003, 0.56921574450005]],
        ["N2", [4.98001838850001, 3.57987289349996, 0.39688556550001]],
        ["N3", [3.77999653349999, 5.70466272150004, 0.49999999950000]],
        ["H3", [5.02076127599995, 7.64373657450001, 0.60758282549997]],
        ["H4", [7.10527023150000, 6.17375954850001, 0.58709578200002]],
        ["H5", [7.28364120750005, 3.63461177700003, 0.41760386850004]],
        ["H6", [4.96865547299997, 2.56629642749996, 0.32842254299997]],
        ["H7", [2.67920894549999, 3.68622618450004, 0.36779496749997]],
        ["H8", [2.90474809949995, 6.22044015000000, 0.50692081349997]],
    ]

    atom = [[a[0], np.asarray(a[1])] for a in eq]

    xy_dist = np.linalg.norm(atom[0][1][:2] - atom[6][1][:2])
    z = np.sqrt(distance**2 - xy_dist**2)
    dzeq = atom[0][1][-1] - atom[6][1][-1]

    for i in range(3):
        atom[i][1][-1] += z - dzeq

    #for a in atom[3:9]:
    #    d = np.linalg.norm(atom[6][1] - a[1])
    #    print(a[0], d)

    #for a in atom[3:]:
    #    d = np.linalg.norm(atom[6][1] - a[1])
    #    print(a[0], d)


    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol

def build_water_boronene(distance, **kwargs):
    """From suppl. mat. of J. Chem. Phys. 147, 044710 (2017)

    Water molecule are O2, H1, H3.
    Nearest nitrogen is N1.
    Nearest hydrogen is H1.
    Next nearest borons are B2 etc.

    Distance measured from O to molecular plane.
    """

    # This data is for 3.4 A
    atoms, coords = load_datafile("boronene.dat")

    #i = 12 # H
    #i = 28 # N1
    #for j, b in enumerate(atoms):
    #    if i == j:
    #        continue

    #    d = np.linalg.norm(coords[i] - coords[j])
    #    print(j, b, d)
    #1/0

    # Move water molecule to distance
    for i in range(len(atoms)):
        #if "*" in atoms[i]:
        if atoms[i] in ("H1", "O2", "H3"):
            coords[i][2] += (distance-3.4)

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]

    #for a in atom:
    #    print(a)

    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol

def build_mn_oxo_porphyrin(distance, **kwargs):
    atoms, coords = load_datafile("mn-oxo-porphyrin.dat")
    dist_eq = np.linalg.norm(coords[1]-coords[0])
    assert np.isclose(dist_eq, 1.493775)
    coords[1][-1] = coords[1][-1] + (distance-dist_eq)

    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]

    #print_distances(atom, 0)

    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol

#def print_distances(atom, origin):
#    if isinstance(origin, int):
#        origin = atom[origin][1]
#    for symbol, coords in atom:
#        print("Distance to %3s: %.5g" % (symbol, np.linalg.norm(coords - origin)))

def build_propane(**kwargs):
    atoms, coords = load_datafile("propane.dat")
    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol


if __name__ == "__main__":
    #build_water_boronene(8.0)
    #build_mn_oxo_porphyrin(4.0, charge=1)
    #print("1")
    #build_azomethane(1.0)
    #print("2")
    #build_azomethane(2.0)
    #print("3.5")
    #build_azomethane(3.5)

    #build_water_dimer(5)
    #build_water_borazine(5)

    mol = build_alkane(4)
    visualize_atoms(mol.atom)

