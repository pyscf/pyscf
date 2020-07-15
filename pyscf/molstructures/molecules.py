"""Molecular systems for testing."""

import logging
import numpy as np

from pyscf import gto
from .counterpoise import mod_for_counterpoise

__all__ = [
        "build_dimer",
        "build_ring",
        "build_methane",
        "build_ethanol",
        "build_chloroethanol",
        "build_ketene",
        "build_biphenyl",
        "build_H2O_NH3",
        "build_H2O_CH6",
        "build_water_borazine",
        ]

log = logging.getLogger(__name__)

def Ry(alpha):
    alpha = np.deg2rad(alpha)
    r = np.asarray([
        [1, 0, 0],
        [0, np.cos(alpha), np.sin(alpha)],
        [0, -np.sin(alpha), np.cos(alpha)],
        ])
    return r

def Rz(alpha):
    alpha = np.deg2rad(alpha)
    r = np.asarray([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1],
        ])
    return r

def build_dimer(d, atoms, **kwargs):
    atom = "{}1 0 0 {}; {}2 0 0 {}".format(atoms[0], -d/2, atoms[1], d/2)
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



def build_ethanol(dOH, **kwargs):
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

def build_water_borazine(distance, counterpoise=None, **kwargs):
    """From suppl. mat. of J. Chem. Phys. 147, 044710 (2017)

    Water molecule are O1, H1, H2.
    Nearest nitrogen is N1.
    Next nearest borons are B1, B3.

    Distance measured from O to molecular plane?
    """

    # At 3.32 A
    eq = [
        ["O1", [6.07642824900005, 5.45929539300001, 3.88230000000000]],
        ["H1", [5.29230349199999, 4.92565706249998, 4.09910000000000]],
        ["H2", [6.08425933200003, 5.49983635649998, 2.90580000000000]],
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

    water = [a[0] for a in atom[:3]]
    borazine = [a[0] for a in atom[3:]]
    basis = None
    if counterpoise == "water":
        atom, basis = mod_for_counterpoise(atom, kwargs.get("basis", None), water, remove_basis=True)
    elif counterpoise == "water-full":
        atom, basis = mod_for_counterpoise(atom, kwargs.get("basis", None), water, remove_basis=False)
    elif counterpoise == "borazine":
        atom, basis = mod_for_counterpoise(atom, kwargs.get("basis", None), borazine, remove_basis=True)
    elif counterpoise == "borazine-full":
        atom, basis = mod_for_counterpoise(atom, kwargs.get("basis", None), borazine, remove_basis=False)
    if basis is not None:
        kwargs["basis"] = basis

    mol = gto.M(
        atom=atom,
        **kwargs)
    return mol
