#!/usr/bin/env python3

import numpy as np

import pyscf
import pyscf.gto

__all__ = [
        "build_dimer",
        "build_ring",
        "build_EtOH",
        "build_biphenyl",
        "build_H2O_NH3",
        ]

def build_dimer(atoms, dist, basis, verbose=0):
    mol = pyscf.gto.M(
        atom = "{}1 0 0 {}; {}2 0 0 {}".format(atoms[0], -dist/2, atoms[1], dist/2),
        basis = basis,
        verbose=verbose)
    return mol

def build_ring(atoms, dist, basis, verbose=0):

    natom = len(atoms)
    angle = np.linspace(0, 2*np.pi, num=natom, endpoint=False)
    theta = 2*np.pi/natom
    r = d / (np.sqrt(2.0 * (1-np.cos(theta))))

    x = r*np.cos(angle)
    y = r*np.sin(angle)
    z = np.zeros_like(x)

    atom = [("%s%d" % (atoms[i], i+1), x[i], y[i], z[i]) for i in range(natom)]

    mol = pyscf.gto.M(
        atom = atom,
        basis = basis,
        verbose=verbose)

    return mol

def build_EtOH(dist, basis, verbose=1):
    geom = np.asarray([
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
    atoms = ["C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6", "O1"]

    natom = len(atoms)
    oidx = -1
    d = [np.linalg.norm(geom[oidx] - geom[i]) for i in range(natom-1)]
    hidx = np.argmin(d)
    v_oh = (geom[hidx] - geom[oidx])
    eqdist = np.linalg.norm(v_oh)
    u_oh = v_oh / eqdist

    geom[hidx] += (dist-eqdist)*u_oh

    atom = [(atoms[idx], geom[idx]) for idx in range(natom)]

    mol = pyscf.gto.M(
        atom = atom,
        basis = basis,
        verbose=verbose)

    return mol

def build_biphenyl(angle, basis, verbose=0):
    pos = np.asarray([
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

    atoms = [
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12",
        "H13","H14","H15","H16","H17","H18","H19","H20","H21","H22"]

    rot_atoms = [2, 5, 6, 9, 10, 12, 15, 16, 19, 20, 22]
    rot_mask = [(int(a[1:]) in rot_atoms) for a in atoms]

    pos[rot_mask] = np.dot(pos[rot_mask], Rz(angle))

    atom = [(atoms[atomidx], pos[atomidx]) for atomidx in range(len(atoms))]

    mol = pyscf.gto.M(
        atom = atom,
        basis = basis,
        verbose=verbose)

    return mol


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


def build_H2O_NH3(dist, basis, verbose=0):
    geom_h2o = np.asarray([
    (0.0000, 0.0000, 0.1173),
    (0.0000, 0.7572, -0.4692),
    (0.0000, -0.7572, -0.4692),
    ])

    angle_h2o = 104.4776

    # rotation along y axis
    geom_h2o = np.dot(geom_h2o, Ry(angle_h2o/2))
    # shift H2 to origin
    geom_h2o -= geom_h2o[2]

    geom_nh3 = np.asarray([
    (0.0000, 0.0000, 0.0000),
    (0.0000, -0.9377, -0.3816),
    (0.8121, 0.4689, -0.3816),
    (-0.8121, 0.4689, -0.3816),
    ])

    atoms = ["O1", "H1", "H2", "N1", "H3", "H4", "H5"]
    #atoms = ["O1", "H1", "H2", "N1", "C3", "C4", "H5"]
    geom = np.vstack((geom_h2o, geom_nh3))

    g = geom.copy()
    g[:3,2] += dist

    atom = [(atoms[idx], g[idx]) for idx in range(len(atoms))]

    mol = pyscf.gto.M(
        atom = atom,
        basis = basis,
        verbose=verbose)

    return mol


