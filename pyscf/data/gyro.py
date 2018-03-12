#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyscf.gto import mole
from pyscf.data import nist

# nuclear magneton are taken from http://easyspin.org/documentation/isotopetable.html
# isotope-mass, spin, nuclear-g-factor
ISOTOPE_GYRO = (
    (0  , 0.  ,         0.0),
    (1  , 1./2,  5.58569468),  # H
    (3  , 1./2, -4.25499544),  # He
    (7  , 3./2,    2.170951),  # Li
    (9  , 3./2,    -0.78495),  # Be
    (11 , 3./2,   1.7924326),  # B
    (13 , 1./2,   1.4048236),  # C
    (14 , 1.  ,  0.40376100),  # N
    (17 , 5./2,   -0.757516),  # O
    (19 , 1./2,    5.257736),  # F
    (21 , 3./2,   -0.441198),  # Ne
    (23 , 3./2,    1.478348),  # Na
    (25 , 5./2,    -0.34218),  # Mg
    (27 , 5./2,   1.4566028),  # Al
    (29 , 1./2,    -1.11058),  # Si
    (31 , 1./2,     2.26320),  # P
    (33 , 3./2,    0.429214),  # S
    (35 , 3./2,   0.5479162),  # Cl
    (40 , 0   ,         0.0),  # Ar
    (39 , 3./2,     0.26098),  # K
    (43 , 7./2,    -0.37637),  # Ca
    (45 , 7./2,     1.35899),  # Sc
    (47 , 5./2,    -0.31539),  # Ti
    (51 , 7./2,     1.47106),  # V
    (53 , 3./2,    -0.31636),  # Cr
    (55 , 5./2,      1.3813),  # Mn
    (57 , 1./2,      0.1809),  # Fe
    (59 , 7./2,       1.322),  # Co
    (61 , 3./2,    -0.50001),  # Ni
    (63 , 3./2,      1.4824),  # Cu
    (67 , 5./2,    0.350192),  # Zn
    (69 , 3./2,     1.34439),  # Ga
    (73 , 9./2,  -0.1954373),  # Ge
    (75 , 3./2,     0.95965),  # As
    (77 , 1./2,     1.07008),  # Se
    (79 , 3./2,    1.404267),  # Br
    (83 , 9./2,   -0.215704),  # Kr
    (85 , 5./2,    0.541192),  # Rb
    (87 , 9./2,    -0.24284),  # Sr
    (89 , 1./2,  -0.2748308),  # Y
    (91 , 5./2,   -0.521448),  # Zr
    (93 , 9./2,      1.3712),  # Nb
    (95 , 5./2,     -0.3657),  # Mo
    (99 , 9./2,      1.2632),  # Tc
    (101, 5./2,      -0.288),  # Ru
    (103, 1./2,     -0.1768),  # Rh
    (105, 5./2,      -0.257),  # Pd
    (107, 1./2,    -0.22714),  # Ag
    (111, 1./2,    -1.18977),  # Cd
    (115, 9./2,      1.2313),  # In
    (119, 1./2,    -2.09456),  # Sn
    (121, 5./2,      1.3454),  # Sb
    (125, 1./2,  -1.7770102),  # Te
    (127, 5./2,     1.12531),  # I
    (129, 1./2,    -1.55595),  # Xe
    (133, 7./2,   0.7377214),  # Cs
    (137, 3./2,     0.62491),  # Ba
    (139, 7./2,    0.795156),  # La
    (140, 0   ,         0.0),  # Ce
    (141, 5./2,      1.7102),  # Pr
    (143, 7./2,     -0.3043),  # Nd
    (147, 7./2,       0.737),  # Pm
    (147, 7./2,      -0.232),  # Sm
    (153, 5./2,      0.6134),  # Eu
    (157, 3./2,     -0.2265),  # Gd
    (159, 3./2,       1.343),  # Tb
    (161, 5./2,      -0.192),  # Dy
    (165, 7./2,       1.668),  # Ho
    (167, 7./2,     -0.1611),  # Er
    (169, 1./2,      -0.462),  # Tm
    (171, 1./2,     0.98734),  # Yb
    (175, 7./2,      0.6378),  # Lu
    (177, 7./2,      0.2267),  # Hf
    (181, 7./2,     0.67729),  # Ta
    (183, 1./2,   0.2355695),  # W
    (187, 5./2,      1.2879),  # Re
    (187, 1./2,   0.1293038),  # Os
    (193, 3./2,      0.1091),  # Ir
    (195, 1./2,      1.2190),  # Pt
    (197, 3./2,    0.097164),  # Au
    (199, 1./2,    1.011771),  # Hg
    (205, 1./2,   3.2764292),  # Tl
    (207, 1./2,     1.18512),  # Pb
    (209, 9./2,      0.9134),  # Bi
    (209, 1./2,         1.5),  # Po
    (210, 0.  ,         0.0),  # At
    (209, 0.  ,         0.0),  # Rn
    (223, 0.  ,         0.0),  # Fr
    (223, 0.  ,         0.0),  # Ra
    (227, 3./2,        0.73),  # Ac
    (229, 5./2,        0.18),  # Th
    (231, 3./2,         0.0),  # Pa
    (235, 7./2,      -0.109),  # U
    (237, 5./2,       1.256),  # Np
    (239, 1./2,       0.406),  # Pu
    (243, 5./2,         0.6),  # Am
    (247, 9./2,         0.0),  # Cm
)

def g_factor_to_gyromagnetic_ratio(g):
    '''Larmor freq in Hz'''
    return nist.NUC_MAGNETON/nist.PLANCK * g

def get_nuc_g_factor(symb, mass=None):
    Z = mole.charge(symb)
# g factor of other isotopes can be found in file nuclear_g_factor.dat
    nuc_spin, g_nuc = ISOTOPE_GYRO[Z][1:3]
    #gyromag = g_factor_to_gyromagnetic_ratio(g_nuc)
    return g_nuc
