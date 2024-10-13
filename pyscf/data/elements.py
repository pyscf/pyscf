#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import numpy

ELEMENTS = [
    'X',  # Ghost
    'H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar', 'K' , 'Ca',
    'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y' , 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I' , 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
]
NUC = {x: i for i,x in enumerate(ELEMENTS)}
NUC.update((x.upper(),i) for i,x in enumerate(ELEMENTS))
NUC['GHOST'] = 0
ELEMENTS_PROTON = NUC

ATOMIC_NAMES = [
    'Ghost',
    # IUPAC version dated 28 November 2016
    'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
    'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon',
    'Sodium', 'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus',
    'Sulfur', 'Chlorine', 'Argon', 'Potassium', 'Calcium',
    'Scandium', 'Titanium', 'Vanadium', 'Chromium', 'Manganese',
    'Iron', 'Cobalt', 'Nickel', 'Copper', 'Zinc',
    'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine',
    'Krypton', 'Rubidium', 'Strontium', 'Yttrium', 'Zirconium',
    'Niobium', 'Molybdenum', 'Technetium', 'Ruthenium', 'Rhodium',
    'Palladium', 'Silver', 'Cadmium', 'Indium', 'Tin',
    'Antimony', 'Tellurium', 'Iodine', 'Xenon', 'Caesium',
    'Barium', 'Lanthanum', 'Cerium', 'Praseodymium', 'Neodymium',
    'Promethium', 'Samarium', 'Europium', 'Gadolinium', 'Terbium',
    'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
    'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
    'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
    'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
    'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
    'Protactinium', 'Uranium', 'Neptunium', 'Plutonium', 'Americium',
    'Curium', 'Berkelium', 'Californium', 'Einsteinium', 'Fermium',
    'Mendelevium', 'Nobelium', 'Lawrencium', 'Rutherfordium', 'Dubnium',
    'Seaborgium', 'Bohrium', 'Hassium', 'Meitnerium', 'Darmastadtium',
    'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium', 'Moscovium',
    'Livermorium', 'Tennessine', 'Oganesson'
]

ISOTOPE_MAIN = [
    0  ,   # GHOST
    1  ,   # H
    4  ,   # He
    7  ,   # Li
    9  ,   # Be
    11 ,   # B
    12 ,   # C
    14 ,   # N
    16 ,   # O
    19 ,   # F
    20 ,   # Ne
    23 ,   # Na
    24 ,   # Mg
    27 ,   # Al
    28 ,   # Si
    31 ,   # P
    32 ,   # S
    35 ,   # Cl
    40 ,   # Ar
    39 ,   # K
    40 ,   # Ca
    45 ,   # Sc
    48 ,   # Ti
    51 ,   # V
    52 ,   # Cr
    55 ,   # Mn
    56 ,   # Fe
    59 ,   # Co
    58 ,   # Ni
    63 ,   # Cu
    64 ,   # Zn
    69 ,   # Ga
    74 ,   # Ge
    75 ,   # As
    80 ,   # Se
    79 ,   # Br
    84 ,   # Kr
    85 ,   # Rb
    88 ,   # Sr
    89 ,   # Y
    90 ,   # Zr
    93 ,   # Nb
    98 ,   # Mo
    98 ,   # 98Tc
    102,   # Ru
    103,   # Rh
    106,   # Pd
    107,   # Ag
    114,   # Cd
    115,   # In
    120,   # Sn
    121,   # Sb
    130,   # Te
    127,   # I
    132,   # Xe
    133,   # Cs
    138,   # Ba
    139,   # La
    140,   # Ce
    141,   # Pr
    144,   # Nd
    145,   # Pm
    152,   # Sm
    153,   # Eu
    158,   # Gd
    159,   # Tb
    162,   # Dy
    162,   # Ho
    168,   # Er
    169,   # Tm
    174,   # Yb
    175,   # Lu
    180,   # Hf
    181,   # Ta
    184,   # W
    187,   # Re
    192,   # Os
    193,   # Ir
    195,   # Pt
    197,   # Au
    202,   # Hg
    205,   # Tl
    208,   # Pb
    209,   # Bi
    209,   # Po
    210,   # At
    222,   # Rn
    223,   # Fr
    226,   # Ra
    227,   # Ac
    232,   # Th
    231,   # Pa
    238,   # U
    237,   # Np
    244,   # Pu
    243,   # Am
    247,   # Cm
    247,   # Bk
    251,   # Cf
    252,   # Es
    257,   # Fm
    258,   # Md
    259,   # No
    262,   # Lr
    261,   # Rf
    262,   # Db
    263,   # Sg
    262,   # Bh
    265,   # Hs
    266,   # Mt
    0  ,   # Ds
    0  ,   # Rg
    0  ,   # Cn
    0  ,   # Nh
    0  ,   # Fl
    0  ,   # Mc
    0  ,   # Lv
    0  ,   # Ts
    0  ,   # Og
]


# From ase code
#
#    https://wiki.fysik.dtu.dk/ase/index.html
#
# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
MASSES = [
    0.,                 # GHOST
    1.008,              # H [1.00784, 1.00811]
    4.002602,           # He
    6.94,               # Li [6.938, 6.997]
    9.0121831,          # Be
    10.81,              # B [10.806, 10.821]
    12.011,             # C [12.0096, 12.0116]
    14.007,             # N [14.00643, 14.00728]
    15.999,             # O [15.99903, 15.99977]
    18.998403163,       # F
    20.1797,            # Ne
    22.98976928,        # Na
    24.305,             # Mg [24.304, 24.307]
    26.9815385,         # Al
    28.085,             # Si [28.084, 28.086]
    30.973761998,       # P
    32.06,              # S [32.059, 32.076]
    35.45,              # Cl [35.446, 35.457]
    39.948,             # Ar
    39.0983,            # K
    40.078,             # Ca
    44.955908,          # Sc
    47.867,             # Ti
    50.9415,            # V
    51.9961,            # Cr
    54.938044,          # Mn
    55.845,             # Fe
    58.933194,          # Co
    58.6934,            # Ni
    63.546,             # Cu
    65.38,              # Zn
    69.723,             # Ga
    72.630,             # Ge
    74.921595,          # As
    78.971,             # Se
    79.904,             # Br [79.901, 79.907]
    83.798,             # Kr
    85.4678,            # Rb
    87.62,              # Sr
    88.90584,           # Y
    91.224,             # Zr
    92.90637,           # Nb
    95.95,              # Mo
    97.90721,           # 98Tc
    101.07,             # Ru
    102.90550,          # Rh
    106.42,             # Pd
    107.8682,           # Ag
    112.414,            # Cd
    114.818,            # In
    118.710,            # Sn
    121.760,            # Sb
    127.60,             # Te
    126.90447,          # I
    131.293,            # Xe
    132.90545196,       # Cs
    137.327,            # Ba
    138.90547,          # La
    140.116,            # Ce
    140.90766,          # Pr
    144.242,            # Nd
    144.91276,          # 145Pm
    150.36,             # Sm
    151.964,            # Eu
    157.25,             # Gd
    158.92535,          # Tb
    162.500,            # Dy
    164.93033,          # Ho
    167.259,            # Er
    168.93422,          # Tm
    173.054,            # Yb
    174.9668,           # Lu
    178.49,             # Hf
    180.94788,          # Ta
    183.84,             # W
    186.207,            # Re
    190.23,             # Os
    192.217,            # Ir
    195.084,            # Pt
    196.966569,         # Au
    200.592,            # Hg
    204.38,             # Tl [204.382, 204.385]
    207.2,              # Pb
    208.98040,          # Bi
    208.98243,          # Po
    209.98715,          # At
    222.01758,          # Rn
    223.01974,          # Fr
    226.02541,          # Ra
    227.02775,          # Ac
    232.0377,           # Th
    231.03588,          # Pa
    238.02891,          # U
    237.04817,          # Np
    244.06421,          # Pu
    243.06138,          # Am
    247.07035,          # Cm
    247.07031,          # Bk
    251.07959,          # Cf
    252.0830,           # Es
    257.09511,          # Fm
    258.09843,          # Md
    259.1010,           # No
    262.110,            # Lr
    267.122,            # Rf
    268.126,            # Db
    271.134,            # Sg
    270.133,            # Bh
    269.1338,           # Hs
    278.156,            # Mt
    281.165,            # Ds
    281.166,            # Rg
    285.177,            # Cn
    286.182,            # Nh
    289.190,            # Fl
    289.194,            # Mc
    293.204,            # Lv
    293.208,            # Ts
    294.214,            # Og
]

# Atomic Weights of the most common isotopes
# From https://chemistry.sciences.ncsu.edu/msf/pdf/IsotopicMass_NaturalAbundance.pdf
COMMON_ISOTOPE_MASSES = [
    0.,       # Ghost
    1.007825, # H
    4.002603, # He
    7.016004, # Li
    9.012182, # Be
    11.009305, # B
    12.000000, # C
    14.003074, # N
    15.994915, # O
    18.998403, # F
    19.992440, # Ne
    22.989770, # Na
    23.985042, # Mg
    26.981538, # Al
    27.976927, # Si
    30.973762, # P
    31.972071, # S
    34.968853, # Cl
    39.962383, # Ar
    38.963707, # K
    39.962591, # Ca
    44.955910, # Sc
    47.947947, # Ti
    50.943964, # V
    51.940512, # Cr
    54.938050, # Mn
    55.934942, # Fe
    58.933200, # Co
    57.935348, # Ni
    62.929601, # Cu
    63.929147, # Zn
    68.925581, # Ga
    73.921178, # Ge
    74.921596, # As
    79.916522, # Se
    78.918338, # Br
    83.911507, # Kr
    84.911789, # Rb
    87.905614, # Sr
    88.905848, # Y
    89.904704, # Zr
    92.906378, # Nb
    97.905408, # Mo
    98.907216, # 98
    101.904350, # Ru
    102.905504, # Rh
    105.903483, # Pd
    106.905093, # Ag
    113.903358, # Cd
    114.903878, # In
    119.902197, # Sn
    120.903818, # Sb
    129.906223, # Te
    126.904468, # I
    131.904154, # Xe
    132.905447, # Cs
    137.905241, # Ba
    138.906348, # La
    139.905435, # Ce
    140.907648, # Pr
    141.907719, # Nd
    144.912744, # 14
    151.919729, # Sm
    152.921227, # Eu
    157.924101, # Gd
    158.925343, # Tb
    163.929171, # Dy
    164.930319, # Ho
    165.930290, # Er
    168.934211, # Tm
    173.938858, # Yb
    174.940768, # Lu
    179.946549, # Hf
    180.947996, # Ta
    183.950933, # W
    186.955751, # Re
    191.961479, # Os
    192.962924, # Ir
    194.964774, # Pt
    196.966552, # Au
    201.970626, # Hg
    204.974412, # Tl
    207.976636, # Pb
    208.980383, # Bi
    208.982416, # Po
    209.987131, # At
    222.017570, # Rn
    223.019731, # Fr
    226.025403, # Ra
    227.027747, # Ac
    232.038050, # Th
    231.035879, # Pa
    238.050783, # U
    237.048167, # Np
    244.064198, # Pu
    243.061373, # Am
    247.070347, # Cm
    247.070299, # Bk
    251.079580, # Cf
    252.082972, # Es
    257.095099, # Fm
    258.098425, # Md
    259.101024, # No
    262.109692, # Lr
    267.,       # Rf
    268.,       # Db
    269.,       # Sg
    270.,       # Bh
    270.,       # Hs
    278.,       # Mt
    281.,       # Ds
    282.,       # Rg
    285.,       # Cn
    286.,       # Nh
    289.,       # Fl
    290.,       # Mc
    293.,       # Lv
    294.,       # Ts
    294.        # Og
]

# ground state configuration = (num. electrons for each irrep./angular moment)
CONFIGURATION = [
    [ 0, 0, 0, 0],     #  0  GHOST
    [ 1, 0, 0, 0],     #  1  H
    [ 2, 0, 0, 0],     #  2  He
    [ 3, 0, 0, 0],     #  3  Li
    [ 4, 0, 0, 0],     #  4  Be
    [ 4, 1, 0, 0],     #  5  B
    [ 4, 2, 0, 0],     #  6  C
    [ 4, 3, 0, 0],     #  7  N
    [ 4, 4, 0, 0],     #  8  O
    [ 4, 5, 0, 0],     #  9  F
    [ 4, 6, 0, 0],     # 10  Ne
    [ 5, 6, 0, 0],     # 11  Na
    [ 6, 6, 0, 0],     # 12  Mg
    [ 6, 7, 0, 0],     # 13  Al
    [ 6, 8, 0, 0],     # 14  Si
    [ 6, 9, 0, 0],     # 15  P
    [ 6,10, 0, 0],     # 16  S
    [ 6,11, 0, 0],     # 17  Cl
    [ 6,12, 0, 0],     # 18  Ar
    [ 7,12, 0, 0],     # 19  K
    [ 8,12, 0, 0],     # 20  Ca
    [ 8,12, 1, 0],     # 21  Sc
    [ 8,12, 2, 0],     # 22  Ti
    [ 8,12, 3, 0],     # 23  V
    [ 7,12, 5, 0],     # 24  Cr
    [ 8,12, 5, 0],     # 25  Mn
    [ 8,12, 6, 0],     # 26  Fe
    [ 8,12, 7, 0],     # 27  Co
    [ 8,12, 8, 0],     # 28  Ni
    [ 7,12,10, 0],     # 29  Cu
    [ 8,12,10, 0],     # 30  Zn
    [ 8,13,10, 0],     # 31  Ga
    [ 8,14,10, 0],     # 32  Ge
    [ 8,15,10, 0],     # 33  As
    [ 8,16,10, 0],     # 34  Se
    [ 8,17,10, 0],     # 35  Br
    [ 8,18,10, 0],     # 36  Kr
    [ 9,18,10, 0],     # 37  Rb
    [10,18,10, 0],     # 38  Sr
    [10,18,11, 0],     # 39  Y
    [10,18,12, 0],     # 40  Zr
    [ 9,18,14, 0],     # 41  Nb
    [ 9,18,15, 0],     # 42  Mo
    [10,18,15, 0],     # 43  Tc
    [ 9,18,17, 0],     # 44  Ru
    [ 9,18,18, 0],     # 45  Rh
    [ 8,18,20, 0],     # 46  Pd
    [ 9,18,20, 0],     # 47  Ag
    [10,18,20, 0],     # 48  Cd
    [10,19,20, 0],     # 49  In
    [10,20,20, 0],     # 50  Sn
    [10,21,20, 0],     # 51  Sb
    [10,22,20, 0],     # 52  Te
    [10,23,20, 0],     # 53  I
    [10,24,20, 0],     # 54  Xe
    [11,24,20, 0],     # 55  Cs
    [12,24,20, 0],     # 56  Ba
    [12,24,21, 0],     # 57  La
    [12,24,21, 1],     # 58  Ce
    [12,24,20, 3],     # 59  Pr
    [12,24,20, 4],     # 60  Nd
    [12,24,20, 5],     # 61  Pm
    [12,24,20, 6],     # 62  Sm
    [12,24,20, 7],     # 63  Eu
    [12,24,21, 7],     # 64  Gd
    [12,24,21, 8],     # 65  Tb
    [12,24,20,10],     # 66  Dy
    [12,24,20,11],     # 67  Ho
    [12,24,20,12],     # 68  Er
    [12,24,20,13],     # 69  Tm
    [12,24,20,14],     # 70  Yb
    [12,24,21,14],     # 71  Lu
    [12,24,22,14],     # 72  Hf
    [12,24,23,14],     # 73  Ta
    [12,24,24,14],     # 74  W
    [12,24,25,14],     # 75  Re
    [12,24,26,14],     # 76  Os
    [12,24,27,14],     # 77  Ir
    [11,24,29,14],     # 78  Pt
    [11,24,30,14],     # 79  Au
    [12,24,30,14],     # 80  Hg
    [12,25,30,14],     # 81  Tl
    [12,26,30,14],     # 82  Pb
    [12,27,30,14],     # 83  Bi
    [12,28,30,14],     # 84  Po
    [12,29,30,14],     # 85  At
    [12,30,30,14],     # 86  Rn
    [13,30,30,14],     # 87  Fr
    [14,30,30,14],     # 88  Ra
    [14,30,31,14],     # 89  Ac
    [14,30,32,14],     # 90  Th
    [14,30,31,16],     # 91  Pa
    [14,30,31,17],     # 92  U
    [14,30,31,18],     # 93  Np
    [14,30,30,20],     # 94  Pu
    [14,30,30,21],     # 95  Am
    [14,30,31,21],     # 96  Cm
    [14,30,31,22],     # 97  Bk
    [14,30,30,24],     # 98  Cf
    [14,30,30,25],     # 99  Es
    [14,30,30,26],     #100  Fm
    [14,30,30,27],     #101  Md
    [14,30,30,28],     #102  No
    [14,30,31,28],     #103  Lr
    [14,30,32,28],     #104  Rf
    [14,30,33,28],     #105  Db
    [14,30,34,28],     #106  Sg
    [14,30,35,28],     #107  Bh
    [14,30,36,28],     #108  Hs
    [14,30,37,28],     #109  Mt
    [14,30,38,28],     #110  Ds
    [14,30,39,28],     #111  Rg
    [14,30,40,28],     #112  Cn
    [14,31,40,28],     #113  Nh
    [14,32,40,28],     #114  Fl
    [14,33,40,28],     #115  Mc
    [14,34,40,28],     #116  Lv
    [14,35,40,28],     #117  Ts
    [14,36,40,28],     #118  Og
]

# Non-relativistic spin-restricted spherically averaged Hartree-Fock
# configurations for use in atomic SAD calculations. Reference
# configurations from Phys. Rev. A 101, 012516 (2020).
NRSRHF_CONFIGURATION = [
    [ 0, 0, 0, 0],     #  0  GHOST
    [ 1, 0, 0, 0],     #  1  H
    [ 2, 0, 0, 0],     #  2  He
    [ 3, 0, 0, 0],     #  3  Li
    [ 4, 0, 0, 0],     #  4  Be
    [ 4, 1, 0, 0],     #  5  B
    [ 4, 2, 0, 0],     #  6  C
    [ 4, 3, 0, 0],     #  7  N
    [ 4, 4, 0, 0],     #  8  O
    [ 4, 5, 0, 0],     #  9  F
    [ 4, 6, 0, 0],     # 10  Ne
    [ 5, 6, 0, 0],     # 11  Na
    [ 6, 6, 0, 0],     # 12  Mg
    [ 6, 7, 0, 0],     # 13  Al
    [ 6, 8, 0, 0],     # 14  Si
    [ 6, 9, 0, 0],     # 15  P
    [ 6,10, 0, 0],     # 16  S
    [ 6,11, 0, 0],     # 17  Cl
    [ 6,12, 0, 0],     # 18  Ar
    [ 7,12, 0, 0],     # 19  K
    [ 8,12, 0, 0],     # 20  Ca
    [ 8,13, 0, 0],     # 21  Sc
    [ 8,12, 2, 0],     # 22  Ti
    [ 8,12, 3, 0],     # 23  V
    [ 8,12, 4, 0],     # 24  Cr
    [ 6,12, 7, 0],     # 25  Mn
    [ 6,12, 8, 0],     # 26  Fe
    [ 6,12, 9, 0],     # 27  Co
    [ 6,12,10, 0],     # 28  Ni
    [ 7,12,10, 0],     # 29  Cu
    [ 8,12,10, 0],     # 30  Zn
    [ 8,13,10, 0],     # 31  Ga
    [ 8,14,10, 0],     # 32  Ge
    [ 8,15,10, 0],     # 33  As
    [ 8,16,10, 0],     # 34  Se
    [ 8,17,10, 0],     # 35  Br
    [ 8,18,10, 0],     # 36  Kr
    [ 9,18,10, 0],     # 37  Rb
    [10,18,10, 0],     # 38  Sr
    [10,19,10, 0],     # 39  Y
    [10,18,12, 0],     # 40  Zr
    [10,18,13, 0],     # 41  Nb
    [ 8,18,16, 0],     # 42  Mo
    [ 8,18,17, 0],     # 43  Tc
    [ 8,18,18, 0],     # 44  Ru
    [ 8,18,19, 0],     # 45  Rh
    [ 8,18,20, 0],     # 46  Pd
    [ 9,18,20, 0],     # 47  Ag
    [10,18,20, 0],     # 48  Cd
    [10,19,20, 0],     # 49  In
    [10,20,20, 0],     # 50  Sn
    [10,21,20, 0],     # 51  Sb
    [10,22,20, 0],     # 52  Te
    [10,23,20, 0],     # 53  I
    [10,24,20, 0],     # 54  Xe
    [11,24,20, 0],     # 55  Cs
    [12,24,20, 0],     # 56  Ba
    [12,24,21, 0],     # 57  La
    [12,24,22, 0],     # 58  Ce
    [12,24,21, 2],     # 59  Pr
    [12,24,20, 4],     # 60  Nd
    [12,24,20, 5],     # 61  Pm
    [12,24,20, 6],     # 62  Sm
    [12,24,20, 7],     # 63  Eu
    [11,24,20, 9],     # 64  Gd
    [10,24,20,11],     # 65  Tb
    [10,24,20,12],     # 66  Dy
    [10,24,20,13],     # 67  Ho
    [10,24,20,14],     # 68  Er
    [11,24,20,14],     # 69  Tm
    [12,24,20,14],     # 70  Yb
    [12,25,20,14],     # 71  Lu
    [12,24,22,14],     # 72  Hf
    [12,24,23,14],     # 73  Ta
    [10,24,26,14],     # 74  W
    [10,24,27,14],     # 75  Re
    [10,24,28,14],     # 76  Os
    [10,24,29,14],     # 77  Ir
    [10,24,30,14],     # 78  Pt
    [11,24,30,14],     # 79  Au
    [12,24,30,14],     # 80  Hg
    [12,25,30,14],     # 81  Tl
    [12,26,30,14],     # 82  Pb
    [12,27,30,14],     # 83  Bi
    [12,28,30,14],     # 84  Po
    [12,29,30,14],     # 85  At
    [12,30,30,14],     # 86  Rn
    [13,30,30,14],     # 87  Fr
    [14,30,30,14],     # 88  Ra
    [14,30,31,14],     # 89  Ac
    [14,30,32,14],     # 90  Th
    [14,30,30,17],     # 91  Pa
    [14,30,30,18],     # 92  U
    [14,30,30,19],     # 93  Np
    [13,30,30,21],     # 94  Pu
    [12,30,30,23],     # 95  Am
    [12,30,30,24],     # 96  Cm
    [12,30,30,25],     # 97  Bk
    [12,30,30,26],     # 98  Cf
    [12,30,30,27],     # 99  Es
    [12,30,30,28],     #100  Fm
    [13,30,30,28],     #101  Md
    [14,30,30,28],     #102  No
    [14,30,31,28],     #103  Lr
    [14,30,32,28],     #104  Rf
    [14,30,33,28],     #105  Db
    [12,30,36,28],     #106  Sg
    [12,30,37,28],     #107  Bh
    [12,30,38,28],     #108  Hs
    [12,30,39,28],     #109  Mt
    [12,30,40,28],     #110  Ds
    [13,30,40,28],     #111  Rg
    [14,30,40,28],     #112  Cn
    [14,31,40,28],     #113  Nh
    [14,32,40,28],     #114  Fl
    [14,33,40,28],     #115  Mc
    [14,34,40,28],     #116  Lv
    [14,35,40,28],     #117  Ts
    [14,36,40,28],     #118  Og
]

# Non-relativistic spin-restricted spherically averaged exchange-only
# LDA a.k.a. Hartree-Fock-Slater configurations for use in atomic SAD
# calculations. Reference configurations from Phys. Rev. A 101, 012516
# (2020).
NRSRHFS_CONFIGURATION = [
    [ 0, 0, 0, 0],     #  0  GHOST
    [ 1, 0, 0, 0],     #  1  H
    [ 2, 0, 0, 0],     #  2  He
    [ 3, 0, 0, 0],     #  3  Li
    [ 4, 0, 0, 0],     #  4  Be
    [ 4, 1, 0, 0],     #  5  B
    [ 4, 2, 0, 0],     #  6  C
    [ 4, 3, 0, 0],     #  7  N
    [ 4, 4, 0, 0],     #  8  O
    [ 4, 5, 0, 0],     #  9  F
    [ 4, 6, 0, 0],     # 10  Ne
    [ 5, 6, 0, 0],     # 11  Na
    [ 6, 6, 0, 0],     # 12  Mg
    [ 6, 7, 0, 0],     # 13  Al
    [ 6, 8, 0, 0],     # 14  Si
    [ 6, 9, 0, 0],     # 15  P
    [ 6,10, 0, 0],     # 16  S
    [ 6,11, 0, 0],     # 17  Cl
    [ 6,12, 0, 0],     # 18  Ar
    [ 7,12, 0, 0],     # 19  K
    [ 8,12, 0, 0],     # 20  Ca
    [ 8,12, 1, 0],     # 21  Sc
    [ 8,12, 2, 0],     # 22  Ti
    [ 8,12, 3, 0],     # 23  V
    [ 8,12, 4, 0],     # 24  Cr
    [ 7,12, 6, 0],     # 25  Mn
    [ 7,12, 7, 0],     # 26  Fe
    [ 7,12, 8, 0],     # 27  Co
    [ 7,12, 9, 0],     # 28  Ni
    [ 7,12,10, 0],     # 29  Cu
    [ 8,12,10, 0],     # 30  Zn
    [ 8,13,10, 0],     # 31  Ga
    [ 8,14,10, 0],     # 32  Ge
    [ 8,15,10, 0],     # 33  As
    [ 8,16,10, 0],     # 34  Se
    [ 8,17,10, 0],     # 35  Br
    [ 8,18,10, 0],     # 36  Kr
    [ 9,18,10, 0],     # 37  Rb
    [10,18,10, 0],     # 38  Sr
    [10,18,11, 0],     # 39  Y
    [10,18,12, 0],     # 40  Zr
    [10,18,13, 0],     # 41  Nb
    [ 9,18,15, 0],     # 42  Mo
    [ 9,18,16, 0],     # 43  Tc
    [ 8,18,18, 0],     # 44  Ru
    [ 8,18,19, 0],     # 45  Rh
    [ 8,18,20, 0],     # 46  Pd
    [ 9,18,20, 0],     # 47  Ag
    [10,18,20, 0],     # 48  Cd
    [10,19,20, 0],     # 49  In
    [10,20,20, 0],     # 50  Sn
    [10,21,20, 0],     # 51  Sb
    [10,22,20, 0],     # 52  Te
    [10,23,20, 0],     # 53  I
    [10,24,20, 0],     # 54  Xe
    [11,24,20, 0],     # 55  Cs
    [12,24,20, 0],     # 56  Ba
    [12,24,20, 1],     # 57  La
    [12,24,20, 2],     # 58  Ce
    [12,24,20, 3],     # 59  Pr
    [12,24,20, 4],     # 60  Nd
    [12,24,20, 5],     # 61  Pm
    [12,24,20, 6],     # 62  Sm
    [12,24,20, 7],     # 63  Eu
    [12,24,20, 8],     # 64  Gd
    [12,24,20, 9],     # 65  Tb
    [12,24,20,10],     # 66  Dy
    [12,24,20,11],     # 67  Ho
    [12,24,20,12],     # 68  Er
    [12,24,20,13],     # 69  Tm
    [12,24,20,14],     # 70  Yb
    [12,24,21,14],     # 71  Lu
    [12,24,22,14],     # 72  Hf
    [12,24,23,14],     # 73  Ta
    [11,24,25,14],     # 74  W
    [11,24,26,14],     # 75  Re
    [10,24,28,14],     # 76  Os
    [10,24,29,14],     # 77  Ir
    [10,24,30,14],     # 78  Pt
    [11,24,30,14],     # 79  Au
    [12,24,30,14],     # 80  Hg
    [12,25,30,14],     # 81  Tl
    [12,26,30,14],     # 82  Pb
    [12,27,30,14],     # 83  Bi
    [12,28,30,14],     # 84  Po
    [12,29,30,14],     # 85  At
    [12,30,30,14],     # 86  Rn
    [13,30,30,14],     # 87  Fr
    [14,30,30,14],     # 88  Ra
    [14,30,30,15],     # 89  Ac
    [14,30,30,16],     # 90  Th
    [14,30,30,17],     # 91  Pa
    [13,30,30,19],     # 92  U
    [13,30,30,20],     # 93  Np
    [13,30,30,21],     # 94  Pu
    [13,30,30,22],     # 95  Am
    [13,30,30,23],     # 96  Cm
    [13,30,30,24],     # 97  Bk
    [13,30,30,25],     # 98  Cf
    [13,30,30,26],     # 99  Es
    [13,30,30,27],     #100  Fm
    [13,30,30,28],     #101  Md
    [14,30,30,28],     #102  No
    [14,30,31,28],     #103  Lr
    [14,30,32,28],     #104  Rf
    [13,30,34,28],     #105  Db
    [12,30,36,28],     #106  Sg
    [12,30,37,28],     #107  Bh
    [12,30,38,28],     #108  Hs
    [12,30,39,28],     #109  Mt
    [12,30,40,28],     #110  Ds
    [13,30,40,28],     #111  Rg
    [14,30,40,28],     #112  Cn
    [14,31,40,28],     #113  Nh
    [14,32,40,28],     #114  Fl
    [14,33,40,28],     #115  Mc
    [14,34,40,28],     #116  Lv
    [14,35,40,28],     #117  Ts
    [14,36,40,28],     #118  Og
]

# This is No. of shells, not the atomic configurations
#     core       core+valence
# core+valence = lambda nuc, l: \
#            int(numpy.ceil(pyscf.lib.parameters.ELEMENTS[nuc][2][l]/(4*l+2.)))
N_CORE_SHELLS = [
    '0s0p0d0f',         #  0  GHOST
    '0s0p0d0f',         #  1  H
    '0s0p0d0f',         #  2  He
    '1s0p0d0f',         #  3  Li
    '1s0p0d0f',         #  4  Be
    '1s0p0d0f',         #  5  B
    '1s0p0d0f',         #  6  C
    '1s0p0d0f',         #  7  N
    '1s0p0d0f',         #  8  O
    '1s0p0d0f',         #  9  F
    '1s0p0d0f',         # 10  Ne
    '2s1p0d0f',         # 11  Na
    '2s1p0d0f',         # 12  Mg
    '2s1p0d0f',         # 13  Al
    '2s1p0d0f',         # 14  Si
    '2s1p0d0f',         # 15  P
    '2s1p0d0f',         # 16  S
    '2s1p0d0f',         # 17  Cl
    '2s1p0d0f',         # 18  Ar
    '3s2p0d0f',         # 19  K
    '3s2p0d0f',         # 20  Ca
    '3s2p0d0f',         # 21  Sc
    '3s2p0d0f',         # 22  Ti
    '3s2p0d0f',         # 23  V
    '3s2p0d0f',         # 24  Cr
    '3s2p0d0f',         # 25  Mn
    '3s2p0d0f',         # 26  Fe
    '3s2p0d0f',         # 27  Co
    '3s2p0d0f',         # 28  Ni
    '3s2p0d0f',         # 29  Cu
    '3s2p0d0f',         # 30  Zn
    '3s2p1d0f',         # 31  Ga
    '3s2p1d0f',         # 32  Ge
    '3s2p1d0f',         # 33  As
    '3s2p1d0f',         # 34  Se
    '3s2p1d0f',         # 35  Br
    '3s2p1d0f',         # 36  Kr
    '4s3p1d0f',         # 37  Rb
    '4s3p1d0f',         # 38  Sr
    '4s3p1d0f',         # 39  Y
    '4s3p1d0f',         # 40  Zr
    '4s3p1d0f',         # 41  Nb
    '4s3p1d0f',         # 42  Mo
    '4s3p1d0f',         # 43  Tc
    '4s3p1d0f',         # 44  Ru
    '4s3p1d0f',         # 45  Rh
    '4s3p1d0f',         # 46  Pd
    '4s3p1d0f',         # 47  Ag
    '4s3p1d0f',         # 48  Cd
    '4s3p2d0f',         # 49  In
    '4s3p2d0f',         # 50  Sn
    '4s3p2d0f',         # 51  Sb
    '4s3p2d0f',         # 52  Te
    '4s3p2d0f',         # 53  I
    '4s3p2d0f',         # 54  Xe
    '5s4p2d0f',         # 55  Cs
    '5s4p2d0f',         # 56  Ba
    '5s4p2d0f',         # 57  La
    '5s4p2d0f',         # 58  Ce
    '5s4p2d0f',         # 59  Pr
    '5s4p2d0f',         # 60  Nd
    '5s4p2d0f',         # 61  Pm
    '5s4p2d0f',         # 62  Sm
    '5s4p2d0f',         # 63  Eu
    '5s4p2d0f',         # 64  Gd
    '5s4p2d0f',         # 65  Tb
    '5s4p2d0f',         # 66  Dy
    '5s4p2d0f',         # 67  Ho
    '5s4p2d0f',         # 68  Er
    '5s4p2d0f',         # 69  Tm
    '5s4p2d0f',         # 70  Yb
    '5s4p2d1f',         # 71  Lu
    '5s4p2d1f',         # 72  Hf
    '5s4p2d1f',         # 73  Ta
    '5s4p2d1f',         # 74  W
    '5s4p2d1f',         # 75  Re
    '5s4p2d1f',         # 76  Os
    '5s4p2d1f',         # 77  Ir
    '5s4p2d1f',         # 78  Pt
    '5s4p2d1f',         # 79  Au
    '5s4p2d1f',         # 80  Hg
    '5s4p3d1f',         # 81  Tl
    '5s4p3d1f',         # 82  Pb
    '5s4p3d1f',         # 83  Bi
    '5s4p3d1f',         # 84  Po
    '5s4p3d1f',         # 85  At
    '5s4p3d1f',         # 86  Rn
    '6s5p3d1f',         # 87  Fr
    '6s5p3d1f',         # 88  Ra
    '6s5p3d1f',         # 89  Ac
    '6s5p3d1f',         # 90  Th
    '6s5p3d1f',         # 91  Pa
    '6s5p3d1f',         # 92  U
    '6s5p3d1f',         # 93  Np
    '6s5p3d1f',         # 94  Pu
    '6s5p3d1f',         # 95  Am
    '6s5p3d1f',         # 96  Cm
    '6s5p3d1f',         # 97  Bk
    '6s5p3d1f',         # 98  Cf
    '6s5p3d1f',         # 99  Es
    '6s5p3d1f',         #100  Fm
    '6s5p3d1f',         #101  Md
    '6s5p3d1f',         #102  No
    '6s5p3d2f',         #103  Lr
    '6s5p3d2f',         #104  Rf
    '6s5p3d2f',         #105  Db
    '6s5p3d2f',         #106  Sg
    '6s5p3d2f',         #107  Bh
    '6s5p3d2f',         #108  Hs
    '6s5p3d2f',         #109  Mt
    '6s5p3d2f',         #110  Ds
    '6s5p3d2f',         #111  Rg
    '6s5p3d2f',         #112  Cn
    '6s5p4d2f',         #113  Nh
    '6s5p4d2f',         #114  Fl
    '6s5p4d2f',         #115  Mc
    '6s5p4d2f',         #116  Lv
    '6s3p4d2f',         #117  Ts
    '6s3p4d2f',         #118  Og
]


N_CORE_VALENCE_SHELLS = [
    '0s0p0d0f',         #  0  GHOST
    '1s0p0d0f',         #  1  H
    '1s0p0d0f',         #  2  He
    '2s0p0d0f',         #  3  Li
    '2s0p0d0f',         #  4  Be
    '2s1p0d0f',         #  5  B
    '2s1p0d0f',         #  6  C
    '2s1p0d0f',         #  7  N
    '2s1p0d0f',         #  8  O
    '2s1p0d0f',         #  9  F
    '2s1p0d0f',         # 10  Ne
    '3s1p0d0f',         # 11  Na
    '3s1p0d0f',         # 12  Mg
    '3s2p0d0f',         # 13  Al
    '3s2p0d0f',         # 14  Si
    '3s2p0d0f',         # 15  P
    '3s2p0d0f',         # 16  S
    '3s2p0d0f',         # 17  Cl
    '3s2p0d0f',         # 18  Ar
    '4s2p0d0f',         # 19  K
    '4s2p0d0f',         # 20  Ca
    '4s2p1d0f',         # 21  Sc
    '4s2p1d0f',         # 22  Ti
    '4s2p1d0f',         # 23  V
    '4s2p1d0f',         # 24  Cr
    '4s2p1d0f',         # 25  Mn
    '4s2p1d0f',         # 26  Fe
    '4s2p1d0f',         # 27  Co
    '4s2p1d0f',         # 28  Ni
    '4s2p1d0f',         # 29  Cu
    '4s2p1d0f',         # 30  Zn
    '4s3p1d0f',         # 31  Ga
    '4s3p1d0f',         # 32  Ge
    '4s3p1d0f',         # 33  As
    '4s3p1d0f',         # 34  Se
    '4s3p1d0f',         # 35  Br
    '4s3p1d0f',         # 36  Kr
    '5s3p1d0f',         # 37  Rb
    '5s3p1d0f',         # 38  Sr
    '5s3p2d0f',         # 39  Y
    '5s3p2d0f',         # 40  Zr
    '5s3p2d0f',         # 41  Nb
    '5s3p2d0f',         # 42  Mo
    '5s3p2d0f',         # 43  Tc
    '5s3p2d0f',         # 44  Ru
    '5s3p2d0f',         # 45  Rh
    '4s3p2d0f',         # 46  Pd
    '5s3p2d0f',         # 47  Ag
    '5s3p2d0f',         # 48  Cd
    '5s4p2d0f',         # 49  In
    '5s4p2d0f',         # 50  Sn
    '5s4p2d0f',         # 51  Sb
    '5s4p2d0f',         # 52  Te
    '5s4p2d0f',         # 53  I
    '5s4p2d0f',         # 54  Xe
    '6s4p2d0f',         # 55  Cs
    '6s4p2d0f',         # 56  Ba
    '6s4p3d0f',         # 57  La
    '6s4p3d1f',         # 58  Ce
    '6s4p2d1f',         # 59  Pr
    '6s4p2d1f',         # 60  Nd
    '6s4p2d1f',         # 61  Pm
    '6s4p2d1f',         # 62  Sm
    '6s4p2d1f',         # 63  Eu
    '6s4p3d1f',         # 64  Gd
    '6s4p3d1f',         # 65  Tb
    '6s4p2d1f',         # 66  Dy
    '6s4p2d1f',         # 67  Ho
    '6s4p2d1f',         # 68  Er
    '6s4p2d1f',         # 69  Tm
    '6s4p2d1f',         # 70  Yb
    '6s4p3d1f',         # 71  Lu
    '6s4p3d1f',         # 72  Hf
    '6s4p3d1f',         # 73  Ta
    '6s4p3d1f',         # 74  W
    '6s4p3d1f',         # 75  Re
    '6s4p3d1f',         # 76  Os
    '6s4p3d1f',         # 77  Ir
    '6s4p3d1f',         # 78  Pt
    '6s4p3d1f',         # 79  Au
    '6s4p3d1f',         # 80  Hg
    '6s5p3d1f',         # 81  Tl
    '6s5p3d1f',         # 82  Pb
    '6s5p3d1f',         # 83  Bi
    '6s5p3d1f',         # 84  Po
    '6s5p3d1f',         # 85  At
    '6s5p3d1f',         # 86  Rn
    '7s5p3d1f',         # 87  Fr
    '7s5p3d1f',         # 88  Ra
    '7s5p4d1f',         # 89  Ac
    '7s5p4d1f',         # 90  Th
    '7s5p4d2f',         # 91  Pa
    '7s5p4d2f',         # 92  U
    '7s5p4d2f',         # 93  Np
    '7s5p3d2f',         # 94  Pu
    '7s5p3d2f',         # 95  Am
    '7s5p4d2f',         # 96  Cm
    '7s5p4d2f',         # 97  Bk
    '7s5p3d2f',         # 98  Cf
    '7s5p3d2f',         # 99  Es
    '7s5p3d2f',         #100  Fm
    '7s5p3d2f',         #101  Md
    '7s5p3d2f',         #102  No
    '7s5p4d2f',         #103  Lr
    '7s5p4d2f',         #104  Rf
    '7s5p4d2f',         #105  Db
    '7s5p4d2f',         #106  Sg
    '7s5p4d2f',         #107  Bh
    '7s5p4d2f',         #108  Hs
    '7s5p4d2f',         #109  Mt
    '7s5p4d2f',         #110  Ds
    '7s5p4d2f',         #111  Rg
    '7s5p4d2f',         #112  Cn
    '7s6p4d2f',         #113  Nh
    '7s6p4d2f',         #114  Fl
    '7s6p4d2f',         #115  Mc
    '7s6p4d2f',         #116  Lv
    '7s6p4d2f',         #117  Ts
    '7s6p4d2f',         #118  Og
]

chemcore_atm = [
    0, # ghost
    0,                                                                  0,
    0,  0,                                          1,  1,  1,  1,  1,  1,
    1,  1,                                          5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  9,  9,  9,  9,  9,  9,
    9,  9, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 18, 18, 18, 18, 18, 18,
    18, 18,
    # lanthanides
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 34, 34, 34, 34, 34, 34,
    34, 34,
    # actinides
    34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55, 55, 55, 55]

def chemcore(mol, spinorb=False):
    '''
    Set spinorb=True for GMP2, GCCSD, etc.
    For R/U ones, spinorb=False is fine.
    '''
    core = 0
    for a in range(mol.natm):
        atm_nelec = mol.atom_charge(a)
        atm_z = charge(mol.atom_symbol(a))
        ne_ecp = atm_z - atm_nelec
        ncore_ecp = ne_ecp // 2
        atm_ncore = chemcore_atm[atm_z]
        if ncore_ecp > atm_ncore:
            core += 0
        else:
            core += atm_ncore - ncore_ecp

    if spinorb:
        core *= 2
    return core

#def chemcore_list(mol):
#    ncore = chemcore(mol)
#    return list(range(ncore))


########################################
#
# Some functions to format atomic symbol
#
########################################

def _rm_digit(symb):
    if symb.isalpha():
        return symb
    else:
        return ''.join([i for i in symb if i.isalpha()])

_ELEMENTS_UPPER = {x.upper(): x for x in ELEMENTS}
_ELEMENTS_UPPER['GHOST'] = 'Ghost'

def charge(symb_or_chg):
    if isinstance(symb_or_chg, str):
        a = str(symb_or_chg.strip().upper())
        if (a[:5] == 'GHOST' or (a[0] == 'X' and a[:2] != 'XE')):
            return 0
        else:
            return ELEMENTS_PROTON[_rm_digit(a)]
    else:
        return symb_or_chg

def _symbol(symb_or_chg):
    if isinstance(symb_or_chg, str):
        return str(symb_or_chg)
    else:
        return ELEMENTS[symb_or_chg]

def _std_symbol(symb_or_chg):
    '''For a given atom symbol (lower case or upper case) or charge, return the
    standardized atom symbol (without the numeric prefix or suffix)
    '''
    if isinstance(symb_or_chg, str):
        symb_or_chg = str(symb_or_chg.upper())
        rawsymb = _rm_digit(symb_or_chg)
        if rawsymb in _ELEMENTS_UPPER:
            return _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 1 and symb_or_chg[0] == 'X' and symb_or_chg[:2] != 'XE':
            rawsymb = rawsymb[1:]  # Remove the prefix X
            return 'X-' + _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 5 and rawsymb[:5] == 'GHOST':
            rawsymb = rawsymb[5:]  # Remove the prefix GHOST
            return 'GHOST-' + _ELEMENTS_UPPER[rawsymb]
        else:
            raise RuntimeError('Unsupported atom symbol %s' % symb_or_chg)
    else:
        return ELEMENTS[symb_or_chg]

def _std_symbol_without_ghost(symb_or_chg):
    '''For a given atom symbol (lower case or upper case) or charge, return the
    standardized atom symbol
    '''
    if isinstance(symb_or_chg, str):
        symb_or_chg = str(symb_or_chg.upper())
        rawsymb = _rm_digit(symb_or_chg)
        if rawsymb in _ELEMENTS_UPPER:
            return _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 1 and symb_or_chg[0] == 'X' and symb_or_chg[:2] != 'XE':
            rawsymb = rawsymb[1:]  # Remove the prefix X
            return _ELEMENTS_UPPER[rawsymb]
        elif len(rawsymb) > 5 and rawsymb[:5] == 'GHOST':
            rawsymb = rawsymb[5:]  # Remove the prefix GHOST
            return _ELEMENTS_UPPER[rawsymb]
        else:
            raise RuntimeError('Unsupported atom symbol %s' % symb_or_chg)
    else:
        return ELEMENTS[symb_or_chg]

def _atom_symbol(symb_or_chg):
    '''For a given atom symbol (lower case or upper case) or charge, return the
    standardized atom symbol (with the numeric prefix or suffix)
    '''
    if isinstance(symb_or_chg, str):
        a = str(symb_or_chg.strip().upper())
        if a.isdigit():
            symb = ELEMENTS[int(a)]
        else:
            rawsymb = _rm_digit(a)
            if rawsymb not in _ELEMENTS_UPPER:  # likely a ghost atom
                if len(rawsymb) > 1 and a[0] == 'X' and a[:2] != 'XE':
                    rawsymb = rawsymb[1:]  # Remove the prefix X
                    # put hyphen between X prefix and the atomic symbol
                    if a[1].isalpha():
                        a = a[0] + '-' + a[1:]
                    else:
                        a = a[0] + '-' + a[2:]
                elif len(rawsymb) > 5 and rawsymb[:5] == 'GHOST':
                    rawsymb = rawsymb[5:]  # Remove the prefix GHOST
                    # put hyphen between Ghost prefix and the atomic symbol
                    if a[5].isalpha():
                        a = a[:5] + '-' + a[5:]
                    elif a[5] != '-':
                        a = a[:5] + '-' + a[6:]
                else:
                    raise RuntimeError('Unsupported atom symbol %s' % a)
            stdsymb = _ELEMENTS_UPPER[rawsymb]
            symb = a.replace(rawsymb, stdsymb)
    else:
        symb = ELEMENTS[symb_or_chg]
    return symb

def is_ghost_atom(symb_or_chg):
    if isinstance(symb_or_chg, (int, numpy.integer)):
        return symb_or_chg == 0
    elif 'GHOST' in symb_or_chg.upper():
        return True
    else:
        return symb_or_chg[0] == 'X' and symb_or_chg[:2].upper() != 'XE'
