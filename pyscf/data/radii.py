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

import numpy
from pyscf.lib.parameters import BOHR

unknown = 1.999999

#########################
# JCP 41, 3199 (1964); DOI:10.1063/1.1725697.
BRAGG = 1/BOHR * numpy.array((unknown,  # Ghost atom
        0.35,                                     1.40,             # 1s
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             # 2s2p
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             # 3s3p
        2.20, 1.80,                                                 # 4s
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, # 3d
                    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             # 4p
        2.35, 2.00,                                                 # 5s
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, # 4d
                    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             # 5p
        2.60, 2.15,                                                 # 6s
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   # La, Ce-Eu
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             # Gd, Tb-Lu
              1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, # 5d
                    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             # 6p
        1.80, 2.15,                                                 # 7s
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                    1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75))

# from Gerald Knizia's CtDftGrid, which is based on
#       http://en.wikipedia.org/wiki/Covalent_radius
# and
#       Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves,
#       Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago
#       Alvarez.  Covalent radii revisited. Dalton Trans., 2008, 2832-2838,
#       doi:10.1039/b801115j
COVALENT = 1/BOHR * numpy.array((unknown,  # Ghost atom
        0.31,                                     0.28,             # 1s
        1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,             # 2s2p
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             # 3s3p
        2.03, 1.76,                                                 # 4s
        1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22, # 3d
                    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,             # 4p
        2.20, 1.95,                                                 # 5s
        1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, # 4d
                    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,             # 5p
        2.44, 2.15,                                                 # 6s
        2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                   # La, Ce-Eu
        1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             # Gd, Tb-Lu
              1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, # 5d
                    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,             # 6p
        2.60, 2.21,                                                 # 7s
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69))


#
# vdw from ASE
#
# Van der Waals radii in [A] taken from
# http://www.webelements.com/periodicity/van_der_waals_radius/
# and the references given there.
# Additional source 5 from http://de.wikipedia.org/wiki/Van-der-Waals-Radius
# 
# 1. A. Bondi, J. Phys. Chem., 1964, 68, 441.
# 
# 2. L. Pauling, The Nature of the Chemical Bond,
#    Cornell University Press, USA, 1945.
# 
# 3. J.E. Huheey, E.A. Keiter, and R.L. Keiter in Inorganic Chemistry
#    Principles of Structure and Reactivity, 4th edition, HarperCollins,
#    New York, USA, 1993.W.W. Porterfield in Inorganic chemistry,
#    a unified approach, Addison Wesley Publishing Co.,
#    Reading Massachusetts, USA, 1984.
# 
# 4. A.M. James and M.P. Lord in Macmillan's Chemical and Physical Data,
#    Macmillan, London, UK, 1992.
# 
# 5. Manjeera Mantina, Adam C. Chamberlin, Rosendo Valero,
#    Christopher J. Cramer, Donald G. Truhlar Consistent van der Waals Radii
#    for the Whole Main Group. In J. Phys. Chem. A. 2009, 113, 5806-5812,
#    doi:10.1021/jp8111556
VDW = 1/BOHR * numpy.array((unknown,  # Ghost atom
    1.20,       #  1 H
    1.40,       #  2 He [1]
    1.82,       #  3 Li [1]
    1.53,       #  4 Be [5]
    1.92,       #  5 B  [5]
    1.70,       #  6 C  [1]
    1.55,       #  7 N  [1]
    1.52,       #  8 O  [1]
    1.47,       #  9 F  [1]
    1.54,       # 10 Ne [1]
    2.27,       # 11 Na [1]
    1.73,       # 12 Mg [1]
    1.84,       # 13 Al [5]
    2.10,       # 14 Si [1]
    1.80,       # 15 P  [1]
    1.80,       # 16 S  [1]
    1.75,       # 17 Cl [1]
    1.88,       # 18 Ar [1]
    2.75,       # 19 K  [1]
    2.31,       # 20 Ca [5]
    unknown,    # 21 Sc
    unknown,    # 22 Ti
    unknown,    # 23 V
    unknown,    # 24 Cr
    unknown,    # 25 Mn
    unknown,    # 26 Fe
    unknown,    # 27 Co
    1.63,       # 28 Ni [1]
    1.40,       # 29 Cu [1]
    1.39,       # 30 Zn [1]
    1.87,       # 31 Ga [1]
    2.11,       # 32 Ge [5]
    1.85,       # 33 As [1]
    1.90,       # 34 Se [1]
    1.85,       # 35 Br [1]
    2.02,       # 36 Kr [1]
    3.03,       # 37 Rb [5]
    2.49,       # 38 Sr [5]
    unknown,    # 39 Y
    unknown,    # 40 Zr
    unknown,    # 41 Nb
    unknown,    # 42 Mo
    unknown,    # 43 Tc
    unknown,    # 44 Ru
    unknown,    # 45 Rh
    1.63,       # 46 Pd [1]
    1.72,       # 47 Ag [1]
    1.58,       # 48 Cd [1]
    1.93,       # 49 In [1]
    2.17,       # 50 Sn [1]
    2.06,       # 51 Sb [5]
    2.06,       # 52 Te [1]
    1.98,       # 53 I  [1]
    2.16,       # 54 Xe [1]
    3.43,       # 55 Cs [5]
    2.49,       # 56 Ba [5]
    unknown,    # 57 La
    unknown,    # 58 Ce
    unknown,    # 59 Pr
    unknown,    # 60 Nd
    unknown,    # 61 Pm
    unknown,    # 62 Sm
    unknown,    # 63 Eu
    unknown,    # 64 Gd
    unknown,    # 65 Tb
    unknown,    # 66 Dy
    unknown,    # 67 Ho
    unknown,    # 68 Er
    unknown,    # 69 Tm
    unknown,    # 70 Yb
    unknown,    # 71 Lu
    unknown,    # 72 Hf
    unknown,    # 73 Ta
    unknown,    # 74 W
    unknown,    # 75 Re
    unknown,    # 76 Os
    unknown,    # 77 Ir
    1.75,       # 78 Pt [1]
    1.66,       # 79 Au [1]
    1.55,       # 80 Hg [1]
    1.96,       # 81 Tl [1]
    2.02,       # 82 Pb [1]
    2.07,       # 83 Bi [5]
    1.97,       # 84 Po [5]
    2.02,       # 85 At [5]
    2.20,       # 86 Rn [5]
    3.48,       # 87 Fr [5]
    2.83,       # 88 Ra [5]
    unknown,    # 89 Ac
    unknown,    # 90 Th
    unknown,    # 91 Pa
    1.86,       # 92 U [1]
    unknown,    # 93 Np
    unknown,    # 94 Pu
    unknown,    # 95 Am
    unknown,    # 96 Cm
    unknown,    # 97 Bk
    unknown,    # 98 Cf
    unknown,    # 99 Es
    unknown,    #100 Fm
    unknown,    #101 Md
    unknown,    #102 No
    unknown,    #103 Lr
))

# Universal Force Field (UFF)
# J. Am. Chem. Soc., 1992, 114 (25), pp 10024-10035
UFF = 1/BOHR * numpy.array((unknown,  # Ghost atom
    1.4430,     #  1  H
    1.8100,     #  2  He
    1.2255,     #  3  Li
    1.3725,     #  4  Be
    2.0415,     #  5  B
    1.9255,     #  6  C
    1.8300,     #  7  N
    1.7500,     #  8  O
    1.6820,     #  9  F
    1.6215,     # 10  Ne
    1.4915,     # 11  Na
    1.5105,     # 12  Mg
    2.2495,     # 13  Al
    2.1475,     # 14  Si
    2.0735,     # 15  P
    2.0175,     # 16  S
    1.9735,     # 17  Cl
    1.9340,     # 18  Ar
    1.9060,     # 19  K
    1.6995,     # 20  Ca
    1.6475,     # 21  Sc
    1.5875,     # 22  Ti
    1.5720,     # 23  V
    1.5115,     # 24  Cr
    1.4805,     # 25  Mn
    1.4560,     # 26  Fe
    1.4360,     # 27  Co
    1.4170,     # 28  Ni
    1.7475,     # 29  Cu
    1.3815,     # 30  Zn
    2.1915,     # 31  Ga
    2.1400,     # 32  Ge
    2.1150,     # 33  As
    2.1025,     # 34  Se
    2.0945,     # 35  Br
    2.0705,     # 36  Kr
    2.0570,     # 37  Rb
    1.8205,     # 38  Sr
    1.6725,     # 39  Y
    1.5620,     # 40  Zr
    1.5825,     # 41  Nb
    1.5260,     # 42  Mo
    1.4990,     # 43  Tc
    1.4815,     # 44  Ru
    1.4645,     # 45  Rh
    1.4495,     # 46  Pd
    1.5740,     # 47  Ag
    1.4240,     # 48  Cd
    2.2315,     # 49  In
    2.1960,     # 50  Sn
    2.2100,     # 51  Sb
    2.2350,     # 52  Te
    2.2500,     # 53  I
    2.2020,     # 54  Xe
    2.2585,     # 55  Cs
    1.8515,     # 56  Ba
    1.7610,     # 57  La
    1.7780,     # 58  Ce
    1.8030,     # 59  Pr
    1.7875,     # 60  Nd
    1.7735,     # 61  Pm
    1.7600,     # 62  Sm
    1.7465,     # 63  Eu
    1.6840,     # 64  Gd
    1.7255,     # 65  Tb
    1.7140,     # 66  Dy
    1.7045,     # 67  Ho
    1.6955,     # 68  Er
    1.6870,     # 69  Tm
    1.6775,     # 70  Yb
    1.8200,     # 71  Lu
    1.5705,     # 72  Hf
    1.5850,     # 73  Ta
    1.5345,     # 74  W
    1.4770,     # 75  Re
    1.5600,     # 76  Os
    1.4200,     # 77  Ir
    1.3770,     # 78  Pt
    1.6465,     # 79  Au
    1.3525,     # 80  Hg
    2.1735,     # 81  Tl
    2.1485,     # 82  Pb
    2.1850,     # 83  Bi
    2.3545,     # 84  Po
    2.3750,     # 85  At
    2.3825,     # 86  Rn
    2.4500,     # 87  Fr
    1.8385,     # 88  Ra
    1.7390,     # 89  Ac
    1.6980,     # 90  Th
    1.7120,     # 91  Pa
    1.6975,     # 92  U
    1.7120,     # 93  Np
    1.7120,     # 94  Pu
    1.6905,     # 95  Am
    1.6630,     # 96  Cm
    1.6695,     # 97  Bk
    1.6565,     # 98  Cf
    1.6495,     # 99  Es
    1.6430,     #100  Fm
    1.6370,     #101  Md
    1.6240,     #102  No
    1.6180,     #103  Lr
    unknown,    #104  Rf
    unknown,    #105  Db
    unknown,    #106  Sg
    unknown,    #107  Bh
    unknown,    #108  Hs
    unknown,    #109  Mt
    unknown,    #110  Ds
    unknown,    #111  Rg
    unknown,    #112  Cn
    unknown,    #113  Nh
    unknown,    #114  Fl
    unknown,    #115  Mc
    unknown,    #116  Lv
    unknown,    #117  Ts
    unknown,    #118  Og
))

# Allinger's MM3 radii
# From http://pcmsolver.readthedocs.io/en/latest/users/input.html
MM3 = 1/BOHR * numpy.array((unknown,  # Ghost atom
    1.62,       #  1  H
    1.53,       #  2  He
    2.55,       #  3  Li
    2.23,       #  4  Be
    2.15,       #  5  B
    2.04,       #  6  C
    1.93,       #  7  N
    1.82,       #  8  O
    1.71,       #  9  F
    1.60,       # 10  Ne
    2.70,       # 11  Na
    2.43,       # 12  Mg
    2.36,       # 13  Al
    2.29,       # 14  Si
    2.22,       # 15  P
    2.15,       # 16  S
    2.07,       # 17  Cl
    1.99,       # 18  Ar
    3.09,       # 19  K
    2.81,       # 20  Ca
    2.61,       # 21  Sc
    2.39,       # 22  Ti
    2.29,       # 23  V
    2.25,       # 24  Cr
    2.24,       # 25  Mn
    2.23,       # 26  Fe
    2.23,       # 27  Co
    2.22,       # 28  Ni
    2.26,       # 29  Cu
    2.29,       # 30  Zn
    2.46,       # 31  Ga
    2.44,       # 32  Ge
    2.36,       # 33  As
    2.29,       # 34  Se
    2.22,       # 35  Br
    2.15,       # 36  Kr
    3.25,       # 37  Rb
    3.00,       # 38  Sr
    2.71,       # 39  Y
    2.54,       # 40  Zr
    2.43,       # 41  Nb
    2.39,       # 42  Mo
    2.36,       # 43  Tc
    2.34,       # 44  Ru
    2.34,       # 45  Rh
    2.37,       # 46  Pd
    2.43,       # 47  Ag
    2.50,       # 48  Cd
    2.64,       # 49  In
    2.59,       # 50  Sn
    2.52,       # 51  Sb
    2.44,       # 52  Te
    2.36,       # 53  I
    2.28,       # 54  Xe
    3.44,       # 55  Cs
    3.07,       # 56  Ba
    2.78,       # 57  La
    2.74,       # 58  Ce
    2.73,       # 59  Pr
    2.73,       # 60  Nd
    2.72,       # 61  Pm
    2.71,       # 62  Sm
    2.94,       # 63  Eu
    2.71,       # 64  Gd
    2.70,       # 65  Tb
    2.69,       # 66  Dy
    2.67,       # 67  Ho
    2.67,       # 68  Er
    2.67,       # 69  Tm
    2.79,       # 70  Yb
    2.65,       # 71  Lu
    2.53,       # 72  Hf
    2.43,       # 73  Ta
    2.39,       # 74  W
    2.37,       # 75  Re
    2.35,       # 76  Os
    2.36,       # 77  Ir
    2.39,       # 78  Pt
    2.43,       # 79  Au
    2.53,       # 80  Hg
    2.59,       # 81  Tl
    2.74,       # 82  Pb
    2.66,       # 83  Bi
    2.59,       # 84  Po
    2.51,       # 85  At
    2.43,       # 86  Rn
    3.64,       # 87  Fr
    3.27,       # 88  Ra
    3.08,       # 89  Ac
    2.74,       # 90  Th
    2.64,       # 91  Pa
    2.52,       # 92  U
    2.52,       # 93  Np
    2.52,       # 94  Pu
    unknown,    # 95  Am
    unknown,    # 96  Cm
    unknown,    # 97  Bk
    unknown,    # 98  Cf
    unknown,    # 99  Es
    unknown,    #100  Fm
    unknown,    #101  Md
    unknown,    #102  No
    unknown,    #103  Lr
    2.73,       #104  Rf
    2.63,       #105  Db
    unknown,    #106  Sg
    1.62,       #107  Bh
    unknown,    #108  Hs
    unknown,    #109  Mt
    unknown,    #110  Ds
    unknown,    #111  Rg
    unknown,    #112  Cn
    unknown,    #113  Nh
    unknown,    #114  Fl
    unknown,    #115  Mc
    unknown,    #116  Lv
    unknown,    #117  Ts
    unknown,    #118  Og
))
del unknown

# flake8: noqa
