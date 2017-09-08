#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os
L_MAX      = 8
MAX_MEMORY = int(os.environ.get('PYSCF_MAX_MEMORY', 4000)) # MB
TMPDIR = os.environ.get('TMPDIR', '.')
TMPDIR = os.environ.get('PYSCF_TMPDIR', TMPDIR)

LIGHT_SPEED = 137.03599967994  #http://physics.nist.gov/cgi-bin/cuu/Value?alph
#LIGHT_SPEED = 137.0359895
ALPHA = 1./ LIGHT_SPEED
LIGHT_SPEED = float(os.environ.get('PYSCF_LIGHT_SPEED', LIGHT_SPEED))
# BOHR = .529 177 210 92(17) e-10m  #http://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
BOHR = 0.52917721092  # Angstroms

G_ELECTRON = 2.00231930436182   # http://physics.nist.gov/cgi-bin/cuu/Value?gem
E_MASS = 9.10938356e-31         # kg https://physics.nist.gov/cgi-bin/cuu/Value?me
PROTON_MASS = 1.672621898e-27   # kg https://physics.nist.gov/cgi-bin/cuu/Value?mp
BOHR_MAGNETON = 927.4009994e-26 # J/T http://physics.nist.gov/cgi-bin/cuu/Value?mub
NUC_MAGNETON = BOHR_MAGNETON * E_MASS / PROTON_MASS
PLANCK = 6.626070040e-34        # J*s http://physics.nist.gov/cgi-bin/cuu/Value?h
HARTREE2J = 4.359744650e-18     # J https://physics.nist.gov/cgi-bin/cuu/Value?hrj
HARTREE2EV = 27.21138602        # eV https://physics.nist.gov/cgi-bin/cuu/Value?threv
E_CHARGE = 1.6021766208e-19     # C https://physics.nist.gov/cgi-bin/cuu/Value?e
LIGHT_SPEED_SI = 299792458      # https://physics.nist.gov/cgi-bin/cuu/Value?c

OUTPUT_DIGITS = int(os.environ.get('PYSCF_OUTPUT_DIGITS', 5))
OUTPUT_COLS   = int(os.environ.get('PYSCF_OUTPUT_COLS', 5))

ANGULAR = 'spdfghij'
ANGULARMAP = {'s': 0,
              'p': 1,
              'd': 2,
              'f': 3,
              'g': 4,
              'h': 5,
              'i': 6,
              'j': 7}

REAL_SPHERIC = (
    ('',), \
    ('x', 'y', 'z'), \
    ('xy', 'yz', 'z^2', 'xz', 'x2-y2',), \
    ('y^3', 'xyz', 'yz^2', 'z^3', 'xz^2', 'zx^2', 'x^3'), \
    ('-4', '-3', '-2', '-1', ' 0', ' 1', ' 2', ' 3', ' 4'),
    ('-5', '-4', '-3', '-2', '-1', ' 0', ' 1', ' 2', ' 3', ' 4', ' 5'),
    ('-6', '-5', '-4', '-3', '-2', '-1', ' 0', ' 1', ' 2', ' 3', ' 4', ' 5',' 6'),
)

ELEMENTS = ('GHOST',
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
)
ELEMENTS_PROTON = NUC = dict([(x,i) for i,x in enumerate(ELEMENTS)])

VERBOSE_DEBUG  = 5
VERBOSE_INFO   = 4
VERBOSE_NOTICE = 3
VERBOSE_WARN   = 2
VERBOSE_ERR    = 1
VERBOSE_QUIET  = 0
VERBOSE_CRIT   = -1
VERBOSE_ALERT  = -2
VERBOSE_PANIC  = -3
TIMER_LEVEL    = VERBOSE_DEBUG

