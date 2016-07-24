#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example has two parts.  The first part applies oscillated electric field
by modifying the 1-electron Hamiltonian.  The second part generate input
script for Jmol to plot the HOMOs.  Running jmol xxxx.spt can output 50
image files of HOMOs under different electric field.
'''

import numpy
from pyscf import gto, scf, tools

mol = gto.Mole() # Benzene
mol.atom = '''
     C    0.000000000000     1.398696930758     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
     H    0.000000000000     2.491406946734     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
  '''
mol.basis = '6-31g'
mol.build()

#
# Pass 1, generate all HOMOs with external field
#
N = 50 # 50 samples in one period of the oscillated field
mo_id = 20  # HOMO
dm_init_guess = [None]

def apply_field(E):
    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + numpy.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    dm_init_guess[0] = mf.make_rdm1()
    mo = mf.mo_coeff[:,mo_id]
    if mo[23] < -1e-5:  # To ensure that all MOs have same phase
        mo *= -1
    return mo

fields = numpy.sin((2*numpy.pi)/N * numpy.arange(N))*.2
mos = [apply_field((i+1e-5,0,0)) for i in fields]


#
# Pass 2, generate molden file and jmol input file
#
moldenfile = 'bz-homo.molden'
tools.molden.from_mo(mol, moldenfile, numpy.array(mos).T)

jmol_script = 'bz-homo.spt'
fspt = open(jmol_script,'w')
fspt.write('''
initialize;
set background [xffffff];
set frank off
set autoBond true;
set bondRadiusMilliAngstroms 66;
set bondTolerance 0.5;
set forceAutoBond false;
load %s
''' % moldenfile)
fspt.write('''
zoom 130;
rotate -20 z
rotate -60 x
axes

MO COLOR [xff0020] [x0060ff];
MO COLOR translucent 0.25;
MO fill noDots noMesh;
MO titleformat "";
''')
for i in range(N):
    fspt.write('MO %d cutoff 0.025;\n' % (i+1))
    fspt.write('write IMAGE 400 400 PNG 90 "bz-homo-%02d.png";\n' % (i+1))
fspt.close()
