#!/usr/bin/env python

# Author: Junzi Liu <latrix1247@gmail.com>

'''
Calculate the effective electronic coupling based on single determinant diabatic
states.

Here the diabatic states are calcuated by mom-SCF(HF or DFT). And the direct
electronic coupling is obtained using Hartree-Fock formlism and it can use both
HF and DFT wave functions because of their single determinant form.  Within
mom-SCF, it is supposed to evaluate the electronic coupling between any two
states.
'''

import os
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import dft


mol = gto.Mole()
mol.verbose = 3
mol.atom = [
 ["C",  ( 0.000000,  0.418626, 0.000000)],
 ["H",  (-0.460595,  1.426053, 0.000000)],
 ["O",  ( 1.196516,  0.242075, 0.000000)],
 ["N",  (-0.936579, -0.568753, 0.000000)],
 ["H",  (-0.634414, -1.530889, 0.000000)],
 ["H",  (-1.921071, -0.362247, 0.000000)]
]
mol.basis = {"H": '6-311++g**',
             "O": '6-311++g**',
             "N": '6-311++g**',
             "C": '6-311++g**',
             }
mol.build()

# First state calculation with DFT
a = dft.UKS(mol)
a.xc='b3lyp'
# Store molecular orbital information into chkfile
a.chkfile='nh2cho_s0.chkfile'
a.scf()
mo0 = a.mo_coeff
occ0 = a.mo_occ

# Set initial ouoccupation pattern for excited state
occ0[0][11] = 0.0
occ0[0][12] = 1.0

# Second state calculation with DFT 
b = dft.UKS(mol)
b.xc='b3lyp'
# Store molecular orbital information into another chkfile
b.chkfile='nh2cho_s1.chkfile'
dm = b.make_rdm1(mo0, occ0)
# Use mom method to determine occupation number
scf.addons.mom_occ_(b, mo0, occ0)
b.scf(dm)

# Read the MO coefficients and occupation numbers from chkfile.
# So the calculation of electronic coupling can be carried out 
# standalone use chkfiles.
mo0 = scf.chkfile.load('nh2cho_s0.chkfile', 'scf/mo_coeff')
occ0 = scf.chkfile.load('nh2cho_s0.chkfile', 'scf/mo_occ')
mo1 = scf.chkfile.load('nh2cho_s1.chkfile', 'scf/mo_coeff')
occ1 = scf.chkfile.load('nh2cho_s1.chkfile', 'scf/mo_occ')

mf = scf.UHF(mol)
# Calculate overlap between two determiant <I|F>
s, x = mf.det_ovlp(mo0, mo1, occ0, occ1)

# Construct density matrix 
dm_s0 = mf.make_rdm1(mo0, occ0)
dm_s1 = mf.make_rdm1(mo1, occ1)
dm_01 = mf.make_asym_dm(mo0, mo1, occ0, occ1, x)

# One-electron part contrbution
h1e = mf.get_hcore(mol)
e1_s0 = numpy.einsum('ji,ji', h1e.conj(), dm_s0[0]+dm_s0[1])
e1_s1 = numpy.einsum('ji,ji', h1e.conj(), dm_s1[0]+dm_s1[1])
e1_01 = numpy.einsum('ji,ji', h1e.conj(), dm_01[0]+dm_01[1])

# Two-electron part contrbution. D_{IF} is asymmetric
vhf_s0 = mf.get_veff(mol, dm_s0)
vhf_s1 = mf.get_veff(mol, dm_s1)
vhf_01 = mf.get_veff(mol, dm_01, hermi=0)

# New total energy: <I|H|I>, <F|H|F>, <I|H|F>
e_s0 = mf.energy_elec(dm_s0, h1e, vhf_s0)
e_s1 = mf.energy_elec(dm_s1, h1e, vhf_s1)
e_01 = mf.energy_elec(dm_01, h1e, vhf_01)

print('The overlap between these two determiants is: %12.8f' % s)
print('E_1e(I),  E_JK(I),  E_tot(I):  %15.7f, %13.7f, %15.7f' % (e1_s0, e_s0[1], e_s0[0]))
print('E_1e(F),  E_JK(F),  E_tot(I):  %15.7f, %13.7f, %15.7f' % (e1_s1, e_s1[1], e_s1[0]))
print('E_1e(IF), E_JK(IF), E_tot(IF): %15.7f, %13.7f, %15.7f' % (e1_01, e_01[1], e_01[0]))
print(' <I|H|F> coupling is: %12.7f a.u.' % (e_01[0]*s))
print('(0.5*s*H_II+H_FF) is: %12.7f a.u.' % (0.5*s*(e_s0[0]+e_s1[0])))

# Calculate the effective electronic coupling
# V_{IF} = \frac{1}{1-S_{IF}^2}\left| H_{IF} - S_{IF}\frac{H_{II}+H_{FF}}{2} \right|
v01 = s*(e_01[0]-(e_s0[0]+e_s1[0])*0.5)/(1.0 - s*s)
print('The effective coupling is: %7.5f eV' % (numpy.abs(v01)*27.211385) )

#remove chkfile if necessary
os.remove('nh2cho_s0.chkfile')
os.remove('nh2cho_s1.chkfile')
