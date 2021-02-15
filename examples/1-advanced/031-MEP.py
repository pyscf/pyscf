#!/usr/bin/env python

'''
Molecular Electrostatic Potential (MEP)
See also http://www.cup.uni-muenchen.de/ch/compchem/pop/mep1.html
'''

import numpy
from pyscf import gto, scf, lib

mol = gto.M(atom='''
C   -0.9741771331,0.,-0.6301926171
C   -0.907244165,0.,0.7404218415
N   0.4034833469,0.,1.1675913068
C   1.124080815,0.,0.0676366081
N   0.3402990549,0.,-1.0525852362
H   0.6597356899,0.,-2.0102490569
H   -1.8002953135,0.,-1.3251503429
H   -1.7267332321,0.,1.4464261172
H   2.2048589411,0.,0.0167357941
''',
verbose = 4,
basis = '631g')
mf = scf.RHF(mol).run()

#
# 1. Define points where to evaluate MEP, eg some points in a cubic box
# Note the unit of the coordinates is atomic unit
#
xs = numpy.arange(-3., 3.01, .5)
ys = numpy.arange(-3., 3.01, .5)
zs = numpy.arange(-3., 3.01, .5)
points = lib.cartesian_prod([xs, ys, zs])

#
# 2. Nuclear potential at given points
#
Vnuc = 0
for i in range(mol.natm):
    r = mol.atom_coord(i)
    Z = mol.atom_charge(i)
    rp = r - points
    Vnuc += Z / numpy.einsum('xi,xi->x', rp, rp)**.5

#
# 3. Potential of electron density
#

# There are two ways to compute this potential.
# Method 1 (slow): looping over r_orig and evaluating 1/|r-r_orig| for each
# point.
dm = mf.make_rdm1()
Vele = []
for p in points:
    mol.set_rinv_orig_(p)
    Vele.append(numpy.einsum('ij,ij', mol.intor('int1e_rinv'), dm))
Vele = numpy.array(Vele)

# Method 2 (fast): Mimicing the points with delta function (steep S-type
# Gaussians) then calling the 3-center integral code to calculate the
# interaction between points and other AOs. Below, fakemol holds the delta
# functions.
from pyscf import df
fakemol = gto.fakemol_for_charges(points)
Vele = numpy.einsum('ijp,ij->p', df.incore.aux_e2(mol, fakemol), mf.make_rdm1())

#
# 4. MEP at each point
#
MEP = Vnuc - Vele

#
# 5. MEP force = -d/dr MEP = -d/dr Vnuc + d/dr Vele
#
Fnuc = 0
for i in range(mol.natm):
    r = mol.atom_coord(i)
    Z = mol.atom_charge(i)
    pr = points - r
    Fnuc += Z / (numpy.einsum('xi,xi->x', pr, pr)**1.5).reshape(-1,1) * pr

# Method 1 (slow)
Fele = []
for p in points:
    mol.set_rinv_orig_(p)
    # <i(x)| d/dr 1/|r-x| |j(x)> = <i(x)|-d/dx 1/|r-x| |j(x)>
    #                            = <d/dx i(x)| 1/|r-x| |j(x)> + <i(x)| 1/|r-x| |d/dx j(x)>
    d_rinv = mol.intor('int1e_iprinv', comp=3)
    d_rinv = d_rinv + d_rinv.transpose(0,2,1)
    Fele.append(numpy.einsum('xij,ij->x', d_rinv, dm))
Fele = numpy.array(Fele)

# Method 2 (fast)
from pyscf import df
fakemol = gto.fakemol_for_charges(points)
ints = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1')
ints = ints + ints.transpose(0,2,1,3)
Fele = numpy.einsum('xijp,ij->px', ints, dm)

F_MEP = Fnuc + Fele
print(F_MEP.shape)
