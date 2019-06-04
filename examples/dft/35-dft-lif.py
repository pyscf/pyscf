#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
from __future__ import print_function
import sys
import numpy 
from pyscf import gto, dft

'''
A simple example to run DFT calculation.

See also pyscf/dft/libxc.py and pyscf/dft/xcfun.py for the complete list of
available XC functionals.
'''

mol = gto.Mole()
mol.build(
    atom = 'Li 0 0 0; F 0 0 2.0',  # in Angstrom
    basis = '631g', spin=0
)

mf = dft.UKS(mol)
mf.xc = 'svwn' # shorthand for slater,vwn
#mf.xc = 'bp86' # shorthand for b88,p86
#mf.xc = 'blyp' # shorthand for b88,lyp
#mf.xc = 'pbe' # shorthand for pbe,pbe
#mf.xc = 'lda,vwn_rpa'
#mf.xc = 'b97,pw91'
#mf.xc = 'pbe0'
#mf.xc = 'b3p86'
#mf.xc = 'wb97x'
#mf.xc = 'b3lyp'
mf.kernel()

# Orbital energies, Mulliken population etc.
mf.analyze()

rdm = mf.make_rdm1()
print(rdm.shape)

a2xyz = mol.atom_coords()
zcc = (a2xyz[0,2]+a2xyz[1,2])/2
h = 0.0025
yy = numpy.arange(-2.0, 2.0+h, h)
zz = numpy.arange(zcc-4.0, zcc+4.0+h, h)
coords = numpy.zeros((len(zz),3))

# AO values values on given grids
coords[:,2] = zz
c2ao = mol.eval_gto('GTOval_sph', coords)
cs2dens = numpy.einsum('ca,sab,cb->cs', c2ao, rdm, c2ao)

f = open('z2spin_dens_lif.txt', 'w')
for iz, (s2d, z) in enumerate(zip(cs2dens, coords[:,2])):
    print(z, *s2d, file=f)
f.close()
sys.exit(1)

f = open('yz2spin_dens_large.txt', 'w')
for iy, y in enumerate(yy):
    print(y)
    coords[:,1] = y
    coords[:,2] = zz

    # AO values values on given grids
    c2ao = mol.eval_gto('GTOval_sph', coords)
    cs2dens = numpy.einsum('ca,sab,cb->cs', c2ao, rdm, c2ao)

    for iz, (s2d, z) in enumerate(zip(cs2dens, coords[:,2])):
        print(iy, iz, *s2d, file=f)

f.close()


numpy.savetxt('yy.txt', yy)
numpy.savetxt('zz.txt', zz)

f = open('yz2spin_dens_large.txt', 'w')
for iy, y in enumerate(yy):
    print(y)
    coords[:,1] = y
    coords[:,2] = zz

    # AO values values on given grids
    c2ao = mol.eval_gto('GTOval_sph', coords)
    cs2dens = numpy.einsum('ca,sab,cb->cs', c2ao, rdm, c2ao)

    for iz, (s2d, z) in enumerate(zip(cs2dens, coords[:,2])):
        print(iy, iz, *s2d, file=f)

f.close()
