#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import tempfile
import numpy
import h5py
from pyscf import gto, scf, ao2mo

'''
Integral transformation for irregular operators
'''

mol = gto.M(
    verbose = 0,
    atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)] ],
    basis = 'ccpvdz',
)

mf = scf.RHF(mol)
e = mf.scf()
print('E = %.15g, ref -76.0267656731' % e)

#
# Given four MOs, compute the MO-integral gradients
#
gradtmp = tempfile.NamedTemporaryFile()
nocc = mol.nelectron // 2
nvir = len(mf.mo_energy) - nocc
co = mf.mo_coeff[:,:nocc]
cv = mf.mo_coeff[:,nocc:]
# Note the AO integrals cint2e_ip1_sph have 3 components (x,y,z) and only have
# permutation symmetry k>=l.
ao2mo.kernel(mol, (co,cv,co,cv), gradtmp.name, intor='cint2e_ip1_sph',
             aosym='s2kl')#, verbose=5)
feri = h5py.File(gradtmp.name, 'r')
grad = feri['eri_mo']
print('gradient integrals (d/dR i j|kl) have shape %s == (3,%dx%d,%dx%d)'
      % (str(grad.shape), nocc,nvir,nocc,nvir))


#
# Hessian integrals have 9 components
#       1       d/dX  d/dX
#       2       d/dX  d/dY
#       3       d/dX  d/dZ
#       4       d/dY  d/dX
#       5       d/dY  d/dY
#       6       d/dY  d/dZ
#       7       d/dZ  d/dX
#       8       d/dZ  d/dY
#       9       d/dZ  d/dZ
#
orb = mf.mo_coeff
hesstmp = tempfile.NamedTemporaryFile()
ao2mo.kernel(mol, orb, hesstmp.name, intor='cint2e_ipvip1_sph',
             dataname='hessints1', aosym='s4')
with ao2mo.load(hesstmp, 'hessints1') as eri:
    print('(d/dR i d/dR j| kl) have shape %s due to the 4-fold permutation '
          'symmetry i >= j, k >= l' % str(eri.shape))

ao2mo.kernel(mol, orb, hesstmp.name, intor='cint2e_ipip1_sph',
             dataname='hessints2', aosym='s2kl')
feri = h5py.File(hesstmp.name, 'r')
print('(d/dR d/dR i j| kl) have shape %s due to the 2-fold permutation '
      'symmetry k >= l' % str(feri['hessints2'].shape))
feri.close()

with ao2mo.load(ao2mo.kernel(mol, orb, hesstmp.name, intor='cint2e_ip1ip2_sph',
                             aosym='s1')) as eri:
    print('(d/dR i j|d/dR k l) have shape %s because there is no permutation '
          'symmetry' % str(eri.shape))
