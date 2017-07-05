#!/usr/bin/env python

import numpy
from pyscf import gto

# Test numerical integration scheme JCP, 73, 5718

n_gauss = 30

# 2-center vec{r}/r^3 type integral

mol = gto.M(atom='H 0 1 2; H 1 .3 .1', basis='ccpvdz')
r0 = (.5, .4, .3)
mol.set_rinv_orig_(r0)
ipv = mol.intor('int1e_iprinv_sph', comp=3)
nablav = ipv + ipv.transpose(0,2,1)
t, w = numpy.polynomial.legendre.leggauss(n_gauss)
#t, w = numpy.polynomial.chebyshev.chebgauss(n_gauss)
a = (1+t)/(1-t) * .8
w *= 2/(1-t)**2 * .8

#a, w = numpy.polynomial.hermite.hermgauss(n_gauss)
#a, w = dft.radi.gauss_chebyshev(n_gauss)
#a, w = dft.radi.treutler_ahlrichs(n_gauss)
#a, w = dft.radi.mura_knowles(n_gauss)
#a, w = dft.radi.delley(n_gauss)

fakemol = gto.Mole()
ptr = 0
fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
ptr += 3
fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
p_cart2sph_factor = 0.488602511902919921
fakemol._env = numpy.hstack((r0, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
fakemol._built = True

pmol = mol + fakemol
i3c = pmol.intor('int3c1e_sph', shls_slice=(mol.nbas,pmol.nbas,0,mol.nbas,0,mol.nbas))
print(numpy.linalg.norm(i3c - nablav))



# 3-center 1/r type integral

mol.set_rinv_orig_(r0)
ipv = mol.intor('int3c1e_rinv_sph')
nablav = ipv
t, w = numpy.polynomial.legendre.leggauss(n_gauss)
a = (1+t)/(1-t) * .8
w *= 2/(1-t)**2 * .8

fakemol = gto.Mole()
ptr = 0
fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
ptr += 3
fakemol._bas = numpy.asarray([[0, 0, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
s_cart2sph_factor = 0.282094791773878143
fakemol._env = numpy.hstack((r0, a**2, w*2/numpy.pi**.5/s_cart2sph_factor))
fakemol._built = True

pmol = mol + fakemol
i3c = pmol.intor('int4c1e_sph', shls_slice=(mol.nbas,pmol.nbas, 0,mol.nbas,0,mol.nbas,0,mol.nbas))
nao = mol.nao_nr()
i3c = i3c.reshape(nao,nao,nao)
print(numpy.linalg.norm(i3c - nablav))



# 3-center vec{r}/r^3 type integral

mol.set_rinv_orig_(r0)
ipv = mol.intor('int3c1e_iprinv_sph', comp=3)
nablav = ipv + ipv.transpose(0,2,1,3) + ipv.transpose(0,2,3,1)
t, w = numpy.polynomial.legendre.leggauss(n_gauss)
a = (1+t)/(1-t) * .8
w *= 2/(1-t)**2 * .8

fakemol = gto.Mole()
ptr = 0
fakemol._atm = numpy.asarray([[0, ptr, 0, 0, 0, 0]], dtype=numpy.int32)
ptr += 3
fakemol._bas = numpy.asarray([[0, 1, n_gauss, 1, 0, ptr, ptr+n_gauss, 0]], dtype=numpy.int32)
p_cart2sph_factor = 0.488602511902919921
fakemol._env = numpy.hstack((r0, a**2, a**2*w*4/numpy.pi**.5/p_cart2sph_factor))
fakemol._built = True

pmol = mol + fakemol
i3c = pmol.intor('int4c1e_sph', shls_slice=(mol.nbas,pmol.nbas, 0,mol.nbas,0,mol.nbas,0,mol.nbas))
nao = mol.nao_nr()
i3c = i3c.reshape(3,nao,nao,nao)
print(numpy.linalg.norm(i3c - nablav))
