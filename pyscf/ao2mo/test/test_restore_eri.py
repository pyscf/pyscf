#!/usr/bin/env python

import numpy
from pyscf import ao2mo

n = 20
np = n*(n+1)//2
a8 = numpy.random.random((np*(np+1)//2))
a4 = numpy.empty((np,np))
a1 = numpy.empty((n,n,n,n))
ij = 0
for i in range(np):
    for j in range(i+1):
        a4[j,i] = a4[i,j] = a8[ij]
        ij += 1
ij = 0
for i in range(n):
    for j in range(i+1):
        kl = 0
        for k in range(n):
            for l in range(k+1):
                a1[i,j,k,l] = \
                a1[j,i,k,l] = \
                a1[i,j,l,k] = \
                a1[j,i,l,k] = a4[ij,kl]
                kl += 1
        ij += 1

print('1->1', numpy.allclose(a1, ao2mo.restore(1, a1, n)))
print('4->1', numpy.allclose(a1, ao2mo.restore(1, a4, n)))
print('8->1', numpy.allclose(a1, ao2mo.restore(1, a8, n)))
print('1->4', numpy.allclose(a4, ao2mo.restore(4, a1, n)))
print('4->4', numpy.allclose(a4, ao2mo.restore(4, a4, n)))
print('8->4', numpy.allclose(a4, ao2mo.restore(4, a8, n)))
print('1->8', numpy.allclose(a8, ao2mo.restore(8, a1, n)))
print('4->8', numpy.allclose(a8, ao2mo.restore(8, a4, n)))
print('8->8', numpy.allclose(a8, ao2mo.restore(8, a8, n)))
