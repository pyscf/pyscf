#!/usr/bin/env python

import unittest
import numpy
from pyscf import ao2mo


class KnownValues(unittest.TestCase):
    def test_restore8(self):
        n = 10
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
        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a1, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a4, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a8, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a1, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a4, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a8, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore(8, a1, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore(8, a4, n)))
        self.assertTrue(numpy.allclose(a8, ao2mo.restore(8, a8, n)))

    def test_restore4(self):
        n = 10
        np = n*(n+1)//2
        a4 = numpy.random.random((np,np))
        a1 = numpy.empty((n,n,n,n))
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

        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a1, n)))
        self.assertTrue(numpy.allclose(a1, ao2mo.restore(1, a4, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a1, n)))
        self.assertTrue(numpy.allclose(a4, ao2mo.restore(4, a4, n)))

if __name__ == '__main__':
    print('Full Tests for ao2mo.restore')
    unittest.main()

