#!/usr/bin/env python

import ctypes
import unittest
from functools import reduce
import tempfile
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo

mol = gto.Mole()
mol.verbose = 0
mol.output = None#'out_h2o'
mol.atom = '''
      o     0    0.       0
      h     0    -0.757   0.587
      h     0    0.757    0.587'''

mol.basis = 'cc-pvdz'
mol.build()
nao = mol.nao_nr()

class KnowValues(unittest.TestCase):
    def test_incore(self):
        from pyscf.scf import _vhf
        numpy.random.seed(15)
        nmo = 12
        mo = numpy.random.random((nao,nmo))
        eri = mol.intor('int2e_sph', aosym='s8')
        eriref = ao2mo.restore(1, eri, nao)
        eriref = numpy.einsum('pjkl,pi->ijkl', eriref, mo)
        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
        eriref = numpy.einsum('ijpl,pk->ijkl', eriref, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)

        eri1 = ao2mo.incore.full(ao2mo.restore(8,eri,nao), mo)
        self.assertTrue(numpy.allclose(ao2mo.restore(1,eri1,nmo), eriref))
        eri1 = ao2mo.incore.full(ao2mo.restore(4,eri,nao), mo, compact=False)
        self.assertTrue(numpy.allclose(eri1.reshape((nmo,)*4), eriref))

        eri1 = ao2mo.incore.general(eri, (mo[:,:2], mo[:,1:3], mo[:,:3], mo[:,2:5]))
        eri1 = eri1.reshape(2,2,3,3)
        self.assertTrue(numpy.allclose(eri1, eriref[:2,1:3,:3,2:5]))


if __name__ == '__main__':
    print('Full Tests for incore')
    unittest.main()


