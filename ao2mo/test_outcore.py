#!/usr/bin/env python

import ctypes
import unittest
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo

mol = gto.Mole()
mol.verbose = 0
mol.output = None#'out_h2o'
mol.atom.extend([
    ['O' , (0. , 0.     , 0.)],
    [1   , (0. , -0.757 , 0.587)],
    [1   , (0. , 0.757  , 0.587)] ])

mol.basis = 'cc-pvdz'
mol.build()
nao = mol.nao_nr()
naopair = nao*(nao+1)/2
numpy.random.seed(15)
mo = numpy.random.random((nao,nao))
mo = mo.copy(order='F')

c_atm = numpy.array(mol._atm, dtype=numpy.int32)
c_bas = numpy.array(mol._bas, dtype=numpy.int32)
c_env = numpy.array(mol._env)
natm = ctypes.c_int(c_atm.shape[0])
nbas = ctypes.c_int(c_bas.shape[0])

class KnowValues(unittest.TestCase):
    def test_nroutcore_grad(self):
        eri_ao = numpy.empty((3,nao,nao,nao,nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                kp = 0
                for k in range(mol.nbas):
                    lp = 0
                    for l in range(mol.nbas):
                        buf = gto.moleintor.getints_by_shell('cint2e_ip1_sph',
                                                             (i,j,k,l), c_atm,
                                                             c_bas, c_env, 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri_ao[:,ip:ip+di,jp:jp+dj,kp:kp+dk,lp:lp+dl] = buf
                        lp += dl
                    kp += dk
                jp += dj
            ip += di
        eriref = numpy.einsum('npjkl,pi->nijkl', eri_ao, mo)
        eriref = numpy.einsum('nipkl,pj->nijkl', eriref, mo)
        eriref = numpy.einsum('nijpl,pk->nijkl', eriref, mo)
        eriref = numpy.einsum('nijkp,pl->nijkl', eriref, mo)
        eriref = eriref.transpose(0,1,2,3,4)

        erifile = 'h2oeri.h5'
        ao2mo.outcore.full(mol, mo, erifile, dataname='eri_mo',
                           intor='cint2e_ip1_sph', aosym='s2kl', comp=3,
                           max_memory=10, ioblk_size=5, compact=False)
        feri = h5py.File(erifile)
        eri1 = numpy.array(feri['eri_mo']).reshape(3,nao,nao,nao,nao)
        self.assertTrue(numpy.allclose(eri1, eriref))


if __name__ == '__main__':
    print 'Full Tests for outcore'
    unittest.main()

