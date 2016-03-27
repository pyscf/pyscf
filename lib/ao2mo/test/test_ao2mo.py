#!/usr/bin/env python

import os
import ctypes
import _ctypes
import unittest
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf import scf
from pyscf import gto
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.scf import _vhf

libao2mo1 = lib.load_library('libao2mo')

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
naopair = nao*(nao+1)//2
numpy.random.seed(15)
mo = numpy.random.random((nao,nao))
mo = mo.copy(order='F')

c_atm = numpy.array(mol._atm, dtype=numpy.int32)
c_bas = numpy.array(mol._bas, dtype=numpy.int32)
c_env = numpy.array(mol._env)
natm = ctypes.c_int(c_atm.shape[0])
nbas = ctypes.c_int(c_bas.shape[0])

def s2kl_to_s1(eri1, norb):
    eri2 = numpy.empty((norb,)*4)
    for i in range(norb):
        for j in range(norb):
            eri2[i,j] = lib.unpack_tril(eri1[i,j])
    return eri2
def s2ij_to_s1(eri1, norb):
    eri2 = numpy.empty((norb,)*4)
    ij = 0
    for i in range(norb):
        for j in range(i+1):
            eri2[i,j] = eri2[j,i] = eri1[ij]
            ij += 1
    return eri2
def f1pointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libao2mo1._handle, name))

class KnowValues(unittest.TestCase):
    def test_nr_transe2(self):
        eri_ao = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
        eri1 = ao2mo.restore(1, eri_ao, nao)
        eriref = numpy.einsum('ijpl,pk->ijkl', eri1, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)

        def e2drv(ftrans2, fmmm, eri1, eri2):
            libao2mo1.AO2MOnr_e2_drv(ftrans2, fmmm,
                                     eri2.ctypes.data_as(ctypes.c_void_p),
                                     eri1.ctypes.data_as(ctypes.c_void_p),
                                     mo.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(nao*nao), ctypes.c_int(nao),
                                     ctypes.c_int(0), ctypes.c_int(nao),
                                     ctypes.c_int(0), ctypes.c_int(nao),
                                     ctypes.c_void_p(), nbas)
            return eri2

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s1_iltj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri1, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s1_igtj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri1, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s2kl')
        fmmm = f1pointer('AO2MOmmm_nr_s2_iltj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        tril = numpy.tril_indices(nao)
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s2kl')
        fmmm = f1pointer('AO2MOmmm_nr_s2_igtj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        tril = numpy.tril_indices(nao)
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s2kl')
        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
        eri2 = numpy.zeros((nao,nao,naopair))
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        eri2 = s2kl_to_s1(eri2, nao)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s2kl')
        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
        eri1p = ao2mo.restore(4, eri1, nao)
        eri2 = numpy.zeros((naopair,naopair))
        libao2mo1.AO2MOnr_e2_drv(ftrans2, fmmm,
                                 eri2.ctypes.data_as(ctypes.c_void_p),
                                 eri1p.ctypes.data_as(ctypes.c_void_p),
                                 mo.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(naopair), ctypes.c_int(nao),
                                 ctypes.c_int(0), ctypes.c_int(nao),
                                 ctypes.c_int(0), ctypes.c_int(nao))
        self.assertTrue(numpy.allclose(eri2, ao2mo.restore(4,eriref,nao)))


###########################################################
        ftrans2 = f1pointer('AO2MOtrans_nr_s1_iltj')
        fmmm = ctypes.c_void_p()
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri1, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtrans_nr_s1_igtj')
        fmmm = ctypes.c_void_p()
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri1, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtrans_nr_s2_iltj')
        fmmm = ctypes.c_void_p()
        eri2 = numpy.zeros((nao,nao,nao,nao))
        tril = numpy.tril_indices(nao)
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtrans_nr_s2_igtj')
        fmmm = ctypes.c_void_p()
        eri2 = numpy.zeros((nao,nao,nao,nao))
        tril = numpy.tril_indices(nao)
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtrans_nr_s2_s2')
        fmmm = ctypes.c_void_p()
        eri2 = numpy.zeros((nao,nao,naopair))
        eri2 = e2drv(ftrans2, fmmm, eri1[:,:,tril[0],tril[1]].copy(), eri2)
        eri2 = s2kl_to_s1(eri2, nao)
        self.assertTrue(numpy.allclose(eri2, eriref))



if __name__ == '__main__':
    print('Full Tests for nr_ao2mo')
    unittest.main()
