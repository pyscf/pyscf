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
import pyscf.scf._vhf as _vhf

libcvhf = lib.load_library('libcvhf')
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
#    def test_nr_transe1_comp1(self):
#        eri_ao = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
#        eriref = ao2mo.restore(1, eri_ao, nao)
#        eriref = numpy.einsum('pjkl,pi->ijkl', eriref, mo)
#        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
#
#        def e1drv(intor, ftrans1, fill, fmmm, eri1):
#            libao2mo1.AO2MOnr_e1_drv(intor, ftrans1, fmmm,
#                                     eri1.ctypes.data_as(ctypes.c_void_p),
#                                     mo.ctypes.data_as(ctypes.c_void_p),
#                                     ctypes.c_int(0), nbas,
#                                     ctypes.c_int(0), ctypes.c_int(nao),
#                                     ctypes.c_int(0), ctypes.c_int(nao),
#                                     ctypes.c_int(1),
#                                     ctypes.c_void_p(0), ctypes.c_void_p(0),
#                                     c_atm.ctypes.data_as(ctypes.c_void_p), natm,
#                                     c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
#                                     c_env.ctypes.data_as(ctypes.c_void_p))
#            return eri1
#
#        intor = f1pointer('cint2e_sph')
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s4')
#        fill = f1pointer('AO2MOfill_nr_s4')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
#        eri1 = numpy.empty((naopair,naopair))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1).T.copy()
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s4')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_igtj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s4')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_iltj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2ij')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
#        eri1 = numpy.empty((nao,nao,naopair))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2ij_to_s1(eri1.transpose(2,0,1), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2kl')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
#        eri1 = numpy.empty((naopair,naopair))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1).T.copy()
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2kl')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_iltj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2kl')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_igtj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2kl')
#        fmmm = f1pointer('AO2MOmmm_nr_s1_iltj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2kl')
#        fmmm = f1pointer('AO2MOmmm_nr_s1_igtj')
#        eri1 = numpy.empty((naopair,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2kl_to_s1(eri1.transpose(1,2,0), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s1')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
#        eri1 = numpy.empty((nao,nao,naopair))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1)
#        eri1 = s2ij_to_s1(eri1.transpose(2,0,1), nao)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s1')
#        fmmm = f1pointer('AO2MOmmm_nr_s1_igtj')
#        eri1 = numpy.empty((nao,nao,nao,nao))
#        eri1 = e1drv(intor, ftrans1, fmmm, eri1).transpose(2,3,0,1)
#        self.assertTrue(numpy.allclose(ao2mo.restore(1, eri1, nao), eriref))
#
    def test_nr_transe2(self):
        eri_ao = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
        eri1 = ao2mo.restore(1, eri_ao, nao)
        eriref = numpy.einsum('ijpl,pk->ijkl', eri1, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)

        def e2drv(ftrans2, fmmm, eri2):
            libao2mo1.AO2MOnr_e2_drv(ftrans2, fmmm,
                                     eri2.ctypes.data_as(ctypes.c_void_p),
                                     eri1.ctypes.data_as(ctypes.c_void_p),
                                     mo.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_int(nao*nao), ctypes.c_int(nao),
                                     ctypes.c_int(0), ctypes.c_int(nao),
                                     ctypes.c_int(0), ctypes.c_int(nao))
            return eri2

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s1_iltj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s1_igtj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s2_iltj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s2_igtj')
        eri2 = numpy.zeros((nao,nao,nao,nao))
        eri2 = e2drv(ftrans2, fmmm, eri2)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
        eri2 = numpy.zeros((nao,nao,naopair))
        eri2 = e2drv(ftrans2, fmmm, eri2)
        eri2 = s2kl_to_s1(eri2, nao)
        self.assertTrue(numpy.allclose(eri2, eriref))

        ftrans2 = f1pointer('AO2MOtranse2_nr_s2kl')
        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
        eri1 = ao2mo.restore(4, eri1, nao)
        eri2 = numpy.zeros((naopair,naopair))
        libao2mo1.AO2MOnr_e2_drv(ftrans2, fmmm,
                                 eri2.ctypes.data_as(ctypes.c_void_p),
                                 eri1.ctypes.data_as(ctypes.c_void_p),
                                 mo.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(naopair), ctypes.c_int(nao),
                                 ctypes.c_int(0), ctypes.c_int(nao),
                                 ctypes.c_int(0), ctypes.c_int(nao))
        self.assertTrue(numpy.allclose(eri2, ao2mo.restore(4,eriref,nao)))

    def test_nr_transe1_compn(self):
        intor = f1pointer('cint2e_ip1_sph')
        eri_ao = numpy.empty((3,nao,nao,nao*nao))
        ip = 0
        for i in range(mol.nbas):
            jp = 0
            for j in range(mol.nbas):
                p0 = 0
                kp = 0
                for k in range(mol.nbas):
                    lp = 0
                    for l in range(mol.nbas):
                        buf = gto.moleintor.getints_by_shell('cint2e_ip1_sph',
                                                             (i,j,k,l), c_atm,
                                                             c_bas, c_env, 3)
                        di,dj,dk,dl = buf.shape[1:]
                        eri_ao[:,ip:ip+di,jp:jp+dj,p0:p0+dk*dl] = buf.reshape(3,di,dj,-1)
                        lp += dl
                        p0 += dk * dl
                    kp += dk
                jp += dj
            ip += di
        eriref = numpy.einsum('npjk,pi->nijk', eri_ao, mo)
        eriref = numpy.einsum('nipk,pj->nijk', eriref, mo)
        eriref = eriref.transpose(0,3,1,2)

        ftrans1 = f1pointer('AO2MOtranse1_nr_s1')
        fill = f1pointer('AO2MOfill_nr_s1')
        fmmm = f1pointer('AO2MOmmm_nr_s1_iltj')
        eri1 = numpy.empty((3,nao*nao,nao,nao))
        libao2mo1.AO2MOnr_e1_drv(intor, fill, ftrans1, fmmm,
                                 eri1.ctypes.data_as(ctypes.c_void_p),
                                 mo.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(0),
                                 ctypes.c_int(nbas.value*nbas.value),
                                 ctypes.c_int(nao*nao),
                                 ctypes.c_int(0), ctypes.c_int(nao),
                                 ctypes.c_int(0), ctypes.c_int(nao),
                                 ctypes.c_int(3),
                                 ctypes.c_void_p(0), ctypes.c_void_p(0),
                                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                 c_env.ctypes.data_as(ctypes.c_void_p))
        self.assertTrue(numpy.allclose(eri1, eriref))

#    def test_nr_transe1_frag(self):
#        eri_ao = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
#        eriref = ao2mo.restore(1, eri_ao, nao)
#        eriref = numpy.einsum('pjkl,pi->ijkl', eriref, mo)
#        eriref = numpy.einsum('ipkl,pj->ijkl', eriref, mo)
#        eriref = eriref.transpose(2,3,0,1)
#
#        intor = f1pointer('cint2e_sph')
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s1')
#        fmmm = f1pointer('AO2MOmmm_nr_s1_iltj')
#        ib1,ib2 = mol.nao_nr_range(1,1+3)
#        eri1 = numpy.empty((ib2-ib1,nao,2,4))
#        libao2mo1.AO2MOnr_e1_drv(intor, ftrans1, fmmm,
#                                 eri1.ctypes.data_as(ctypes.c_void_p),
#                                 mo.ctypes.data_as(ctypes.c_void_p),
#                                 ctypes.c_int(1), ctypes.c_int(3),
#                                 ctypes.c_int(0), ctypes.c_int(2),
#                                 ctypes.c_int(2), ctypes.c_int(4),
#                                 ctypes.c_int(1),
#                                 ctypes.c_void_p(0), ctypes.c_void_p(0),
#                                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
#                                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
#                                 c_env.ctypes.data_as(ctypes.c_void_p))
#        self.assertTrue(numpy.allclose(eri1, eriref[ib1:ib2,:,:2,2:6]))
#
#        intor = f1pointer('cint2e_sph')
#        ftrans1 = f1pointer('AO2MOtranse1_nr_s2ij')
#        fmmm = f1pointer('AO2MOmmm_nr_s2_igtj')
#        ib1,ib2 = mol.nao_nr_range(1,1+3)
#        eri1 = numpy.empty((ib2-ib1,nao,2,4))
#        libao2mo1.AO2MOnr_e1_drv(intor, ftrans1, fmmm,
#                                 eri1.ctypes.data_as(ctypes.c_void_p),
#                                 mo.ctypes.data_as(ctypes.c_void_p),
#                                 ctypes.c_int(1), ctypes.c_int(3),
#                                 ctypes.c_int(0), ctypes.c_int(2),
#                                 ctypes.c_int(2), ctypes.c_int(4),
#                                 ctypes.c_int(1),
#                                 ctypes.c_void_p(0), ctypes.c_void_p(0),
#                                 c_atm.ctypes.data_as(ctypes.c_void_p), natm,
#                                 c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
#                                 c_env.ctypes.data_as(ctypes.c_void_p))
#        self.assertTrue(numpy.allclose(eri1, eriref[ib1:ib2,:,:2,2:6]))

    def test_nr_transe1incore(self):
        eri_ao = _vhf.int2e_sph(mol._atm, mol._bas, mol._env)
        eriref = ao2mo.restore(1, eri_ao, nao)
        eriref = numpy.einsum('ijpl,pk->ijkl', eriref, mo)
        eriref = numpy.einsum('ijkp,pl->ijkl', eriref, mo)
        eriref = ao2mo.restore(4, eriref, nao)

        eri_ao = ao2mo.restore(8, eri_ao, nao)
        ftrans1 = f1pointer('AO2MOtranse1_incore_s8')
        fmmm = f1pointer('AO2MOmmm_nr_s2_s2')
        eri1 = numpy.empty((naopair,naopair))
        libao2mo1.AO2MOnr_e1incore_drv(ftrans1, fmmm,
                                       eri1.ctypes.data_as(ctypes.c_void_p),
                                       eri_ao.ctypes.data_as(ctypes.c_void_p),
                                       mo.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_int(0), ctypes.c_int(naopair),
                                       ctypes.c_int(nao),
                                       ctypes.c_int(0), ctypes.c_int(nao),
                                       ctypes.c_int(0), ctypes.c_int(nao))
        self.assertTrue(numpy.allclose(eri1, eriref))



if __name__ == '__main__':
    print('Full Tests for nr_ao2mo_o5')
    unittest.main()
