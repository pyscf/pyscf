#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
import ctypes
import numpy
import numpy as np
from pyscf.pbc import gto as pgto
import pyscf.gto
import pyscf.gto.moleintor


L = 1.5
n = 20
cl = pgto.Cell()
cl.build(
    h = [[L,0,0], [0,L,0], [0,0,L]],
    gs = [n,n,n],
    atom = 'He %f %f %f' % ((L/2.,)*3),
    basis = 'ccpvdz')

numpy.random.seed(1)
cl1 = pgto.Cell()
cl1.build(h = numpy.random.random((3,3)),
          gs = [n,n,n],
          atom ='''He .1 .0 .0
                   He .5 .1 .0
                   He .0 .5 .0
                   He .1 .3 .2''',
          basis = 'ccpvdz')

def intor_cross(intor, cell1, cell2, comp=1, hermi=0, kpts=None, kpt=None):
    r'''1-electron integrals from two cells like

    .. math::

        \langle \mu | intor | \nu \rangle, \mu \in cell1, \nu \in cell2
    '''
    if '2c2e' in intor:
        drv_name = 'GTOint2c2e'
    else:
        assert('2e' not in intor)
        drv_name = 'GTOint2c'

    drv = getattr(pyscf.gto.moleintor.libcgto, drv_name)
    fintor = getattr(pyscf.gto.moleintor.libcgto, intor)
    intopt = lib.c_null_ptr()

    if kpts is None:
        if kpt is not None:
            kpts_lst = np.reshape(kpt, (1,3))
        else:
            kpts_lst = np.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    atm, bas, env = pyscf.gto.conc_env(cell1._atm, cell1._bas, cell1._env,
                                       cell2._atm, cell2._bas, cell2._env)
    atm = np.asarray(atm, dtype=np.int32)
    bas = np.asarray(bas, dtype=np.int32)
    env = np.asarray(env, dtype=np.double)
    natm = len(atm)
    nbas = len(bas)
    shls_slice = (0, cell1.nbas, cell1.nbas, nbas)
    ao_loc = pyscf.gto.moleintor.make_loc(bas, intor)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    buf = np.empty((ni,nj,comp), order='F')
    c_buf = buf.ctypes.data_as(ctypes.c_void_p)
    c_comp = ctypes.c_int(comp)
    c_hermi = ctypes.c_int(hermi)
    c_shls_slice = (ctypes.c_int*4)(*(shls_slice[:4]))
    c_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_natm = ctypes.c_int(natm)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_nbas = ctypes.c_int(nbas)
    c_env = env.ctypes.data_as(ctypes.c_void_p)

    nimgs = np.max((cell1.nimgs, cell2.nimgs), axis=0)
    Ls = cell1.get_lattice_Ls(nimgs)
    expkL = np.exp(1j*np.dot(Ls, kpts_lst.T))

    xyz = cell2.atom_coords()
    ptr_coord = atm[cell1.natm:,pyscf.gto.PTR_COORD]
    ptr_coord = np.vstack((ptr_coord,ptr_coord+1,ptr_coord+2)).T.copy('C')
    nkpts = len(kpts_lst)
    out = [0] * nkpts
    for l,L in enumerate(Ls):
        env[ptr_coord] = xyz + L
        drv(fintor, c_buf, c_comp, c_hermi, c_shls_slice, c_ao_loc, intopt,
            c_atm, c_natm, c_bas, c_nbas, c_env)

        for k in range(nkpts):
            out[k] += buf * expkL[l,k]

    def trans(out):
        out = out.transpose(2,0,1)
        if hermi == lib.HERMITIAN:
            # GTOint2c fills the upper triangular of the F-order array.
            idx = np.triu_indices(ni)
            for i in range(comp):
                out[i,idx[1],idx[0]] = out[i,idx[0],idx[1]].conj()
        elif hermi == lib.ANTIHERMI:
            idx = np.triu_indices(ni)
            for i in range(comp):
                out[i,idx[1],idx[0]] = -out[i,idx[0],idx[1]].conj()
        elif hermi == lib.SYMMETRIC:
            idx = np.triu_indices(ni)
            for i in range(comp):
                out[i,idx[1],idx[0]] = out[i,idx[0],idx[1]]
        if comp == 1:
            out = out.reshape(ni,nj)
        return out

    if abs(kpts_lst).sum() < 1e-8:  # gamma_point
        out = [out[k].real.copy(order='F') for k in range(nkpts)]
    if kpts is None or np.shape(kpts) == (3,):
# A single k-point
        out = trans(out[0])
    else:
        out = [trans(out[k]) for k in range(nkpts)]
    return out

def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_nimgs(self):
        self.assertTrue(numpy.all(cl.get_nimgs(9e-1)==[3,3,3]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-2)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-4)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-6)==[4,4,4]))
        self.assertTrue(numpy.all(cl.get_nimgs(1e-9)==[5,5,5]))

    def test_Gv(self):
        a = cl1.get_Gv()
        self.assertAlmostEqual(finger(a), -99.791927068519939, 10)

    def test_SI(self):
        a = cl1.get_SI()
        self.assertAlmostEqual(finger(a), (16.506917823339265+1.6393578329869585j), 10)

    def test_bounding_sphere(self):
        self.assertTrue(numpy.all(cl1.get_bounding_sphere(4.5)==[12,8,7]))

    def test_mixed_basis(self):
        cl = pgto.Cell()
        cl.build(
            h = [[L,0,0], [0,L,0], [0,0,L]],
            gs = [n,n,n],
            atom = 'C1 %f %f %f; C2 %f %f %f' % ((L/2.,)*6),
            basis = {'C1':'ccpvdz', 'C2':'gthdzv'})

    def test_dumps_loads(self):
        cl1.loads(cl1.dumps())

    def test_get_lattice_Ls(self):
        self.assertEqual((1  , 3), cl1.get_lattice_Ls([0,0,0]).shape)
        self.assertEqual((7  , 3), cl1.get_lattice_Ls([1,1,1]).shape)
        self.assertEqual((33 , 3), cl1.get_lattice_Ls([2,2,2]).shape)
        self.assertEqual((123, 3), cl1.get_lattice_Ls([3,3,3]).shape)
        self.assertEqual((257, 3), cl1.get_lattice_Ls([4,4,4]).shape)
        self.assertEqual((515, 3), cl1.get_lattice_Ls([5,5,5]).shape)

    def test_ewald(self):
        cell = pgto.Cell()
        cell.unit = 'B'
        Lx = Ly = Lz = 5.
        cell.h = numpy.diag([Lx,Ly,Lz])
        cell.gs = numpy.array([20,20,20])
        cell.atom = [['He', (2, 0.5*Ly, 0.5*Lz)],
                     ['He', (3, 0.5*Ly, 0.5*Lz)]]
        cell.basis = {'He': [[0, (1.0, 1.0)]]}
        cell.verbose = 5
        cell.output = '/dev/null'
        cell.build()

        ew_cut = (20,20,20)
        self.assertAlmostEqual(cell.ewald(.05, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.1, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(0.2, ew_cut), -0.468640671931, 9)
        self.assertAlmostEqual(cell.ewald(1  , ew_cut), -0.468640671931, 9)

        def check(precision, eta_ref, ewald_ref):
            ew_eta0, ew_cut0 = cell.get_ewald_params(precision)
            self.assertAlmostEqual(ew_eta0, eta_ref)
            self.assertAlmostEqual(cell.ewald(ew_eta0, ew_cut0), ewald_ref, 9)
        check(0.001, 4.78124933741, -0.469112631739)
        check(1e-05, 3.70353981157, -0.468642153932)
        check(1e-07, 3.1300624293 , -0.468640678042)
        check(1e-09, 2.76045559201, -0.468640671959)

    def test_pbc_intor(self):
        numpy.random.seed(12)
        kpts = numpy.random.random((4,3))
        kpts[0] = 0
        s0 = cl1.pbc_intor('cint1e_ovlp_sph', hermi=0, kpts=kpts)
        self.assertAlmostEqual(finger(s0[0]), 492.28169269619838, 10)
        self.assertAlmostEqual(finger(s0[1]), 37.814902167003254-28.972661123811193j, 10)
        self.assertAlmostEqual(finger(s0[2]), -26.113652771749571-34.448650717254417j, 10)
        self.assertAlmostEqual(finger(s0[3]), 186.58447442096966+123.89977363497668j, 10)

        s1 = cl1.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts[0])
        self.assertAlmostEqual(finger(s1), 492.28169269619838, 10)


if __name__ == '__main__':
    print("Full Tests for pbc.gto.cell")
    unittest.main()

