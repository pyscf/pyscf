#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import reduce
import unittest
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.x2c import sfx2c1e
from pyscf.x2c import sfx2c1e_grad

def _sqrt0(a):
    w, v = scipy.linalg.eigh(a)
    return numpy.dot(v*numpy.sqrt(w), v.conj().T)

def _invsqrt0(a):
    w, v = scipy.linalg.eigh(a)
    return numpy.dot(v/numpy.sqrt(w), v.conj().T)

def _sqrt1(a0, a1):
    '''Solving first order of x^2 = a'''
    w, v = scipy.linalg.eigh(a0)
    w = numpy.sqrt(w)
    a1 = reduce(numpy.dot, (v.conj().T, a1, v))
    x1 = a1 / (w[:,None] + w)
    x1 = reduce(numpy.dot, (v, x1, v.conj().T))
    return x1

def _invsqrt1(a0, a1):
    '''Solving first order of x^2 = a^{-1}'''
    w, v = scipy.linalg.eigh(a0)
    w = 1./numpy.sqrt(w)
    a1 = -reduce(numpy.dot, (v.conj().T, a1, v))
    x1 = numpy.einsum('i,ij,j->ij', w**2, a1, w**2) / (w[:,None] + w)
    x1 = reduce(numpy.dot, (v, x1, v.conj().T))
    return x1

def get_R(mol):
    s0 = mol.intor('int1e_ovlp')
    t0 = mol.intor('int1e_kin')
    s0sqrt = _sqrt0(s0)
    s0invsqrt = _invsqrt0(s0)
    x0 = get_x0(mol)
    c = lib.param.LIGHT_SPEED
    stild = s0 + reduce(numpy.dot, (x0.T, t0*(.5/c**2), x0))
    R = _invsqrt0(reduce(numpy.dot, (s0invsqrt, stild, s0invsqrt)))
    R = reduce(numpy.dot, (s0invsqrt, R, s0sqrt))
    return R

def get_r1(mol, atm_id, pos):
# See JCP 135 084114, Eq (34)
    c = lib.param.LIGHT_SPEED
    aoslices = mol.aoslice_by_atom()
    ish0, ish1, p0, p1 = aoslices[atm_id]
    s0 = mol.intor('int1e_ovlp')
    t0 = mol.intor('int1e_kin')
    s1all = mol.intor('int1e_ipovlp', comp=3)
    t1all = mol.intor('int1e_ipkin', comp=3)
    s1 = numpy.zeros_like(s0)
    t1 = numpy.zeros_like(t0)
    s1[p0:p1,:]  =-s1all[pos][p0:p1]
    s1[:,p0:p1] -= s1all[pos][p0:p1].T
    t1[p0:p1,:]  =-t1all[pos][p0:p1]
    t1[:,p0:p1] -= t1all[pos][p0:p1].T
    x0 = get_x0(mol)
    x1 = get_x1(mol, atm_id)[pos]
    sa0 = s0 + reduce(numpy.dot, (x0.T, t0*(.5/c**2), x0))
    sa1 = s1 + reduce(numpy.dot, (x0.T, t1*(.5/c**2), x0))
    sa1+= reduce(numpy.dot, (x1.T, t0*(.5/c**2), x0))
    sa1+= reduce(numpy.dot, (x0.T, t0*(.5/c**2), x1))

    s0_sqrt = _sqrt0(s0)
    s0_invsqrt = _invsqrt0(s0)
    s1_sqrt = _sqrt1(s0, s1)
    s1_invsqrt = _invsqrt1(s0, s1)
    R0_part = reduce(numpy.dot, (s0_invsqrt, sa0, s0_invsqrt))
    R1_part = (reduce(numpy.dot, (s0_invsqrt, sa1, s0_invsqrt)) +
               reduce(numpy.dot, (s1_invsqrt, sa0, s0_invsqrt)) +
               reduce(numpy.dot, (s0_invsqrt, sa0, s1_invsqrt)))
    R1  = reduce(numpy.dot, (s0_invsqrt, _invsqrt1(R0_part, R1_part), s0_sqrt))
    R1 += reduce(numpy.dot, (s1_invsqrt, _invsqrt0(R0_part), s0_sqrt))
    R1 += reduce(numpy.dot, (s0_invsqrt, _invsqrt0(R0_part), s1_sqrt))
    return R1

def get_h0_s0(mol):
    s = mol.intor_symmetric('int1e_ovlp')
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    w = mol.intor_symmetric('int1e_pnucp')
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    c = lib.param.LIGHT_SPEED
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    return h, m

def get_h1_s1(mol, ia):
    aoslices = mol.aoslice_by_atom()
    ish0, ish1, p0, p1 = aoslices[0]
    nao = mol.nao_nr()
    s1 = mol.intor('int1e_ipovlp', comp=3)
    t1 = mol.intor('int1e_ipkin', comp=3)
    v1 = mol.intor('int1e_ipnuc', comp=3)
    w1 = mol.intor('int1e_ipspnucsp', comp=12).reshape(3,4,nao,nao)[:,3]
    with mol.with_rinv_origin(mol.atom_coord(ia)):
        rinv1 = -8*mol.intor('int1e_iprinv', comp=3)
        prinvp1 = -8*mol.intor('int1e_ipsprinvsp', comp=12).reshape(3,4,nao,nao)[:,3]
    n2 = nao * 2
    h = numpy.zeros((3,n2,n2), dtype=v1.dtype)
    m = numpy.zeros((3,n2,n2), dtype=v1.dtype)
    rinv1[:,p0:p1,:] -= v1[:,p0:p1]
    rinv1 = rinv1 + rinv1.transpose(0,2,1).conj()
    prinvp1[:,p0:p1,:] -= w1[:,p0:p1]
    prinvp1 = prinvp1 + prinvp1.transpose(0,2,1).conj()

    s1ao = numpy.zeros_like(s1)
    t1ao = numpy.zeros_like(t1)
    s1ao[:,p0:p1,:] = -s1[:,p0:p1]
    s1ao[:,:,p0:p1]+= -s1[:,p0:p1].transpose(0,2,1)
    t1ao[:,p0:p1,:] = -t1[:,p0:p1]
    t1ao[:,:,p0:p1]+= -t1[:,p0:p1].transpose(0,2,1)

    c = lib.param.LIGHT_SPEED
    h[:,:nao,:nao] = rinv1
    h[:,:nao,nao:] = t1ao
    h[:,nao:,:nao] = t1ao
    h[:,nao:,nao:] = prinvp1 * (.25/c**2) - t1ao
    m[:,:nao,:nao] = s1ao
    m[:,nao:,nao:] = t1ao * (.5/c**2)
    return h, m

def get_x0(mol):
    c = lib.param.LIGHT_SPEED
    h0, s0 = get_h0_s0(mol)
    e, c = scipy.linalg.eigh(h0, s0)
    nao = mol.nao_nr()
    cl = c[:nao,nao:]
    cs = c[nao:,nao:]
    x0 = scipy.linalg.solve(cl.T, cs.T).T
    return x0

def get_x1(mol, ia):
    h0, s0 = get_h0_s0(mol)
    h1, s1 = get_h1_s1(mol, ia)
    e0, c0 = scipy.linalg.eigh(h0, s0)
    nao = mol.nao_nr()
    cl0 = c0[:nao,nao:]
    cs0 = c0[nao:,nao:]
    x0 = scipy.linalg.solve(cl0.T, cs0.T).T
    h1 = numpy.einsum('pi,xpq,qj->xij', c0.conj(), h1, c0[:,nao:])
    s1 = numpy.einsum('pi,xpq,qj->xij', c0.conj(), s1, c0[:,nao:])
    epi = e0[:,None] - e0[nao:]
    degen_mask = abs(epi) < 1e-7
    epi[degen_mask] = 1e200
    c1 = (h1 - s1 * e0[nao:]) / -epi
    c1[:,degen_mask] = -.5 * s1[:,degen_mask]
    c1 = numpy.einsum('pq,xqi->xpi', c0, c1)
    cl1 = c1[:,:nao]
    cs1 = c1[:,nao:]
    x1 = [scipy.linalg.solve(cl0.T, (cs1[i] - x0.dot(cl1[i])).T).T
          for i in range(3)]
    return numpy.asarray(x1)

def setUpModule():
    global mol, mol1, mol2
    mol1 = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

    mol2 = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     ,-0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.   )],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

def tearDownModule():
    global mol, mol1, mol2
    del mol, mol1, mol2


class KnownValues(unittest.TestCase):
    def test_x1(self):
        with lib.light_speed(10) as c:
            x_1 = get_x0(mol1)
            x_2 = get_x0(mol2)
            x1_ref = (x_1 - x_2) / 0.0002 * lib.param.BOHR
            x1t = get_x1(mol, 0)
            self.assertAlmostEqual(abs(x1t[2]-x1_ref).max(), 0, 7)

            x0 = get_x0(mol)
            h0, s0 = get_h0_s0(mol)
            e0, c0 = scipy.linalg.eigh(h0, s0)
            get_h1_etc = sfx2c1e_grad._gen_first_order_quantities(mol, e0, c0, x0)
            x1 = get_h1_etc(0)[4]
            self.assertAlmostEqual(abs(x1-x1t).max(), 0, 9)

    def test_R1(self):
        with lib.light_speed(10) as c:
            R_1 = get_R(mol1)
            R_2 = get_R(mol2)
            R1_ref = (R_1 - R_2) / 0.0002 * lib.param.BOHR
            R1t = get_r1(mol, 0, 2)
            self.assertAlmostEqual(abs(R1t-R1_ref).max(), 0, 7)

            x0 = get_x0(mol)
            h0, s0 = get_h0_s0(mol)
            e0, c0 = scipy.linalg.eigh(h0, s0)
            get_h1_etc = sfx2c1e_grad._gen_first_order_quantities(mol, e0, c0, x0)
            R1 = get_h1_etc(0)[6][2]
            self.assertAlmostEqual(abs(R1-R1t).max(), 0, 9)

    def test_hfw(self):
        with lib.light_speed(10) as c:
            x2c_1 = sfx2c1e.SpinFreeX2C(mol1)
            x2c_2 = sfx2c1e.SpinFreeX2C(mol2)
            x2cobj = sfx2c1e.SpinFreeX2C(mol)
            fh_ref = (x2c_1.get_hcore() - x2c_2.get_hcore()) / 0.0002 * lib.param.BOHR
            fh = x2cobj.hcore_deriv_generator(deriv=1)
            self.assertAlmostEqual(abs(fh(0)[2] - fh_ref).max(), 0, 7)

            x2c_1.xuncontract = 0
            x2c_2.xuncontract = 0
            x2cobj.xuncontract =0
            fh_ref = (x2c_1.get_hcore() - x2c_2.get_hcore()) / 0.0002 * lib.param.BOHR
            fh = x2cobj.hcore_deriv_generator(deriv=1)
            self.assertAlmostEqual(abs(fh(0)[2] - fh_ref).max(), 0, 7)
            x2c_1.xuncontract = 1
            x2c_2.xuncontract = 1
            x2cobj.xuncontract =1

            x2c_1.approx = 'ATOM1E'
            x2c_2.approx = 'ATOM1E'
            x2cobj.approx = 'ATOM1E'
            fh_ref = (x2c_1.get_hcore() - x2c_2.get_hcore()) / 0.0002 * lib.param.BOHR
            fh = x2cobj.hcore_deriv_generator(deriv=1)
            self.assertAlmostEqual(abs(fh(0)[2] - fh_ref).max(), 0, 7)

if __name__ == "__main__":
    print("Full Tests for sfx2c1e gradients")
    unittest.main()
