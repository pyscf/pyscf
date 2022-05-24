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
from pyscf.x2c import sfx2c1e_hess

def _sqrt0(a):
    w, v = scipy.linalg.eigh(a)
    return numpy.dot(v*numpy.sqrt(w), v.conj().T)

def _invsqrt0(a):
    w, v = scipy.linalg.eigh(a)
    return numpy.dot(v/numpy.sqrt(w), v.conj().T)

def _sqrt1(a0, a1):
    '''Solving first order derivative of x^2 = a'''
    w, v = scipy.linalg.eigh(a0)
    w = numpy.sqrt(w)
    a1 = reduce(numpy.dot, (v.conj().T, a1, v))
    x1 = a1 / (w[:,None] + w)
    x1 = reduce(numpy.dot, (v, x1, v.conj().T))
    return x1

def _sqrt2(a0, a1i, a1j, a2ij):
    '''Solving second order derivative of x^2 = a'''
    w, v = scipy.linalg.eigh(a0)
    w = numpy.sqrt(w)
    a1i = reduce(numpy.dot, (v.conj().T, a1i, v))
    x1i = a1i / (w[:,None] + w)

    a1j = reduce(numpy.dot, (v.conj().T, a1j, v))
    x1j = a1j / (w[:,None] + w)

    a2ij = reduce(numpy.dot, (v.conj().T, a2ij, v))
    tmp = x1i.dot(x1j)
    a2ij -= tmp + tmp.conj().T
    x2 = a2ij / (w[:,None] + w)
    x2 = reduce(numpy.dot, (v, x2, v.conj().T))
    return x2

def _invsqrt1(a0, a1):
    '''Solving first order derivative of x^2 = a^{-1}'''
    w, v = scipy.linalg.eigh(a0)
    w = 1./numpy.sqrt(w)
    a1 = -reduce(numpy.dot, (v.conj().T, a1, v))
    x1 = numpy.einsum('i,ij,j->ij', w**2, a1, w**2) / (w[:,None] + w)
    x1 = reduce(numpy.dot, (v, x1, v.conj().T))
    return x1

def _invsqrt2(a0, a1i, a1j, a2ij):
    '''Solving first order derivative of x^2 = a^{-1}'''
    w, v = scipy.linalg.eigh(a0)
    w = 1./numpy.sqrt(w)
    a1i = reduce(numpy.dot, (v.conj().T, a1i, v))
    x1i = numpy.einsum('i,ij,j->ij', w**2, a1i, w**2) / (w[:,None] + w)

    a1j = reduce(numpy.dot, (v.conj().T, a1j, v))
    x1j = numpy.einsum('i,ij,j->ij', w**2, a1j, w**2) / (w[:,None] + w)

    a2ij = reduce(numpy.dot, (v.conj().T, a2ij, v))
    tmp = (a1i*w**2).dot(a1j)
    a2ij -= tmp + tmp.conj().T
    a2ij = -numpy.einsum('i,ij,j->ij', w**2, a2ij, w**2)
    tmp = x1i.dot(x1j)
    a2ij -= tmp + tmp.conj().T
    x2 = a2ij / (w[:,None] + w)
    x2 = reduce(numpy.dot, (v, x2, v.conj().T))
    return x2

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
    ish0, ish1, p0, p1 = aoslices[ia]
    nao = mol.nao_nr()
    s1 = mol.intor('int1e_ipovlp', comp=3)
    t1 = mol.intor('int1e_ipkin', comp=3)
    v1 = mol.intor('int1e_ipnuc', comp=3)
    w1 = mol.intor('int1e_ipspnucsp', comp=12).reshape(3,4,nao,nao)[:,3]
    with mol.with_rinv_origin(mol.atom_coord(ia)):
        z = -mol.atom_charge(ia)
        rinv1 = z*mol.intor('int1e_iprinv', comp=3)
        prinvp1 = z*mol.intor('int1e_ipsprinvsp', comp=12).reshape(3,4,nao,nao)[:,3]
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

def get_h2_s2(mol, ia, ja):
    aoslices = mol.aoslice_by_atom()
    ish0, ish1, i0, i1 = aoslices[ia]
    jsh0, jsh1, j0, j1 = aoslices[ja]
    nao = mol.nao_nr()
    s2aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    t2aa = mol.intor('int1e_ipipkin', comp=9).reshape(3,3,nao,nao)
    v2aa = mol.intor('int1e_ipipnuc', comp=9).reshape(3,3,nao,nao)
    w2aa = mol.intor('int1e_ipippnucp', comp=9).reshape(3,3,nao,nao)
    s2ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    t2ab = mol.intor('int1e_ipkinip', comp=9).reshape(3,3,nao,nao)
    v2ab = mol.intor('int1e_ipnucip', comp=9).reshape(3,3,nao,nao)
    w2ab = mol.intor('int1e_ippnucpip', comp=9).reshape(3,3,nao,nao)

    n2 = nao * 2
    h2 = numpy.zeros((3,3,n2,n2), dtype=v2aa.dtype)
    m2 = numpy.zeros((3,3,n2,n2), dtype=v2aa.dtype)
    s2ao = numpy.zeros_like(s2aa)
    t2ao = numpy.zeros_like(s2aa)
    v2ao = numpy.zeros_like(s2aa)
    w2ao = numpy.zeros_like(s2aa)
    if ia == ja:
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv2aa = z*mol.intor('int1e_ipiprinv', comp=9).reshape(3,3,nao,nao)
            rinv2ab = z*mol.intor('int1e_iprinvip', comp=9).reshape(3,3,nao,nao)
            prinvp2aa = z*mol.intor('int1e_ipipprinvp', comp=9).reshape(3,3,nao,nao)
            prinvp2ab = z*mol.intor('int1e_ipprinvpip', comp=9).reshape(3,3,nao,nao)
        s2ao[:,:,i0:i1      ] = s2aa[:,:,i0:i1      ]
        s2ao[:,:,i0:i1,j0:j1]+= s2ab[:,:,i0:i1,j0:j1]
        t2ao[:,:,i0:i1      ] = t2aa[:,:,i0:i1      ]
        t2ao[:,:,i0:i1,j0:j1]+= t2ab[:,:,i0:i1,j0:j1]
        v2ao -= rinv2aa + rinv2ab
        v2ao[:,:,i0:i1      ]+= v2aa[:,:,i0:i1      ]
        v2ao[:,:,i0:i1,j0:j1]+= v2ab[:,:,i0:i1,j0:j1]
        v2ao[:,:,i0:i1      ]+= rinv2aa[:,:,i0:i1] * 2
        v2ao[:,:,i0:i1      ]+= rinv2ab[:,:,i0:i1] * 2
        w2ao -= prinvp2aa + prinvp2ab
        w2ao[:,:,i0:i1      ]+= w2aa[:,:,i0:i1      ]
        w2ao[:,:,i0:i1,j0:j1]+= w2ab[:,:,i0:i1,j0:j1]
        w2ao[:,:,i0:i1      ]+= prinvp2aa[:,:,i0:i1] * 2
        w2ao[:,:,i0:i1      ]+= prinvp2ab[:,:,i0:i1] * 2
        s2ao = s2ao + s2ao.transpose(0,1,3,2)
        t2ao = t2ao + t2ao.transpose(0,1,3,2)
        v2ao = v2ao + v2ao.transpose(0,1,3,2)
        w2ao = w2ao + w2ao.transpose(0,1,3,2)
    else:
        s2ao[:,:,i0:i1,j0:j1] = s2ab[:,:,i0:i1,j0:j1]
        t2ao[:,:,i0:i1,j0:j1] = t2ab[:,:,i0:i1,j0:j1]
        v2ao[:,:,i0:i1,j0:j1] = v2ab[:,:,i0:i1,j0:j1]
        w2ao[:,:,i0:i1,j0:j1] = w2ab[:,:,i0:i1,j0:j1]
        zi = mol.atom_charge(ia)
        zj = mol.atom_charge(ja)
        with mol.with_rinv_at_nucleus(ia):
            shls_slice = (jsh0, jsh1, 0, mol.nbas)
            rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
            rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
            prinvp2aa = mol.intor('int1e_ipipprinvp', comp=9, shls_slice=shls_slice)
            prinvp2ab = mol.intor('int1e_ipprinvpip', comp=9, shls_slice=shls_slice)
            rinv2aa = zi * rinv2aa.reshape(3,3,j1-j0,nao)
            rinv2ab = zi * rinv2ab.reshape(3,3,j1-j0,nao)
            prinvp2aa = zi * prinvp2aa.reshape(3,3,j1-j0,nao)
            prinvp2ab = zi * prinvp2ab.reshape(3,3,j1-j0,nao)
            v2ao[:,:,j0:j1] += rinv2aa
            v2ao[:,:,j0:j1] += rinv2ab.transpose(1,0,2,3)
            w2ao[:,:,j0:j1] += prinvp2aa
            w2ao[:,:,j0:j1] += prinvp2ab.transpose(1,0,2,3)

        with mol.with_rinv_at_nucleus(ja):
            shls_slice = (ish0, ish1, 0, mol.nbas)
            rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
            rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
            prinvp2aa = mol.intor('int1e_ipipprinvp', comp=9, shls_slice=shls_slice)
            prinvp2ab = mol.intor('int1e_ipprinvpip', comp=9, shls_slice=shls_slice)
            rinv2aa = zj * rinv2aa.reshape(3,3,i1-i0,nao)
            rinv2ab = zj * rinv2ab.reshape(3,3,i1-i0,nao)
            prinvp2aa = zj * prinvp2aa.reshape(3,3,i1-i0,nao)
            prinvp2ab = zj * prinvp2ab.reshape(3,3,i1-i0,nao)
            v2ao[:,:,i0:i1] += rinv2aa
            v2ao[:,:,i0:i1] += rinv2ab
            w2ao[:,:,i0:i1] += prinvp2aa
            w2ao[:,:,i0:i1] += prinvp2ab
        s2ao = s2ao + s2ao.transpose(0,1,3,2)
        t2ao = t2ao + t2ao.transpose(0,1,3,2)
        v2ao = v2ao + v2ao.transpose(0,1,3,2)
        w2ao = w2ao + w2ao.transpose(0,1,3,2)

    c = lib.param.LIGHT_SPEED
    h2[:,:,:nao,:nao] = v2ao
    h2[:,:,:nao,nao:] = t2ao
    h2[:,:,nao:,:nao] = t2ao
    h2[:,:,nao:,nao:] = w2ao * (.25/c**2) - t2ao
    m2[:,:,:nao,:nao] = s2ao
    m2[:,:,nao:,nao:] = t2ao * (.5/c**2)
    return h2, m2

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
    c0[:,c0[1]<0] *= -1
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
#    c1[:,degen_mask] = -.5 * s1[:,degen_mask]
    c1 = numpy.einsum('pq,xqi->xpi', c0, c1)
    cl1 = c1[:,:nao]
    cs1 = c1[:,nao:]
    x1 = [scipy.linalg.solve(cl0.T, (cs1[i] - x0.dot(cl1[i])).T).T
          for i in range(3)]
    return numpy.asarray(x1)

def get_x2(mol, ia, ja):
    h0, s0 = get_h0_s0(mol)
    e0, c0 = scipy.linalg.eigh(h0, s0)
    c0[:,c0[1]<0] *= -1
    nao = mol.nao_nr()
    cl0 = c0[:nao,nao:]
    cs0 = c0[nao:,nao:]
    x0 = scipy.linalg.solve(cl0.T, cs0.T).T
    epq = e0[:,None] - e0
    degen_mask = abs(epq) < 1e-7
    epq[degen_mask] = 1e200

    h1i, s1i = get_h1_s1(mol, ia)
    h1i = numpy.einsum('pi,xpq,qj->xij', c0.conj(), h1i, c0)
    s1i = numpy.einsum('pi,xpq,qj->xij', c0.conj(), s1i, c0)
    c1i = (h1i - s1i * e0) / -epq
    c1i[:,degen_mask] = -.5 * s1i[:,degen_mask]
    e1i = h1i - s1i * e0
    e1i[:,~degen_mask] = 0
    c1i_ao = numpy.einsum('pq,xqi->xpi', c0, c1i[:,:,nao:])
    cl1i = c1i_ao[:,:nao]
    cs1i = c1i_ao[:,nao:]
    x1i = [scipy.linalg.solve(cl0.T, (cs1i[i] - x0.dot(cl1i[i])).T).T
           for i in range(3)]

    h1j, s1j = get_h1_s1(mol, ja)
    h1j = numpy.einsum('pi,xpq,qj->xij', c0.conj(), h1j, c0)
    s1j = numpy.einsum('pi,xpq,qj->xij', c0.conj(), s1j, c0)
    c1j = (h1j - s1j * e0) / -epq
    c1j[:,degen_mask] = -.5 * s1j[:,degen_mask]
    e1j = h1j - s1j * e0
    e1j[:,~degen_mask] = 0
    c1j_ao = numpy.einsum('pq,xqi->xpi', c0, c1j[:,:,nao:])
    cl1j = c1j_ao[:,:nao]
    cs1j = c1j_ao[:,nao:]
    x1j = [scipy.linalg.solve(cl0.T, (cs1j[i] - x0.dot(cl1j[i])).T).T
           for i in range(3)]

    h2, s2 = get_h2_s2(mol, ia, ja)
    h2 = numpy.einsum('pi,xypq,qj->xyij', c0.conj(), h2, c0[:,nao:])
    s2 = numpy.einsum('pi,xypq,qj->xyij', c0.conj(), s2, c0[:,nao:])
    f2 = h2 + numpy.einsum('xip,ypj->xyij', h1i, c1j[:,:,nao:])
    f2+= numpy.einsum('yip,xpj->xyij', h1j, c1i[:,:,nao:])
    f2-=(s2 + numpy.einsum('xip,ypj->xyij', s1i, c1j[:,:,nao:]) +
         numpy.einsum('yip,xpj->xyij', s1j, c1i[:,:,nao:])) * e0[nao:]
    f2-= numpy.einsum('xip,ypj->xyij', s1i + c1i, e1j[:,:,nao:])
    f2-= numpy.einsum('yip,xpj->xyij', s1j + c1j, e1i[:,:,nao:])
    c2 = f2 / -epq[:,nao:]
    s2pp = numpy.einsum('xip,ypj->xyij', s1i, c1j)
    s2pp+= numpy.einsum('yip,xpj->xyij', s1j, c1i)
    s2pp+= numpy.einsum('xpi,ypj->xyij', c1i, c1j)
    s2pp = s2pp + s2pp.transpose(0,1,3,2)
    s2pp = s2pp[:,:,:,nao:] + s2
    c2[:,:,degen_mask[:,nao:]] = -.5 * s2pp[:,:,degen_mask[:,nao:]]

    c2_ao = numpy.einsum('pq,xyqi->xypi', c0, c2)
    cl2 = c2_ao[:,:,:nao]
    cs2 = c2_ao[:,:,nao:]
    x2 = numpy.zeros((3,3,nao,nao))
    for i in range(3):
        for j in range(3):
            tmp = cs2[i,j] - x0.dot(cl2[i,j])
            tmp -= x1i[i].dot(cl1j[j])
            tmp -= x1j[j].dot(cl1i[i])
            x2[i,j] = scipy.linalg.solve(cl0.T, tmp.T).T
    return numpy.asarray(x2).reshape(3,3,nao,nao)


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

def get_r2(mol, ia, ja, ipos, jpos):
# See JCP 135 084114, Eq (34)
    c = lib.param.LIGHT_SPEED
    aoslices = mol.aoslice_by_atom()
    ish0, ish1, i0, i1 = aoslices[ia]
    jsh0, jsh1, j0, j1 = aoslices[ja]

    s0 = mol.intor('int1e_ovlp')
    t0 = mol.intor('int1e_kin')
    s1all = mol.intor('int1e_ipovlp', comp=3)
    t1all = mol.intor('int1e_ipkin', comp=3)
    s1i = numpy.zeros_like(s0)
    t1i = numpy.zeros_like(t0)
    s1i[i0:i1,:]  =-s1all[ipos][i0:i1]
    s1i[:,i0:i1] -= s1all[ipos][i0:i1].T
    t1i[i0:i1,:]  =-t1all[ipos][i0:i1]
    t1i[:,i0:i1] -= t1all[ipos][i0:i1].T

    s1j = numpy.zeros_like(s0)
    t1j = numpy.zeros_like(t0)
    s1j[j0:j1,:]  =-s1all[jpos][j0:j1]
    s1j[:,j0:j1] -= s1all[jpos][j0:j1].T
    t1j[j0:j1,:]  =-t1all[jpos][j0:j1]
    t1j[:,j0:j1] -= t1all[jpos][j0:j1].T

    x0 = get_x0(mol)
    x1i = get_x1(mol, ia)[ipos]
    x1j = get_x1(mol, ja)[jpos]
    x2 = get_x2(mol, ia, ja)[ipos,jpos]
    sa0 = s0 + reduce(numpy.dot, (x0.T, t0*(.5/c**2), x0))
    sa1i = s1i + reduce(numpy.dot, (x0.T, t1i*(.5/c**2), x0))
    sa1i+= reduce(numpy.dot, (x1i.T, t0*(.5/c**2), x0))
    sa1i+= reduce(numpy.dot, (x0.T, t0*(.5/c**2), x1i))
    sa1j = s1j + reduce(numpy.dot, (x0.T, t1j*(.5/c**2), x0))
    sa1j+= reduce(numpy.dot, (x1j.T, t0*(.5/c**2), x0))
    sa1j+= reduce(numpy.dot, (x0.T, t0*(.5/c**2), x1j))

    nao = mol.nao_nr()
    s2aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    t2aa = mol.intor('int1e_ipipkin', comp=9).reshape(3,3,nao,nao)
    s2ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    t2ab = mol.intor('int1e_ipkinip', comp=9).reshape(3,3,nao,nao)
    s2ao = numpy.zeros_like(s0)
    t2ao = numpy.zeros_like(t0)
    if ia == ja:
        s2ao[i0:i1      ] = s2aa[ipos,jpos,i0:i1      ]
        s2ao[i0:i1,j0:j1]+= s2ab[ipos,jpos,i0:i1,j0:j1]
        t2ao[i0:i1      ] = t2aa[ipos,jpos,i0:i1      ]
        t2ao[i0:i1,j0:j1]+= t2ab[ipos,jpos,i0:i1,j0:j1]
    else:
        s2ao[i0:i1,j0:j1] = s2ab[ipos,jpos,i0:i1,j0:j1]
        t2ao[i0:i1,j0:j1] = t2ab[ipos,jpos,i0:i1,j0:j1]
    s2ao = s2ao + s2ao.T
    t2ao = t2ao + t2ao.T
    sa2  = reduce(numpy.dot, (x2.T, t0*(.5/c**2), x0))
    sa2 += reduce(numpy.dot, (x1i.T, t1j*(.5/c**2), x0))
    sa2 += reduce(numpy.dot, (x0.T, t1i*(.5/c**2), x1j))
    sa2 += reduce(numpy.dot, (x1i.T, t0*(.5/c**2), x1j))
    sa2  = sa2 + sa2.T
    sa2 += s2ao + reduce(numpy.dot, (x0.T, t2ao*(.5/c**2), x0))

    s0_sqrt = _sqrt0(s0)
    s0_invsqrt = _invsqrt0(s0)
    s1i_sqrt = _sqrt1(s0, s1i)
    s1i_invsqrt = _invsqrt1(s0, s1i)
    s1j_sqrt = _sqrt1(s0, s1j)
    s1j_invsqrt = _invsqrt1(s0, s1j)
    s2_sqrt = _sqrt2(s0, s1i, s1j, s2ao)
    s2_invsqrt = _invsqrt2(s0, s1i, s1j, s2ao)

    R0_mid = reduce(numpy.dot, (s0_invsqrt, sa0, s0_invsqrt))
    R1i_mid = (reduce(numpy.dot, (s0_invsqrt, sa1i, s0_invsqrt)) +
               reduce(numpy.dot, (s1i_invsqrt, sa0, s0_invsqrt)) +
               reduce(numpy.dot, (s0_invsqrt, sa0, s1i_invsqrt)))
    R1j_mid = (reduce(numpy.dot, (s0_invsqrt, sa1j, s0_invsqrt)) +
               reduce(numpy.dot, (s1j_invsqrt, sa0, s0_invsqrt)) +
               reduce(numpy.dot, (s0_invsqrt, sa0, s1j_invsqrt)))
# second derivative of (s_invsqrt * sa * s_invsqrt), 9 terms
    R2_mid = (reduce(numpy.dot, (s0_invsqrt, sa0, s2_invsqrt)) +
              reduce(numpy.dot, (s1i_invsqrt, sa1j, s0_invsqrt)) +
              reduce(numpy.dot, (s0_invsqrt, sa1i, s1j_invsqrt)) +
              reduce(numpy.dot, (s1i_invsqrt, sa0, s1j_invsqrt)))
    R2_mid  = R2_mid + R2_mid.T
    R2_mid += reduce(numpy.dot, (s0_invsqrt, sa2, s0_invsqrt))
    R2_mid = _invsqrt2(R0_mid, R1i_mid, R1j_mid, R2_mid)
    R1i_mid = _invsqrt1(R0_mid, R1i_mid)
    R1j_mid = _invsqrt1(R0_mid, R1j_mid)
    R0_mid = _invsqrt0(R0_mid)

    R2  = reduce(numpy.dot, (s2_invsqrt, R0_mid, s0_sqrt))
    R2 += reduce(numpy.dot, (s1i_invsqrt, R1j_mid, s0_sqrt))
    R2 += reduce(numpy.dot, (s1i_invsqrt, R0_mid, s1j_sqrt))
    R2 += reduce(numpy.dot, (s1j_invsqrt, R1i_mid, s0_sqrt))
    R2 += reduce(numpy.dot, (s0_invsqrt, R2_mid, s0_sqrt))
    R2 += reduce(numpy.dot, (s0_invsqrt, R1i_mid, s1j_sqrt))
    R2 += reduce(numpy.dot, (s1j_invsqrt, R0_mid, s1i_sqrt))
    R2 += reduce(numpy.dot, (s0_invsqrt, R1j_mid, s1i_sqrt))
    R2 += reduce(numpy.dot, (s0_invsqrt, R0_mid, s2_sqrt))
    return R2

def setUpModule():
    global mol, mol1, mol2
    mol1 = gto.M(
        verbose = 0,
        atom = [["He" , (0. , 0.     , 0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

    mol2 = gto.M(
        verbose = 0,
        atom = [["He" , (0. , 0.     ,-0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

    mol = gto.M(
        verbose = 0,
        atom = [["He" , (0. , 0.     , 0.   )],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )

def tearDownModule():
    global mol, mol1, mol2
    del mol, mol1, mol2

class KnownValues(unittest.TestCase):
    def test_sqrt_second_order(self):
        with lib.light_speed(10) as c:
            nao = mol.nao_nr()
            aoslices = mol.aoslice_by_atom()
            p0, p1 = aoslices[0][2:]
            s1p1 = mol1.intor('int1e_ipovlp', comp=3)
            s1p2 = mol2.intor('int1e_ipovlp', comp=3)
            s1_1 = numpy.zeros((3,nao,nao))
            s1_1[:,p0:p1] = -s1p1[:,p0:p1]
            s1_1 = s1_1 + s1_1.transpose(0,2,1)
            s1_2 = numpy.zeros((3,nao,nao))
            s1_2[:,p0:p1] = -s1p2[:,p0:p1]
            s1_2 = s1_2 + s1_2.transpose(0,2,1)
            s2sqrt_ref = (_sqrt1(mol1.intor('int1e_ovlp'), s1_1[2]) -
                          _sqrt1(mol2.intor('int1e_ovlp'), s1_2[2])) / 0.0002 * lib.param.BOHR
            s2invsqrt_ref = (_invsqrt1(mol1.intor('int1e_ovlp'), s1_1[2]) -
                             _invsqrt1(mol2.intor('int1e_ovlp'), s1_2[2])) / 0.0002 * lib.param.BOHR

            s1p = mol.intor('int1e_ipovlp', comp=3)
            s1i = numpy.zeros((3,nao,nao))
            s1i[:,p0:p1] = -s1p[:,p0:p1]
            s1i = s1i + s1i.transpose(0,2,1)
            s2aap = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
            s2abp = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
            s2 = numpy.zeros((3,3,nao,nao))
            s2[:,:,p0:p1]        = s2aap[:,:,p0:p1]
            s2[:,:,p0:p1,p0:p1] += s2abp[:,:,p0:p1,p0:p1]
            s2 = s2 + s2.transpose(0,1,3,2)
            s2sqrt = _sqrt2(mol.intor('int1e_ovlp'), s1i[2], s1i[2], s2[2,2])
            s2invsqrt = _invsqrt2(mol.intor('int1e_ovlp'), s1i[2], s1i[2], s2[2,2])

            self.assertAlmostEqual(abs(s2sqrt-s2sqrt_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(s2invsqrt-s2invsqrt_ref).max(), 0, 7)

            p0, p1 = aoslices[1][2:]
            s1_1 = numpy.zeros((3,nao,nao))
            s1_1[:,p0:p1] = -s1p1[:,p0:p1]
            s1_1 = s1_1 + s1_1.transpose(0,2,1)
            s1_2 = numpy.zeros((3,nao,nao))
            s1_2[:,p0:p1] = -s1p2[:,p0:p1]
            s1_2 = s1_2 + s1_2.transpose(0,2,1)
            s2sqrt_ref = (_sqrt1(mol1.intor('int1e_ovlp'), s1_1[2]) -
                          _sqrt1(mol2.intor('int1e_ovlp'), s1_2[2])) / 0.0002 * lib.param.BOHR
            s2invsqrt_ref = (_invsqrt1(mol1.intor('int1e_ovlp'), s1_1[2]) -
                             _invsqrt1(mol2.intor('int1e_ovlp'), s1_2[2])) / 0.0002 * lib.param.BOHR
            q0, q1 = aoslices[0][2:]
            s1i = numpy.zeros((3,nao,nao))
            s1i[:,p0:p1] = -s1p[:,p0:p1]
            s1i = s1i + s1i.transpose(0,2,1)
            s1j = numpy.zeros((3,nao,nao))
            s1j[:,q0:q1] = -s1p[:,q0:q1]
            s1j = s1j + s1j.transpose(0,2,1)
            s2 = numpy.zeros((3,3,nao,nao))
            s2[:,:,p0:p1,q0:q1] = s2abp[:,:,p0:p1,q0:q1]
            s2 = s2 + s2.transpose(0,1,3,2)
            s2sqrt = _sqrt2(mol.intor('int1e_ovlp'), s1i[2], s1j[2], s2[2,2])
            s2invsqrt = _invsqrt2(mol.intor('int1e_ovlp'), s1i[2], s1j[2], s2[2,2])

            self.assertAlmostEqual(abs(s2sqrt-s2sqrt_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(s2invsqrt-s2invsqrt_ref).max(), 0, 7)

    def test_h2(self):
        with lib.light_speed(10) as c:
            h1_1, s1_1 = get_h1_s1(mol1, 0)
            h1_2, s1_2 = get_h1_s1(mol2, 0)
            h2_ref = (h1_1[2] - h1_2[2]) / 0.0002 * lib.param.BOHR
            s2_ref = (s1_1[2] - s1_2[2]) / 0.0002 * lib.param.BOHR

            h2, s2 = get_h2_s2(mol, 0, 0)
            self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(s2[2,2]-s2_ref).max(), 0, 7)

            h1_1, s1_1 = get_h1_s1(mol1, 1)
            h1_2, s1_2 = get_h1_s1(mol2, 1)
            h2_ref = (h1_1[2] - h1_2[2]) / 0.0002 * lib.param.BOHR
            s2_ref = (s1_1[2] - s1_2[2]) / 0.0002 * lib.param.BOHR

            h2, s2 = get_h2_s2(mol, 1, 0)
            self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)
            self.assertAlmostEqual(abs(s2[2,2]-s2_ref).max(), 0, 7)

    def test_x2(self):
        with lib.light_speed(10) as c:
            x1_1 = get_x1(mol1, 0)
            x1_2 = get_x1(mol2, 0)
            x2_ref = (x1_1[2] - x1_2[2]) / 0.0002 * lib.param.BOHR
            x2 = get_x2(mol, 0, 0)
            self.assertAlmostEqual(abs(x2[2,2]-x2_ref).max(), 0, 7)

            x1_1 = get_x1(mol1, 1)
            x1_2 = get_x1(mol2, 1)
            x2_ref = (x1_1[2] - x1_2[2]) / 0.0002 * lib.param.BOHR
            x2 = get_x2(mol, 1, 0)
            self.assertAlmostEqual(abs(x2[2,2]-x2_ref).max(), 0, 7)

    def test_r2(self):
        with lib.light_speed(10) as c:
            r1_1 = get_r1(mol1, 0, 2)
            r1_2 = get_r1(mol2, 0, 2)
            r2_ref = (r1_1 - r1_2) / 0.0002 * lib.param.BOHR
            r2 = get_r2(mol, 0, 0, 2, 2)
            self.assertAlmostEqual(abs(r2-r2_ref).max(), 0, 7)

            r1_1 = get_r1(mol1, 1, 2)
            r1_2 = get_r1(mol2, 1, 2)
            r2_ref = (r1_1 - r1_2) / 0.0002 * lib.param.BOHR
            r2 = get_r2(mol, 1, 0, 2, 2)
            self.assertAlmostEqual(abs(r2-r2_ref).max(), 0, 7)

    def test_hfw2(self):
        h1_deriv_1 = sfx2c1e_grad.gen_sf_hfw(mol1, approx='1E')
        h1_deriv_2 = sfx2c1e_grad.gen_sf_hfw(mol2, approx='1E')
        h2_deriv = sfx2c1e_hess.gen_sf_hfw(mol, approx='1E')

        h2 = h2_deriv(0,0)
        h2_ref = (h1_deriv_1(0)[2] - h1_deriv_2(0)[2]) / 0.0002 * lib.param.BOHR
        self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)

        h2 = h2_deriv(1,0)
        h2_ref = (h1_deriv_1(1)[2] - h1_deriv_2(1)[2]) / 0.0002 * lib.param.BOHR
        self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)

        h1_deriv_1 = sfx2c1e_grad.gen_sf_hfw(mol1, approx='ATOM1E')
        h1_deriv_2 = sfx2c1e_grad.gen_sf_hfw(mol2, approx='ATOM1E')
        h2_deriv = sfx2c1e_hess.gen_sf_hfw(mol, approx='ATOM1E')

        h2 = h2_deriv(0,0)
        h2_ref = (h1_deriv_1(0)[2] - h1_deriv_2(0)[2]) / 0.0002 * lib.param.BOHR
        self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)

        h2 = h2_deriv(1,0)
        h2_ref = (h1_deriv_1(1)[2] - h1_deriv_2(1)[2]) / 0.0002 * lib.param.BOHR
        self.assertAlmostEqual(abs(h2[2,2]-h2_ref).max(), 0, 7)


if __name__ == "__main__":
    print("Full Tests for sfx2c1e gradients")
    unittest.main()
