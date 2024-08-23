#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
Analytical nuclear hessian for 1-electron spin-free x2c method

Ref.
JCP 135, 244104 (2011); DOI:10.1063/1.3667202
JCTC 8, 2617 (2012); DOI:10.1021/ct300127e
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.x2c import x2c
from pyscf.x2c import sfx2c1e_grad

def hcore_hess_generator(x2cobj, mol=None):
    '''nuclear gradients of 1-component X2c hcore Hamiltonian  (spin-free part only)
    '''
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)

    if x2cobj.basis is not None:
        s22 = xmol.intor_symmetric('int1e_ovlp')
        s21 = gto.intor_cross('int1e_ovlp', xmol, mol)
        contr_coeff = lib.cho_solve(s22, s21)

    get_h1_xmol = gen_sf_hfw(xmol, x2cobj.approx)
    def hcore_deriv(ia, ja):
        h1 = get_h1_xmol(ia, ja)
        if contr_coeff is not None:
            h1 = lib.einsum('pi,xypq,qj->xyij', contr_coeff, h1, contr_coeff)
        return numpy.asarray(h1)
    return hcore_deriv


def gen_sf_hfw(mol, approx='1E'):
    approx = approx.upper()
    c = lib.param.LIGHT_SPEED

    h0, s0 = sfx2c1e_grad._get_h0_s0(mol)
    e0, c0 = scipy.linalg.eigh(h0, s0)
    c0[:,c0[1]<0] *= -1

    aoslices = mol.aoslice_by_atom()
    nao = mol.nao_nr()
    if 'ATOM' in approx:
        x0 = numpy.zeros((nao,nao))
        for ia in range(mol.natm):
            ish0, ish1, p0, p1 = aoslices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            t1 = mol.intor('int1e_kin', shls_slice=shls_slice)
            s1 = mol.intor('int1e_ovlp', shls_slice=shls_slice)
            with mol.with_rinv_at_nucleus(ia):
                z = -mol.atom_charge(ia)
                v1 = z * mol.intor('int1e_rinv', shls_slice=shls_slice)
                w1 = z * mol.intor('int1e_prinvp', shls_slice=shls_slice)
            x0[p0:p1,p0:p1] = x2c._x2c1e_xmatrix(t1, v1, w1, s1, c)
    else:
        cl0 = c0[:nao,nao:]
        cs0 = c0[nao:,nao:]
        x0 = scipy.linalg.solve(cl0.T, cs0.T).T

    t0x0 = numpy.dot(s0[nao:,nao:], x0)
    s_nesc0 = s0[:nao,:nao] + numpy.dot(x0.T, t0x0)

    w_s, v_s = scipy.linalg.eigh(s0[:nao,:nao])
    w_sqrt = numpy.sqrt(w_s)
    s_nesc0_vbas = reduce(numpy.dot, (v_s.T, s_nesc0, v_s))
    R0_mid = numpy.einsum('i,ij,j->ij', 1./w_sqrt, s_nesc0_vbas, 1./w_sqrt)
    wr0, vr0 = scipy.linalg.eigh(R0_mid)
    wr0_sqrt = numpy.sqrt(wr0)
    # R0 in v_s basis
    R0 = numpy.dot(vr0/wr0_sqrt, vr0.T)
    R0 *= w_sqrt
    R0 /= w_sqrt[:,None]
    # Transform R0 back
    R0 = reduce(numpy.dot, (v_s, R0, v_s.T))
    R0 = x2c._get_r(s0[:nao,:nao], s_nesc0)
    c_fw0 = numpy.vstack((R0, numpy.dot(x0, R0)))
    h0_fw_half = numpy.dot(h0, c_fw0)

    epq = e0[:,None] - e0
    degen_mask = abs(epq) < 1e-7
    epq[degen_mask] = 1e200
    s2aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    t2aa = mol.intor('int1e_ipipkin', comp=9).reshape(3,3,nao,nao)
    v2aa = mol.intor('int1e_ipipnuc', comp=9).reshape(3,3,nao,nao)
    w2aa = mol.intor('int1e_ipippnucp', comp=9).reshape(3,3,nao,nao)
    s2ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    t2ab = mol.intor('int1e_ipkinip', comp=9).reshape(3,3,nao,nao)
    v2ab = mol.intor('int1e_ipnucip', comp=9).reshape(3,3,nao,nao)
    w2ab = mol.intor('int1e_ippnucpip', comp=9).reshape(3,3,nao,nao)
    n2 = nao * 2
    h2ao = numpy.zeros((3,3,n2,n2), dtype=v2aa.dtype)
    s2ao = numpy.zeros((3,3,n2,n2), dtype=v2aa.dtype)

    get_h1_etc = sfx2c1e_grad._gen_first_order_quantities(mol, e0, c0, x0, approx)

    def hcore_deriv(ia, ja):
        ish0, ish1, i0, i1 = aoslices[ia]
        jsh0, jsh1, j0, j1 = aoslices[ja]

        s2cc = numpy.zeros_like(s2aa)
        t2cc = numpy.zeros_like(s2aa)
        v2cc = numpy.zeros_like(s2aa)
        w2cc = numpy.zeros_like(s2aa)
        if ia == ja:
            with mol.with_rinv_origin(mol.atom_coord(ia)):
                z = mol.atom_charge(ia)
                rinv2aa = z*mol.intor('int1e_ipiprinv', comp=9).reshape(3,3,nao,nao)
                rinv2ab = z*mol.intor('int1e_iprinvip', comp=9).reshape(3,3,nao,nao)
                prinvp2aa = z*mol.intor('int1e_ipipprinvp', comp=9).reshape(3,3,nao,nao)
                prinvp2ab = z*mol.intor('int1e_ipprinvpip', comp=9).reshape(3,3,nao,nao)
            s2cc[:,:,i0:i1      ] = s2aa[:,:,i0:i1      ]
            s2cc[:,:,i0:i1,j0:j1]+= s2ab[:,:,i0:i1,j0:j1]
            t2cc[:,:,i0:i1      ] = t2aa[:,:,i0:i1      ]
            t2cc[:,:,i0:i1,j0:j1]+= t2ab[:,:,i0:i1,j0:j1]
            v2cc -= rinv2aa + rinv2ab
            v2cc[:,:,i0:i1      ]+= v2aa[:,:,i0:i1      ]
            v2cc[:,:,i0:i1,j0:j1]+= v2ab[:,:,i0:i1,j0:j1]
            v2cc[:,:,i0:i1      ]+= rinv2aa[:,:,i0:i1]
            v2cc[:,:,i0:i1      ]+= rinv2ab[:,:,i0:i1]
            v2cc[:,:,:    ,i0:i1]+= rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
            v2cc[:,:,:    ,i0:i1]+= rinv2ab[:,:,:,i0:i1]
            w2cc -= prinvp2aa + prinvp2ab
            w2cc[:,:,i0:i1      ]+= w2aa[:,:,i0:i1      ]
            w2cc[:,:,i0:i1,j0:j1]+= w2ab[:,:,i0:i1,j0:j1]
            w2cc[:,:,i0:i1      ]+= prinvp2aa[:,:,i0:i1]
            w2cc[:,:,i0:i1      ]+= prinvp2ab[:,:,i0:i1]
            w2cc[:,:,:    ,i0:i1]+= prinvp2aa[:,:,i0:i1].transpose(0,1,3,2)
            w2cc[:,:,:    ,i0:i1]+= prinvp2ab[:,:,:,i0:i1]

        else:
            s2cc[:,:,i0:i1,j0:j1] = s2ab[:,:,i0:i1,j0:j1]
            t2cc[:,:,i0:i1,j0:j1] = t2ab[:,:,i0:i1,j0:j1]
            v2cc[:,:,i0:i1,j0:j1] = v2ab[:,:,i0:i1,j0:j1]
            w2cc[:,:,i0:i1,j0:j1] = w2ab[:,:,i0:i1,j0:j1]
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
                v2cc[:,:,j0:j1] += rinv2aa
                v2cc[:,:,j0:j1] += rinv2ab.transpose(1,0,2,3)
                w2cc[:,:,j0:j1] += prinvp2aa
                w2cc[:,:,j0:j1] += prinvp2ab.transpose(1,0,2,3)

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
                v2cc[:,:,i0:i1] += rinv2aa
                v2cc[:,:,i0:i1] += rinv2ab
                w2cc[:,:,i0:i1] += prinvp2aa
                w2cc[:,:,i0:i1] += prinvp2ab
        s2cc = s2cc + s2cc.transpose(0,1,3,2)
        t2cc = t2cc + t2cc.transpose(0,1,3,2)
        v2cc = v2cc + v2cc.transpose(0,1,3,2)
        w2cc = w2cc + w2cc.transpose(0,1,3,2)
        h2ao[:,:,:nao,:nao] = v2cc
        h2ao[:,:,:nao,nao:] = t2cc
        h2ao[:,:,nao:,:nao] = t2cc
        h2ao[:,:,nao:,nao:] = w2cc * (.25/c**2) - t2cc
        s2ao[:,:,:nao,:nao] = s2cc
        s2ao[:,:,nao:,nao:] = t2cc * (.5/c**2)

        h1i, s1i, e1i, c1i, x1i, s_nesc1i, R1i, c_fw1i = get_h1_etc(ia)
        h1j, s1j, e1j, c1j, x1j, s_nesc1j, R1j, c_fw1j = get_h1_etc(ja)
        if 'ATOM' not in approx:
            f2 = lib.einsum('xypq,qj->xypj', h2ao, c0[:,nao:])
            f2+= lib.einsum('xpq,yqj->xypj', h1i, c1j)
            f2+= lib.einsum('ypq,xqj->xypj', h1j, c1i)
            sc2 = lib.einsum('xypq,qj->xypj', s2ao, c0[:,nao:])
            sc2+= lib.einsum('xpq,yqj->xypj', s1i, c1j)
            sc2+= lib.einsum('ypq,xqj->xypj', s1j, c1i)
            f2-= sc2 * e0[nao:]
            sc1i = lib.einsum('xpq,qj->xpj', s1i, c0[:,nao:])
            sc1j = lib.einsum('xpq,qj->xpj', s1j, c0[:,nao:])
            sc1i+= lib.einsum('pq,xqj->xpj', s0, c1i)
            sc1j+= lib.einsum('pq,xqj->xpj', s0, c1j)
            f2-= lib.einsum('xpq,yqj->xypj', sc1i, e1j)
            f2-= lib.einsum('ypq,xqj->xypj', sc1j, e1i)

            c2 = lib.einsum('pi,xypj->xyij', c0.conj(), f2) / -epq[:,nao:]
            c2_ao = lib.einsum('pq,xyqi->xypi', c0, c2)
            cl2 = c2_ao[:,:,:nao]
            cs2 = c2_ao[:,:,nao:]

            tmp = cs2 - lib.einsum('pq,xyqi->xypi', x0, cl2)
            tmp-= lib.einsum('xpq,yqi->xypi', x1i, c1j[:,:nao])
            tmp-= lib.einsum('ypq,xqi->xypi', x1j, c1i[:,:nao])
            x2 = scipy.linalg.solve(cl0.T, tmp.reshape(-1,nao).T).T.reshape(3,3,nao,nao)

        hfw2 = numpy.empty((3,3,nao,nao))
        for i in range(3):
            for j in range(3):
                if 'ATOM' in approx:
                    s_nesc2  = reduce(numpy.dot, (x0.T, s2ao[i,j,nao:,nao:], x0))
                    s_nesc2 += s2ao[i,j,:nao,:nao]
                    R2 = _get_r2((w_sqrt,v_s), s_nesc0,
                                 s1i[i,:nao,:nao], s_nesc1i[i],
                                 s1j[j,:nao,:nao], s_nesc1j[j],
                                 s2ao[i,j,:nao,:nao], s_nesc2, (wr0_sqrt,vr0))
                    c_fw2 = numpy.vstack((R2, numpy.dot(x0, R2)))
                else:
                    s_nesc2  = numpy.dot(x2[i,j].T, t0x0)
                    s_nesc2 += reduce(numpy.dot, (x1i[i].T, s1j[j,nao:,nao:], x0))
                    s_nesc2 += reduce(numpy.dot, (x0.T, s1i[i,nao:,nao:], x1j[j]))
                    s_nesc2 += reduce(numpy.dot, (x1i[i].T, s0[nao:,nao:], x1j[j]))
                    s_nesc2  = s_nesc2 + s_nesc2.T
                    s_nesc2 += reduce(numpy.dot, (x0.T, s2ao[i,j,nao:,nao:], x0))
                    s_nesc2 += s2ao[i,j,:nao,:nao]
                    R2 = _get_r2((w_sqrt,v_s), s_nesc0,
                                 s1i[i,:nao,:nao], s_nesc1i[i],
                                 s1j[j,:nao,:nao], s_nesc1j[j],
                                 s2ao[i,j,:nao,:nao], s_nesc2, (wr0_sqrt,vr0))
                    c_fw_s = (numpy.dot(x0, R2) + numpy.dot(x1i[i], R1j[j]) +
                              numpy.dot(x1j[j], R1i[i]) + numpy.dot(x2[i,j], R0))
                    c_fw2 = numpy.vstack((R2, c_fw_s))
                tmp  = numpy.dot(c_fw2.T, h0_fw_half)
                tmp += reduce(numpy.dot, (c_fw1i[i].T, h1j[j], c_fw0))
                tmp += reduce(numpy.dot, (c_fw0.T, h1i[i], c_fw1j[j]))
                tmp += reduce(numpy.dot, (c_fw1i[i].T, h0, c_fw1j[j]))
                hfw2[i,j] = tmp + tmp.T
                hfw2[i,j]+= reduce(numpy.dot, (c_fw0.T, h2ao[i,j], c_fw0))
        return hfw2

    return hcore_deriv

def _get_r2(s0_roots, sa0, s1i, sa1i, s1j, sa1j, s2, sa2, r0_roots):
    w_sqrt, v_s = s0_roots
    w_invsqrt = 1. / w_sqrt
    wr0_sqrt, vr0 = r0_roots
    wr0_invsqrt = 1. / wr0_sqrt

    sa0  = lib.einsum('pi,pq,qj->ij', v_s, sa0 , v_s)
    s1i  = lib.einsum('pi,pq,qj->ij', v_s, s1i , v_s)
    s1j  = lib.einsum('pi,pq,qj->ij', v_s, s1j , v_s)
    s2   = lib.einsum('pi,pq,qj->ij', v_s, s2  , v_s)
    sa1i = lib.einsum('pi,pq,qj->ij', v_s, sa1i, v_s)
    sa1j = lib.einsum('pi,pq,qj->ij', v_s, sa1j, v_s)
    sa2  = lib.einsum('pi,pq,qj->ij', v_s, sa2 , v_s)

    s1i_sqrt = s1i / (w_sqrt[:,None] + w_sqrt)
    s1i_invsqrt = (numpy.einsum('i,ij,j->ij', w_invsqrt**2, s1i, w_invsqrt**2)
                   / -(w_invsqrt[:,None] + w_invsqrt))
    s1j_sqrt = s1j / (w_sqrt[:,None] + w_sqrt)
    s1j_invsqrt = (numpy.einsum('i,ij,j->ij', w_invsqrt**2, s1j, w_invsqrt**2)
                   / -(w_invsqrt[:,None] + w_invsqrt))

    tmp = numpy.dot(s1i_sqrt, s1j_sqrt)
    s2_sqrt = (s2 - tmp - tmp.T) / (w_sqrt[:,None] + w_sqrt)
    tmp = numpy.dot(s1i*w_invsqrt**2, s1j)
    tmp = s2 - tmp - tmp.T
    tmp = -numpy.einsum('i,ij,j->ij', w_invsqrt**2, tmp, w_invsqrt**2)
    tmp1 = numpy.dot(s1i_invsqrt, s1j_invsqrt)
    s2_invsqrt = (tmp - tmp1 - tmp1.T) / (w_invsqrt[:,None] + w_invsqrt)

    R1i_mid = lib.einsum('ip,pj,j->ij', s1i_invsqrt, sa0, w_invsqrt)
    R1i_mid = R1i_mid + R1i_mid.T
    R1i_mid+= numpy.einsum('i,ij,j->ij', w_invsqrt, sa1i, w_invsqrt)
    R1i_mid = tmpi = lib.einsum('pi,pq,qj->ij', vr0, R1i_mid, vr0)
    R1i_mid = (numpy.einsum('i,ij,j->ij', wr0_invsqrt**2, R1i_mid, wr0_invsqrt**2)
               / -(wr0_invsqrt[:,None] + wr0_invsqrt))

    R1j_mid = lib.einsum('ip,pj,j->ij', s1j_invsqrt, sa0, w_invsqrt)
    R1j_mid = R1j_mid + R1j_mid.T
    R1j_mid+= numpy.einsum('i,ij,j->ij', w_invsqrt, sa1j, w_invsqrt)
    R1j_mid = tmpj = lib.einsum('pi,pq,qj->ij', vr0, R1j_mid, vr0)
    R1j_mid = (numpy.einsum('i,ij,j->ij', wr0_invsqrt**2, R1j_mid, wr0_invsqrt**2)
               / -(wr0_invsqrt[:,None] + wr0_invsqrt))

# second derivative of (s_invsqrt * sa * s_invsqrt), 9 terms
    R2_mid = lib.einsum('ip,pj,j->ij', s2_invsqrt , sa0 , w_invsqrt)
    R2_mid+= lib.einsum('ip,pj,j->ij', s1i_invsqrt, sa1j, w_invsqrt)
    R2_mid+= lib.einsum('i,ip,pj->ij', w_invsqrt  , sa1i, s1j_invsqrt)
    R2_mid+= lib.einsum('ip,pq,qj->ij', s1i_invsqrt, sa0 , s1j_invsqrt)
    R2_mid  = R2_mid + R2_mid.T
    R2_mid+= numpy.einsum('i,ij,j->ij', w_invsqrt, sa2, w_invsqrt)
    R2_mid = lib.einsum('pi,pq,qj->ij', vr0, R2_mid, vr0)
    tmp = numpy.dot(tmpi*wr0_invsqrt**2, tmpj)
    tmp = R2_mid - tmp - tmp.T
    tmp = -numpy.einsum('i,ij,j->ij', wr0_invsqrt**2, tmp, wr0_invsqrt**2)
    tmp1 = numpy.dot(R1i_mid, R1j_mid)
    R2_mid = (tmp - tmp1 - tmp1.T) / (wr0_invsqrt[:,None] + wr0_invsqrt)

    R0_mid = numpy.dot(vr0*wr0_invsqrt, vr0.T)
    R1i_mid = reduce(numpy.dot, (vr0, R1i_mid, vr0.T))
    R1j_mid = reduce(numpy.dot, (vr0, R1j_mid, vr0.T))
    R2_mid = reduce(numpy.dot, (vr0, R2_mid, vr0.T))

    R2  = lib.einsum('ip,pj,j->ij' , s2_invsqrt , R0_mid , w_sqrt)
    R2 += lib.einsum('ip,pj,j->ij' , s1i_invsqrt, R1j_mid, w_sqrt)
    R2 += lib.einsum('ip,pq,qj->ij', s1i_invsqrt, R0_mid , s1j_sqrt)
    R2 += lib.einsum('ip,pj,j->ij' , s1j_invsqrt, R1i_mid, w_sqrt)
    R2 += numpy.einsum('i,ij,j->ij', w_invsqrt  , R2_mid , w_sqrt)
    R2 += lib.einsum('i,iq,qj->ij' , w_invsqrt  , R1i_mid, s1j_sqrt)
    R2 += lib.einsum('ip,pq,qj->ij', s1j_invsqrt, R0_mid , s1i_sqrt)
    R2 += lib.einsum('i,iq,qj->ij' , w_invsqrt  , R1j_mid, s1i_sqrt)
    R2 += lib.einsum('i,iq,qj->ij' , w_invsqrt  , R0_mid , s2_sqrt)
    R2 = reduce(numpy.dot, (v_s, R2, v_s.T))
    return R2


if __name__ == '__main__':
    bak = lib.param.LIGHT_SPEED
    lib.param.LIGHT_SPEED = 10

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    h1_deriv_1 = sfx2c1e_grad.gen_sf_hfw(mol, approx='1E')

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     ,-0.0001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    h1_deriv_2 = sfx2c1e_grad.gen_sf_hfw(mol, approx='1E')

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.   )],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    h2_deriv = gen_sf_hfw(mol)

    h2 = h2_deriv(0,0)
    h2_ref = (h1_deriv_1(0)[2] - h1_deriv_2(0)[2]) / 0.0002 * lib.param.BOHR
    print(abs(h2[2,2]-h2_ref).max())
    print(lib.finger(h2) - 33.71188112440316)

    h2 = h2_deriv(1,0)
    h2_ref = (h1_deriv_1(1)[2] - h1_deriv_2(1)[2]) / 0.0002 * lib.param.BOHR
    print(abs(h2[2,2]-h2_ref).max())
    print(lib.finger(h2) - -23.609411428378138)
    lib.param.LIGHT_SPEED = bak
