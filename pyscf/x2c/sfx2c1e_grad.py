#!/usr/bin/env python

'''
Analytical nuclear gradients for 1-electron spin-free x2c method

Ref.
JCP 135 084114
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.x2c import x2c

def hcore_grad_generator(x2cobj, mol=None):
    '''nuclear gradients of 1-component X2c hcore Hamiltonian  (spin-free part only)
    '''
    if mol is None: mol = x2cobj.mol
    xmol, contr_coeff = x2cobj.get_xmol(mol)

    if x2cobj.basis is not None:
        s22 = xmol.intor_symmetric('int1e_ovlp')
        s21 = gto.intor_cross('int1e_ovlp', xmol, mol)
        contr_coeff = lib.cho_solve(s22, s21)

    get_h1_xmol = gen_sf_hfw(xmol, x2cobj.approx)
    def hcore_deriv(atm_id):
        h1 = get_h1_xmol(atm_id)
        if contr_coeff is not None:
            h1 = [reduce(numpy.dot, (contr_coeff.T, h1[i], contr_coeff))
                  for i in range(3)]
        return numpy.asarray(h1)
    return hcore_deriv


def gen_sf_hfw(mol, approx='1E'):
    approx = approx.upper()
    c = lib.param.LIGHT_SPEED

    h0, s0 = _get_h0_s0(mol)
    e0, c0 = scipy.linalg.eigh(h0, s0)

    aoslices = mol.aoslice_by_atom()
    nao = mol.nao_nr()
    if 'ATOM' in approx:
        x0 = numpy.zeros((nao,nao))
        for ia in range(mol.natm):
            ish0, ish1, p0, p1 = aoslices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            t1 = mol.intor('int1e_kin', shls_slice=shls_slice)
            s1 = mol.intor('int1e_ovlp', shls_slice=shls_slice)
            with mol.with_rinv_as_nucleus(ia):
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

    h_nesc0 = numpy.dot(h0[:nao,nao:], x0)
    h_nesc0 = h_nesc0 + h_nesc0.T
    h_nesc0+= h0[:nao,:nao]
    h_nesc0_half = numpy.dot(h0[nao:,nao:], x0)
    h_nesc0+= numpy.dot(x0.T, h_nesc0_half)
    h_nesc0_half += h0[nao:,:nao]

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
    h0 = s0 = None

    s1 = mol.intor('int1e_ipovlp', comp=3)
    t1 = mol.intor('int1e_ipkin', comp=3)
    v1 = mol.intor('int1e_ipnuc', comp=3)
    w1 = mol.intor('int1e_ippnucp', comp=3)
    n2 = nao * 2
    h1 = numpy.zeros((n2,n2), dtype=v1.dtype)
    m1 = numpy.zeros((n2,n2), dtype=v1.dtype)

    def hcore_deriv(ia):
        ish0, ish1, p0, p1 = aoslices[ia]
        with mol.with_rinv_origin(mol.atom_coord(ia)):
            z = mol.atom_charge(ia)
            rinv1   = -z*mol.intor('int1e_iprinv', comp=3)
            prinvp1 = -z*mol.intor('int1e_ipprinvp', comp=3)
        rinv1  [:,p0:p1,:] -= v1[:,p0:p1]
        prinvp1[:,p0:p1,:] -= w1[:,p0:p1]

        hfw1 = numpy.empty((3,nao,nao))
        for i in range(3):
            s1cc = numpy.zeros((nao,nao))
            t1cc = numpy.zeros((nao,nao))
            s1cc[p0:p1,:] =-s1[i,p0:p1]
            s1cc[:,p0:p1]-= s1[i,p0:p1].T
            t1cc[p0:p1,:] =-t1[i,p0:p1]
            t1cc[:,p0:p1]-= t1[i,p0:p1].T
            v1cc = rinv1[i]   + rinv1[i].T
            w1cc = prinvp1[i] + prinvp1[i].T

            h1[:nao,:nao] = v1cc
            h1[:nao,nao:] = t1cc
            h1[nao:,:nao] = t1cc
            h1[nao:,nao:] = w1cc * (.25/c**2) - t1cc
            m1[:nao,:nao] = s1cc
            m1[nao:,nao:] = t1cc * (.5/c**2)

            if 'ATOM' in approx:
                s_nesc1 = m1[:nao,:nao] + reduce(numpy.dot, (x0.T, m1[nao:,nao:], x0))
                R1 = _get_r1(s1cc, s_nesc1, s_nesc0_vbas,
                             (w_sqrt,v_s), (wr0_sqrt,vr0))

                h_nesc1 = numpy.dot(h1[:nao,nao:], x0)
                h_nesc1 = h_nesc1 + h_nesc1.T
                h_nesc1+= h1[:nao,:nao]
                h_nesc1+= reduce(numpy.dot, (x0.T, h1[nao:,nao:], x0))
            else:
                x1 = _get_x1(e0, c0, h1, m1, x0)

                s_nesc1 = numpy.dot(x1.T, t0x0)
                s_nesc1 = s_nesc1 + s_nesc1.T
                s_nesc1+= reduce(numpy.dot, (x0.T, m1[nao:,nao:], x0))
                s_nesc1+= m1[:nao,:nao]
                R1 = _get_r1(s1cc, s_nesc1, s_nesc0_vbas,
                             (w_sqrt,v_s), (wr0_sqrt,vr0))

                h_nesc1 = numpy.dot(x1.T, h_nesc0_half)
                h_nesc1+= numpy.dot(h1[:nao,nao:], x0)
                h_nesc1 = h_nesc1 + h_nesc1.T
                h_nesc1+= h1[:nao,:nao]
                h_nesc1+= reduce(numpy.dot, (x0.T, h1[nao:,nao:], x0))

            tmp = reduce(numpy.dot, (R0.T, h_nesc0, R1))
            hfw1[i] = tmp + tmp.T
            hfw1[i]+= reduce(numpy.dot, (R0.T, h_nesc1, R0))
        return hfw1

    return hcore_deriv

def _get_x1(e0, c0, h1, s1, x0):
    nao = e0.size // 2
    cl0 = c0[:nao,nao:]
    cs0 = c0[nao:,nao:]
    h1mo = lib.einsum('pi,pq,qj->ij', c0.conj(), h1, c0[:,nao:])
    s1mo = lib.einsum('pi,pq,qj->ij', c0.conj(), s1, c0[:,nao:])

    epi = e0[:,None] - e0[nao:]
    degen_mask = abs(epi) < 1e-7
    epi[degen_mask] = 1e200
    c1 = (h1mo - s1mo * e0[nao:]) / -epi
    c1[degen_mask] = -.5 * s1mo[degen_mask]
    c1 = lib.einsum('pq,qi->pi', c0, c1)
    cl1 = c1[:nao]
    cs1 = c1[nao:]

    x1 = scipy.linalg.solve(cl0.T, (cs1 - x0.dot(cl1)).T).T
    return x1

def _get_h0_s0(mol):
    c = lib.param.LIGHT_SPEED
    s = mol.intor_symmetric('int1e_ovlp')
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    w = mol.intor_symmetric('int1e_pnucp')
    nao = s.shape[0]
    n2 = nao * 2
    h = numpy.zeros((n2,n2), dtype=v.dtype)
    m = numpy.zeros((n2,n2), dtype=v.dtype)
    h[:nao,:nao] = v
    h[:nao,nao:] = t
    h[nao:,:nao] = t
    h[nao:,nao:] = w * (.25/c**2) - t
    m[:nao,:nao] = s
    m[nao:,nao:] = t * (.5/c**2)
    return h, m

def _get_r1(s1, s_nesc1, s_nesc0, s0_roots, r0_roots):
# See JCP 135 084114, Eq (34)
    w_sqrt, v_s = s0_roots
    wr0_sqrt, vr0 = r0_roots

    s1 = reduce(numpy.dot, (v_s.T, s1, v_s))
    s_nesc1 = reduce(numpy.dot, (v_s.T, s_nesc1, v_s))

    s1_sqrt = s1 / (w_sqrt[:,None] + w_sqrt)
    s1_invsqrt = s1 / -(1./w_sqrt[:,None] + 1./w_sqrt)
    s1_invsqrt *= 1./w_sqrt**2
    s1_invsqrt *= 1./w_sqrt[:,None]**2
    R1_mid = numpy.dot(s1_invsqrt, s_nesc0) / w_sqrt
    R1_mid = R1_mid + R1_mid.T
    R1_mid += numpy.einsum('i,ij,j->ij', 1./w_sqrt, s_nesc1, 1./w_sqrt)

    R1_mid = reduce(numpy.dot, (vr0.T, R1_mid, vr0))
    R1_mid /= -(1./wr0_sqrt[:,None] + 1./wr0_sqrt)
    R1_mid *= 1./wr0_sqrt**2
    R1_mid *= 1./wr0_sqrt[:,None]**2
    vr0_wr0_sqrt = vr0 / wr0_sqrt
    vr0_s0_sqrt = vr0.T * w_sqrt
    vr0_s0_invsqrt = vr0.T / w_sqrt

    R1  = reduce(numpy.dot, (vr0_s0_invsqrt.T, R1_mid, vr0_s0_sqrt))
    R1 += reduce(numpy.dot, (s1_invsqrt, vr0_wr0_sqrt, vr0_s0_sqrt))
    R1 += reduce(numpy.dot, (vr0_s0_invsqrt.T, vr0_wr0_sqrt.T, s1_sqrt))
    R1 = reduce(numpy.dot, (v_s, R1, v_s.T))
    return R1


if __name__ == '__main__':
    from pyscf import gto
    bak = lib.param.LIGHT_SPEED
    lib.param.LIGHT_SPEED = 10
    def get_h(mol):
        c = lib.param.LIGHT_SPEED
        t = mol.intor_symmetric('int1e_kin')
        v = mol.intor_symmetric('int1e_nuc')
        s = mol.intor_symmetric('int1e_ovlp')
        w = mol.intor_symmetric('int1e_pnucp')
        return x2c._x2c1e_get_hcore(t, v, w, s, c)

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    h_1 = get_h(mol)

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     ,-0.001)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    h_2 = get_h(mol)
    h_ref = (h_1 - h_2) / 0.002 * lib.param.BOHR

    mol = gto.M(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.   )],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)]],
        basis = '3-21g',
    )
    hcore_deriv = gen_sf_hfw(mol)
    h1 = hcore_deriv(0)
    print(abs(h1[2]-h_ref).max())
    lib.param.LIGHT_SPEED = bak
