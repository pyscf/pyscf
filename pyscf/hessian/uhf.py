#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic UHF analytical Hessian
'''

from functools import reduce
import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.soscf.newton_ah import _gen_uhf_response
from pyscf.hessian import rhf as rhf_hess
_get_jk = rhf_hess._get_jk
_make_vhfopt = rhf_hess._make_vhfopt

# import pyscf.grad.uhf to activate nuc_grad_method method
from pyscf.grad import uhf


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, hessobj.chkfile, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    if isinstance(h1ao, str):
        h1ao = lib.chkfile.load(h1ao, 'scf_f1ao')
        h1aoa = h1ao['0']
        h1aob = h1ao['1']
        h1aoa = dict([(int(k), h1aoa[k]) for k in h1aoa])
        h1aob = dict([(int(k), h1aob[k]) for k in h1aob])
    else:
        h1aoa, h1aob = h1ao
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1a = mo1['0']
        mo1b = mo1['1']
        mo1a = dict([(int(k), mo1a[k]) for k in mo1a])
        mo1b = dict([(int(k), mo1b[k]) for k in mo1b])
    else:
        mo1a, mo1b = mo1
    mo_e1a, mo_e1b = mo_e1

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1ooa = numpy.einsum('xpq,pi,qj->xij', s1ao, mocca, mocca)
        s1oob = numpy.einsum('xpq,pi,qj->xij', s1ao, moccb, moccb)

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            dm1a = numpy.einsum('ypi,qi->ypq', mo1a[ja], mocca)
            dm1b = numpy.einsum('ypi,qi->ypq', mo1b[ja], moccb)
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1aoa[ia], dm1a) * 2
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1aob[ia], dm1b) * 2
            dm1a = numpy.einsum('ypi,qi,i->ypq', mo1a[ja], mocca, mo_ea)
            dm1b = numpy.einsum('ypi,qi,i->ypq', mo1b[ja], moccb, mo_eb)
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1a) * 2
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1b) * 2
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ooa, mo_e1a[ja])
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oob, mo_e1b[ja])

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek  # (A,B,dR_A,dR_B)

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (time.clock(), time.time())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    dm0 = dm0a + dm0b
    # Energy weighted density matrix
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    dme0 = numpy.einsum('pi,qi,i->pq', mocca, mocca, mo_ea)
    dme0+= numpy.einsum('pi,qi,i->pq', moccb, moccb, mo_eb)

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)

    vj1_diag, vk1a_diag, vk1b_diag = \
            _get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                    ['lk->s1ij', dm0,
                     'jk->s1il', dm0a, 'jk->s1il', dm0b],
                    vhfopt=_make_vhfopt(mol, dm0, 'ipip1', 'int2e_ipip1ipip2'))
    vj1_diag = vj1_diag.reshape(3,3,nao,nao)
    vk1a_diag = vk1a_diag.reshape(3,3,nao,nao)
    vk1b_diag = vk1b_diag.reshape(3,3,nao,nao)
    vj1a = vj1b = None
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    ip1ip2_opt = _make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
    ipvip1_opt = _make_vhfopt(mol, dm0, 'ipvip1', 'int2e_ipvip1ipvip2')
    aoslices = mol.aoslice_by_atom()
    e1 = numpy.zeros((mol.natm,mol.natm,3,3))  # (A,B,dR_A,dR_B)
    ej = numpy.zeros((mol.natm,mol.natm,3,3))
    ek = numpy.zeros((mol.natm,mol.natm,3,3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vk1a, vk1b, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                        ['ji->s1kl', dm0 [:,p0:p1],
                         'li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1],
                         'lj->s1ki', dm0a         , 'lj->s1ki', dm0b         ],
                        shls_slice=shls_slice, vhfopt=ip1ip2_opt)
        vk1a[:,:,p0:p1] += vk2a
        vk1b[:,:,p0:p1] += vk2b
        t1 = log.timer_debug1('contracting int2e_ip1ip2 for atom %d'%ia, *t1)
        vj2, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                        ['lk->s1ij', dm0          ,
                         'li->s1kj', dm0a[:,p0:p1], 'li->s1kj', dm0b[:,p0:p1]],
                        shls_slice=shls_slice, vhfopt=ipvip1_opt)
        vj1[:,:,p0:p1] += vj2.transpose(0,2,1) * .5
        vk1a += vk2a.transpose(0,2,1)
        vk1b += vk2b.transpose(0,2,1)
        t1 = log.timer_debug1('contracting int2e_ipvip1 for atom %d'%ia, *t1)
        vj1 = vj1.reshape(3,3,nao,nao)
        vk1a = vk1a.reshape(3,3,nao,nao)
        vk1b = vk1b.reshape(3,3,nao,nao)

        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1ooa = numpy.einsum('xpq,pi,qj->xij', s1ao, mocca, mocca)
        s1oob = numpy.einsum('xpq,pi,qj->xij', s1ao, moccb, moccb)

        ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1_diag[:,:,p0:p1], dm0[p0:p1])*2
        ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1a_diag[:,:,p0:p1], dm0a[p0:p1])*2
        ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1b_diag[:,:,p0:p1], dm0b[p0:p1])*2
        e1[i0,i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1])*4
            ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1a[:,:,q0:q1], dm0a[q0:q1])*2
            ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1b[:,:,q0:q1], dm0b[q0:q1])*2
            e1[i0,j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T

    log.timer('UHF partial hessian', *time0)
    return e1, ej, ek

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    time0 = t1 = (time.clock(), time.time())
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    aoslices = mol.aoslice_by_atom()
    h1aoa = [None] * mol.natm
    h1aob = [None] * mol.natm
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1a, vj1b, vj2a, vj2b, vk1a, vk1b, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                        ['ji->s2kl', -dm0a[:,p0:p1], 'ji->s2kl', -dm0b[:,p0:p1],
                         'lk->s1ij', -dm0a         , 'lk->s1ij', -dm0b         ,
                         'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                         'jk->s1il', -dm0a         , 'jk->s1il', -dm0b         ],
                        shls_slice=shls_slice)
        vj1 = vj1a + vj1b
        vj2 = vj2a + vj2b
        vhfa = vj1 - vk1a
        vhfb = vj1 - vk1b
        vhfa[:,p0:p1] += vj2 - vk2a
        vhfb[:,p0:p1] += vj2 - vk2b
        h1 = hcore_deriv(ia)
        h1a = h1 + vhfa + vhfa.transpose(0,2,1)
        h1b = h1 + vhfb + vhfb.transpose(0,2,1)

        if chkfile is None:
            h1aoa[ia] = h1a
            h1aob[ia] = h1b
        else:
            lib.chkfile.save(chkfile, 'scf_f1ao/0/%d' % ia, h1a)
            lib.chkfile.save(chkfile, 'scf_f1ao/1/%d' % ia, h1b)
    if chkfile is None:
        return (h1aoa,h1aob)
    else:
        return chkfile

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
              fx=None, atmlst=None, max_memory=4000, verbose=None):
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nao*(nocca+noccb)*3*6)))
    mo1sa = [None] * mol.natm
    mo1sb = [None] * mol.natm
    e1sa = [None] * mol.natm
    e1sb = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1voa = []
        s1vob = []
        h1voa = []
        h1vob = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1voa.append(_ao2mo(s1ao, mo_coeff[0], mocca))
            s1vob.append(_ao2mo(s1ao, mo_coeff[1], moccb))
            if isinstance(h1ao_or_chkfile, str):
                h1aoa = lib.chkfile.load(h1ao_or_chkfile, 'scf_f1ao/0/%d'%ia)
                h1aob = lib.chkfile.load(h1ao_or_chkfile, 'scf_f1ao/1/%d'%ia)
            else:
                h1aoa = h1ao_or_chkfile[0][ia]
                h1aob = h1ao_or_chkfile[1][ia]
            h1voa.append(_ao2mo(h1aoa, mo_coeff[0], mocca))
            h1vob.append(_ao2mo(h1aob, mo_coeff[1], moccb))

        h1vo = (numpy.vstack(h1voa), numpy.vstack(h1vob))
        s1vo = (numpy.vstack(s1voa), numpy.vstack(s1vob))
        mo1, e1 = ucphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo)
        mo1a = numpy.einsum('pq,xqi->xpi', mo_coeff[0], mo1[0]).reshape(-1,3,nao,nocca)
        mo1b = numpy.einsum('pq,xqi->xpi', mo_coeff[1], mo1[1]).reshape(-1,3,nao,noccb)
        e1a = e1[0].reshape(-1,3,nocca,nocca)
        e1b = e1[1].reshape(-1,3,noccb,noccb)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            if isinstance(h1ao_or_chkfile, str):
                lib.chkfile.save(h1ao_or_chkfile, 'scf_mo1/0/%d'%ia, mo1a[k])
                lib.chkfile.save(h1ao_or_chkfile, 'scf_mo1/1/%d'%ia, mo1b[k])
            else:
                mo1sa[ia] = mo1a[k]
                mo1sb[ia] = mo1b[k]
            e1sa[ia] = e1a[k].reshape(3,nocca,nocca)
            e1sb[ia] = e1b[k].reshape(3,noccb,noccb)
        mo1 = e1 = mo1a = mo1b = e1a = e1b = None

    if isinstance(h1ao_or_chkfile, str):
        return h1ao_or_chkfile, (e1sa,e1sb)
    else:
        return (mo1sa,mo1sb), (e1sa,e1sb)

def gen_vind(mf, mo_coeff, mo_occ):
    nao, nmoa = mo_coeff[0].shape
    nmob = mo_coeff[1].shape[1]
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    vresp = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)
        dm1 = numpy.empty((2,nset,nao,nao))
        for i, x in enumerate(mo1):
            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
            xb = x[nmoa*nocca:].reshape(nmob,noccb)
            dma = reduce(numpy.dot, (mo_coeff[0], xa, mocca.T))
            dmb = reduce(numpy.dot, (mo_coeff[1], xb, moccb.T))
            dm1[0,i] = dma + dma.T
            dm1[1,i] = dmb + dmb.T
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1)
        for i in range(nset):
            v1vo[i,:nmoa*nocca] = reduce(numpy.dot, (mo_coeff[0].T, v1[0,i], mocca)).ravel()
            v1vo[i,nmoa*nocca:] = reduce(numpy.dot, (mo_coeff[1].T, v1[1,i], moccb)).ravel()
        return v1vo
    return fx


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj.base

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    atmlst = range(natm)
    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    # Compute H1 integrals and store in hobj.chkfile
    hobj.make_h1(mo_coeff, mo_occ, hobj.chkfile, atmlst, log)

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1aoa = 0
        h1aob = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao_i = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/0/%d' % ia)
            h1aoa += numpy.einsum('x,xij->ij', x[ia], h1ao_i)
            h1ao_i = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/1/%d' % ia)
            h1aob += numpy.einsum('x,xij->ij', x[ia], h1ao_i)
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1voa = reduce(numpy.dot, (mo_coeff[0].T, s1ao, mocca))
        s1vob = reduce(numpy.dot, (mo_coeff[1].T, s1ao, moccb))
        h1voa = reduce(numpy.dot, (mo_coeff[0].T, h1aoa, mocca))
        h1vob = reduce(numpy.dot, (mo_coeff[1].T, h1aob, moccb))
        mo1, mo_e1 = ucphf.solve(fvind, mo_energy, mo_occ,
                                 (h1voa,h1vob), (s1voa,s1vob))
        mo1a = numpy.dot(mo_coeff[0], mo1[0])
        mo1b = numpy.dot(mo_coeff[1], mo1[1])
        mo_e1a = mo_e1[0].reshape(nocca,nocca)
        mo_e1b = mo_e1[1].reshape(noccb,noccb)
        dm1a = numpy.einsum('pi,qi->pq', mo1a, mocca)
        dm1b = numpy.einsum('pi,qi->pq', mo1b, moccb)
        dme1a = numpy.einsum('pi,qi,i->pq', mo1a, mocca, mo_ea)
        dme1a = dme1a + dme1a.T + reduce(numpy.dot, (mocca, mo_e1a, mocca.T))
        dme1b = numpy.einsum('pi,qi,i->pq', mo1b, moccb, mo_eb)
        dme1b = dme1b + dme1b.T + reduce(numpy.dot, (moccb, mo_e1b, moccb.T))
        dme1 = dme1a + dme1b

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            h1aoa = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/0/%d' % ja)
            h1aob = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/1/%d' % ja)
            hx[ja] += numpy.einsum('xpq,pq->x', h1aoa, dm1a) * 2
            hx[ja] += numpy.einsum('xpq,pq->x', h1aob, dm1b) * 2
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1])
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1])
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


class Hessian(rhf_hess.Hessian):
    '''Non-relativistic UHF hessian'''
    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1
    gen_hop = gen_hop

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose)

from pyscf import scf
scf.uhf.UHF.Hessian = lib.class_as_method(Hessian)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = mf.Hessian()
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.finger(e2) - -0.50693144355876429)

    mol.spin = 2
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = Hessian(mf)
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = scf.UHF(mol).run(conv_tol=1e-14)
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-4))

# \partial^2 E / \partial R \partial R'
    e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    e2 += hobj.hess_nuc(mol)
    e2 = e2.transpose(0,2,1,3).reshape(n3,n3)
    def grad_partial_R(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_partial_R(ia, .5e-4) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-8))
