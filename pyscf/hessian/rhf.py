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
Non-relativistic RHF analytical Hessian
'''

from functools import reduce
import ctypes
import time
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.scf import cphf
from pyscf.soscf.newton_ah import _gen_rhf_response


# import pyscf.grad.rhf to activate nuc_grad_method method
from pyscf.grad import rhf


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
        h1ao = dict([(int(k), h1ao[k]) for k in h1ao])
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1 = dict([(int(k), mo1[k]) for k in mo1])

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        for j0 in range(i0+1):
            ja = atmlst[j0]
            q0, q1 = aoslices[ja][2:]
# *2 for double occupancy, *2 for +c.c.
            dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
            de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 4
            dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[mo_occ>0])
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 4
            de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja]) * 2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    '''Partial derivative
    '''
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

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    # Energy weighted density matrix
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0]) * 2

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = get_ovlp(mol)

    vj1_diag, vk1_diag = \
            _get_jk(mol, 'int2e_ipip1', 9, 's2kl',
                    ['lk->s1ij', dm0,   # vj1
                     'jk->s1il', dm0],  # vk1
                    vhfopt=_make_vhfopt(mol, dm0, 'ipip1', 'int2e_ipip1ipip2'))
    vj1_diag = vj1_diag.reshape(3,3,nao,nao)
    vk1_diag = vk1_diag.reshape(3,3,nao,nao)
    t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

    ip1ip2_opt = _make_vhfopt(mol, dm0, 'ip1ip2', 'int2e_ip1ip2')
    ipvip1_opt = _make_vhfopt(mol, dm0, 'ipvip1', 'int2e_ipvip1ipvip2')
    aoslices = mol.aoslice_by_atom()
    e1 = numpy.zeros((mol.natm,mol.natm,3,3))
    ej = numpy.zeros((mol.natm,mol.natm,3,3))
    ek = numpy.zeros((mol.natm,mol.natm,3,3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vk1, vk2 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1',
                                ['ji->s1kl', dm0[:,p0:p1],  # vj1
                                 'li->s1kj', dm0[:,p0:p1],  # vk1
                                 'lj->s1ki', dm0         ], # vk2
                                shls_slice=shls_slice, vhfopt=ip1ip2_opt)
        vk1[:,:,p0:p1] += vk2
        t1 = log.timer_debug1('contracting int2e_ip1ip2 for atom %d'%ia, *t1)
        vj2, vk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl',
                           ['lk->s1ij', dm0         ,  # vj1
                            'li->s1kj', dm0[:,p0:p1]], # vk1
                           shls_slice=shls_slice, vhfopt=ipvip1_opt)
        vj1[:,:,p0:p1] += vj2.transpose(0,2,1) * .5
        vk1 += vk2.transpose(0,2,1)
        vj1 = vj1.reshape(3,3,nao,nao)
        vk1 = vk1.reshape(3,3,nao,nao)
        t1 = log.timer_debug1('contracting int2e_ipvip1 for atom %d'%ia, *t1)

        s1ao = numpy.zeros((3,nao,nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
        s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

        ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1_diag[:,:,p0:p1], dm0[p0:p1])*2
        ek[i0,i0] += numpy.einsum('xypq,pq->xy', vk1_diag[:,:,p0:p1], dm0[p0:p1])
        e1[i0,i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            # *2 for +c.c.
            ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm0[q0:q1])*4
            ek[i0,j0] += numpy.einsum('xypq,pq->xy', vk1[:,:,q0:q1], dm0[q0:q1])
            e1[i0,j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T
            ek[j0,i0] = ek[i0,j0].T

    log.timer('RHF partial hessian', *time0)
    return e1, ej, ek

def _make_vhfopt(mol, dms, key, vhf_intor):
    if not hasattr(_vhf.libcvhf, vhf_intor):
        return None

    vhfopt = _vhf.VHFOpt(mol, vhf_intor, 'CVHF'+key+'_prescreen',
                         'CVHF'+key+'_direct_scf')
    dms = numpy.asarray(dms, order='C')
    if dms.ndim == 3:
        n_dm = dms.shape[0]
    else:
        n_dm = 1
    ao_loc = mol.ao_loc_nr()
    fsetdm = getattr(_vhf.libcvhf, 'CVHF'+key+'_direct_scf_dm')
    fsetdm(vhfopt._this,
           dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
           ao_loc.ctypes.data_as(ctypes.c_void_p),
           mol._atm.ctypes.data_as(ctypes.c_void_p), mol.natm,
           mol._bas.ctypes.data_as(ctypes.c_void_p), mol.nbas,
           mol._env.ctypes.data_as(ctypes.c_void_p))

    # Update the vhfopt's attributes intor.  Function direct_mapdm needs
    # vhfopt._intor and vhfopt._cintopt to compute J/K.
    if vhf_intor != 'int2e_'+key:
        vhfopt._intor = mol._add_suffix('int2e_'+key)
        vhfopt._cintopt = None
    return vhfopt


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    time0 = t1 = (time.clock(), time.time())
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    aoslices = mol.aoslice_by_atom()
    h1ao = [None] * mol.natm
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vj2, vk1, vk2 = _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice)
        vhf = vj1 - vk1*.5
        vhf[:,p0:p1] += vj2 - vk2*.5
        h1 = vhf + vhf.transpose(0,2,1)
        h1 += hcore_deriv(ia)

        if chkfile is None:
            h1ao[ia] = h1
        else:
            key = 'scf_f1ao/%d' % ia
            lib.chkfile.save(chkfile, key, h1)
    if chkfile is None:
        return h1ao
    else:
        return chkfile

def get_hcore(mol):
    '''Part of the second derivatives of core Hamiltonian'''
    h1aa = mol.intor('int1e_ipipkin', comp=9)
    h1ab = mol.intor('int1e_ipkinip', comp=9)
    if mol._pseudo:
        NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa+= mol.intor('int1e_ipipnuc', comp=9)
        h1ab+= mol.intor('int1e_ipnucip', comp=9)
    if mol.has_ecp():
        h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
    nao = h1aa.shape[-1]
    return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

def get_ovlp(mol):
    s1a =-mol.intor('int1e_ipovlp', comp=3)
    nao = s1a.shape[-1]
    s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    return s1aa, s1ab, s1a

def _get_jk(mol, intor, comp, aosym, script_dms,
            shls_slice=None, cintopt=None, vhfopt=None):
    intor = mol._add_suffix(intor)
    scripts = script_dms[::2]
    dms = script_dms[1::2]
    vs = _vhf.direct_bindm(intor, aosym, scripts, dms, comp,
                           mol._atm, mol._bas, mol._env, vhfopt=vhfopt,
                           cintopt=cintopt, shls_slice=shls_slice)
    for k, script in enumerate(scripts):
        if 's2' in script:
            hermi = 1
        elif 'a2' in script:
            hermi = 2
        else:
            continue

        shape = vs[k].shape
        if shape[-2] == shape[-1]:
            if comp > 1:
                for i in range(comp):
                    lib.hermi_triu(vs[k][i], hermi=hermi, inplace=True)
            else:
                lib.hermi_triu(vs[k], hermi=hermi, inplace=True)
    return vs

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
              fx=None, atmlst=None, max_memory=4000, verbose=None):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nmo*nocc*3*6)))
    mo1s = [None] * mol.natm
    e1s = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo = []
        h1vo = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo.append(_ao2mo(s1ao))
            if isinstance(h1ao_or_chkfile, str):
                key = 'scf_f1ao/%d' % ia
                h1ao = lib.chkfile.load(h1ao_or_chkfile, key)
            else:
                h1ao = h1ao_or_chkfile[ia]
            h1vo.append(_ao2mo(h1ao))

        h1vo = numpy.vstack(h1vo)
        s1vo = numpy.vstack(s1vo)
        mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo)
        mo1 = numpy.einsum('pq,xqi->xpi', mo_coeff, mo1).reshape(-1,3,nao,nocc)
        e1 = e1.reshape(-1,3,nocc,nocc)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            if isinstance(h1ao_or_chkfile, str):
                key = 'scf_mo1/%d' % ia
                lib.chkfile.save(h1ao_or_chkfile, key, mo1[k])
            else:
                mo1s[ia] = mo1[k]
            e1s[ia] = e1[k].reshape(3,nocc,nocc)
        mo1 = e1 = None

    if isinstance(h1ao_or_chkfile, str):
        return h1ao_or_chkfile, e1s
    else:
        return mo1s, e1s

def gen_vind(mf, mo_coeff, mo_occ):
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    vresp = _gen_rhf_response(mf, mo_coeff, mo_occ, hermi=1)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmo,nocc)
        nset = len(mo1)
        dm1 = numpy.empty((nset,nao,nao))
        for i, x in enumerate(mo1):
            dm = reduce(numpy.dot, (mo_coeff, x*2, mocc.T)) # *2 for double occupancy
            dm1[i] = dm + dm.T
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1)
        for i, x in enumerate(v1):
            v1vo[i] = reduce(numpy.dot, (mo_coeff.T, x, mocc))
        return v1vo
    return fx

def hess_nuc(mol, atmlst=None):
    h = numpy.zeros((mol.natm,mol.natm,3,3))
    qs = numpy.asarray([mol.atom_charge(i) for i in range(mol.natm)])
    rs = numpy.asarray([mol.atom_coord(i) for i in range(mol.natm)])
    for i in range(mol.natm):
        r12 = rs[i] - rs
        s12 = numpy.sqrt(numpy.einsum('ki,ki->k', r12, r12))
        s12[i] = 1e60
        tmp1 = qs[i] * qs / s12**3
        tmp2 = numpy.einsum('k, ki,kj->kij',-3*qs[i]*qs/s12**5, r12, r12)

        h[i,i,0,0] = \
        h[i,i,1,1] = \
        h[i,i,2,2] = -tmp1.sum()
        h[i,i] -= numpy.einsum('kij->ij', tmp2)

        h[i,:,0,0] += tmp1
        h[i,:,1,1] += tmp1
        h[i,:,2,2] += tmp1
        h[i,:] += tmp2

    if atmlst is not None:
        h = h[atmlst][:,atmlst]
    return h


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj.base

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

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
        h1ao = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao_i = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/%d' % ia)
            h1ao += numpy.einsum('x,xij->ij', x[ia], h1ao_i)
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1vo = reduce(numpy.dot, (mo_coeff.T, s1ao, mocc))
        h1vo = reduce(numpy.dot, (mo_coeff.T, h1ao, mocc))
        mo1, mo_e1 = cphf.solve(fvind, mo_energy, mo_occ, h1vo, s1vo)
        mo1 = numpy.dot(mo_coeff, mo1)
        mo_e1 = mo_e1.reshape(nocc,nocc)
        dm1 = numpy.einsum('pi,qi->pq', mo1, mocc)
        dme1 = numpy.einsum('pi,qi,i->pq', mo1, mocc, mo_energy[mo_occ>0])
        dme1 = dme1 + dme1.T + reduce(numpy.dot, (mocc, mo_e1.T, mocc.T))

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            h1ao = lib.chkfile.load(hobj.chkfile, 'scf_f1ao/%s'%ja)
            hx[ja] += numpy.einsum('xpq,pq->x', h1ao, dm1) * 4
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1]) * 2
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1]) * 2
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


class Hessian(lib.StreamObject):
    '''Non-relativistic restricted Hartree-Fock hessian'''
    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.base = scf_method
        self.chkfile = scf_method.chkfile
        self.max_memory = self.mol.max_memory

        self.atmlst = range(self.mol.natm)
        self.de = numpy.zeros((0,0,3,3))  # (A,B,dR_A,dR_B)
        self._keys = set(self.__dict__.keys())

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            return with_x2c.hcore_deriv_generator(deriv=2)

        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        nbas = mol.nbas
        nao = mol.nao_nr()
        h1aa, h1ab = self.get_hcore(mol)
        def get_hcore(iatm, jatm):
            ish0, ish1, i0, i1 = aoslices[iatm]
            jsh0, jsh1, j0, j1 = aoslices[jatm]
            zi = mol.atom_charge(iatm)
            zj = mol.atom_charge(jatm)
            if iatm == jatm:
                with mol.with_rinv_at_nucleus(iatm):
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    if with_ecp and iatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9)
                rinv2aa = rinv2aa.reshape(3,3,nao,nao)
                rinv2ab = rinv2ab.reshape(3,3,nao,nao)
                hcore = -rinv2aa - rinv2ab
                hcore[:,:,i0:i1] += h1aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2ab[:,:,i0:i1]
                hcore[:,:,:,i0:i1] += rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
                hcore[:,:,:,i0:i1] += rinv2ab[:,:,:,i0:i1]
                hcore[:,:,i0:i1,i0:i1] += h1ab[:,:,i0:i1,i0:i1]

            else:
                hcore = numpy.zeros((3,3,nao,nao))
                hcore[:,:,i0:i1,j0:j1] += h1ab[:,:,i0:i1,j0:j1]
                with mol.with_rinv_at_nucleus(iatm):
                    shls_slice = (jsh0, jsh1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    if with_ecp and iatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,j0:j1] += rinv2aa.reshape(3,3,j1-j0,nao)
                    hcore[:,:,j0:j1] += rinv2ab.reshape(3,3,j1-j0,nao).transpose(1,0,2,3)

                with mol.with_rinv_at_nucleus(jatm):
                    shls_slice = (ish0, ish1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zj
                    rinv2ab *= zj
                    if with_ecp and jatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,i0:i1] += rinv2aa.reshape(3,3,i1-i0,nao)
                    hcore[:,:,i0:i1] += rinv2ab.reshape(3,3,i1-i0,nao)
            return hcore + hcore.conj().transpose(0,1,3,2)
        return get_hcore

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao_or_chkfile,
                         fx, atmlst, max_memory, verbose)

    def hess_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        return hess_nuc(mol, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (time.clock(), time.time())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
        self.de = de + self.hess_nuc(self.mol, atmlst=atmlst)
        return self.de
    hess = kernel

    gen_hop = gen_hop

# Inject to RHF class
from pyscf import scf
scf.hf.RHF.Hessian = lib.class_as_method(Hessian)


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
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = mf.Hessian()
    e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
    print(lib.finger(e2) - -0.50693144355876429)
    #from hessian import rhf_o0
    #e2ref = rhf_o0.Hessian(mf).kernel().transpose(0,2,1,3).reshape(n3,n3)
    #print numpy.linalg.norm(e2-e2ref)
    #print numpy.allclose(e2,e2ref)

    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            mf = scf.RHF(mol).run(conv_tol=1e-14)
            e1a = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i] - inc
            mf = scf.RHF(mol).run(conv_tol=1e-14)
            e1b = mf.nuc_grad_method().kernel()
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    e2ref = [grad_full(ia, .5e-4) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(n3,n3)
    print(numpy.linalg.norm(e2-e2ref))
    print(abs(e2-e2ref).max())
    print(numpy.allclose(e2,e2ref,atol=1e-6))

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
