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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Co-iterative augmented hessian second order SCF solver (CIAH-SOSCF)
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.pbc.scf import _response_functions  # noqa
from pyscf.soscf import newton_ah

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    cell = mf.cell
    nkpts = len(mo_occ)
    occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
    viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]
    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpts)]
    orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpts)]

    if fock_ao is None:
        if h1e is None: h1e = mf.get_hcore()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = h1e + mf.get_veff(cell, dm0)
    fock = [reduce(numpy.dot, (mo_coeff[k].conj().T, fock_ao[k], mo_coeff[k]))
            for k in range(nkpts)]

    g = [fock[k][viridx[k][:,None],occidx[k]] * 2 for k in range(nkpts)]

    foo = [fock[k][occidx[k][:,None],occidx[k]] for k in range(nkpts)]
    fvv = [fock[k][viridx[k][:,None],viridx[k]] for k in range(nkpts)]

    h_diag = [(fvv[k].diagonal().real[:,None]-foo[k].diagonal().real) * 2
              for k in range(nkpts)]

    vind = mf.gen_response(mo_coeff, mo_occ, singlet=None, hermi=1)

    def h_op(x1):
        x1 = _unpack(x1, mo_occ)
        dm1 = []
        for k in range(nkpts):
            # *2 for double occupancy
            d1 = reduce(numpy.dot, (orbv[k], x1[k]*2, orbo[k].conj().T))
            dm1.append(d1+d1.conj().T)

        v1 = vind(lib.asarray(dm1))
        x2 = [0] * nkpts
        for k in range(nkpts):
            x2[k] = numpy.einsum('ps,sq->pq', fvv[k], x1[k]) * 2
            x2[k]-= numpy.einsum('ps,rp->rs', foo[k], x1[k]) * 2
            x2[k] += reduce(numpy.dot, (orbv[k].conj().T, v1[k], orbo[k])) * 2
        return numpy.hstack([x.ravel() for x in x2])

    return (numpy.hstack([x.ravel() for x in g]), h_op,
            numpy.hstack([x.ravel() for x in h_diag]))

def gen_g_hop_uhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    cell = mf.cell
    nkpts = len(mo_occ[0])
    occidxa = [numpy.where(mo_occ[0][k]>0)[0] for k in range(nkpts)]
    occidxb = [numpy.where(mo_occ[1][k]>0)[0] for k in range(nkpts)]
    viridxa = [numpy.where(mo_occ[0][k]==0)[0] for k in range(nkpts)]
    viridxb = [numpy.where(mo_occ[1][k]==0)[0] for k in range(nkpts)]
    moa, mob = mo_coeff
    orboa = [moa[k][:,occidxa[k]] for k in range(nkpts)]
    orbva = [moa[k][:,viridxa[k]] for k in range(nkpts)]
    orbob = [mob[k][:,occidxb[k]] for k in range(nkpts)]
    orbvb = [mob[k][:,viridxb[k]] for k in range(nkpts)]
    tot_vopair_a = sum(len(occidxa[k])*len(viridxa[k]) for k in range(nkpts))

    if fock_ao is None:
        if h1e is None: h1e = mf.get_hcore()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = h1e + mf.get_veff(cell, dm0)
    focka = [reduce(numpy.dot, (moa[k].conj().T, fock_ao[0][k], moa[k]))
             for k in range(nkpts)]
    fockb = [reduce(numpy.dot, (mob[k].conj().T, fock_ao[1][k], mob[k]))
             for k in range(nkpts)]
    fooa = [focka[k][occidxa[k][:,None],occidxa[k]] for k in range(nkpts)]
    fvva = [focka[k][viridxa[k][:,None],viridxa[k]] for k in range(nkpts)]
    foob = [fockb[k][occidxb[k][:,None],occidxb[k]] for k in range(nkpts)]
    fvvb = [fockb[k][viridxb[k][:,None],viridxb[k]] for k in range(nkpts)]

    g = ([focka[k][viridxa[k][:,None],occidxa[k]] for k in range(nkpts)] +
         [fockb[k][viridxb[k][:,None],occidxb[k]] for k in range(nkpts)])

    h_diag = ([fvva[k].diagonal().real[:,None]-fooa[k].diagonal().real for k in range(nkpts)] +
              [fvvb[k].diagonal().real[:,None]-foob[k].diagonal().real for k in range(nkpts)])

    vind = mf.gen_response(mo_coeff, mo_occ, hermi=1)
    nao = orboa[0].shape[0]

    def h_op(x1):
        x1a = _unpack(x1[:tot_vopair_a], mo_occ[0])
        x1b = _unpack(x1[tot_vopair_a:], mo_occ[1])
        dm1 = numpy.empty((2,nkpts,nao,nao), dtype=x1.dtype)
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (orbva[k], x1a[k], orboa[k].conj().T))
            dm1[0,k] = d1+d1.conj().T
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (orbvb[k], x1b[k], orbob[k].conj().T))
            dm1[1,k] = d1+d1.conj().T
        v1 = vind(dm1)

        x2a = [0] * nkpts
        x2b = [0] * nkpts
        for k in range(nkpts):
            x2a[k] = numpy.einsum('ps,sq->pq', fvva[k], x1a[k])
            x2a[k]-= numpy.einsum('ps,rp->rs', fooa[k], x1a[k])
            x2b[k] = numpy.einsum('ps,sq->pq', fvvb[k], x1b[k])
            x2b[k]-= numpy.einsum('ps,rp->rs', foob[k], x1b[k])

            x2a[k] += reduce(numpy.dot, (orbva[k].conj().T, v1[0][k], orboa[k]))
            x2b[k] += reduce(numpy.dot, (orbvb[k].conj().T, v1[1][k], orbob[k]))
        return numpy.hstack([x.ravel() for x in (x2a+x2b)])

    return (numpy.hstack([x.ravel() for x in g]), h_op,
            numpy.hstack([x.ravel() for x in h_diag]))

def gen_g_hop_rohf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    if getattr(fock_ao, 'focka', None) is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock_ao = fock_ao.focka, fock_ao.fockb
    mo_occa = occidxa = [occ > 0 for occ in mo_occ]
    mo_occb = occidxb = [occ ==2 for occ in mo_occ]
    ug, uh_op, uh_diag = gen_g_hop_uhf(mf, (mo_coeff,)*2, (mo_occa,mo_occb),
                                       fock_ao, None)

    nkpts = len(mo_occ)
    idx_var_a = []
    idx_var_b = []
    p0 = 0
    for k in range(nkpts):
        viridxa = ~occidxa[k]
        viridxb = ~occidxb[k]
        uniq_var_a = viridxa[:,None] & occidxa[k]
        uniq_var_b = viridxb[:,None] & occidxb[k]
        uniq_ab = uniq_var_a | uniq_var_b
        nmo = len(mo_occ[k])

        n_uniq_ab = numpy.count_nonzero(uniq_ab)
        idx_array = numpy.zeros((nmo,nmo), dtype=int)
        idx_array[uniq_ab] = numpy.arange(n_uniq_ab)
        idx_var_a.append(p0 + idx_array[uniq_var_a])
        idx_var_b.append(p0 + idx_array[uniq_var_b])
        p0 += n_uniq_ab

    idx_var_a = numpy.hstack(idx_var_a)
    idx_var_b = numpy.hstack(idx_var_b)
    nvars = p0

    def sum_ab(x):
        x1 = numpy.zeros(nvars, dtype=x.dtype)
        x1[idx_var_a]  = x[:len(idx_var_a)]
        x1[idx_var_b] += x[len(idx_var_a):]
        return x1

    g = sum_ab(ug)
    h_diag = sum_ab(uh_diag)
    def h_op(x):
        # unpack ROHF rotation parameters
        x1 = numpy.hstack((x[idx_var_a], x[idx_var_b]))
        return sum_ab(uh_op(x1))

    return g, h_op, h_diag


# Be careful with the parameter ordering conventions are different for the
# _unpack function here and the one in tdscf.krhf
def _unpack(vo, mo_occ):
    z = []
    p1 = 0
    for k, occ in enumerate(mo_occ):
        no = numpy.count_nonzero(occ > 0)
        nv = occ.size - no
        p0, p1 = p1, p1 + nv * no
        z.append(vo[p0:p1].reshape(nv,no))
    return z

class _SecondOrderKRHF(newton_ah._CIAH_SOSCF):
    gen_g_hop = gen_g_hop_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        p0 = 0
        u = []
        for k, occ in enumerate(mo_occ):
            occidx = occ > 0
            viridx = ~occidx
            nocc = occidx.sum()
            nvir = viridx.sum()
            nmo = nocc + nvir
            dr = numpy.zeros((nmo,nmo), dtype=dx.dtype)
            dr[viridx[:,None] & occidx] = dx[p0:p0+nocc*nvir]
            dr = dr - dr.conj().T
            p0 += nocc * nvir

            u1 = newton_ah.expmat(dr)
            if isinstance(u0, int) and u0 == 1:
                u.append(u1)
            else:
                u.append(numpy.dot(u0[k], u1))
        return lib.asarray(u)

    def rotate_mo(self, mo_coeff, u, log=None):
        return lib.asarray([numpy.dot(mo, u[k]) for k,mo in enumerate(mo_coeff)])

class _SecondOrderKUHF(newton_ah._CIAH_SOSCF):
    gen_g_hop = gen_g_hop_uhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        nkpts = len(mo_occ[0])
        p0 = 0
        u = []
        for occ in mo_occ:
            ua = []
            for k in range(nkpts):
                occidx = occ[k] > 0
                viridx = ~occidx
                nocc = occidx.sum()
                nvir = viridx.sum()
                nmo = nocc + nvir
                dr = numpy.zeros((nmo,nmo), dtype=dx.dtype)
                dr[viridx[:,None] & occidx] = dx[p0:p0+nocc*nvir]
                dr = dr - dr.conj().T
                p0 += nocc * nvir
                u1 = newton_ah.expmat(dr)
                if isinstance(u0, int) and u0 == 1:
                    ua.append(u1)
                else:
                    ua.append(numpy.dot(u0[k], u1))
            u.append(ua)
        return lib.asarray(u)

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = ([numpy.dot(mo, u[0][k]) for k, mo in enumerate(mo_coeff[0])],
              [numpy.dot(mo, u[1][k]) for k, mo in enumerate(mo_coeff[1])])
        return lib.asarray(mo)

class _SecondOrderKROHF(_SecondOrderKRHF):
    gen_g_hop = gen_g_hop_rohf

def newton(mf):
    from pyscf.pbc import scf as pscf
    if not isinstance(mf, pscf.khf.KSCF):
        # Note for single k-point other than gamma point (mf.kpt != 0) mf object,
        # orbital hessian is approximated by gamma point hessian.
        return newton_ah.newton(mf)

    if isinstance(mf, newton_ah._CIAH_SOSCF):
        return mf

    if isinstance(mf, pscf.kuhf.KUHF):
        cls = _SecondOrderKUHF
    elif isinstance(mf, pscf.krohf.KROHF):
        cls = _SecondOrderKROHF
    elif isinstance(mf, pscf.kghf.KGHF):
        raise NotImplementedError
    else:
        cls = _SecondOrderKRHF
    return lib.set_class(cls(mf), (cls, mf.__class__))

if __name__ == '__main__':
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = 'ccpvdz'
    cell.a = numpy.eye(3) * 4
    cell.mesh = [11] * 3
    cell.verbose = 4
    cell.build()
    nks = [2,1,1]
    mf = pscf.KRHF(cell, cell.make_kpts(nks))
    mf.max_cycle = 2
    mf.kernel()
    mf.max_cycle = 5
    pscf.newton(mf).kernel()

    mf = pscf.KUHF(cell, cell.make_kpts(nks))
    mf.max_cycle = 2
    mf.kernel()
    mf.max_cycle = 5
    pscf.newton(mf).kernel()
