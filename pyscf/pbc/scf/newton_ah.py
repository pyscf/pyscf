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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Co-iterative augmented hessian second order SCF solver (CIAH-SOSCF)
'''

from functools import reduce
import numpy
from pyscf import lib
from pyscf.pbc.scf import khf, kuhf

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
    fock = [reduce(numpy.dot, (mo_coeff[k].T.conj(), fock_ao[k], mo_coeff[k]))
            for k in range(nkpts)]

    g = [fock[k][viridx[k][:,None],occidx[k]] * 2 for k in range(nkpts)]

    foo = [fock[k][occidx[k][:,None],occidx[k]] for k in range(nkpts)]
    fvv = [fock[k][viridx[k][:,None],viridx[k]] for k in range(nkpts)]

    h_diag = [(fvv[k].diagonal().reshape(-1,1)-foo[k].diagonal()) * 2
              for k in range(nkpts)]

    vind = _gen_rhf_response(mf, mo_coeff, mo_occ, singlet=None, hermi=1)

    def h_op(x1):
        x1 = _unpack(x1, mo_occ)
        dm1 = []
        for k in range(nkpts):
            # *2 for double occupancy
            d1 = reduce(numpy.dot, (orbv[k], x1[k]*2, orbo[k].T.conj()))
            dm1.append(d1+d1.T.conj())

        v1 = vind(lib.asarray(dm1))
        x2 = [0] * nkpts
        for k in range(nkpts):
            x2[k] = numpy.einsum('ps,sq->pq', fvv[k], x1[k]) * 2
            x2[k]-= numpy.einsum('ps,rp->rs', foo[k], x1[k]) * 2
            x2[k] += reduce(numpy.dot, (orbv[k].T.conj(), v1[k], orbo[k])) * 2
        return numpy.hstack([x.ravel() for x in x2])

    return (numpy.hstack([x.ravel() for x in g]), h_op,
            numpy.hstack([x.ravel().real for x in h_diag]))

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
    focka = [reduce(numpy.dot, (moa[k].T.conj(), fock_ao[0][k], moa[k]))
             for k in range(nkpts)]
    fockb = [reduce(numpy.dot, (mob[k].T.conj(), fock_ao[1][k], mob[k]))
             for k in range(nkpts)]
    fooa = [focka[k][occidxa[k][:,None],occidxa[k]] for k in range(nkpts)]
    fvva = [focka[k][viridxa[k][:,None],viridxa[k]] for k in range(nkpts)]
    foob = [fockb[k][occidxb[k][:,None],occidxb[k]] for k in range(nkpts)]
    fvvb = [fockb[k][viridxb[k][:,None],viridxb[k]] for k in range(nkpts)]

    g = ([focka[k][viridxa[k][:,None],occidxa[k]] for k in range(nkpts)] +
         [fockb[k][viridxb[k][:,None],occidxb[k]] for k in range(nkpts)])

    h_diag = ([fvva[k].diagonal().reshape(-1,1)-fooa[k].diagonal() for k in range(nkpts)] +
              [fvvb[k].diagonal().reshape(-1,1)-foob[k].diagonal() for k in range(nkpts)])

    vind = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)
    nao = orboa[0].shape[0]

    def h_op(x1):
        x1a = _unpack(x1[:tot_vopair_a], mo_occ[0])
        x1b = _unpack(x1[tot_vopair_a:], mo_occ[1])
        dm1 = numpy.empty((2,nkpts,nao,nao), dtype=x1.dtype)
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (orbva[k], x1a[k], orboa[k].T.conj()))
            dm1[0,k] = d1+d1.T.conj()
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (orbvb[k], x1b[k], orbob[k].T.conj()))
            dm1[1,k] = d1+d1.T.conj()
        v1 = vind(dm1)

        x2a = [0] * nkpts
        x2b = [0] * nkpts
        for k in range(nkpts):
            x2a[k] = numpy.einsum('ps,sq->pq', fvva[k], x1a[k])
            x2a[k]-= numpy.einsum('ps,rp->rs', fooa[k], x1a[k])
            x2b[k] = numpy.einsum('ps,sq->pq', fvvb[k], x1b[k])
            x2b[k]-= numpy.einsum('ps,rp->rs', foob[k], x1b[k])

            x2a[k] += reduce(numpy.dot, (orbva[k].T.conj(), v1[0][k], orboa[k]))
            x2b[k] += reduce(numpy.dot, (orbvb[k].T.conj(), v1[1][k], orbob[k]))

        return numpy.hstack([x.ravel() for x in (x2a+x2b)])

    return (numpy.hstack([x.ravel() for x in g]), h_op,
            numpy.hstack([x.ravel() for x in h_diag]))

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    from pyscf.pbc.dft import numint
    assert(isinstance(mf, khf.KRHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if hasattr(mf, 'xc') and hasattr(mf, '_numint'):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        if singlet is None:  # for newton solver
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                                mo_occ, 0, kpts)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2,
                                                spin=1, kpts=kpts)
        dm0 = None #mf.make_rdm1(mo_coeff, mo_occ)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, kpts, max_memory=max_memory)
                if abs(hyb) > 1e-10:
                    if hermi != 2:
                        vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                        v1 += vj - .5 * hyb * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if abs(hyb) > 1e-10:
                    if hermi != 2:
                        vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                        v1 += vj - .5 * hyb * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                return v1
        else:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if abs(hyb) > 1e-10:
                    v1 += -.5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)

    return vind

def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None,
                      with_j=True, hermi=0, max_memory=None):
    from pyscf.pbc.dft import numint
    assert(isinstance(mf, kuhf.KUHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if hasattr(mf, 'xc') and hasattr(mf, '_numint'):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1, kpts)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].T.conj()),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].T.conj()))
        dm0 = None

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, kpts, max_memory=max_memory)
            if abs(hyb) < 1e-10:
                if with_j:
                    vj = mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                    v1 += vj[0] + vj[1]
            else:
                if with_j:
                    vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                    v1 += vj[0] + vj[1] - vk * hyb
                else:
                    v1 -= hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)

    return vind


def _unpack(vo, mo_occ):
    z = []
    ip = 0
    for occ in mo_occ:
        nmo = occ.size
        no = numpy.count_nonzero(occ>0)
        nv = nmo - no
        z.append(vo[ip:ip+nv*no].reshape(nv,no))
        ip += nv * no
    return z


def newton(mf):
    from pyscf.soscf import newton_ah
    from pyscf.pbc import scf as pscf
    if not isinstance(mf, pscf.khf.KSCF):
# Note for single k-point other than gamma point (mf.kpt != 0) mf object,
# orbital hessian is approximated by gamma point hessian.
        return newton_ah.newton(mf)

    if isinstance(mf, newton_ah._CIAH_SOSCF):
        return mf

    if mf.__doc__ is None:
        mf_doc = ''
    else:
        mf_doc = mf.__doc__

    if isinstance(mf, pscf.kuhf.KUHF):
        class SecondOrderKUHF(mf.__class__, newton_ah._CIAH_SOSCF):
            __doc__ = mf_doc + newton_ah._CIAH_SOSCF.__doc__
            __init__ = newton_ah._CIAH_SOSCF.__init__
            dump_flags = newton_ah._CIAH_SOSCF.dump_flags
            build = newton_ah._CIAH_SOSCF.build
            kernel = newton_ah._CIAH_SOSCF.kernel

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
                        dr[viridx[:,None]&occidx] = dx[p0:p0+nocc*nvir]
                        dr = dr - dr.T.conj()
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

        return SecondOrderKUHF(mf)

    elif isinstance(mf, pscf.krohf.KROHF):
        raise NotImplementedError

    elif isinstance(mf, pscf.kghf.KGHF):
        raise NotImplementedError

    else:
        class SecondOrderKRHF(mf.__class__, newton_ah._CIAH_SOSCF):
            __doc__ = mf_doc + newton_ah._CIAH_SOSCF.__doc__
            __init__ = newton_ah._CIAH_SOSCF.__init__
            dump_flags = newton_ah._CIAH_SOSCF.dump_flags
            build = newton_ah._CIAH_SOSCF.build
            kernel = newton_ah._CIAH_SOSCF.kernel

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
                    dr[viridx[:,None]&occidx] = dx[p0:p0+nocc*nvir]
                    dr = dr - dr.T.conj()
                    p0 += nocc * nvir

                    u1 = newton_ah.expmat(dr)
                    if isinstance(u0, int) and u0 == 1:
                        u.append(u1)
                    else:
                        u.append(numpy.dot(u0[k], u1))
                return lib.asarray(u)

            def rotate_mo(self, mo_coeff, u, log=None):
                return lib.asarray([numpy.dot(mo, u[k]) for k,mo in enumerate(mo_coeff)])

        return SecondOrderKRHF(mf)

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


