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
from pyscf.pbc.scf import khf, kuhf, krohf

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

    vind = _gen_rhf_response(mf, mo_coeff, mo_occ, singlet=None, hermi=1)

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

    vind = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)
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
        nocca = numpy.count_nonzero(mo_occa[k])
        noccb = numpy.count_nonzero(mo_occb[k])
        nvira = nmo - nocca
        nvirb = nmo - noccb

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

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None,
                      singlet=None, hermi=0, max_memory=None):
    from pyscf.pbc.dft import numint, multigrid
    assert(isinstance(mf, khf.KRHF))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = abs(hyb) > 1e-10
        if abs(omega) > 1e-10:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc, mo_coeff,
                                                mo_occ, 0, kpts)
        else:
            if isinstance(mo_occ, numpy.ndarray):
                mo_occ = mo_occ*.5
            else:
                mo_occ = [x*.5 for x in mo_occ]
            rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ]*2,
                                                spin=1, kpts=kpts)
        dm0 = None #mf.make_rdm1(mo_coeff, mo_occ)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:  # Without specify singlet, general case
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, kpts, max_memory=max_memory)
                if hybrid:
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
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts)
                        v1 += vj - .5 * hyb * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(cell, dm1, hermi=hermi, kpts=kpts)
                elif hermi != 2:
                    v1 += mf.get_j(cell, dm1, hermi=hermi, kpts=kpts)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1
                    v1 = numint.nr_rks_fxc_st(ni, cell, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc, kpts,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
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
    from pyscf.pbc.dft import numint, multigrid
    assert(isinstance(mf, (kuhf.KUHF, krohf.KROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    cell = mf.cell
    kpts = mf.kpts
    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)

        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
        hybrid = abs(hyb) > 1e-10
        if abs(omega) > 1e-10:  # For range separated Coulomb
            raise NotImplementedError

        if not hybrid and isinstance(mf.with_df, multigrid.MultiGridFFTDF):
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_uhf_response(mf, dm0, with_j, hermi)

        rho0, vxc, fxc = ni.cache_xc_kernel(cell, mf.grids, mf.xc,
                                            mo_coeff, mo_occ, 1, kpts)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].conj().T),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].conj().T))
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
            if not hybrid:
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

        return SecondOrderKUHF(mf)

    elif isinstance(mf, pscf.krohf.KROHF):
        class SecondOrderKROHF(SecondOrderKRHF):
            gen_g_hop = gen_g_hop_rohf

        return SecondOrderKROHF(mf)

    elif isinstance(mf, pscf.kghf.KGHF):
        raise NotImplementedError

    else:
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


