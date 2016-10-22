#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Co-iterative augmented hessian (CIAH) second order SCF solver
'''

import sys
import time
import copy
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib

def gen_g_hop_rhf(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    cell = mf.cell
    nkpts = len(mo_occ)
    occidx = [numpy.where(mo_occ[k]==2)[0] for k in range(nkpts)]
    viridx = [numpy.where(mo_occ[k]==0)[0] for k in range(nkpts)]

    if fock_ao is None:
        if h1e is None: h1e = mf.get_hcore()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = h1e + mf.get_veff(cell, dm0)
    fock = [reduce(numpy.dot, (mo_coeff[k].T.conj(), fock_ao[k],
                               mo_coeff[k])) for k in range(nkpts)]

    g = [fock[k][occidx[k][:,None],viridx[k]].T * 2 for k in range(nkpts)]

    foo = [fock[k][occidx[k][:,None],occidx[k]] for k in range(nkpts)]
    fvv = [fock[k][viridx[k][:,None],viridx[k]] for k in range(nkpts)]

    h_diag = [(fvv[k].diagonal().reshape(-1,1)-foo[k].diagonal()) * 2
              for k in range(nkpts)]

    if hasattr(mf, 'xc') and hasattr(mf, '_numint'):
        raise NotImplementedError
        if mf.grids.coords is None:
            mf.grids.build()
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=(cell.spin>0)+1)
        rho0, vxc, fxc = mf._numint.cache_xc_kernel(cell, mf.grids, mf.xc,
                                                    mo_coeff, mo_occ, 0)
        dm0 = None #mf.make_rdm1(mo_coeff, mo_occ)
    else:
        hyb = None

    def h_op(x1):
        xtmp = []
        p0 = 0
        for k in range(nkpts):
            nocc = len(occidx[k])
            nvir = len(viridx[k])
            xtmp.append(x1[p0:p0+nocc*nvir].reshape(nvir,nocc))
            p0 += nocc * nvir
        x1 = xtmp

        dm1 = []
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (mo_coeff[k][:,viridx[k]], x1[k],
                                    mo_coeff[k][:,occidx[k]].T.conj()))
            dm1.append(d1+d1.T.conj())
        dm1 = lib.asarray(dm1)
        if hyb is None:
            v1 = mf.get_veff(cell, dm1)
        else:
            v1 = mf._numint.nr_rks_fxc(cell, mf.grids, mf.xc, dm0, dm1,
                                       0, 1, rho0, vxc, fxc)
            if abs(hyb) < 1e-10:
                v1 += mf.get_j(cell, dm1)
            else:
                vj, vk = mf.get_jk(cell, dm1)
                v1 += vj - vk * hyb * .5

        x2 = [0] * nkpts
        for k in range(nkpts):
            x2[k] = numpy.einsum('sp,sq->pq', fvv[k], x1[k].conj()) * 2
            x2[k]-= numpy.einsum('sp,rp->rs', foo[k], x1[k].conj()) * 2

            x2[k] += reduce(numpy.dot, (mo_coeff[k][:,occidx[k]].T.conj(), v1[k].T,
                                        mo_coeff[k][:,viridx[k]])).T * 4
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

    if fock_ao is None:
        if h1e is None: h1e = mf.get_hcore()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = h1e + mf.get_veff(cell, dm0)
    focka = [reduce(numpy.dot, (moa[k].T.conj(), fock_ao[0][k], moa[k]))
             for k in range(nkpts)]
    fockb = [reduce(numpy.dot, (mob[k].T.conj(), fock_ao[1][k], mob[k]))
             for k in range(nkpts)]

    g = ([focka[k][occidxa[k][:,None],viridxa[k]].T for k in range(nkpts)] +
         [fockb[k][occidxb[k][:,None],viridxb[k]].T for k in range(nkpts)])

    def make_hdiaga(k):
        return (focka[k][viridxa[k],viridxa[k]].reshape(-1,1) -
                focka[k][occidxa[k],occidxa[k]])
    def make_hdiagb(k):
        return (fockb[k][viridxb[k],viridxb[k]].reshape(-1,1) -
                fockb[k][occidxb[k],occidxb[k]])
    h_diag = ([make_hdiaga(k) for k in range(nkpts)] +
              [make_hdiagb(k) for k in range(nkpts)])

    if hasattr(mf, 'xc') and hasattr(mf, '_numint'):
        raise NotImplementedError
        if mf.grids.coords is None:
            mf.grids.build()
        hyb = mf._numint.hybrid_coeff(mf.xc, spin=(cell.spin>0)+1)
        rho0, vxc, fxc = mf._numint.cache_xc_kernel(cell, mf.grids, mf.xc,
                                                    mo_coeff, mo_occ, 1)
        #dm0 =(numpy.dot(mo_coeff[0]*mo_occ[0], mo_coeff[0].T),
        #      numpy.dot(mo_coeff[1]*mo_occ[1], mo_coeff[1].T))
        dm0 = None
    else:
        hyb = None

    def h_op(x1):
        p0 = 0
        x1a = []
        for k in range(nkpts):
            nocc = len(occidxa[k])
            nvir = len(viridxa[k])
            x1a.append(x1[p0:p0+nocc*nvir].reshape(nvir,nocc))
            p0 += nocc * nvir
        x1b = []
        for k in range(nkpts):
            nocc = len(occidxb[k])
            nvir = len(viridxb[k])
            x1b.append(x1[p0:p0+nocc*nvir].reshape(nvir,nocc))
            p0 += nocc * nvir

        dm1a = []
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (moa[k][:,viridxa[k]], x1a[k],
                                    moa[k][:,occidxa[k]].T.conj()))
            dm1a.append(d1+d1.T.conj())
        dm1b = []
        for k in range(nkpts):
            d1 = reduce(numpy.dot, (mob[k][:,viridxa[k]], x1b[k],
                                    mob[k][:,occidxa[k]].T.conj()))
            dm1b.append(d1+d1.T.conj())
        dm1 = lib.asarray([dm1a,dm1b])
        dm1a = dm1b = None
        if hyb is None:
            v1 = mf.get_veff(cell, dm1)
        else:
            v1 = mf._numint.nr_uks_fxc(cell, mf.grids, mf.xc, dm0, dm1,
                                       0, 1, rho0, vxc, fxc)
            if abs(hyb) < 1e-10:
                vj = mf.get_j(cell, dm1)
                v1 += vj[0] + vj[1]
            else:
                vj, vk = mf.get_jk(cell, dm1)
                v1 += vj[0]+vj[1] - vk * hyb * .5

        x2a = [0] * nkpts
        x2b = [0] * nkpts
        for k in range(nkpts):
            x2a[k] = numpy.einsum('sp,sq->pq', focka[k][viridxa[k][:,None],viridxa[k]], x1a[k].conj())
            x2a[k]-= numpy.einsum('sp,rp->rs', focka[k][occidxa[k][:,None],occidxa[k]], x1a[k].conj())
            x2b[k] = numpy.einsum('sp,sq->pq', fockb[k][viridxb[k][:,None],viridxb[k]], x1b[k].conj())
            x2b[k]-= numpy.einsum('sp,rp->rs', fockb[k][occidxb[k][:,None],occidxb[k]], x1b[k].conj())

            x2a[k] += reduce(numpy.dot, (moa[k][:,occidxa[k]].T.conj(), v1[0][k],
                                         moa[k][:,viridxa[k]])).T
            x2b[k] += reduce(numpy.dot, (mob[k][:,occidxb[k]].T.conj(), v1[1][k],
                                         mob[k][:,viridxb[k]])).T

        return numpy.hstack([x.ravel() for x in (x2a+x2b)])

    return (numpy.hstack([x.ravel() for x in g]), h_op,
            numpy.hstack([x.ravel() for x in h_diag]))


def newton(mf):
    from pyscf.scf import newton_ah
    from pyscf.pbc import scf as pscf
    if not isinstance(mf, (pscf.khf.KRHF, pscf.kuhf.KUHF)):
        return scf.newton_ah.newton(mf)

    KSCF = newton_ah.newton_SCF_class(mf)

    if isinstance(mf, pscf.kuhf.KUHF):
        class KUHF(KSCF):
            def build(self, cell=None):
                KSCF.build(self, cell)

            gen_g_hop = gen_g_hop_uhf

            def update_rotate_matrix(self, dx, mo_occ, u0=1):
                nkpts = len(mo_occ[0])
                nmo = mo_occ[0].shape[-1]
                p0 = 0
                u = []
                for occ in mo_occ:
                    ua = []
                    for k in range(nkpts):
                        occidx = occ[k] > 0
                        viridx = ~occidx
                        nocc = occidx.sum()
                        nvir = nmo - nocc
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

        return KUHF()

    else:
        class KRHF(KSCF):
            def build(self, cell=None):
                KSCF.build(self, cell)

            gen_g_hop = gen_g_hop_rhf

            def update_rotate_matrix(self, dx, mo_occ, u0=1):
                nmo = mo_occ.shape[-1]
                p0 = 0
                u = []
                for k, occ in enumerate(mo_occ):
                    occidx = occ > 0
                    viridx = ~occidx
                    nocc = occidx.sum()
                    nvir = nmo - nocc
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

        return KRHF()

if __name__ == '__main__':
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = 'ccpvdz'
    cell.h = numpy.eye(3) * 4
    cell.gs = [8] * 3
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


