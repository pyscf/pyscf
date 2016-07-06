#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc import tools


def get_eri(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    elif numpy.shape(kpts) == (3,):
        kptijkl = numpy.vstack([kpts]*4)
    else:
        kptijkl = numpy.reshape(kpts, (4,3))
    if mydf._cderi is None:
        mydf.build()

    kpti, kptj, kptk, kptl = kptijkl
    auxcell = mydf.auxcell
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    for Lpq in mydf.load_Lpq(kptijkl[:2]):
        pass
    for jpq in mydf.load_j3c(kptijkl[:2]):
        pass

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < 1e-9:
        j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN, kptl-kptk)
        eriR = lib.dot(jpq.T, Lpq)
        eriR = lib.transpose_sum(eriR, inplace=True)
        lib.dot(lib.dot(Lpq.T, j2c), Lpq, -1, eriR, 1)
        jpq = j2c = None

        coulG = tools.get_coulG(cell, kptj-kpti, gs=mydf.gs) / cell.vol
        max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
        trilidx = numpy.tril_indices(nao)
        for pqkR, LkR, pqkI, LkI, p0, p1 \
                in mydf.pw_loop(cell, auxcell, mydf.gs, kptijkl[:2], max_memory):
            pqkR = numpy.asarray(pqkR.reshape(nao,nao,-1)[trilidx], order='C')
            pqkI = numpy.asarray(pqkI.reshape(nao,nao,-1)[trilidx], order='C')
            # Lpq is real here
            lib.dot(Lpq.T, LkR, -1, pqkR, 1)
            lib.dot(Lpq.T, LkI, -1, pqkI, 1)
            vG = numpy.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
            lib.dot(pqkR, pqkR.T, 1, eriR, 1)
            lib.dot(pqkI, pqkI.T, 1, eriR, 1)
        return eriR

####################
# (kpt) i == j == k == l != 0
#
# (kpt) i == l && j == k && i != j && j != k  =>
# both vbar and ovlp are zero. It corresponds to the exchange integral.
#
# complex integrals, N^4 elements
    elif (abs(kpti-kptl).sum() < 1e-9) and (abs(kptj-kptk).sum() < 1e-9):
        if Lpq.shape[1] == nao_pair:
            Lpq = lib.unpack_tril(Lpq).reshape(naux,-1)
        if jpq.shape[1] == nao_pair:
            jpq = lib.unpack_tril(jpq).reshape(naux,-1)

        j2c = auxcell.pbc_intor('cint2c2e_sph', 1, lib.HERMITIAN, kptl-kptk)
        Lpqc = Lpq.conj()
        eriR  = lib.dot(jpq.T, Lpqc)
        eriR += lib.dot(Lpq.T, jpq.conj())
        eriR -= reduce(lib.dot, (Lpq.T, j2c, Lpqc))
        eriI = eriR.imag.copy()
        eriR = eriR.real.copy()
        jpq = j2c = None

        LpqR = Lpq.real.copy()
        LpqI = Lpq.imag.copy()
        Lpq = Lpqc = None

        coulG = tools.get_coulG(cell, kptj-kpti, gs=mydf.gs) / cell.vol
        max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8
        for pqkR, LkR, pqkI, LkI, p0, p1 \
                in mydf.pw_loop(cell, auxcell, mydf.gs, kptijkl[:2], max_memory):
            vG = numpy.sqrt(coulG[p0:p1])
            lib.dot(LpqR.T, LkR, -1, pqkR, 1)
            lib.dot(LpqR.T, LkI, -1, pqkI, 1)
            lib.dot(LpqI.T, LkR, -1, pqkI, 1)
            lib.dot(LpqI.T, LkI,  1, pqkR, 1)
            pqkR *= vG
            pqkI *= vG
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            lib.dot(pqkR, pqkR.T, 1, eriR, 1)
            lib.dot(pqkI, pqkI.T, 1, eriR, 1)
            lib.dot(pqkI, pqkR.T, 1, eriI, 1)
            lib.dot(pqkR, pqkI.T,-1, eriI, 1)
# transpose(0,1,3,2) because
# j == k && i == l  =>
# (L|ij).transpose(0,2,1).conj() = (L^*|ji) = (L^*|kl)  =>  (M|kl)
# rho_rs(-G+k_rs) = conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        return (eriR.reshape((nao,)*4).transpose(0,1,3,2) +
                eriI.reshape((nao,)*4).transpose(0,1,3,2)*1j).reshape(nao**2,-1)

####################
# aosym = s1, complex integrals
#
# kpti == kptj  =>  kptl == kptk
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.
#
    else:
        if Lpq.shape[1] == nao_pair:
            Lpq = lib.unpack_tril(Lpq).reshape(naux, -1)
        if jpq.shape[1] == nao_pair:
            jpq = lib.unpack_tril(jpq).reshape(naux, -1)
        for Mrs in mydf.load_Lpq(kptijkl[2:]):
            if Mrs.shape[1] == nao_pair:
                Mrs = lib.unpack_tril(Mrs).reshape(naux, -1)
        for jrs in mydf.load_j3c(kptijkl[2:]):
            if jrs.shape[1] == nao_pair:
                jrs = lib.unpack_tril(jrs).reshape(naux, -1)

        j2c = auxcell.pbc_intor('cint2c2e_sph', 1, 0, kptl-kptk)
        if numpy.iscomplexobj(jpq):
            eriR  = lib.dot(jpq.T, Mrs)
            eriR += lib.dot(Lpq.T, jrs)
        else:
            eriR  = lib.dot(Lpq.T, jrs)
            eriR += lib.dot(jpq.T, Mrs)
        eriR -= reduce(lib.dot, (Lpq.T, j2c, Mrs))
        eriI = eriR.imag.copy()
        eriR = eriR.real.copy()
        jpq = jrs = j2c = None

        LpqR = Lpq.real.copy()
        LpqI = Lpq.imag.copy()
        MrsR = Mrs.real.copy()
        MrsI = Mrs.imag.copy()
        Lpq = Mrs = None

        coulG = tools.get_coulG(cell, kptj-kpti, gs=mydf.gs) / cell.vol
        max_memory = (mydf.max_memory - lib.current_memory()[0]) * .4

#:        for (pqkR, LkR, pqkI, LkI, coulG), (rskR, MkR, rskI, MkI, coulG1) in \
#:                lib.izip(mydf.pw_loop(cell, auxcell, mydf.gs, kptijkl[:2], max_memory),
#:                         mydf.pw_loop(cell, auxcell, mydf.gs,-kptijkl[2:], max_memory)):
#:            coulG = numpy.sqrt(coulG)
#:            pqk = pqkR + pqkI*1j
#:            Lk  = LkR  + LkI *1j
#:            pqk -= numpy.dot(Lpq.T, Lk)
#:            pqk *= coulG
#:            rsk = rskR + rskI*1j
#:            Mk  = MkR  + MkI *1j
#:            rsk -= numpy.dot(Mrs.T, Mk)
#:            rsk *= coulG
#:            v = numpy.dot(pqk, rsk.conj().T)
#:            eriR += v.real
#:            eriI += v.imag

        for (pqkR, LkR, pqkI, LkI, p0, p1), (rskR, MkR, rskI, MkI, q0, q1) in \
                lib.izip(mydf.pw_loop(cell, auxcell, mydf.gs, kptijkl[:2], max_memory),
                         mydf.pw_loop(cell, auxcell, mydf.gs,-kptijkl[2:], max_memory)):
            lib.dot(LpqR.T, LkR, -1, pqkR, 1)
            lib.dot(LpqR.T, LkI, -1, pqkI, 1)
            lib.dot(LpqI.T, LkR, -1, pqkI, 1)
            lib.dot(LpqI.T, LkI,  1, pqkR, 1)
            pqkR *= coulG[p0:p1]
            pqkI *= coulG[p0:p1]
            lib.dot(MrsR.T, MkR, -1, rskR, 1)
            lib.dot(MrsR.T, MkI, -1, rskI, 1)
            lib.dot(MrsI.T, MkR, -1, rskI, 1)
            lib.dot(MrsI.T, MkI,  1, rskR, 1)
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            lib.dot(pqkR, rskR.T, 1, eriR, 1)
            lib.dot(pqkI, rskI.T, 1, eriR, 1)
            lib.dot(pqkI, rskR.T, 1, eriI, 1)
            lib.dot(pqkR, rskI.T,-1, eriI, 1)
        return eriR + eriI*1j


def general(mydf, mo_coeffs, kpts=None, compact=True):
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4

    eri = mydf.get_eri(kpts)

####################
# gamma point, the integral is real and with s4 symmetry
    if eri.dtype == numpy.float64:
        return ao2mo.general(eri, mo_coeffs, compact=compact)
    else:
        mokl, klslice = ao2mo.incore._conc_mos(mo_coeffs[2], mo_coeffs[3],
                                               False)[2:]
        if mokl.dtype == numpy.float64:
            mokl = mokl + 0j
        nao = mo_coeffs[0].shape[0]
        nmoi = mo_coeffs[0].shape[1]
        nmoj = mo_coeffs[1].shape[1]
        nmok = mo_coeffs[2].shape[1]
        nmol = mo_coeffs[3].shape[1]
        moi = numpy.asarray(mo_coeffs[0], order='F')
        moj = numpy.asarray(mo_coeffs[1], order='F')
        tao = [0]
        ao_loc = None
        pqkl = _ao2mo.r_e2(eri.reshape(-1,nao**2), mokl, klslice, tao, ao_loc, aosym='s1')
        pqkl = pqkl.reshape(nao,nao,nmok*nmol)
        pjkl = numpy.empty((nao,nmoj,nmok*nmol), dtype=numpy.complex128)
        for i in range(nao):
            lib.dot(moj.T, pqkl[i], 1, pjkl[i], 0)
        pqkl = None
        eri_mo = lib.dot(moi.T.conj(), pjkl.reshape(nao,-1))
        return eri_mo.reshape(nmoi*nmoj,-1)


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc.df import xdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)

    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = numpy.random.random((4,3))
    kpts[3] = -numpy.einsum('ij->j', kpts[:3])
    with_df = xdf.XDF(cell)
    with_df.kpts = kpts
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, kpts)
    print abs(eri1-eri0).sum()
