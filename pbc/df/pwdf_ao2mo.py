#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Integral transformation with analytic Fourier transformation
'''

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.pbc.df.df_ao2mo import _mo_as_complex, _dtrans, _ztrans


def get_eri(mydf, kpts=None, compact=True):
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .8)

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < 1e-9:
        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        eriR = numpy.zeros((nao_pair,nao_pair))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory,
                                aosym='s2'):
            vG = numpy.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
            lib.ddot(pqkR, pqkR.T, 1, eriR, 1)
            lib.ddot(pqkI, pqkI.T, 1, eriR, 1)
            pqkR = pqkI = None
        if not compact:
            eriR = ao2mo.restore(1, eriR, nao).reshape(nao**2,-1)
        return eriR

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
# complex integrals, N^4 elements
    elif (abs(kpti-kptl).sum() < 1e-9) and (abs(kptj-kptk).sum() < 1e-9):
        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory):
            vG = numpy.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            zdotNC(pqkR, pqkI, pqkR.T, pqkI.T, 1, eriR, eriI, 1)
            pqkR = pqkI = None
        pqkR = pqkI = coulG = None
# transpose(0,1,3,2) because
# j == k && i == l  =>
# (L|ij).transpose(0,2,1).conj() = (L^*|ji) = (L^*|kl)  =>  (M|kl)
# rho_rs(-G+k_rs) = conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        eri = lib.transpose((eriR+eriI*1j).reshape(-1,nao,nao), axes=(0,2,1))
        return eri.reshape(nao**2,-1)

####################
# aosym = s1, complex integrals
#
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.  =>  kptl == kptk
#
    else:
        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory*.5),
                         mydf.pw_loop(mydf.gs,-kptijkl[2:], max_memory=max_memory*.5)):
            pqkR *= coulG[p0:p1]
            pqkI *= coulG[p0:p1]
# rho'_rs(G-k_rs) = conj(rho_rs(-G+k_rs))
#                 = conj(rho_rs(-G+k_rs) - d_{k_rs:Q,rs} * Q(-G+k_rs))
#                 = rho_rs(G-k_rs) - conj(d_{k_rs:Q,rs}) * Q(G-k_rs)
# rho_pq(G+k_pq) * conj(rho'_rs(G-k_rs))
            zdotNC(pqkR, pqkI, rskR.T, rskI.T, 1, eriR, eriI, 1)
            pqkR = pqkI = rskR = rskI = None
        return (eriR+eriI*1j)


def general(mydf, mo_coeffs, kpts=None, compact=True):
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    all_real = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .5)

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < 1e-9 and all_real:
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        eri_mo = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))

        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        ijR = ijI = klR = klI = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory,
                                aosym='s2'):
            vG = numpy.sqrt(coulG[p0:p1])
            pqkR *= vG
            pqkI *= vG
            buf = lib.transpose(pqkR, out=buf)
            ijR, klR = _dtrans(buf, ijR, ijmosym, moij, ijslice,
                               buf, klR, klmosym, mokl, klslice, sym)
            lib.ddot(ijR.T, klR, 1, eri_mo, 1)
            buf = lib.transpose(pqkI, out=buf)
            ijI, klI = _dtrans(buf, ijI, ijmosym, moij, ijslice,
                               buf, klI, klmosym, mokl, klslice, sym)
            lib.ddot(ijI.T, klI, 1, eri_mo, 1)
            pqkR = pqkI = None
        return eri_mo

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
    elif (abs(kpti-kptl).sum() < 1e-9) and (abs(kptj-kptk).sum() < 1e-9):
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])[1:]
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        zij = zlk = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            buf *= numpy.sqrt(coulG[p0:p1]).reshape(-1,1)
            zij, zlk = _ztrans(buf, zij, moij, ijslice,
                               buf, zlk, molk, lkslice, sym)
            lib.dot(zij.T, zlk.conj(), 1, eri_mo, 1)
            pqkR = pqkI = None
        nmok = mo_coeffs[2].shape[1]
        nmol = mo_coeffs[3].shape[1]
        eri_mo = lib.transpose(eri_mo.reshape(-1,nmol,nmok), axes=(0,2,1))
        return eri_mo.reshape(nij_pair,nlk_pair)

####################
# aosym = s1, complex integrals
#
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.  =>  kptl == kptk
#
    else:
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3])[1:]
        eri_mo = numpy.zeros((nij_pair,nkl_pair), dtype=numpy.complex)

        tao = []
        ao_loc = None
        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        zij = zkl = buf = None
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory*.5),
                         mydf.pw_loop(mydf.gs,-kptijkl[2:], max_memory=max_memory*.5)):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            zij = _ao2mo.r_e2(buf, moij, ijslice, tao, ao_loc, out=zij)
            buf = lib.transpose(rskR-rskI*1j, out=buf)
            zkl = _ao2mo.r_e2(buf, mokl, klslice, tao, ao_loc, out=zkl)
            zij *= coulG[p0:p1].reshape(-1,1)
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            pqkR = pqkI = rskR = rskI = None
        return eri_mo

if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc import df

    L = 5.
    n = 1
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
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
    kpts = numpy.random.random((4,3)) * .5
    kpts[3] = -numpy.einsum('ij->j', kpts[:3])
    with_df = df.PWDF(cell)
    with_df.kpts = kpts
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, kpts)
    print abs(eri1.reshape(eri0.shape)-eri0).sum()

    kpts[3] = kpts[0]
    kpts[2] = kpts[1]
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, kpts)
    print abs(eri1.reshape(eri0.shape)-eri0).sum()

    with_df.kpts *= 0
    eri = ao2mo.restore(1, with_df.get_eri(with_df.kpts), nao)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, with_df.kpts)
    print abs(eri1.reshape(eri0.shape)-eri0).sum()

    mo = mo.real
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, with_df.kpts, compact=False)
    print abs(eri1.reshape(eri0.shape)-eri0).sum()
