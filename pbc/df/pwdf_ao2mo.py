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


def get_eri(mydf, kpts=None, compact=True):
    cell = mydf.cell
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    elif numpy.shape(kpts) == (3,):
        kptijkl = numpy.vstack([kpts]*4)
    else:
        kptijkl = numpy.reshape(kpts, (4,3))

    kpti, kptj, kptk, kptl = kptijkl
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .8

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
            lib.dot(pqkR, pqkR.T, 1, eriR, 1)
            lib.dot(pqkI, pqkI.T, 1, eriR, 1)
        pqkR = LkR = pqkI = LkI = coulG = None
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
        return (eriR.reshape((nao,)*4).transpose(0,1,3,2) +
                eriI.reshape((nao,)*4).transpose(0,1,3,2)*1j).reshape(nao**2,-1)

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
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            zdotNC(pqkR, pqkI, rskR.T, rskI.T, 1, eriR, eriI, 1)
        return (eriR+eriI*1j)


def general(mydf, mo_coeffs, kpts=None, compact=True):
#    from pyscf.pbc.df import mdf_ao2mo
#    return mdf_ao2mo.general(mydf, mo_coeffs, kpts, compact)
    cell = mydf.cell
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    elif numpy.shape(kpts) == (3,):
        kptijkl = numpy.vstack([kpts]*4)
    else:
        kptijkl = numpy.reshape(kpts, (4,3))

    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    nao = mo_coeffs[0].shape[0]
    nao_pair = nao * (nao+1) // 2
    all_real = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = (mydf.max_memory - lib.current_memory()[0]) * .5

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < 1e-9 and all_real:
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        eri_mo = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        def trans(pqkR, ijR, klR, buf):
            buf = lib.transpose(pqkR, out=buf)
            ijR = _ao2mo.nr_e2(buf, moij, ijslice, aosym='s2', mosym=ijmosym, out=ijR)
            if sym:
                klR = ijR
            else:
                klR = _ao2mo.nr_e2(buf, mokl, klslice, aosym='s2', mosym=klmosym, out=klR)
            return ijR, klR, buf

        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        ijR = ijI = klR = klI = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory,
                                aosym='s2'):
            ijR, klR, buf = trans(pqkR, ijR, klR, buf)
            ijI, klI, buf = trans(pqkI, ijI, klI, buf)
            vG = numpy.sqrt(coulG[p0:p1]).reshape(-1,1)
            ijR *= vG
            ijI *= vG
            lib.dot(ijR.T, klR, 1, eri_mo, 1)
            lib.dot(ijI.T, klI, 1, eri_mo, 1)
        return eri_mo

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
    elif (abs(kpti-kptl).sum() < 1e-9) and (abs(kptj-kptk).sum() < 1e-9):
        mos = []
        for c in mo_coeffs:
            if c.dtype == numpy.float64:
                mos.append(c+0j)
            else:
                mos.append(c)
        mo_coeffs = mos
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])
        lkmosym, nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        tao = []
        ao_loc = None
        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        ijz = lkz = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            ijz = _ao2mo.r_e2(buf, moij, ijslice, tao, ao_loc, out=ijz)
            if sym:
                lkz = ijz
            else:
                lkz = _ao2mo.r_e2(buf, molk, lkslice, tao, ao_loc, out=lkz)
            vG = numpy.sqrt(coulG[p0:p1]).reshape(-1,1)
            ijz *= vG
            lib.dot(ijz.T, lkz.conj(), 1, eri_mo, 1)
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
        mos = []
        for c in mo_coeffs:
            if c.dtype == numpy.float64:
                mos.append(c+0j)
            else:
                mos.append(c)
        mo_coeffs = mos
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])
        lkmosym, nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)

        coulG = mydf.weighted_coulG(kptj-kpti, False, mydf.gs)
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mydf.gs, kptijkl[:2], max_memory=max_memory*.5),
                         mydf.pw_loop(mydf.gs,-kptijkl[2:], max_memory=max_memory*.5)):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            ijz = _ao2mo.r_e2(buf, moij, ijslice, tao, ao_loc, out=ijz)
            buf = lib.transpose(rskR+rskI*1j, out=buf)
            lkz = _ao2mo.r_e2(buf, molk, lkslice, tao, ao_loc, out=lkz)
            vG = numpy.sqrt(coulG[p0:p1]).reshape(-1,1)
            ijz *= vG
            lib.dot(ijz.T, lkz.conj(), 1, eri_mo, 1)
        eri_mo = lib.transpose(eri_mo.reshape(-1,nmol,nmok), axes=(0,2,1))
        return eri_mo.reshape(nij_pair,nlk_pair)


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
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
    kpts = numpy.random.random((4,3))
    #kpts[3] = -numpy.einsum('ij->j', kpts[:3])
    kpts[3] = kpts[0]
    kpts[2] = kpts[1]
    with_df = pwdf.PWDF(cell)
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
