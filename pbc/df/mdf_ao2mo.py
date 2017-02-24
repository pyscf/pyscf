#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC, KPT_DIFF_TOL
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.pbc.df.df_ao2mo import _mo_as_complex, _dtrans, _ztrans
from pyscf.pbc.df import df
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df import pwdf_ao2mo


def get_eri(mydf, kpts=None, compact=True):
    if mydf._cderi is None:
        mydf.build()

    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    eri = pwdf_ao2mo.get_eri(mydf, kptijkl, compact=True)
    if mydf.metric is None:
        mydf.__class__, cls_bak = df.DF, mydf.__class__
        eri += df_ao2mo.get_eri(mydf, kptijkl, compact)
        mydf.__class__ = cls_bak
        return eri

    nao = cell.nao_nr()
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0] - nao**4*8/1e6) * .8)

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < KPT_DIFF_TOL:
        eri *= .5  # because we'll do +cc later
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptijkl[:2], max_memory, True):
            lib.ddot(j3cR.T, LpqR, 1, eri, 1)
            LpqR = LpqI = j3cR = j3cI = None
        eri = lib.transpose_sum(eri, inplace=True)
        if not compact:
            eri = ao2mo.restore(1, eri, nao).reshape(nao**2,-1)
        return eri

####################
# (kpt) i == j == k == l != 0
#
# (kpt) i == l && j == k && i != j && j != k  =>
# both vbar and ovlp are zero. It corresponds to the exchange integral.
#
# complex integrals, N^4 elements
    elif (abs(kpti-kptl).sum() < KPT_DIFF_TOL) and (abs(kptj-kptk).sum() < KPT_DIFF_TOL):
        eriR = numpy.zeros((nao*nao,nao*nao))
        eriI = numpy.zeros((nao*nao,nao*nao))
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptijkl[:2], max_memory, False):
            zdotNC(j3cR.T, j3cI.T, LpqR, LpqI, 1, eriR, eriI, 1)
# eri == eri.transpose(3,2,1,0).conj()
#            zdotNC(LpqR.T, LpqI.T, j3cR, j3cI, 1, eriR, eriI, 1)
            LpqR = LpqI = j3cR = j3cI = None
# eri == eri.transpose(3,2,1,0).conj()
        eriR = lib.transpose_sum(eriR, inplace=True)
        buf = lib.transpose(eriI)
        eriI -= buf

        eriR = lib.transpose(eriR.reshape(-1,nao,nao), axes=(0,2,1), out=buf)
        eri += eriR.reshape(eri.shape)
        eriI = lib.transpose(eriI.reshape(-1,nao,nao), axes=(0,2,1), out=buf)
        eri += eriI.reshape(eri.shape)*1j
        return eri

####################
# aosym = s1, complex integrals
#
# kpti == kptj  =>  kptl == kptk
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.
#
    else:
        eriR = numpy.zeros((nao*nao,nao*nao))
        eriI = numpy.zeros((nao*nao,nao*nao))
        max_memory *= .5
        for (LpqR, LpqI, jpqR, jpqI), (LrsR, LrsI, jrsR, jrsI) in \
                lib.izip(mydf.sr_loop(kptijkl[:2], max_memory, False),
                         mydf.sr_loop(kptijkl[2:], max_memory, False)):
            zdotNN(jpqR.T, jpqI.T, LrsR, LrsI, 1, eriR, eriI, 1)
            zdotNN(LpqR.T, LpqI.T, jrsR, jrsI, 1, eriR, eriI, 1)
            LpqR = LpqI = jpqR = jpqI = LrsR = LrsI = jrsR = jrsI = None
        eri += eriR
        eri += eriI*1j
        return eri


def general(mydf, mo_coeffs, kpts=None, compact=True):
    if mydf._cderi is None:
        mydf.build()

    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    eri_mo = pwdf_ao2mo.general(mydf, mo_coeffs, kptijkl, compact)
    if mydf.metric is None:
        mydf.__class__, cls_bak = df.DF, mydf.__class__
        eri_mo += df_ao2mo.general(mydf, mo_coeffs, kptijkl, compact)
        mydf.__class__ = cls_bak
        return eri

    all_real = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .5)

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < KPT_DIFF_TOL and all_real:
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        if sym:
            eri_mo *= .5  # because we'll do +cc later

        ijR = klR = None
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptijkl[:2], max_memory, True):
            ijR, klR = _dtrans(LpqR, ijR, ijmosym, moij, ijslice,
                               j3cR, klR, klmosym, mokl, klslice, False)
            lib.ddot(ijR.T, klR, 1, eri_mo, 1)
            if not sym:
                ijR, klR = _dtrans(j3cR, ijR, ijmosym, moij, ijslice,
                                   LpqR, klR, klmosym, mokl, klslice, False)
                lib.ddot(ijR.T, klR, 1, eri_mo, 1)
            LpqR = LpqI = j3cR = j3cI = None
        if sym:
            eri_mo = lib.transpose_sum(eri_mo, inplace=True)
        return eri_mo

####################
# (kpt) i == j == k == l != 0
#
# (kpt) i == l && j == k && i != j && j != k  =>
# both vbar and ovlp are zero. It corresponds to the exchange integral.
#
# complex integrals, N^4 elements
    elif (abs(kpti-kptl).sum() < KPT_DIFF_TOL) and (abs(kptj-kptk).sum() < KPT_DIFF_TOL):
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])[1:]
        eri_lk = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        zij = zlk = buf = None
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptijkl[:2], max_memory, False):
            bufL = LpqR+LpqI*1j
            bufj = j3cR+j3cI*1j
            zij, zlk = _ztrans(bufL, zij, moij, ijslice,
                               bufj, zlk, molk, lkslice, False)
            lib.dot(zij.T, zlk.conj(), 1, eri_lk, 1)
            if not sym:
                zij, zlk = _ztrans(bufj, zij, moij, ijslice,
                                   bufL, zlk, molk, lkslice, False)
                lib.dot(zij.T, zlk.conj(), 1, eri_lk, 1)
            LpqR = LpqI = j3cR = j3cI = bufL = bufj = None
        if sym:
            eri_lk += lib.transpose(eri_lk).conj()

        nmok = mo_coeffs[2].shape[1]
        nmol = mo_coeffs[3].shape[1]
        eri_lk = lib.transpose(eri_lk.reshape(-1,nmol,nmok), axes=(0,2,1))
        eri_mo += eri_lk.reshape(nij_pair,nlk_pair)
        return eri_mo

####################
# aosym = s1, complex integrals
#
# kpti == kptj  =>  kptl == kptk
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.
#
    else:
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3])[1:]
        max_memory *= .5

        zij = zkl = None
        for (LpqR, LpqI, jpqR, jpqI), (LrsR, LrsI, jrsR, jrsI) in \
                lib.izip(mydf.sr_loop(kptijkl[:2], max_memory, False),
                         mydf.sr_loop(kptijkl[2:], max_memory, False)):
            zij, zkl = _ztrans(LpqR+LpqI*1j, zij, moij, ijslice,
                               jrsR+jrsI*1j, zkl, mokl, klslice, False)
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            zij, zkl = _ztrans(jpqR+jpqI*1j, zij, moij, ijslice,
                               LrsR+LrsI*1j, zkl, mokl, klslice, False)
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            LpqR = LpqI = jpqR = jpqI = LrsR = LrsI = jrsR = jrsI = None
        return eri_mo

