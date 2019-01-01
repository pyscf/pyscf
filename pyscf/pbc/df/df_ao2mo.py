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

import warnings
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc.df.df_jk import zdotNN, zdotCN, zdotNC
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member, unique
from pyscf import __config__


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):
    if mydf._cderi is None:
        mydf.build()

    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'df_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0]-nao**4*16/1e6)

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl):
        eriR = numpy.zeros((nao_pair,nao_pair))
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, True):
            lib.ddot(LpqR.T, LpqR, sign, eriR, 1)
            LpqR = LpqI = None
        if not compact:
            eriR = ao2mo.restore(1, eriR, nao).reshape(nao**2,-1)
        return eriR

    elif is_zero(kpti-kptk) and is_zero(kptj-kptl):
        eriR = numpy.zeros((nao*nao,nao*nao))
        eriI = numpy.zeros((nao*nao,nao*nao))
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, False):
            zdotNN(LpqR.T, LpqI.T, LpqR, LpqI, sign, eriR, eriI, 1)
            LpqR = LpqI = None
        return eriR + eriI*1j

####################
# (kpt) i == j == k == l != 0
#
# (kpt) i == l && j == k && i != j && j != k  =>
# both vbar and ovlp are zero. It corresponds to the exchange integral.
#
# complex integrals, N^4 elements
    elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
        eriR = numpy.zeros((nao*nao,nao*nao))
        eriI = numpy.zeros((nao*nao,nao*nao))
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, False):
            zdotNC(LpqR.T, LpqI.T, LpqR, LpqI, sign, eriR, eriI, 1)
            LpqR = LpqI = None
# transpose(0,1,3,2) because
# j == k && i == l  =>
# (L|ij).transpose(0,2,1).conj() = (L^*|ji) = (L^*|kl)  =>  (M|kl)
        eri = lib.transpose((eriR+eriI*1j).reshape(-1,nao,nao), axes=(0,2,1))
        return eri.reshape(nao**2,-1)

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
        blksize = int(max_memory*.4e6/16/nao**2)
        for (LpqR, LpqI, sign), (LrsR, LrsI, sign1) in \
                lib.izip(mydf.sr_loop(kptijkl[:2], max_memory, False, blksize),
                         mydf.sr_loop(kptijkl[2:], max_memory, False, blksize)):
            zdotNN(LpqR.T, LpqI.T, LrsR, LrsI, sign, eriR, eriI, 1)
            LpqR = LpqI = LrsR = LrsI = None
        return eriR + eriI*1j


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    warn_pbc2d_eri(mydf)
    if mydf._cderi is None:
        mydf.build()

    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'df_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    all_real = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]))

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl) and all_real:
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        eri_mo = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))
        ijR = klR = None
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, True):
            ijR, klR = _dtrans(LpqR, ijR, ijmosym, moij, ijslice,
                               LpqR, klR, klmosym, mokl, klslice, sym)
            lib.ddot(ijR.T, klR, sign, eri_mo, 1)
            LpqR = LpqI = None
        return eri_mo

    elif is_zero(kpti-kptk) and is_zero(kptj-kptl):
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3])[1:]
        eri_mo = numpy.zeros((nij_pair,nkl_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))

        zij = zkl = None
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, False):
            buf = LpqR+LpqI*1j
            zij, zkl = _ztrans(buf, zij, moij, ijslice,
                               buf, zkl, mokl, klslice, sym)
            lib.dot(zij.T, zkl, sign, eri_mo, 1)
            LpqR = LpqI = buf = None
        return eri_mo

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
    elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])[1:]
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        zij = zlk = None
        for LpqR, LpqI, sign in mydf.sr_loop(kptijkl[:2], max_memory, False):
            buf = LpqR+LpqI*1j
            zij, zlk = _ztrans(buf, zij, moij, ijslice,
                               buf, zlk, molk, lkslice, sym)
            lib.dot(zij.T, zlk.conj(), sign, eri_mo, 1)
            LpqR = LpqI = buf = None
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
        nao = mo_coeffs[0].shape[0]
        eri_mo = numpy.zeros((nij_pair,nkl_pair), dtype=numpy.complex)

        blksize = int(min(max_memory*.3e6/16/nij_pair,
                          max_memory*.3e6/16/nkl_pair,
                          max_memory*.3e6/16/nao**2))
        zij = zkl = None
        for (LpqR, LpqI, sign), (LrsR, LrsI, sign1) in \
                lib.izip(mydf.sr_loop(kptijkl[:2], max_memory, False, blksize),
                         mydf.sr_loop(kptijkl[2:], max_memory, False, blksize)):
            zij, zkl = _ztrans(LpqR+LpqI*1j, zij, moij, ijslice,
                               LrsR+LrsI*1j, zkl, mokl, klslice, False)
            lib.dot(zij.T, zkl, sign, eri_mo, 1)
            LpqR = LpqI = LrsR = LrsI = None
        return eri_mo

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    cell = mydf.cell
    if kpts is None:
        kpts = mydf.kpts
    nkpts = len(kpts)

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    # Shape of the orbitals can be different on different k-points. The
    # orbital coefficients must be formatted (padded by zeros) so that the
    # shape of the orbital coefficients are the same on all k-points. This can
    # be achieved by calling pbc.mp.kmp2.padded_mo_coeff function
    nmoi, nmoj, nmok, nmol = [x.shape[2] for x in mo_coeff_kpts]
    eri_shape = (nkpts, nkpts, nkpts, nmoi, nmoj, nmok, nmol)
    if gamma_point(kpts):
        dtype = numpy.result_type(*mo_coeff_kpts)
    else:
        dtype = numpy.complex128

    if out is None:
        out = numpy.empty(eri_shape, dtype=dtype)
    else:
        assert(out.shape == eri_shape)

    kptij_lst = numpy.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = numpy.prod(mydf.mesh)

    nao = cell.nao_nr()
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0]-nao**4*16/1e6) * .5

    tao = []
    ao_loc = None
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for uniq_id, kpt in enumerate(uniq_kpts):
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_id)[0]

        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts

            moij, ijslice = _conc_mos(mo_coeff_kpts[0][ki], mo_coeff_kpts[1][kj])[2:]
            zij = []
            for LpqR, LpqI, sign in mydf.sr_loop(kpts[[ki,kj]], max_memory, False, mydf.blockdim):
                zij.append(_ao2mo.r_e2(LpqR+LpqI*1j, moij, ijslice, tao, ao_loc))

            for kk in range(nkpts):
                kl = kconserv[ki, kj, kk]
                mokl, klslice = _conc_mos(mo_coeff_kpts[2][kk], mo_coeff_kpts[3][kl])[2:]
                eri_mo = numpy.zeros((nmoi*nmoj,nmok*nmol), dtype=numpy.complex128)
                for i, (LrsR, LrsI, sign) in \
                        enumerate(mydf.sr_loop(kpts[[kk,kl]], max_memory, False, mydf.blockdim)):
                    zkl = _ao2mo.r_e2(LrsR+LrsI*1j, mokl, klslice, tao, ao_loc)
                    lib.dot(zij[i].T, zkl, sign*factor, eri_mo, 1)

                if dtype == numpy.double:
                    eri_mo = eri_mo.real
                out[ki,kj,kk] = eri_mo.reshape(eri_shape[3:])
    return out


def _mo_as_complex(mo_coeffs):
    mos = []
    for c in mo_coeffs:
        if c.dtype == numpy.float64:
            mos.append(c+0j)
        else:
            mos.append(c)
    return mos

def _dtrans(Lpq, Lij, ijmosym, moij, ijslice,
            Lrs, Lkl, klmosym, mokl, klslice, sym):
    Lij = _ao2mo.nr_e2(Lpq, moij, ijslice, aosym='s2', mosym=ijmosym, out=Lij)
    if sym:
        Lkl = Lij
    else:
        Lkl = _ao2mo.nr_e2(Lrs, mokl, klslice, aosym='s2', mosym=klmosym, out=Lkl)
    return Lij, Lkl

def _ztrans(Lpq, zij, moij, ijslice, Lrs, zkl, mokl, klslice, sym):
    tao = []
    ao_loc = None
    zij = _ao2mo.r_e2(Lpq, moij, ijslice, tao, ao_loc, out=zij)
    if sym:
        zkl = zij
    else:
        zkl = _ao2mo.r_e2(Lrs, mokl, klslice, tao, ao_loc, out=zkl)
    return zij, zkl


class PBC2DIntegralsWarning(RuntimeWarning):
    pass
def warn_pbc2d_eri(mydf):
    cell = mydf.cell
    if cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum':
        with warnings.catch_warnings():
            warnings.simplefilter('once', PBC2DIntegralsWarning)
            warnings.warn('\nERIs of PBC-2D systems with infinity vacuum are '
                          'singular.  cell.low_dim_ft_type = None  should be '
                          'set.\n')


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import DF

    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

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
    with_df = DF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.mesh = [n] * 3
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, kpts)
    print(abs(eri1-eri0).sum())
