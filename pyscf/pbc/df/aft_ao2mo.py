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
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.df.df_ao2mo import _mo_as_complex, _dtrans, _ztrans
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf import __config__


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'aft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    q = kptj - kpti
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(q, False, mesh)
    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .8)

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl):
        eriR = numpy.zeros((nao_pair,nao_pair))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory,
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
    elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory):
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
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
#
#       (pq|rs) = \sum_G 4\pi rho_pq rho_rs / |G+k_{pq}|^2
#       rho_pq = 1/N \sum_{Tp,Tq} \int exp(-i(G+k_{pq})*r) p(r-Tp) q(r-Tq) dr
#              = \sum_{Tq} exp(i k_q*Tq) \int exp(-i(G+k_{pq})*r) p(r) q(r-Tq) dr
# Note the k-point wrap-around for rho_rs, which leads to G+k_{pq} in FT
#       rho_rs = 1/N \sum_{Tr,Ts} \int exp( i(G+k_{pq})*r) r(r-Tr) s(r-Ts) dr
#              = \sum_{Ts} exp(i k_s*Ts) \int exp( i(G+k_{pq})*r) r(r) s(r-Ts) dr
# rho_pq can be directly evaluated by AFT (function pw_loop)
#       rho_pq = pw_loop(k_q, G+k_{pq})
# Assuming r(r) and s(r) are real functions, rho_rs is evaluated
#       rho_rs = 1/N \sum_{Tr,Ts} \int exp( i(G+k_{pq})*r) r(r-Tr) s(r-Ts) dr
#              = conj(\sum_{Ts} exp(-i k_s*Ts) \int exp(-i(G+k_{pq})*r) r(r) s(r-Ts) dr)
#              = conj( pw_loop(-k_s, G+k_{pq}) )
#
# TODO: For complex AO function r(r) and s(r), pw_loop function needs to be
# extended to include Gv vector in the arguments
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory*.5),
                         mydf.pw_loop(mesh,-kptijkl[2:], q, max_memory=max_memory*.5)):
            pqkR *= coulG[p0:p1]
            pqkI *= coulG[p0:p1]
            zdotNC(pqkR, pqkI, rskR.T, rskI.T, 1, eriR, eriI, 1)
            pqkR = pqkI = rskR = rskI = None
        return (eriR+eriI*1j)


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'aft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    q = kptj - kpti
    mesh = mydf.mesh
    coulG = mydf.weighted_coulG(q, False, mesh)
    all_real = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .5)

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl) and all_real:
        ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1], compact)
        klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_coeffs[2], mo_coeffs[3], compact)
        eri_mo = numpy.zeros((nij_pair,nkl_pair))
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[3]))

        ijR = ijI = klR = klI = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory,
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
    elif is_zero(kpti-kptl) and is_zero(kptj-kptk):
        mo_coeffs = _mo_as_complex(mo_coeffs)
        nij_pair, moij, ijslice = _conc_mos(mo_coeffs[0], mo_coeffs[1])[1:]
        nlk_pair, molk, lkslice = _conc_mos(mo_coeffs[3], mo_coeffs[2])[1:]
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        zij = zlk = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory):
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
        zij = zkl = buf = None
        for (pqkR, pqkI, p0, p1), (rskR, rskI, q0, q1) in \
                lib.izip(mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory*.5),
                         mydf.pw_loop(mesh,-kptijkl[2:], q, max_memory=max_memory*.5)):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            zij = _ao2mo.r_e2(buf, moij, ijslice, tao, ao_loc, out=zij)
            buf = lib.transpose(rskR-rskI*1j, out=buf)
            zkl = _ao2mo.r_e2(buf, mokl, klslice, tao, ao_loc, out=zkl)
            zij *= coulG[p0:p1].reshape(-1,1)
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            pqkR = pqkI = rskR = rskI = None
        return eri_mo


def get_ao_pairs_G(mydf, kpts=numpy.zeros((2,3)), q=None, shls_slice=None,
                   compact=getattr(__config__, 'pbc_df_ao_pairs_compact', False)):
    '''Calculate forward Fourier tranform (G|ij) of all AO pairs.

    Returns:
        ao_pairs_G : 2D complex array
            For gamma point, the shape is (ngrids, nao*(nao+1)/2); otherwise the
            shape is (ngrids, nao*nao)
    '''
    if kpts is None: kpts = numpy.zeros((2,3))
    cell = mydf.cell
    kpts = numpy.asarray(kpts)
    q = kpts[1] - kpts[0]
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    ish0, ish1, jsh0, jsh1 = shls_slice
    ao_loc = cell.ao_loc_nr()
    i0 = ao_loc[ish0]
    i1 = ao_loc[ish1]
    j0 = ao_loc[jsh0]
    j1 = ao_loc[jsh1]
    compact = compact and (i0 == j0) and (i1 == j1)

    if compact and gamma_point(kpts):  # gamma point
        aosym = 's2'
    else:
        aosym = 's1'

    ao_pairs_G = numpy.empty((ngrids,(i1-i0)*(j1-j0)), dtype=numpy.complex)
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.mesh, kptijkl[:2], q, shls_slice,
                            max_memory=max_memory, aosym=aosym):
        ao_pairs_G[p0:p1] = pqkR.T + pqkI.T * 1j
    return ao_pairs_G

def get_mo_pairs_G(mydf, mo_coeffs, kpts=numpy.zeros((2,3)), q=None,
                   compact=getattr(__config__, 'pbc_df_mo_pairs_compact', False)):
    '''Calculate forward fourier transform (G|ij) of all MO pairs.

    Args:
        mo_coeff: length-2 list of (nao,nmo) ndarrays
            The two sets of MO coefficients to use in calculating the
            product |ij).

    Returns:
        mo_pairs_G : (ngrids, nmoi*nmoj) ndarray
            The FFT of the real-space MO pairs.
    '''
    if kpts is None: kpts = numpy.zeros((2,3))
    cell = mydf.cell
    kpts = numpy.asarray(kpts)
    q = kpts[1] - kpts[0]
    coords = cell.gen_uniform_grids(mydf.mesh)
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    ngrids = len(coords)

    mo_pairs_G = numpy.empty((ngrids,nmoi,nmoj), dtype=numpy.complex)
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.mesh, kptijkl[:2], q,
                            max_memory=max_memory, aosym='s2'):
        pqk = (pqkR + pqkI*1j).reshape(nao,nao,-1)
        mo_pairs_G[p0:p1] = lib.einsum('pqk,pi,qj->kij', pqk, *mo_coeffs[:2])
    return mo_pairs_G.reshape(ngrids,nmoi*nmoj)

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import AFTDF

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
    with_df = AFTDF(cell, kpts)
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
