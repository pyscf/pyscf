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
Integral transformation with analytic Fourier transformation
'''

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc.df.df_jk import zdotNC
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.pbc.df.df_ao2mo import _mo_as_complex, _dtrans, _ztrans
from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
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
            lib.ddot(pqkR*coulG[p0:p1], pqkR.T, 1, eriR, 1)
            lib.ddot(pqkI*coulG[p0:p1], pqkI.T, 1, eriR, 1)
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
            # rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            zdotNC(pqkR*coulG[p0:p1], pqkI*coulG[p0:p1], pqkR.T, pqkI.T,
                   1, eriR, eriI, 1)
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
            buf = lib.transpose(pqkR, out=buf)
            ijR, klR = _dtrans(buf, ijR, ijmosym, moij, ijslice,
                               buf, klR, klmosym, mokl, klslice, sym)
            lib.ddot(ijR.T, klR*coulG[p0:p1,None], 1, eri_mo, 1)
            buf = lib.transpose(pqkI, out=buf)
            ijI, klI = _dtrans(buf, ijI, ijmosym, moij, ijslice,
                               buf, klI, klmosym, mokl, klslice, sym)
            lib.ddot(ijI.T, klI*coulG[p0:p1,None], 1, eri_mo, 1)
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
        eri_mo = numpy.zeros((nij_pair,nlk_pair), dtype=numpy.complex128)
        sym = (iden_coeffs(mo_coeffs[0], mo_coeffs[3]) and
               iden_coeffs(mo_coeffs[1], mo_coeffs[2]))

        zij = zlk = buf = None
        for pqkR, pqkI, p0, p1 \
                in mydf.pw_loop(mesh, kptijkl[:2], q, max_memory=max_memory):
            buf = lib.transpose(pqkR+pqkI*1j, out=buf)
            zij, zlk = _ztrans(buf, zij, moij, ijslice,
                               buf, zlk, molk, lkslice, sym)
            lib.dot(zij.T, zlk.conj()*coulG[p0:p1,None], 1, eri_mo, 1)
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
        eri_mo = numpy.zeros((nij_pair,nkl_pair), dtype=numpy.complex128)

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
            zij *= coulG[p0:p1,None]
            lib.dot(zij.T, zkl, 1, eri_mo, 1)
            pqkR = pqkI = rskR = rskI = None
        return eri_mo


def get_ao_pairs_G(mydf, kpts=numpy.zeros((2,3)), q=None, shls_slice=None,
                   compact=getattr(__config__, 'pbc_df_ao_pairs_compact', False)):
    '''Calculate forward Fourier transform (G|ij) of all AO pairs.

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
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .5)

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

    ao_pairs_G = numpy.empty((ngrids,(i1-i0)*(j1-j0)), dtype=numpy.complex128)
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.mesh, kpts, q, shls_slice,
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
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .5)

    mo_pairs_G = numpy.empty((ngrids,nmoi,nmoj), dtype=numpy.complex128)
    nao = cell.nao
    for pqkR, pqkI, p0, p1 \
            in mydf.pw_loop(mydf.mesh, kpts, q,
                            max_memory=max_memory, aosym='s2'):
        pqk = lib.unpack_tril(pqkR + pqkI*1j, axis=0).reshape(nao,nao,-1)
        mo_pairs_G[p0:p1] = lib.einsum('pqk,pi,qj->kij', pqk, *mo_coeffs[:2])
    return mo_pairs_G.reshape(ngrids,nmoi*nmoj)

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
        assert (out.shape == eri_shape)

    kptij_lst = numpy.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:,0]
    kptjs_lst = kptij_lst[:,1]
    kpt_ji = kptjs_lst - kptis_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    ngrids = numpy.prod(mydf.mesh)
    nao = cell.nao
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0]-nao**4*16/1e6) * .5

    # To hold intermediates
    fswap = lib.H5TmpFile()
    tao = []
    ao_loc = None
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for uniq_id, kpt in enumerate(uniq_kpts):
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_id)[0]
        kptjs = kptjs_lst[adapted_ji_idx]
        coulG = mydf.weighted_coulG(q, False, mydf.mesh)
        coulG *= factor

        moij_list = []
        ijslice_list = []
        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            moij, ijslice = _conc_mos(mo_coeff_kpts[0][ki], mo_coeff_kpts[1][kj])[2:]
            moij_list.append(moij)
            ijslice_list.append(ijslice)
            fswap.create_dataset('zij/'+str(ji), (ngrids,nmoi*nmoj), 'D')

        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, kptjs,
                                           max_memory=max_memory):
            for ji, aoao in enumerate(aoaoks):
                ki = adapted_ji_idx[ji] // nkpts
                kj = adapted_ji_idx[ji] %  nkpts
                buf = aoao.transpose(1,2,0).reshape(nao**2,p1-p0)
                zij = _ao2mo.r_e2(lib.transpose(buf), moij_list[ji],
                                  ijslice_list[ji], tao, ao_loc)
                zij *= coulG[p0:p1,None]
                fswap['zij/'+str(ji)][p0:p1] = zij
                buf = zij = None

        mokl_list = []
        klslice_list = []
        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            mokl, klslice = _conc_mos(mo_coeff_kpts[2][kk], mo_coeff_kpts[3][kl])[2:]
            mokl_list.append(mokl)
            klslice_list.append(klslice)
            fswap.create_dataset('zkl/'+str(kk), (ngrids,nmok*nmol), 'D')

        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts
        kptls = kpts[kconserv[ki, kj, :]]
        for aoaoks, p0, p1 in mydf.ft_loop(mydf.mesh, q, -kptls,
                                           max_memory=max_memory):
            for kk, aoao in enumerate(aoaoks):
                buf = aoao.conj().transpose(1,2,0).reshape(nao**2,p1-p0)
                zkl = _ao2mo.r_e2(lib.transpose(buf), mokl_list[kk],
                                  klslice_list[kk], tao, ao_loc)
                fswap['zkl/'+str(kk)][p0:p1] = zkl
                buf = zkl = None

        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            for kk in range(nkpts):
                zij = numpy.asarray(fswap['zij/'+str(ji)])
                zkl = numpy.asarray(fswap['zkl/'+str(kk)])
                tmp = lib.dot(zij.T, zkl)
                if dtype == numpy.double:
                    tmp = tmp.real
                out[ki,kj,kk] = tmp.reshape(eri_shape[3:])
        del (fswap['zij'])
        del (fswap['zkl'])

    return out


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
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
