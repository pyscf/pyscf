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

r'''
Integral transformation with FFT

(ij|kl) = \int dr1 dr2 i*(r1) j(r1) v(r12) k*(r2) l(r2)
        = (ij|G) v(G) (G|kl)

i*(r) j(r) = 1/N \sum_G e^{iGr}  (G|ij)
           = 1/N \sum_G e^{-iGr} (ij|G)

"forward" FFT:
    (G|ij) = \sum_r e^{-iGr} i*(r) j(r) = fft[ i*(r) j(r) ]
"inverse" FFT:
    (ij|G) = \sum_r e^{iGr} i*(r) j(r) = N * ifft[ i*(r) j(r) ]
           = conj[ \sum_r e^{-iGr} j*(r) i(r) ]
'''

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'fft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    q = kptj - kpti
    coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl):
        #:ao_pairs_G = get_ao_pairs_G(mydf, kptijkl[:2], q, compact=compact)
        #:ao_pairs_G *= numpy.sqrt(coulG).reshape(-1,1)
        #:eri = lib.dot(ao_pairs_G.T, ao_pairs_G, cell.vol/ngrids**2)
        ao = mydf._numint.eval_ao(cell, coords, kpti)[0]
        ao = numpy.asarray(ao.T, order='C')
        eri = _contract_compact(mydf, (ao,ao), coulG, max_memory=max_memory)
        if not compact:
            eri = ao2mo.restore(1, eri, nao).reshape(nao**2,nao**2)
        return eri

####################
# aosym = s1, complex integrals
    else:
        #:ao_pairs_G = get_ao_pairs_G(mydf, kptijkl[:2], q, compact=False)
        #:# ao_pairs_invG = rho_kl(-(G+k_ij)) = conj(rho_lk(G+k_ij)).swap(r,s)
        #:#=get_ao_pairs_G(mydf, [kptl,kptk], q, compact=False).transpose(0,2,1).conj()
        #:ao_pairs_invG = get_ao_pairs_G(mydf, -kptijkl[2:], q, compact=False).conj()
        #:ao_pairs_G *= coulG.reshape(-1,1)
        #:eri = lib.dot(ao_pairs_G.T, ao_pairs_invG, cell.vol/ngrids**2)
        if is_zero(kpti-kptl) and is_zero(kptj-kptk):
            if is_zero(kpti-kptj):
                aoi = mydf._numint.eval_ao(cell, coords, kpti)[0]
                aoi = aoj = numpy.asarray(aoi.T, order='C')
            else:
                aoi, aoj = mydf._numint.eval_ao(cell, coords, kptijkl[:2])
                aoi = numpy.asarray(aoi.T, order='C')
                aoj = numpy.asarray(aoj.T, order='C')
            aos = (aoi, aoj, aoj, aoi)
        else:
            aos = mydf._numint.eval_ao(cell, coords, kptijkl)
            aos = [numpy.asarray(x.T, order='C') for x in aos]
        fac = numpy.exp(-1j * numpy.dot(coords, q))
        max_memory = max_memory - aos[0].nbytes*4*1e-6
        eri = _contract_plain(mydf, aos, coulG, fac, max_memory=max_memory)
        return eri


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    '''General MO integral transformation'''
    from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    mo_coeffs = [numpy.asarray(mo, order='F') for mo in mo_coeffs]
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'fft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    allreal = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    q = kptj - kpti
    coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

    if gamma_point(kptijkl) and allreal:
        ao = mydf._numint.eval_ao(cell, coords, kpti)[0]
        if ((iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[3]))):
            moiT = mojT = numpy.asarray(lib.dot(mo_coeffs[0].T,ao.T), order='C')
            ao = None
            max_memory = max_memory - moiT.nbytes*1e-6
            eri = _contract_compact(mydf, (moiT,mojT), coulG, max_memory=max_memory)
            if not compact:
                nmo = moiT.shape[0]
                eri = ao2mo.restore(1, eri, nmo).reshape(nmo**2,nmo**2)
        else:
            mos = [numpy.asarray(lib.dot(c.T, ao.T), order='C') for c in mo_coeffs]
            ao = None
            fac = numpy.array(1.)
            max_memory = max_memory - sum([x.nbytes for x in mos])*1e-6
            eri = _contract_plain(mydf, mos, coulG, fac, max_memory=max_memory).real
        return eri

    else:
        aos = mydf._numint.eval_ao(cell, coords, kptijkl)
        mos = [numpy.asarray(lib.dot(c.T, aos[i].T), order='C')
               for i,c in enumerate(mo_coeffs)]
        aos = None
        fac = numpy.exp(-1j * numpy.dot(coords, q))
        max_memory = max_memory - sum([x.nbytes for x in mos])*1e-6
        eri = _contract_plain(mydf, mos, coulG, fac, max_memory=max_memory)
        return eri


def _contract_compact(mydf, mos, coulG, max_memory):
    cell = mydf.cell
    moiT, mokT = mos
    nmoi, ngrids = moiT.shape
    nmok = mokT.shape[0]
    wcoulG = coulG * (cell.vol/ngrids)

    def fill_orbital_pair(moT, i0, i1, buf):
        npair = i1*(i1+1)//2 - i0*(i0+1)//2
        out = numpy.ndarray((npair,ngrids), dtype=buf.dtype, buffer=buf)
        ij = 0
        for i in range(i0, i1):
            numpy.einsum('p,jp->jp', moT[i], moT[:i+1], out=out[ij:ij+i+1])
            ij += i + 1
        return out

    eri = numpy.empty((nmoi*(nmoi+1)//2,nmok*(nmok+1)//2))
    blksize = int(min(max(nmoi*(nmoi+1)//2, nmok*(nmok+1)//2),
                      (max_memory*1e6/8 - eri.size)/2/ngrids+1))
    buf = numpy.empty((blksize,ngrids))
    for p0, p1 in lib.prange_tril(0, nmoi, blksize):
        mo_pairs_G = tools.fft(fill_orbital_pair(moiT, p0, p1, buf), mydf.mesh)
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        vR = numpy.asarray(v.real, order='C')
        for q0, q1 in lib.prange_tril(0, nmok, blksize):
            mo_pairs = numpy.asarray(fill_orbital_pair(mokT, q0, q1, buf), order='C')
            eri[p0*(p0+1)//2:p1*(p1+1)//2,
                q0*(q0+1)//2:q1*(q1+1)//2] = lib.ddot(vR, mo_pairs.T)
        v = None
    return eri

def _contract_plain(mydf, mos, coulG, phase, max_memory):
    cell = mydf.cell
    moiT, mojT, mokT, molT = mos
    nmoi, nmoj, nmok, nmol = [x.shape[0] for x in mos]
    ngrids = moiT.shape[1]
    wcoulG = coulG * (cell.vol/ngrids)
    dtype = numpy.result_type(phase, *mos)
    eri = numpy.empty((nmoi*nmoj,nmok*nmol), dtype=dtype)

    blksize = int(min(max(nmoi,nmok), (max_memory*1e6/16 - eri.size)/2/ngrids/max(nmoj,nmol)+1))
    assert blksize > 0
    buf0 = numpy.empty((blksize,max(nmoj,nmol),ngrids), dtype=dtype)
    buf1 = numpy.ndarray((blksize,nmoj,ngrids), dtype=dtype, buffer=buf0)
    buf2 = numpy.ndarray((blksize,nmol,ngrids), dtype=dtype, buffer=buf0)
    for p0, p1 in lib.prange(0, nmoi, blksize):
        mo_pairs = numpy.einsum('ig,jg->ijg', moiT[p0:p1].conj()*phase,
                                mojT, out=buf1[:p1-p0])
        mo_pairs_G = tools.fft(mo_pairs.reshape(-1,ngrids), mydf.mesh)
        mo_pairs = None
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        mo_pairs_G = None
        v *= phase.conj()
        if dtype == numpy.double:
            v = numpy.asarray(v.real, order='C')
        for q0, q1 in lib.prange(0, nmok, blksize):
            mo_pairs = numpy.einsum('ig,jg->ijg', mokT[q0:q1].conj(),
                                    molT, out=buf2[:q1-q0])
            eri[p0*nmoj:p1*nmoj,q0*nmol:q1*nmol] = lib.dot(v, mo_pairs.reshape(-1,ngrids).T)
        v = None
    return eri


def get_ao_pairs_G(mydf, kpts=numpy.zeros((2,3)), q=None, shls_slice=None,
                   compact=getattr(__config__, 'pbc_df_ao_pairs_compact', False)):
    '''Calculate forward (G|ij) FFT of all AO pairs.

    Returns:
        ao_pairs_G : 2D complex array
            For gamma point, the shape is (ngrids, nao*(nao+1)/2); otherwise the
            shape is (ngrids, nao*nao)
    '''
    if kpts is None: kpts = numpy.zeros((2,3))
    cell = mydf.cell
    kpts = numpy.asarray(kpts)
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)

    if shls_slice is None:
        i0, i1 = j0, j1 = (0, cell.nao_nr())
    else:
        ish0, ish1, jsh0, jsh1 = shls_slice
        ao_loc = cell.ao_loc_nr()
        i0 = ao_loc[ish0]
        i1 = ao_loc[ish1]
        j0 = ao_loc[jsh0]
        j1 = ao_loc[jsh1]

    def trans(aoi, aoj, fac=1):
        if id(aoi) == id(aoj):
            aoi = aoj = numpy.asarray(aoi.T, order='C')
        else:
            aoi = numpy.asarray(aoi.T, order='C')
            aoj = numpy.asarray(aoj.T, order='C')
        ni = aoi.shape[0]
        nj = aoj.shape[0]
        ao_pairs_G = numpy.empty((ni,nj,ngrids), dtype=numpy.complex128)
        for i in range(ni):
            ao_pairs_G[i] = tools.fft(fac * aoi[i].conj() * aoj, mydf.mesh)
        ao_pairs_G = ao_pairs_G.reshape(-1,ngrids).T
        return ao_pairs_G

    if compact and gamma_point(kpts):  # gamma point
        ao = mydf._numint.eval_ao(cell, coords, kpts[:1])[0]
        ao = numpy.asarray(ao.T, order='C')
        npair = i1*(i1+1)//2 - i0*(i0+1)//2
        ao_pairs_G = numpy.empty((npair,ngrids), dtype=numpy.complex128)
        ij = 0
        for i in range(i0, i1):
            ao_pairs_G[ij:ij+i+1] = tools.fft(ao[i] * ao[:i+1], mydf.mesh)
            ij += i + 1
        ao_pairs_G = ao_pairs_G.T

    elif is_zero(kpts[0]-kpts[1]):
        ao = mydf._numint.eval_ao(cell, coords, kpts[:1])[0]
        ao_pairs_G = trans(ao[:,i0:i1], ao[:,j0:j1])

    else:
        if q is None:
            q = kpts[1] - kpts[0]
        aoi, aoj = mydf._numint.eval_ao(cell, coords, kpts[:2])
        fac = numpy.exp(-1j * numpy.dot(coords, q))
        ao_pairs_G = trans(aoi[:,i0:i1], aoj[:,j0:j1], fac)

    return ao_pairs_G

def get_mo_pairs_G(mydf, mo_coeffs, kpts=numpy.zeros((2,3)), q=None,
                   compact=getattr(__config__, 'pbc_df_mo_pairs_compact', False)):
    '''Calculate forward (G|ij) FFT of all MO pairs.

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
    coords = cell.gen_uniform_grids(mydf.mesh)
    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    ngrids = len(coords)

    def trans(aoi, aoj, fac=1):
        if id(aoi) == id(aoj) and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
            moi = moj = numpy.asarray(lib.dot(mo_coeffs[0].T,aoi.T), order='C')
        else:
            moi = numpy.asarray(lib.dot(mo_coeffs[0].T, aoi.T), order='C')
            moj = numpy.asarray(lib.dot(mo_coeffs[1].T, aoj.T), order='C')
        mo_pairs_G = numpy.empty((nmoi,nmoj,ngrids), dtype=numpy.complex128)
        for i in range(nmoi):
            mo_pairs_G[i] = tools.fft(fac * moi[i].conj() * moj, mydf.mesh)
        mo_pairs_G = mo_pairs_G.reshape(-1,ngrids).T
        return mo_pairs_G

    if gamma_point(kpts):  # gamma point, real
        ao = mydf._numint.eval_ao(cell, coords, kpts[:1])[0]
        if compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
            mo = numpy.asarray(lib.dot(mo_coeffs[0].T, ao.T), order='C')
            npair = nmoi*(nmoi+1)//2
            mo_pairs_G = numpy.empty((npair,ngrids), dtype=numpy.complex128)
            ij = 0
            for i in range(nmoi):
                mo_pairs_G[ij:ij+i+1] = tools.fft(mo[i].conj() * mo[:i+1], mydf.mesh)
                ij += i + 1
            mo_pairs_G = mo_pairs_G.T
        else:
            mo_pairs_G = trans(ao, ao)

    elif is_zero(kpts[0]-kpts[1]):
        ao = mydf._numint.eval_ao(cell, coords, kpts[:1])[0]
        mo_pairs_G = trans(ao, ao)

    else:
        if q is None:
            q = kpts[1] - kpts[0]
        aoi, aoj = mydf._numint.eval_ao(cell, coords, kpts)
        fac = numpy.exp(-1j * numpy.dot(coords, q))
        mo_pairs_G = trans(aoi, aoj, fac)

    return mo_pairs_G

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    cell = mydf.cell
    if kpts is None:
        kpts = mydf.kpts
    nkpts = len(kpts)

    if isinstance(mo_coeff_kpts, numpy.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    mo_ids = [id(x) for x in mo_coeff_kpts]
    moTs = []
    coords = cell.gen_uniform_grids(mydf.mesh)
    aos = mydf._numint.eval_ao(cell, coords, kpts)
    for n, mo_id in enumerate(mo_ids):
        if mo_id in mo_ids[:n]:
            moTs.append(moTs[mo_ids[:n].index(mo_id)])
        else:
            moTs.append([lib.dot(mo.T, aos[k].T) for k,mo in enumerate(mo_coeff_kpts[n])])

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

    # To hold intermediates
    fswap = lib.H5TmpFile()
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for uniq_id, kpt in enumerate(uniq_kpts):
        q = uniq_kpts[uniq_id]
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_id)[0]
        ki = adapted_ji_idx[0] // nkpts
        kj = adapted_ji_idx[0] % nkpts

        coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
        coulG *= (cell.vol/ngrids) * factor
        phase = numpy.exp(-1j * numpy.dot(coords, q))

        for kk in range(nkpts):
            kl = kconserv[ki, kj, kk]
            mokT = moTs[2][kk]
            molT = moTs[3][kl]
            mo_pairs = numpy.einsum('ig,g,jg->ijg', mokT.conj(), phase.conj(), molT)
            v = tools.ifft(mo_pairs.reshape(-1,ngrids), mydf.mesh)
            v *= coulG
            v = tools.fft(v.reshape(-1,ngrids), mydf.mesh)
            v *= phase
            fswap['zkl/'+str(kk)] = v

        for ji_idx in adapted_ji_idx:
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts
            for kk in range(nkpts):
                moiT = moTs[0][ki]
                mojT = moTs[1][kj]
                mo_pairs = numpy.einsum('ig,jg->ijg', moiT.conj(), mojT)
                tmp = lib.dot(mo_pairs.reshape(-1,ngrids),
                              numpy.asarray(fswap['zkl/'+str(kk)]).T)
                if dtype == numpy.double:
                    tmp = tmp.real
                out[ki,kj,kk] = tmp.reshape(eri_shape[3:])
        del(fswap['zkl'])

    return out

def _format_kpts(kpts):
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    else:
        kpts = numpy.asarray(kpts)
        if kpts.size == 3:
            kptijkl = numpy.vstack([kpts]*4).reshape(4,3)
        else:
            kptijkl = kpts.reshape(4,3)
    return kptijkl

def _iskconserv(cell, kpts):
    dk = kpts[1] - kpts[0] + kpts[3] - kpts[2]
    if abs(dk).sum() < 1e-9:
        return True
    else:
        s = 1./(2*numpy.pi)*numpy.dot(dk, cell.lattice_vectors().T)
        s_int = s.round(0)
        return abs(s - s_int).sum() < 1e-9


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc import df

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
    with_df = df.FFTDF(cell)
    with_df.kpts = kpts
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       ).reshape(nao**2,-1)
    eri1 = with_df.ao2mo(mo, kpts)
    print(abs(eri1-eri0.reshape(eri1.shape)).sum())
