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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Functions and classes to handle the ERIs in KRAGF2
'''

import time
import copy
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.pbc import tools, df
#from pyscf.agf2 import mpi_helper
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.ccsd import _adjust_occ
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, padding_k_idx, \
                              padded_mo_coeff, padded_mo_energy

#from . import mpi_helper
import mpi_helper

#TODO: parallelise get_hcore and get_veff
class _ChemistsERIs:
    ''' (pq|rs)
    Stored as (pq|J)(J|rs) if agf2.direct is True.
    '''

    def __init__(self, cell=None):
        self.mol = self.cell = cell
        self.mo_coeff = None
        self.nmo = None
        self.nocc = None
        self.kpts = None
        self.nonzero_padding = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        #self.mo_coeff = padded_mo_coeff(agf2, mo_coeff)
        #self.mo_energy = padded_mo_energy(agf2, agf2.mo_energy)
        self.mo_coeff = np.asarray(mo_coeff)
        self.mo_energy = agf2.mo_energy

        self.nmo = agf2.nmo
        self.nocc = agf2.nocc

        #self.nonzero_padding = padding_k_idx(agf2, kind='joint')
        self.mol = self.cell = agf2.cell
        self.kpts = agf2.kpts

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        exxdiv = agf2._scf.exxdiv if agf2.keep_exxdiv else None
        with lib.temporary_env(agf2._scf, exxdiv=exxdiv):
            h1e_ao = agf2._scf.get_hcore()
            veff_ao = agf2._scf.get_veff(agf2.cell, dm)
            fock_ao = h1e_ao + veff_ao

        self.h1e = []
        self.fock = []
        for ki, ci in enumerate(self.mo_coeff):
            self.h1e.append(np.dot(np.dot(ci.conj().T, h1e_ao[ki]), ci))
            self.fock.append(np.dot(np.dot(ci.conj().T, fock_ao[ki]), ci))

        if not agf2.keep_exxdiv:
            madelung = tools.madelung(agf2.cell, agf2.kpts)
            self.mo_energy = [f.diagonal().real for f in self.fock]
            self.mo_energy = [_adjust_occ(mo_e, self.nocc[k], -madelung) 
                              for k, mo_e in enumerate(self.mo_energy)]

        self.e_hf = agf2._scf.e_tot

    @property
    def naux(self):
        if not hasattr(self, 'eri') or not isinstance(self.eri, (tuple, list)):
            raise AttributeError
        return self.eri[0].shape[2]

#TODO dtype - do we just inherit the mo_coeff.dtype or use np.result_type(eri, mo_coeff) ?
#TODO blksize, max_memory
#TODO symmetry broken (commented)
def _make_mo_eris_incore(agf2, mo_coeff=None):
    # (pq|rs) incore
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    with_df = agf2.with_df
    dtype = complex

    nmo = agf2.nmo
    npair = nmo * (nmo+1) // 2

    kpts = eris.kpts
    nkpts = len(kpts)
    khelper = agf2.khelper
    kconserv = khelper.kconserv

    eri = np.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=dtype)

    for kpqr in mpi_helper.nrange(nkpts**3):
        kpq, kr = divmod(kpqr, nkpts)
        kp, kq = divmod(kpq, nkpts)
        ks = kconserv[kp,kq,kr]

        coeffs = eris.mo_coeff[[kp,kq,kr,ks]]
        kijkl = kpts[[kp,kq,kr,ks]]

        eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

        if dtype in [np.float, np.float64]:
            eri_kpt = eri_kpt.real

        eri[kp,kq,kr] = eri_kpt / nkpts

    #for ikp, ikq, ikr in khelper.symm_map.keys():
    #    iks = kconserv[ikp,ikq,ikr]

    #    coeffs = eris.mo_coeff[[ikp,ikq,ikr,iks]]
    #    kijkl = kpts[[ikp,ikq,ikr,iks]]

    #    eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
    #    eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

    #    if dtype in [np.float, np.float64]: 
    #        eri_kpt = eri_kpt.real

    #    for kp, kq, kr in khelper.symm_map[(ikp, ikq, ikr)]:
    #        eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr)
    #        eri[kp,kr,kq] = eri_kpt_symm.transpose(0,2,1,3) / nkpts

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(eri)

    eris.eri = eri

    log.timer('MO integral transformation', *cput0)

    return eris

    ##NOTE Uncomment to convert to incore:
    #eris = _make_mo_eris_direct(agf2, mo_coeff)
    #eris.eri = lib.einsum('ablp,abclq->abcpq', eris.qij.conj(), eris.qkl).reshape((agf2.nkpts,)*3+(agf2.nmo,)*4)
    #del eris.qij, eris.qkl

    #return eris

def _get_naux_from_cderi(with_df):
    ''' Get the largest possible naux from the _cderi object. There
        can be different numbers at each k-point.
    '''

    cell = with_df.cell
    load3c = df.df._load3c
    kpts = with_df.kpts
    nkpts = len(kpts)

    naux = 0

    for kp in range(nkpts):
        for kq in range(nkpts):
            with load3c(with_df._cderi, 'j3c', kpts[[kp,kq]], 'j3c-kptij') as j3c:
                naux = max(naux, j3c.shape[0])

                if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                    with load3c(with_df._cderi, 'j3c-', kpts[[kp,kq]], 'j3c-kptij',
                                ignore_key_error=True) as j3c:
                        naux = max(naux, j3c.shape[0])

    return naux

##TODO: fix block size?? ft_loop has max_memory argument
def _make_ao_eris_direct_aftdf(agf2, eris):
    ''' Get the 3c AO tensors for AFTDF '''

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    ngrids = len(cell.gen_uniform_grids(with_df.mesh))
    nao = cell.nao
    kconserv = tools.get_kconserv(cell, kpts)
    
    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        kjs = kij[:,1][adapted_ji]
        fac = with_df.weighted_coulG(q, False, with_df.mesh) / nkpts

        for aoaoks, p0, p1 in with_df.ft_loop(with_df.mesh, q, kjs):
            for ji, aoao in enumerate(aoaoks):
                ki, kj = divmod(adapted_ji[ji], nkpts)
                bra[ki,kj,p0:p1] = fac[p0:p1,None] * aoao.reshape(p1-p0, -1)

        ki, kj = divmod(adapted_ji[0], nkpts)
        kls = kpts[kconserv[ki,kj,:]]

        for aoaoks, p0, p1 in with_df.ft_loop(with_df.mesh, q, -kls):
            for kk, aoao in enumerate(aoaoks):
                for ji, ji_idx in enumerate(adapted_ji):
                    ki, kj = divmod(ji_idx, nkpts)
                    ket[ki,kj,kk,p0:p1] = aoao.conj().reshape(p1-p0, -1)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra.conj(), ket

#TODO: blksize loop over mesh?
def _make_ao_eris_direct_fftdf(agf2, eris):
    ''' Get the 3c AO tensors for FFTDF '''

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    coords = cell.gen_uniform_grids(with_df.mesh)
    aos = with_df._numint.eval_ao(cell, coords, kpts)
    ngrids = len(coords)

    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        ki, kj = divmod(adapted_ji[0], nkpts)

        fac = tools.get_coulG(cell, q, mesh=with_df.mesh)
        fac *= (cell.vol / ngrids) / nkpts
        phase = np.exp(-1j * np.dot(coords, q))

        for ji_idx in adapted_ji:
            ki, kj = divmod(ji_idx, nkpts)

            buf = lib.einsum('gi,gj->gij', aos[ki].conj(), aos[kj])
            buf = buf.reshape(ngrids, -1)
            bra[ki,kj] = buf

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]

                buf = lib.einsum('gi,g,gj->ijg', aos[kk].conj(), phase.conj(), aos[kl])
                buf = tools.ifft(buf.reshape(-1, ngrids), with_df.mesh) * fac
                buf = tools.fft(buf.reshape(-1, ngrids), with_df.mesh) * phase
                ket[ki,kj,kk] = lib.transpose(buf)

                buf = None

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra, ket

#TODO: I think gdf can have different naux at different k-points, but we just pad with zeros
#TODO bra-ket symmetry? Must split 1/nkpts factor and sign (could be complex?)
def _make_ao_eris_direct_gdf(agf2, eris):
    ''' Get the 3c AO tensors for GDF '''

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    ngrids = _get_naux_from_cderi(with_df)

    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    for uid in mpi_helper.nrange(len(ukpts)):
        adapted_ji = np.where(uinv == uid)[0]

        for ji_idx in adapted_ji:
            ki, kj = divmod(ji_idx, nkpts)

            p1 = 0
            for qij_r, qij_i, sign in with_df.sr_loop(kpts[[ki,kj]], compact=False):
                p0, p1 = p1, p1 + qij_r.shape[0] 
                bra[ki,kj,p0:p1] = (qij_r - qij_i * 1j) / np.sqrt(nkpts)

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]

                q1 = 0
                for qkl_r, qkl_i, sign in with_df.sr_loop(kpts[[kk,kl]], compact=False):
                    q0, q1 = q1, q1 + qkl_r.shape[0]
                    ket[ki,kj,kk,q0:q1] = (qkl_r + qkl_i * 1j) * sign / np.sqrt(nkpts)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra, ket

#TODO: write out custom function combining aftdf and gdf to avoid copying
def _make_ao_eris_direct_mdf(agf2, eris):
    ''' Get the 3c AO tensors for MDF '''

    eri0 = _make_ao_eris_direct_aftdf(agf2, eris)
    eri1 = _make_ao_eris_direct_gdf(agf2, eris)

    bra = np.concatenate((eri0[0], eri1[0]), axis=-2)
    ket = np.concatenate((eri0[1], eri1[1]), axis=-2)

    return bra, ket

def _fao2mo(eri, cp, cq, dtype, out=None):
    ''' DF ao2mo '''
    #return np.einsum('lpq,pi,qj->lij', eri.reshape(-2, cp.shape[0], cq.shape[0]), cp.conj(), cq).reshape(-1, cp.shape[1]*cq.shape[1])

    npq, cpq, spq = ao2mo.incore._conc_mos(cp, cq, compact=False)[1:]
    sym = dict(aosym='s1', mosym='s1')
    naux = eri.shape[0]

    if out is None:
        out = np.zeros((naux*cp.shape[1]*cq.shape[1]), dtype=dtype)
    out = out.reshape(naux, cp.shape[1]*cq.shape[1])
    out = out.astype(dtype)

    if dtype in [np.float, np.float64]:
        out = ao2mo._ao2mo.nr_e2(eri, cpq, spq, out=out, **sym)
    else:
        cpq = np.asarray(cpq, dtype=complex)
        out = ao2mo._ao2mo.r_e2(eri, cpq, spq, [], None, out=out)

    return out.reshape(naux, npq)

def _make_mo_eris_direct(agf2, mo_coeff=None):
    # (pq|L)(L|rs) incore

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    cell = agf2.cell
    with_df = agf2.with_df
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    nmo = eris.nmo

    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    if isinstance(with_df, df.MDF):
        bra, ket = _make_ao_eris_direct_mdf(agf2, eris)
    elif isinstance(with_df, df.GDF):
        bra, ket = _make_ao_eris_direct_gdf(agf2, eris)
    elif isinstance(with_df, df.AFTDF):
        bra, ket = _make_ao_eris_direct_aftdf(agf2, eris)
    elif isinstance(with_df, df.FFTDF):
        bra, ket = _make_ao_eris_direct_fftdf(agf2, eris)
    else:
        raise ValueError('Unknown DF type %s' % type(with_df))

    dtype = complex
    naux = bra.shape[2]

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    qij = np.zeros((nkpts, nkpts, naux, nmo**2), dtype=dtype)
    qkl = np.zeros((nkpts, nkpts, nkpts, naux, nmo**2), dtype=dtype)

    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        kjs = kij[:,1][adapted_ji]

        for ji, ji_idx in enumerate(adapted_ji):
            ki, kj = divmod(adapted_ji[ji], nkpts)
            ci = mo_coeff[ki].conj()
            cj = mo_coeff[kj].conj()
            qij[ki,kj] = _fao2mo(bra[ki,kj], ci, cj, dtype, out=qij[ki,kj])

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]
                ck = mo_coeff[kk]
                cl = mo_coeff[kl]
                qkl[ki,kj,kk] = _fao2mo(ket[ki,kj,kk], ck, cl, dtype, out=qkl[ki,kj,kk])

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qij)
    mpi_helper.allreduce_safe_inplace(qkl)

    eris.qij = qij
    eris.qkl = qkl
    eris.eri = (qij, qkl)

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs, kpts):
    ''' (xi|ja)
    '''
    #return np.einsum('pqrs,qi,rj,sa->pija', eri.eri[kpts[0],kpts[1],kpts[2]], coeffs[0], coeffs[1].conj(), coeffs[2])

    #cput0 = (time.clock(), time.time())
    #log = logger.Logger(agf2.stdout, agf2.verbose)

    kx, ki, kj, ka = kpts
    ci, cj, ca = coeffs
    nmo = eri.nmo

    dtype = np.result_type(eri.eri.dtype, ci.dtype, cj.dtype, ca.dtype)
    xija = np.zeros((nmo*nmo, cj.shape[1], ca.shape[1]), dtype=dtype)

    mo = eri.eri[kx,ki,kj].reshape(-1, nmo*nmo)
    xija = _fao2mo(mo, cj, ca, dtype, out=xija)

    xija = xija.reshape(nmo, nmo, cj.shape[1], ca.shape[1])
    xija = lib.einsum('xyja,yi->xija', xija, ci)

    #log.timer('QMO integral transformation', *cput0)

    return xija

def _make_qmo_eris_direct(agf2, eri, coeffs, kpts):
    ''' (xi|L)(L|rs) incore
    '''
    #return (np.einsum('lpq,qi->lpi', eri.qij[kpts[0],kpts[1]].reshape(-1, eri.nmo, eri.nmo), coeffs[0].conj()).reshape(-1, eri.nmo*coeffs[0].shape[1]), 
    #        np.einsum('lpq,pi,qj->lij', eri.qkl[kpts[0],kpts[1],kpts[2]].reshape(-1, eri.nmo, eri.nmo), coeffs[1].conj(), coeffs[2]).reshape(-1, coeffs[1].shape[1]*coeffs[2].shape[1]))

    #cput0 = (time.clock(), time.time())
    #log = logger.Logger(agf2.stdout, agf2.verbose)

    kx, ki, kj, ka = kpts
    ci, cj, ca = coeffs
    nmo = eri.nmo
    
    dtype = np.result_type(eri.qij.dtype, eri.qkl.dtype, ci.dtype, cj.dtype, ca.dtype)
    naux = eri.eri[0].shape[2]

    qwx = eri.eri[0][kx,ki].reshape(-1, nmo)
    qwi = np.dot(qwx, ci.conj()).reshape(naux, -1)

    qyz = eri.eri[1][kx,ki,kj].reshape(-1, nmo, nmo)
    qja = _fao2mo(qyz, cj, ca, dtype).reshape(naux, -1)

    #log.timer('QMO integral transformation', *cput0)

    return qwi, qja
