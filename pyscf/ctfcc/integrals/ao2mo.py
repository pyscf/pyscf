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
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

'''core module for CC/k-CC ao2mo transformation'''
import numpy
import ctf
import time
from pyscf import gto, ao2mo, lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ctfcc import mpi_helper
from symtensor.sym_ctf import tensor, einsum
import pyscf.pbc.tools.pbc as tools

comm = mpi_helper.comm
rank = mpi_helper.rank
size = mpi_helper.size

def _make_ao_ints(mol, mo_coeff, nocc, dtype):
    '''
    partial ao2mo transformation, complex mo_coeff not supported
    returns:
      ppoo,     ppov,     ppvv
    (uv|ij),  (uv|ia),  (uv|ab)
    '''
    NS = ctf.SYM.NS
    SY = ctf.SYM.SY

    ao_loc = mol.ao_loc_nr()
    mo = numpy.asarray(mo_coeff, order='F')
    nao, nmo = mo.shape
    nvir = nmo - nocc

    ppoo = ctf.tensor((nao,nao,nocc,nocc), sym=[SY,NS,NS,NS], dtype=dtype)
    ppov = ctf.tensor((nao,nao,nocc,nvir), sym=[SY,NS,NS,NS], dtype=dtype)
    ppvv = ctf.tensor((nao,nao,nvir,nvir), sym=[SY,NS,SY,NS], dtype=dtype)
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    blksize = int(max(4, min(nao/3, nao/size**.5, 2000e6/8/nao**3)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
    tasks = []
    for k, (ish0, ish1, di) in enumerate(sh_ranges):
        for jsh0, jsh1, dj in sh_ranges[:k+1]:
            tasks.append((ish0,ish1,jsh0,jsh1))

    sqidx = numpy.arange(nao**2).reshape(nao,nao)
    trilidx = sqidx[numpy.tril_indices(nao)]
    vsqidx = numpy.arange(nvir**2).reshape(nvir,nvir)
    vtrilidx = vsqidx[numpy.tril_indices(nvir)]

    subtasks = list(mpi_helper.static_partition(tasks))
    ntasks = max(comm.allgather(len(subtasks)))
    for itask in range(ntasks):
        if itask >= len(subtasks):
            ppoo.write([], [])
            ppov.write([], [])
            ppvv.write([], [])
            continue

        shls_slice = subtasks[itask]
        ish0, ish1, jsh0, jsh1 = shls_slice
        i0, i1 = ao_loc[ish0], ao_loc[ish1]
        j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
        di = i1 - i0
        dj = j1 - j0
        if i0 != j0:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s2kl',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            idx = sqidx[i0:i1,j0:j1].ravel()

            eri = _ao2mo.nr_e2(eri.reshape(di*dj,-1), mo, (0,nmo,0,nmo), 's2kl', 's1')
        else:
            eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                          shls_slice=shls_slice, aosym='s4',
                                          ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
            eri = _ao2mo.nr_e2(eri, mo, (0,nmo,0,nmo), 's4', 's1')
            idx = sqidx[i0:i1,j0:j1][numpy.tril_indices(i1-i0)]

        ooidx = idx[:,None] * nocc**2 + numpy.arange(nocc**2)
        ovidx = idx[:,None] * (nocc*nvir) + numpy.arange(nocc*nvir)
        vvidx = idx[:,None] * nvir**2 + vtrilidx
        eri = eri.reshape(-1,nmo,nmo)
        ppoo.write(ooidx.ravel(), eri[:,:nocc,:nocc].ravel())
        ppov.write(ovidx.ravel(), eri[:,:nocc,nocc:].ravel())
        ppvv.write(vvidx.ravel(), lib.pack_tril(eri[:,nocc:,nocc:]).ravel())
        idx = eri = None
    return ppoo, ppov, ppvv

def _make_fftdf_eris(mycc, mo_a, mo_b, nocca, noccb, out=None):
    mydf = mycc._scf.with_df
    kpts = mycc.kpts
    cell = mydf.cell
    gvec = cell.reciprocal_vectors()
    nao = cell.nao_nr()
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)
    nkpts = len(kpts)
    nmoa, nmob = mo_a.shape[-1], mo_b.shape[-1]
    nvira, nvirb = nmoa-nocca, nmob-noccb
    if mo_a.shape==mo_b.shape:
        RESTRICTED = numpy.linalg.norm(mo_a-mo_b) < 1e-10
    else:
        RESTRICTED = False
    cput1 = cput0 = (time.clock(), time.time())
    ijG = ctf.zeros([nkpts,nkpts,nocca,nocca,ngrids], dtype=numpy.complex128)
    iaG = ctf.zeros([nkpts,nkpts,nocca,nvira,ngrids], dtype=numpy.complex128)
    abG = ctf.zeros([nkpts,nkpts,nvira,nvira,ngrids], dtype=numpy.complex128)

    ijR = ctf.zeros([nkpts,nkpts,noccb,noccb,ngrids], dtype=numpy.complex128)
    iaR = ctf.zeros([nkpts,nkpts,noccb,nvirb,ngrids], dtype=numpy.complex128)
    aiR = ctf.zeros([nkpts,nkpts,nvirb,noccb,ngrids], dtype=numpy.complex128)
    abR = ctf.zeros([nkpts,nkpts,nvirb,nvirb,ngrids], dtype=numpy.complex128)

    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])
    tasks = mpi_helper.static_partition(jobs)
    ntasks = max(comm.allgather(len(tasks)))

    idx_ooG = numpy.arange(nocca*nocca*ngrids)
    idx_ovG = numpy.arange(nocca*nvira*ngrids)
    idx_vvG = numpy.arange(nvira*nvira*ngrids)

    idx_ooR = numpy.arange(noccb*noccb*ngrids)
    idx_ovR = numpy.arange(noccb*nvirb*ngrids)
    idx_vvR = numpy.arange(nvirb*nvirb*ngrids)

    for itask in range(ntasks):
        if itask >= len(tasks):
            ijR.write([], [])
            iaR.write([], [])
            aiR.write([], [])
            abR.write([], [])
            ijR.write([], [])
            iaR.write([], [])
            aiR.write([], [])
            abR.write([], [])

            ijG.write([], [])
            iaG.write([], [])
            abG.write([], [])
            ijG.write([], [])
            iaG.write([], [])
            abG.write([], [])
            continue
        ki, kj = tasks[itask]
        kpti, kptj = kpts[ki], kpts[kj]
        ao_kpti = mydf._numint.eval_ao(cell, coords, kpti)[0]
        ao_kptj = mydf._numint.eval_ao(cell, coords, kptj)[0]
        q = kptj - kpti
        coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
        wcoulG = coulG * (cell.vol/ngrids)
        fac = numpy.exp(-1j * numpy.dot(coords, q))
        mo_kpti_b = numpy.dot(ao_kpti, mo_b[ki]).T
        mo_kptj_b = numpy.dot(ao_kptj, mo_b[kj]).T
        mo_pairs_b = numpy.einsum('ig,jg->ijg', mo_kpti_b.conj(), mo_kptj_b)
        if RESTRICTED:
            mo_pairs_a = mo_pairs_b
        else:
            mo_kpti_a = numpy.dot(ao_kpti, mo_a[ki]).T
            mo_kptj_a = numpy.dot(ao_kptj, mo_a[kj]).T
            mo_pairs_a = numpy.einsum('ig,jg->ijg', mo_kpti_a.conj(), mo_kptj_a)

        mo_pairs_G = tools.fft(mo_pairs_a.reshape(-1,ngrids)*fac, mydf.mesh)

        off = ki * nkpts + kj
        ijR.write(off*idx_ooR.size+idx_ooR, mo_pairs_b[:noccb,:noccb].ravel())
        iaR.write(off*idx_ovR.size+idx_ovR, mo_pairs_b[:noccb,noccb:].ravel())
        aiR.write(off*idx_ovR.size+idx_ovR, mo_pairs_b[noccb:,:noccb].ravel())
        abR.write(off*idx_vvR.size+idx_vvR, mo_pairs_b[noccb:,noccb:].ravel())

        off = kj * nkpts + ki
        mo_pairs_b = mo_pairs_b.transpose(1,0,2).conj()
        ijR.write(off*idx_ooR.size+idx_ooR, mo_pairs_b[:noccb,:noccb].ravel())
        iaR.write(off*idx_ovR.size+idx_ovR, mo_pairs_b[:noccb,noccb:].ravel())
        aiR.write(off*idx_ovR.size+idx_ovR, mo_pairs_b[noccb:,:noccb].ravel())
        abR.write(off*idx_vvR.size+idx_vvR, mo_pairs_b[noccb:,noccb:].ravel())

        mo_pairs_a = mo_pairs_b = None
        mo_pairs_G*= wcoulG
        v = tools.ifft(mo_pairs_G, mydf.mesh)
        v *= fac.conj()
        v = v.reshape(nmob,nmob,ngrids)

        off = ki * nkpts + kj
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocca,:nocca].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocca,nocca:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocca:,nocca:].ravel())

        off = kj * nkpts + ki
        v = v.transpose(1,0,2).conj()
        ijG.write(off*idx_ooG.size+idx_ooG, v[:nocca,:nocca].ravel())
        iaG.write(off*idx_ovG.size+idx_ovG, v[:nocca,nocca:].ravel())
        abG.write(off*idx_vvG.size+idx_vvG, v[nocca:,nocca:].ravel())

    cput1 = logger.timer(mycc, "generating (pq|G)", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooG = tensor(ijG, sym1, verbose=mycc.SYMVERBOSE)
    ovG = tensor(iaG, sym1, verbose=mycc.SYMVERBOSE)
    vvG = tensor(abG, sym1, verbose=mycc.SYMVERBOSE)

    ooR = tensor(ijR, sym2, verbose=mycc.SYMVERBOSE)
    ovR = tensor(iaR, sym2, verbose=mycc.SYMVERBOSE)
    voR = tensor(aiR, sym2, verbose=mycc.SYMVERBOSE)
    vvR = tensor(abR, sym2, verbose=mycc.SYMVERBOSE)

    oooo = einsum('ijg,klg->ijkl', ooG, ooR)/ nkpts
    ooov = einsum('ijg,kag->ijka', ooG, ovR)/ nkpts
    oovv = einsum('ijg,abg->ijab', ooG, vvR)/ nkpts
    ooG = ooR = ijG = ijR = None
    ovvo = einsum('iag,bjg->iabj', ovG, voR)/ nkpts
    ovov = einsum('iag,jbg->iajb', ovG, ovR)/ nkpts
    ovR = iaR = voR = aiR = None
    ovvv = einsum('iag,bcg->iabc', ovG, vvR)/ nkpts
    ovG = iaG = None
    vvvv = einsum('abg,cdg->abcd', vvG, vvR)/ nkpts
    cput1 = logger.timer(mycc, "(pq|G) to (pq|rs)", *cput1)
    if out is None:
        return oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv
    else:
        return oooo+out[0], ooov+out[1], oovv+out[2], \
               ovvo+out[3], ovov+out[4], ovvv+out[5], vvvv+out[6]

def make_fftdf_eris_rhf(mycc, eris):
    mo_coeff = eris.mo_coeff
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                _make_fftdf_eris(mycc, mo_coeff, mo_coeff, nocc, nocc)
    eris.oooo = oooo
    eris.ooov = ooov
    eris.oovv = oovv
    eris.ovvo = ovvo
    eris.ovov = ovov
    eris.ovvv = ovvv
    eris.vvvv = vvvv

def make_fftdf_eris_uhf(mycc, eris):
    mo_a, mo_b = eris.mo_coeff[0], eris.mo_coeff[1]
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                _make_fftdf_eris(mycc, mo_a, mo_a, nocca, nocca)
    eris.oooo = oooo
    eris.ooov = ooov
    eris.oovv = oovv
    eris.ovov = ovov
    eris.voov = ovvo.transpose(1,0,3,2).conj()
    eris.vovv = ovvv.transpose(1,0,3,2).conj()
    eris.vvvv = vvvv

    OOOO, OOOV, OOVV, OVVO, OVOV, OVVV, VVVV = \
                _make_fftdf_eris(mycc, mo_b, mo_b, noccb, noccb)
    eris.OOOO = OOOO
    eris.OOOV = OOOV
    eris.OOVV = OOVV
    eris.OVOV = OVOV
    eris.VOOV = OVVO.transpose(1,0,3,2).conj()
    eris.VOVV = OVVV.transpose(1,0,3,2).conj()
    eris.VVVV = VVVV

    ooOO, ooOV, ooVV, ovVO, ovOV, ovVV, vvVV = \
                _make_fftdf_eris(mycc, mo_a, mo_b, nocca, noccb)
    eris.ooOO = ooOO
    eris.ooOV = ooOV
    eris.ooVV = ooVV
    eris.ovOV = ovOV
    eris.voOV = ovVO.transpose(1,0,3,2).conj()
    eris.voVV = ovVV.transpose(1,0,3,2).conj()
    eris.vvVV = vvVV

    _, OOov, OOvv, OVvo, OVov, OVvv, _ = \
                _make_fftdf_eris(mycc, mo_b, mo_a, noccb, nocca)
    eris.OOov = OOov
    eris.OOvv = OOvv
    eris.OVov = OVov
    eris.VOov = OVvo.transpose(1,0,3,2).conj()
    eris.VOvv = OVvv.transpose(1,0,3,2).conj()

def make_fftdf_eris_ghf(mycc, eris):
    nocc = mycc.nocc
    nkpts = mycc.nkpts
    nao = mycc._scf.cell.nao_nr()
    if getattr(eris.mo_coeff[0], 'orbspin', None) is None:
        # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
        mo_a_coeff = numpy.asarray([mo[:nao] for mo in eris.mo_coeff])
        mo_b_coeff = numpy.asarray([mo[nao:] for mo in eris.mo_coeff])
        eri = _make_fftdf_eris(mycc, mo_a_coeff, mo_a_coeff, nocc, nocc)
        eri = _make_fftdf_eris(mycc, mo_b_coeff, mo_b_coeff, nocc, nocc, eri)
        eri = _make_fftdf_eris(mycc, mo_a_coeff, mo_b_coeff, nocc, nocc, eri)
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv =\
                            _make_fftdf_eris(mycc, mo_b_coeff, mo_a_coeff, nocc, nocc, eri)
        eri = None
    else:
        mo_a_coeff = numpy.asarray([mo[:nao] + mo[nao:] for mo in eris.mo_coeff])
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                        _make_fftdf_eris(mycc, mo_a_coeff, mo_a_coeff, nocc, nocc)
        jobs = numpy.arange(nkpts**3)
        tasks = mpi_helper.static_partition(jobs)
        ntasks = max(comm.allgather(len(tasks)))
        def _force_sym(eri, symbols, kp, kq, kr):
            off = (kp*nkpts**2+kq*nkpts+kr)*numpy.prod(eri.array.shape[3:])
            pq_size = int(numpy.prod(eri.array.shape[3:5]))
            rs_size = int(numpy.prod(eri.array.shape[5:]))
            ks = eris.kconserv[kp,kq,kr]
            orb =[]
            for sx, kx in zip(symbols, [kp,kq,kr,ks]):
                if sx=='o':
                    orb.append(getattr(eris.mo_coeff[kx], 'orbspin')[:nocc])
                elif sx=='v':
                    orb.append(getattr(eris.mo_coeff[kx], 'orbspin')[nocc:])
                else:
                    raise ValueError("orbital label %s not recognized " %sx)
            pqforbid = numpy.asarray(numpy.where((orb[0][:,None] != orb[1]).ravel()), dtype=int)[0]
            rsforbid = numpy.asarray(numpy.where((orb[2][:,None] != orb[3]).ravel()), dtype=int)[0]
            idxpq = off + pqforbid[:,None] * rs_size + numpy.arange(rs_size)
            idxrs = off + numpy.arange(pq_size)[:,None]*rs_size + rsforbid
            idx = numpy.concatenate((idxpq.ravel(), idxrs.ravel()))
            eri.write(idx, numpy.zeros(idx.size))

        for itask in range(ntasks):
            if itask >= len(tasks):
                oooo.write([],[])
                ooov.write([],[])
                oovv.write([],[])
                ovvo.write([],[])
                ovov.write([],[])
                ovvv.write([],[])
                vvvv.write([],[])
                continue
            kp, kq, kr = mpi_helper.unpack_idx(tasks[itask], nkpts, nkpts, nkpts)
            _force_sym(oooo, 'oooo', kp, kq, kr)
            _force_sym(ooov, 'ooov', kp, kq, kr)
            _force_sym(oovv, 'oovv', kp, kq, kr)
            _force_sym(ovvo, 'ovvo', kp, kq, kr)
            _force_sym(ovov, 'ovov', kp, kq, kr)
            _force_sym(ovvv, 'ovvv', kp, kq, kr)
            _force_sym(vvvv, 'vvvv', kp, kq, kr)

    eris.vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(2,0,1,3)
    eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    del vvvv, ovvv
    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(2,0,1,3)
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    eris.ovov = oovv.transpose(0,2,1,3) - ovvo.transpose(0,2,3,1)
    eris.ovvo = ovvo.transpose(0,2,1,3) - oovv.transpose(0,2,3,1)
    del oooo, ooov, ovov, oovv, ovvo

def _ao2mo_j3c(mydf, mo_coeff, nocc):
    '''
    ao2mo on density-fitted j3c integrals
    returns:
      ijL,     iaL,     aiL,    abL
    (ij|L),  (ia|L),  (ai|L), (ab|L)
    '''
    from pyscf.ctfcc.integrals import mpigdf
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    if getattr(mydf, 'j3c', None) is None: mydf.build()
    kpts = mydf.kpts
    nkpts = len(kpts)
    nao, naux = mydf.j3c.shape[2:]
    ijL = ctf.zeros([nkpts,nkpts,nocc,nocc,naux], dtype=mydf.j3c.dtype)
    iaL = ctf.zeros([nkpts,nkpts,nocc,nvir,naux], dtype=mydf.j3c.dtype)
    aiL = ctf.zeros([nkpts,nkpts,nvir,nocc,naux], dtype=mydf.j3c.dtype)
    abL = ctf.zeros([nkpts,nkpts,nvir,nvir,naux], dtype=mydf.j3c.dtype)
    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])
    tasks = mpi_helper.static_partition(jobs)
    ntasks = max(comm.allgather((len(tasks))))
    idx_j3c = numpy.arange(nao**2*naux)
    idx_ooL = numpy.arange(nocc**2*naux)
    idx_ovL = numpy.arange(nocc*nvir*naux)
    idx_vvL = numpy.arange(nvir**2*naux)
    cput1 = cput0 = (time.clock(), time.time())
    for itask in range(ntasks):
        if itask >= len(tasks):
            mydf.j3c.read([])
            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])

            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])
            continue
        ki, kj = tasks[itask]
        ijid, ijdagger = mpigdf.get_member(kpts[ki], kpts[kj], mydf.kptij_lst)
        uvL = mydf.j3c.read(ijid*idx_j3c.size+idx_j3c).reshape(nao,nao,naux)
        if ijdagger: uvL = uvL.transpose(1,0,2).conj()
        pvL = numpy.einsum("up,uvL->pvL", mo_coeff[ki].conj(), uvL, optimize=True)
        uvL = None
        pqL = numpy.einsum('vq,pvL->pqL', mo_coeff[kj], pvL, optimize=True)

        off = ki * nkpts + kj
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        pqL = pqL.transpose(1,0,2).conj()
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

    return ijL, iaL, aiL, abL

def make_df_eris_rhf(mycc, eris):
    mydf = mycc._scf.with_df
    mo_coeff = eris.mo_coeff
    nocc = mycc.nocc
    kpts = mycc.kpts
    nkpts = len(kpts)
    gvec = mydf.cell.reciprocal_vectors()
    cput1 = (time.clock(), time.time())
    ijL, iaL, aiL, abL = _ao2mo_j3c(mydf, mo_coeff, nocc)
    cput1 = logger.timer(mycc, "j3c transformation", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooL = tensor(ijL, sym1, verbose=mycc.SYMVERBOSE)
    ovL = tensor(iaL, sym1, verbose=mycc.SYMVERBOSE)
    voL = tensor(aiL, sym1, verbose=mycc.SYMVERBOSE)
    vvL = tensor(abL, sym1, verbose=mycc.SYMVERBOSE)

    ooL2 = tensor(ijL, sym2, verbose=mycc.SYMVERBOSE)
    ovL2 = tensor(iaL, sym2, verbose=mycc.SYMVERBOSE)
    voL2 = tensor(aiL, sym2, verbose=mycc.SYMVERBOSE)
    vvL2 = tensor(abL, sym2, verbose=mycc.SYMVERBOSE)

    eris.oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
    eris.ooov = einsum('ijg,kag->ijka', ooL, ovL2) / nkpts
    eris.oovv = einsum('ijg,abg->ijab', ooL, vvL2) / nkpts
    eris.ovvo = einsum('iag,bjg->iabj', ovL, voL2) / nkpts
    eris.ovov = einsum('iag,jbg->iajb', ovL, ovL2) / nkpts
    eris.ovvv = einsum('iag,bcg->iabc', ovL, vvL2) / nkpts
    eris.vvvv = einsum('abg,cdg->abcd', vvL, vvL2) / nkpts

    cput1 = logger.timer(mycc, "integral transformation", *cput1)

def make_df_eris_uhf(mycc, eris):
    mydf = mycc._scf.with_df
    mo_a, mo_b = eris.mo_coeff[0], eris.mo_coeff[1]
    nocca, noccb = mycc.nocc
    kpts = mycc.kpts
    nkpts = len(kpts)
    gvec = mydf.cell.reciprocal_vectors()
    cput1 = (time.clock(), time.time())
    ijL, iaL, aiL, abL = _ao2mo_j3c(mydf, mo_a, nocca)
    IJL, IAL, AIL, ABL = _ao2mo_j3c(mydf, mo_b, noccb)
    cput1 = logger.timer(mycc, "(uv|L)->(pq|L)", *cput1)
    sym1 = ["+-+", [kpts,]*3, None, gvec]
    sym2 = ["+--", [kpts,]*3, None, gvec]

    ooL = tensor(ijL, sym1, verbose=mycc.SYMVERBOSE)
    ovL = tensor(iaL, sym1, verbose=mycc.SYMVERBOSE)
    voL = tensor(aiL, sym1, verbose=mycc.SYMVERBOSE)
    vvL = tensor(abL, sym1, verbose=mycc.SYMVERBOSE)

    ooL2 = tensor(ijL, sym2, verbose=mycc.SYMVERBOSE)
    ovL2 = tensor(iaL, sym2, verbose=mycc.SYMVERBOSE)
    voL2 = tensor(aiL, sym2, verbose=mycc.SYMVERBOSE)
    vvL2 = tensor(abL, sym2, verbose=mycc.SYMVERBOSE)

    OOL = tensor(IJL, sym1, verbose=mycc.SYMVERBOSE)
    OVL = tensor(IAL, sym1, verbose=mycc.SYMVERBOSE)
    VOL = tensor(AIL, sym1, verbose=mycc.SYMVERBOSE)
    VVL = tensor(ABL, sym1, verbose=mycc.SYMVERBOSE)

    OOL2 = tensor(IJL, sym2, verbose=mycc.SYMVERBOSE)
    OVL2 = tensor(IAL, sym2, verbose=mycc.SYMVERBOSE)
    VOL2 = tensor(AIL, sym2, verbose=mycc.SYMVERBOSE)
    VVL2 = tensor(ABL, sym2, verbose=mycc.SYMVERBOSE)

    eris.oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
    eris.ooov = einsum('ijg,kag->ijka', ooL, ovL2) / nkpts
    eris.oovv = einsum('ijg,abg->ijab', ooL, vvL2) / nkpts
    eris.voov = einsum('iag,jbg->aibj', ovL, voL2).conj() /nkpts
    eris.ovov = einsum('iag,jbg->iajb', ovL, ovL2) / nkpts
    eris.vovv = einsum('iag,bcg->aicb', ovL, vvL2).conj() / nkpts
    eris.vvvv = einsum('abg,cdg->abcd', vvL, vvL2) / nkpts

    eris.OOOO = einsum('ijg,klg->ijkl', OOL, OOL2) / nkpts
    eris.OOOV = einsum('ijg,kag->ijka', OOL, OVL2) / nkpts
    eris.OOVV = einsum('ijg,abg->ijab', OOL, VVL2) / nkpts
    eris.VOOV = einsum('iag,jbg->aibj', OVL, VOL2).conj() /nkpts
    eris.OVOV = einsum('iag,jbg->iajb', OVL, OVL2) / nkpts
    eris.VOVV = einsum('iag,bcg->aicb', OVL, VVL2).conj() / nkpts
    eris.VVVV = einsum('abg,cdg->abcd', VVL, VVL2) / nkpts

    eris.ooOO = einsum('ijg,klg->ijkl', ooL, OOL2) / nkpts
    eris.ooOV = einsum('ijg,kag->ijka', ooL, OVL2) / nkpts
    eris.ooVV = einsum('ijg,abg->ijab', ooL, VVL2) / nkpts
    eris.ovOV = einsum('iag,jbg->iajb', ovL, OVL2) / nkpts
    eris.voOV = einsum('iag,jbg->aibj', ovL, VOL2).conj() /nkpts
    eris.voVV = einsum('iag,bcg->aicb', ovL, VVL2).conj() / nkpts
    eris.vvVV = einsum('abg,cdg->abcd', vvL, VVL2) / nkpts

    eris.OOov = einsum('ijg,kag->ijka', OOL, ovL2) / nkpts
    eris.OOvv = einsum('ijg,abg->ijab', OOL, vvL2) / nkpts
    eris.OVov = einsum('iag,jbg->iajb', OVL, ovL2) / nkpts
    eris.VOov = einsum('iag,jbg->aibj', OVL, voL2).conj() /nkpts
    eris.VOvv = einsum('iag,bcg->aicb', OVL, vvL2).conj() / nkpts

    cput1 = logger.timer(mycc, "integral transformation", *cput1)

def make_df_eris_ghf(mycc, eris):
    nocc = mycc.nocc
    nkpts = mycc.nkpts
    nao = mycc._scf.cell.nao_nr()
    nocc = mycc.nocc
    mydf = mycc._scf.with_df
    if getattr(eris.mo_coeff[0], 'orbspin', None) is None:
        # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
        mo_a_coeff = numpy.asarray([mo[:nao] for mo in eris.mo_coeff])
        mo_b_coeff = numpy.asarray([mo[nao:] for mo in eris.mo_coeff])

        ijL, iaL, aiL, abL = _ao2mo_j3c(mydf, mo_a_coeff, nocc)
        IJL, IAL, AIL, ABL = _ao2mo_j3c(mydf, mo_b_coeff, nocc)
        ijL += IJL
        iaL += IAL
        aiL += AIL
        abL += ABL
        del IJL, IAL, AIL, ABL
        ooL = tensor(ijL, sym1, verbose=mycc.SYMVERBOSE)
        ovL = tensor(iaL, sym1, verbose=mycc.SYMVERBOSE)
        voL = tensor(aiL, sym1, verbose=mycc.SYMVERBOSE)
        vvL = tensor(abL, sym1, verbose=mycc.SYMVERBOSE)

        ooL2 = tensor(ijL, sym2, verbose=mycc.SYMVERBOSE)
        ovL2 = tensor(iaL, sym2, verbose=mycc.SYMVERBOSE)
        voL2 = tensor(aiL, sym2, verbose=mycc.SYMVERBOSE)
        vvL2 = tensor(abL, sym2, verbose=mycc.SYMVERBOSE)

        oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
        ooov = einsum('ijg,klg->ijkl', ooL, ovL2) / nkpts
        oovv = einsum('ijg,klg->ijkl', ooL, vvL2) / nkpts
        ovvo = einsum('ijg,klg->ijkl', ovL, voL2) / nkpts
        ovov = einsum('ijg,klg->ijkl', ovL, ovL2) / nkpts
        ovvv = einsum('ijg,klg->ijkl', ovL, vvL2) / nkpts
        vvvv = einsum('ijg,klg->ijkl', vvL, vvL2) / nkpts
        del ooL, ovL, voL, vvL, ooL2, ovL2, voL2, vvL2, ijL, iaL, aiL, abL
    else:
        mo_a_coeff = numpy.asarray([mo[:nao] + mo[nao:] for mo in eris.mo_coeff])
        ijL, iaL, aiL, abL = _ao2mo_j3c(mydf, mo_a_coeff, nocc)
        ooL = tensor(ijL, sym1, verbose=mycc.SYMVERBOSE)
        ovL = tensor(iaL, sym1, verbose=mycc.SYMVERBOSE)
        voL = tensor(aiL, sym1, verbose=mycc.SYMVERBOSE)
        vvL = tensor(abL, sym1, verbose=mycc.SYMVERBOSE)

        ooL2 = tensor(ijL, sym2, verbose=mycc.SYMVERBOSE)
        ovL2 = tensor(iaL, sym2, verbose=mycc.SYMVERBOSE)
        voL2 = tensor(aiL, sym2, verbose=mycc.SYMVERBOSE)
        vvL2 = tensor(abL, sym2, verbose=mycc.SYMVERBOSE)

        oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
        ooov = einsum('ijg,klg->ijkl', ooL, ovL2) / nkpts
        oovv = einsum('ijg,klg->ijkl', ooL, vvL2) / nkpts
        ovvo = einsum('ijg,klg->ijkl', ovL, voL2) / nkpts
        ovov = einsum('ijg,klg->ijkl', ovL, ovL2) / nkpts
        ovvv = einsum('ijg,klg->ijkl', ovL, vvL2) / nkpts
        vvvv = einsum('ijg,klg->ijkl', vvL, vvL2) / nkpts
        jobs = numpy.arange(nkpts**3)
        tasks = mpi_helper.static_partition(jobs)
        ntasks = max(comm.allgather(len(tasks)))
        def _force_sym(eri, symbols, kp, kq, kr):
            off = (kp*nkpts**2+kq*nkpts+kr)*numpy.prod(eri.array.shape[3:])
            pq_size = int(numpy.prod(eri.array.shape[3:5]))
            rs_size = int(numpy.prod(eri.array.shape[5:]))
            ks = eris.kconserv[kp,kq,kr]
            orb =[]
            for sx, kx in zip(symbols, [kp,kq,kr,ks]):
                if sx=='o':
                    orb.append(getattr(eris.mo_coeff[kx], 'orbspin')[:nocc])
                elif sx=='v':
                    orb.append(getattr(eris.mo_coeff[kx], 'orbspin')[nocc:])
                else:
                    raise ValueError("orbital label %s not recognized " %sx)
            pqforbid = numpy.asarray(numpy.where((orb[0][:,None] != orb[1]).ravel()), dtype=int)[0]
            rsforbid = numpy.asarray(numpy.where((orb[2][:,None] != orb[3]).ravel()), dtype=int)[0]
            idxpq = off + pqforbid[:,None] * rs_size + numpy.arange(rs_size)
            idxrs = off + numpy.arange(pq_size)[:,None]*rs_size + rsforbid
            idx = numpy.concatenate((idxpq.ravel(), idxrs.ravel()))
            eri.write(idx, numpy.zeros(idx.size))

        for itask in range(ntasks):
            if itask >= len(tasks):
                oooo.write([],[])
                ooov.write([],[])
                oovv.write([],[])
                ovvo.write([],[])
                ovov.write([],[])
                ovvv.write([],[])
                vvvv.write([],[])
                continue
            kp, kq, kr = mpi_helper.unpack_idx(tasks[itask], nkpts, nkpts, nkpts)
            _force_sym(oooo, 'oooo', kp, kq, kr)
            _force_sym(ooov, 'ooov', kp, kq, kr)
            _force_sym(oovv, 'oovv', kp, kq, kr)
            _force_sym(ovvo, 'ovvo', kp, kq, kr)
            _force_sym(ovov, 'ovov', kp, kq, kr)
            _force_sym(ovvv, 'ovvv', kp, kq, kr)
            _force_sym(vvvv, 'vvvv', kp, kq, kr)
    eris.vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(2,0,1,3)
    eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    del vvvv, ovvv
    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(2,0,1,3)
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    eris.ovov = oovv.transpose(0,2,1,3) - ovvo.transpose(0,2,3,1)
    eris.ovvo = ovvo.transpose(0,2,1,3) - oovv.transpose(0,2,3,1)
    del oooo, ooov, ovov, oovv, ovvo
