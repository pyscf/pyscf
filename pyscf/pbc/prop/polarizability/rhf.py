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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

'''
Non-relativistic static polarizability
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import cphf
from pyscf.prop.polarizability.rhf import Polarizability as mol_polar
from functools import reduce

def ifft(cell, fk, kpts, Ls):
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_scaled = np.mod(np.mod(kpts_scaled, 1), 1) 
    kpts = cell.get_abs_kpts(kpts_scaled)
    expkL = np.asarray(np.exp(-1j*np.dot(kpts, Ls.T)))
    Nk = len(kpts)
    fg = 1.0 / Nk * np.dot(expkL.T, fk.reshape(Nk, -1))
    return fg

def fft(cell, fg, kpts, Ls):
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_scaled = np.mod(np.mod(kpts_scaled, 1), 1)
    kpts = cell.get_abs_kpts(kpts_scaled)
    expkL = np.asarray(np.exp(1j*np.dot(kpts, Ls.T)))
    Nk = len(kpts)
    fk = np.dot(expkL, fg)
    return fk

def fft_k_deriv(cell, fg, kpts, Ls):
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_scaled = np.mod(np.mod(kpts_scaled, 1), 1)
    kpts = cell.get_abs_kpts(kpts_scaled)
    expkL = np.asarray(np.exp(1j*np.dot(kpts, Ls.T)))
    fk = []
    for beta in range(3):
        fk.append(1j*np.dot(expkL*Ls[:,beta], fg))
    return np.asarray(fk)

def get_h1(polobj):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    kpts = polobj.kpts
    cell = mf.cell
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpt)]
    orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpt)]

    with cell.with_common_orig((0., 0., 0.)):
        Z = cell.pbc_intor('int1e_r', comp=3, hermi=0, kpts=kpts, 
                           pbcopt=lib.c_null_ptr()) 
    Zji = []
    for k in range(nkpt):
        Zji.append(lib.einsum('xpq,pi,qj->xij', Z[k], 
                              orbv[k].conj(), orbo[k]))

    Ls = cell.get_lattice_Ls(discard=False)
    Rji = []
    Rao = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=True)
    for k in range(nkpt):
        Rji.append(lib.einsum('xpq,pi,qj->xij', Rao[k],
                              orbv[k].conj(), orbo[k]))

    Zji_tilde = []
    for k in range(nkpt):
        Zji_tilde.append(Zji[k] + 1j*0.5*Rji[k])

    fock = mf.get_fock()
    fockg = ifft(cell, fock, kpts, Ls)
    fockk = fft(cell, fockg, kpts, Ls)
    error = np.linalg.norm(fock.flatten() - fockk.flatten())
    if error > 1e-6:
        log.warn("k-mesh may be too small: FFT error = %.6e", error)

    Kao = fft_k_deriv(cell, fockg, kpts, Ls).reshape(3,nkpt,nao,nao)
    Kji = []
    for k in range(nkpt):
        Kji.append(lib.einsum('xpq,pi,qj->xij', Kao[:,k], 
                              orbv[k].conj(), orbo[k]))

    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_ai = [1 / lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]
    e_ai_plus = [lib.direct_sum('a+i->ai', e_a[k], e_i[k]) for k in range(nkpt)]    

    Qji_tilde = []
    for k in range(nkpt):
        Qji_k = 1j*(Kji[k] - 0.5 * Rji[k] * e_ai_plus[k])
        Qji_k[:] *= -e_ai[k]
        Qji_tilde.append(Qji_k)

    h1 = []
    s1 = []
    for k in range(nkpt):
        h1.append(Zji_tilde[k] + Qji_tilde[k])

    return h1

def polarizability(polobj, with_cphf=True):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    h1 = get_h1(polobj)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1=None,
                         max_cycle=polobj.max_cycle_cphf, tol=polobj.conv_tol,
                         verbose=log)[0]
    else:
        raise NotImplementedError

    e2 = 0    
    for k in range(nkpt):
        e2 += np.einsum('xpi,ypi->xy', h1[k].conj(), mo1[k]) / nkpt
    e2 = (e2 + e2.T.conj()) * -2
    if np.linalg.norm(e2.imag) > 1e-8:
        raise RuntimeError("Imaginary polarizability found! Something may be wrong.")
    e2 = e2.real
    if mf.verbose >= logger.INFO:
        xx, yy, zz = e2.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug('Static polarizability tensor\n%s', e2)
    return e2

def hyper_polarizability(polobj, with_cphf=True):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    kpts = polobj.kpts
    cell = mf.cell
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpt)]

    with cell.with_common_orig((0., 0., 0.)):
        Z = cell.pbc_intor('int1e_r', comp=3, hermi=0, kpts=kpts,
                           pbcopt=lib.c_null_ptr())
    Zji = []
    for k in range(nkpt):
        Zji.append(lib.einsum('xpq,pi,qj->xij', Z[k],
                              mo_coeff[k].conj(), orbo[k]))

    Ls = cell.get_lattice_Ls(discard=False)
    Rji = []
    Rao = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=True)
    for k in range(nkpt):
        Rji.append(lib.einsum('xpq,pi,qj->xij', Rao[k],
                              mo_coeff[k].conj(), orbo[k]))

    Zji_tilde = []
    for k in range(nkpt):
        Zji_tilde.append(Zji[k] + 1j*0.5*Rji[k])

    fock = mf.get_fock()
    fockg = ifft(cell, fock, kpts, Ls)
    fockk = fft(cell, fockg, kpts, Ls)
    error = np.linalg.norm(fock.flatten() - fockk.flatten())
    if error > 1e-6:
        log.warn("k-mesh may be too small: FFT error = %.6e", error)

    Kao = fft_k_deriv(cell, fockg, kpts, Ls).reshape(3,nkpt,nao,nao)
    Kji = []
    for k in range(nkpt):
        Kji.append(lib.einsum('xpq,pi,qj->xij', Kao[:,k],
                              mo_coeff[k].conj(), orbo[k]))

    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_all = [mo_energy[k] for k in range(nkpt)]
    e_ai = [1 / lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]
    e_ji_plus = [lib.direct_sum('a+i->ai', e_all[k], e_i[k]) for k in range(nkpt)]

    Qji_tilde = []
    for k in range(nkpt):
        Qji_k = 1j*(Kji[k] - 0.5 * Rji[k] * e_ji_plus[k])
        Qji_k[:,viridx[k]] *= -e_ai[k]
        Qji_k[:,occidx[k]] = 0.0
        Qji_tilde.append(Qji_k)

    h1 = []
    s1 = []
    for k in range(nkpt):
        h1.append(Zji_tilde[k] + Qji_tilde[k])
        s1.append(np.zeros_like(h1[k]))
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1, e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                             polobj.max_cycle_cphf, polobj.conv_tol,
                             verbose=log)
    else:
        raise NotImplementedError


    mo_coeff1 = [lib.einsum('xqi,pq->xpi', mo1[k], mo_coeff[k]) for k in range(nkpt)]
    dm1 = []
    for k in range(nkpt):
        dm1_k = lib.einsum('xpi,qi->xpq', mo_coeff1[k], orbo[k].conj()) * 2
        dm1_k = dm1_k + dm1_k.transpose(0,2,1).conj()
        dm1.append(dm1_k)
    dm1 = np.asarray(dm1).transpose(1,0,2,3)

    vresp = mf.gen_response(hermi=1)
    Bao = vresp(dm1)
    Bao_dk = []
    for i in range(3):
        Baog = ifft(cell, Bao[i], kpts, Ls)
        Baok = fft(cell, Baog, kpts, Ls)
        error = np.linalg.norm(Bao[i].flatten() - Baok.flatten())
        if error > 1e-6:
            log.warn("k-mesh may be too small: FFT error = %.6e", error)
        Bao_dk.append(fft_k_deriv(cell, Baog, kpts, Ls).reshape(3,nkpt,nao,nao))
    Bao_dk = np.asarray(Bao_dk)

    v1ao = Bao.transpose(1,0,2,3)
    v1mo = []
    for k in range(nkpt):
        v1mo.append(lib.einsum('xpq,pi,qj->xij', v1ao[k],
                               mo_coeff[k].conj(), mo_coeff[k]))

    h1_all = []
    for k in range(nkpt):
        h1_all_k = np.empty((3, nmo, nmo), dtype=np.complex128)
        h1_all_k[:, :, occidx[k]] = h1[k]
        mask_vv = np.outer(viridx[k], viridx[k])
        mask = np.stack([mask_vv,]*3, axis=0)
        h1_all_k[mask] = 0.0
        mask_ov = np.outer(occidx[k], viridx[k])
        mask = np.stack([mask_ov,]*3, axis=0)
        h1_all_k[mask] = h1[k][:, viridx[k]].transpose(0,2,1).conj().flatten()
        h1_all.append(h1_all_k) 
    G1 = [h1_all[k] + v1mo[k] for k in range(nkpt)]

    ##############
    # dU/dk part #
    ##############
    Z_dk = []
    Z = np.asarray(Z).transpose(1,0,2,3)
    for i in range(3):
        Zg = ifft(cell, Z[i], kpts, Ls)
        Zk = fft(cell, Zg, kpts, Ls)
        error = np.linalg.norm(Z[i].flatten() - Zk.flatten())
        if error > 1e-6:
            log.warn("k-mesh may be too small: FFT error = %.6e", error)
        Z_dk.append(fft_k_deriv(cell, Zg, kpts, Ls).reshape(3,nkpt,nao,nao))
    Z_dk = np.asarray(Z_dk)

    R_dk = []
    R = np.asarray(Rao).transpose(1,0,2,3)
    for i in range(3):
        Rg = ifft(cell, R[i], kpts, Ls)
        Rk = fft(cell, Rg, kpts, Ls)
        error = np.linalg.norm(R[i].flatten() - Rk.flatten())
        if error > 1e-6:
            log.warn("k-mesh may be too small: FFT error = %.6e", error)
        R_dk.append(fft_k_deriv(cell, Rg, kpts, Ls).reshape(3,nkpt,nao,nao))
    R_dk = np.asarray(R_dk)

    Kji = []
    for k in range(nkpt):
        Kji.append(lib.einsum('xpq,pi,qj->xij', Kao[:,k],
                              mo_coeff[k].conj(), mo_coeff[k]))

    Rji = []
    for k in range(nkpt):
        Rji.append(lib.einsum('xpq,pi,qj->xij', Rao[k],
                              mo_coeff[k].conj(), mo_coeff[k]))

    mo_dk = []
    Qji = []
    for k in range(nkpt):
        Qji_k = Kji[k] - Rji[k] * mo_energy[k]
        mask_ov = np.outer(occidx[k], viridx[k])
        mask_vo = np.outer(viridx[k], occidx[k])
        mask_oo = np.outer(occidx[k], occidx[k])
        mask_vv = np.outer(viridx[k], viridx[k])
        for i in range(3):
            Qji_k[i,mask_vo] *= -e_ai[k].flatten()
            Qji_k[i,mask_ov] *= e_ai[k].T.flatten()
            Qji_k[i,mask_oo] = -0.5 * Rji[k][i][mask_oo].flatten()
            Qji_k[i,mask_vv] = -0.5 * Rji[k][i][mask_vv].flatten()
        Qji.append(Qji_k)
        mo_dk.append(lib.einsum('uq,xqp->xup', mo_coeff[k], Qji_k))

    dedk = []
    for k in range(nkpt):
        dedk_k = Kji[k].diagonal(0,1,2) - Rji[k].diagonal(0,1,2) * mo_energy[k]
        dedk.append(dedk_k)

    #Rji_d2k
    Rji_d2k = []
    for k in range(nkpt):
        tmp = lib.einsum('yup,zuv,vi->yzpi', mo_dk[k].conj(), Rao[k], orbo[k])
        tmp += lib.einsum('up,zuv,yvi->yzpi', mo_coeff[k].conj(), Rao[k], mo_dk[k][:,:,occidx[k]])
        tmp += lib.einsum('up,zyuv,vi->yzpi', mo_coeff[k].conj(), R_dk[:,:,k], orbo[k])
        Rji_d2k.append(tmp)

    ##############
    # mo_d2k
    ##############
    Kao_dk = []
    for i in range(3):
        Kaog = ifft(cell, Kao[i], kpts, Ls)
        Kaok = fft(cell, Kaog, kpts, Ls)
        error = np.linalg.norm(Kao[i].flatten() - Kaok.flatten())
        if error > 1e-6:
            log.warn("k-mesh may be too small: FFT error = %.6e", error)
        Kao_dk.append(fft_k_deriv(cell, Kaog, kpts, Ls).reshape(3,nkpt,nao,nao))
    Kao_dk = np.asarray(Kao_dk)

    Kji_d2k = []
    for k in range(nkpt):
        tmp = lib.einsum('yup,zuv,vi->yzpi', mo_dk[k].conj(), Kao[:,k], orbo[k])
        tmp += lib.einsum('up,zuv,yvi->yzpi', mo_coeff[k].conj(), Kao[:,k], mo_dk[k][:,:,occidx[k]])
        tmp += lib.einsum('up,zyuv,vi->yzpi', mo_coeff[k].conj(), Kao_dk[:,:,k], orbo[k])
        Kji_d2k.append(tmp)

    dQdk = []
    for k in range(nkpt):
        tmp = Kji_d2k[k]
        tmp -= lib.einsum('zpi,yi->yzpi', Rji[k][:,:,occidx[k]], dedk[k][:,occidx[k]])
        tmp -= Rji_d2k[k] * mo_energy[k][occidx[k]]

        tmp -= lib.einsum('zpi,yi->yzpi', Qji[k][:,:,occidx[k]], dedk[k][:,occidx[k]])
        tmp += lib.einsum('zpi,yp->yzpi', Qji[k][:,:,occidx[k]], dedk[k])

        tmp[:,:,viridx[k]] *= -e_ai[k]
        tmp[:,:,occidx[k]] = -0.5 * Rji_d2k[k][:,:,occidx[k]]
        dQdk.append(tmp)

    mo_d2k = []
    for k in range(nkpt):
        tmp  = lib.einsum('yup,zpi->yzui', mo_dk[k], Qji[k][:,:,occidx[k]])
        tmp += lib.einsum('up,yzpi->yzui', mo_coeff[k], dQdk[k])
        mo_d2k.append(tmp)
    ##############

    S = mf.get_ovlp(kpts=kpts)
    dUdk = []
    for k in range(nkpt):
        tmp  = lib.einsum('yup,zuv,vi->yzpi', mo_dk[k].conj(), Bao[:,k]+Z[:,k]+1j*R[:,k], orbo[k])
        tmp += lib.einsum('up,zuv,yvi->yzpi', mo_coeff[k].conj(), Bao[:,k]+Z[:,k]+1j*R[:,k], mo_dk[k][:,:,occidx[k]])
        tmp += lib.einsum('up,zyuv,vi->yzpi', mo_coeff[k].conj(), Bao_dk[:,:,k]+Z_dk[:,:,k]+1j*R_dk[:,:,k], orbo[k])
        tmp += 1j * lib.einsum('yup,zvi,uv->yzpi', mo_dk[k].conj(), mo_dk[k][:,:,occidx[k]], S[k])
        tmp += 1j * lib.einsum('up,zvi,yuv->yzpi', mo_coeff[k].conj(), mo_dk[k][:,:,occidx[k]], Rao[k])
        tmp += 1j * lib.einsum('up,yzvi,uv->yzpi', mo_coeff[k].conj(), mo_d2k[k], S[k])

        tmp1 = -lib.einsum('zpi,yi->yzpi', G1[k][:,:,occidx[k]], dedk[k][:,occidx[k]])
        tmp1 += lib.einsum('zpi,yp->yzpi', G1[k][:,:,occidx[k]], dedk[k])
        tmp1[:,:,viridx[k]] *= -e_ai[k]
        tmp1[:,:,occidx[k]]  = 0.0

        tmp += tmp1
        tmp[:,:,viridx[k]] *= -e_ai[k]
        tmp[:,:,occidx[k]]  = 0.0
        dUdk.append(tmp)
    ##############

    e3 = 0.
    for k in range(nkpt):
        # *2 for double occupancy
        e3_k  = lib.einsum('xpi,ypq,zqi->xyz', mo1[k].conj(), G1[k], mo1[k]) * 2
        e3_k -= lib.einsum('xpi,yji,zpj->xyz', mo1[k].conj(), e1[k], mo1[k]) * 2
        e3_k += 1j * lib.einsum('xpi,yzpi->xyz', mo1[k].conj(), dUdk[k]) * 2
        e3_k = e3_k.real
        e3_k = (e3_k + e3_k.transpose(1,2,0) + e3_k.transpose(2,0,1) +
                e3_k.transpose(0,2,1) + e3_k.transpose(1,0,2) + e3_k.transpose(2,1,0))
        e3 += e3_k / nkpt
    e3 = -e3
    log.debug('Static hyper polarizability tensor\n%s', e3)
    return e3

def cphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                   max_cycle=20, tol=1e-6, hermi=False, verbose=logger.WARN):
    # lib.krylov often fails, newton_krylov solver from relatively new scipy
    # library is needed.
    from scipy.optimize import newton_krylov
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    mo_coeff = np.asarray(mf.mo_coeff)
    nkpt, nao, nmo = mo_coeff.shape

    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpt)]
    orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpt)]


    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_ai = [lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]

    # e_ai - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    LEVEL_SHIF = 0.1
    diag = []
    for k in range(nkpt):
        diag.append((e_ai[k] - freq, e_ai[k] + freq))
        diag[k][0][diag[k][0] < LEVEL_SHIF] += LEVEL_SHIF
        diag[k][1][diag[k][1] < LEVEL_SHIF] += LEVEL_SHIF

    nvir = [e_ai[k].shape[0] for k in range(nkpt)]
    nocc = [e_ai[k].shape[1] for k in range(nkpt)]
    ncomp = h1[0].shape[0]

    rhs_k = []
    mo1base_k = []
    for k in range(nkpt):
        rhs = np.stack((-h1[k], -h1[k].conj()), axis=1)
        rhs = rhs.reshape(ncomp, nocc[k]*nvir[k]*2)
        rhs_k.append(rhs)
        mo1base = np.stack((-h1[k]/diag[k][0], -h1[k].conj()/diag[k][1]), axis=1)
        mo1base = mo1base.reshape(ncomp, nocc[k]*nvir[k]*2)
        mo1base_k.append(mo1base)

    vresp = mf.gen_response(hermi=0)
    def vind(xys):
        nkpt = len(xys)
        ncomp = xys[0].shape[0]
        dms_k = []
        for k in range(nkpt):
            dms = np.empty((ncomp,nao,nao), dtype=np.complex128)
            for i in range(ncomp):
                x, y = xys[k][i].reshape(2,nvir[k],nocc[k])
                # *2 for double occupancy
                dmx = reduce(np.dot, (orbv[k], x  *2, orbo[k].T.conj()))
                dmy = reduce(np.dot, (orbo[k], y.T*2, orbv[k].T.conj()))
                dms[i] = dmx + dmy  # AX + BY
            dms_k.append(dms)

        dms_k = np.asarray(dms_k).transpose(1,0,2,3)
        v1ao = vresp(dms_k).transpose(1,0,2,3)
        res = []
        for k in range(nkpt):
            v1vo = lib.einsum('xpq,pi,qj->xij', v1ao[k], orbv[k].conj(), orbo[k])  # ~c1
            v1ov = lib.einsum('xpq,pi,qj->xji', v1ao[k], orbo[k].conj(), orbv[k])  # ~c1^T

            for i in range(ncomp):
                x, y = xys[k][i].reshape(2,nvir[k],nocc[k])
                v1vo[i] += (e_ai[k] - freq) * x
                v1ov[i] += (e_ai[k] + freq) * y
            v = np.stack((v1vo, v1ov), axis=1)
            res.append(v.reshape(ncomp,-1) - rhs_k[k])
        return res

    _mo1 = newton_krylov(vind, mo1base_k, f_tol=tol)
    mo1 = []
    for k in range(nkpt):
        mo1.append(_mo1[k].reshape(ncomp, 2, nvir[k], nocc[k]))
    log.timer('krylov solver in CPHF', *t0)

    dms_k = []
    for k in range(nkpt):
        dms = np.empty((ncomp,nao,nao), dtype=np.complex128)
        for i in range(ncomp):
            x, y = mo1[k][i]
            dmx = reduce(np.dot, (orbv[k], x  *2, orbo[k].T.conj()))
            dmy = reduce(np.dot, (orbo[k], y.T*2, orbv[k].T.conj()))
            dms[i] = dmx + dmy
        dms_k.append(dms)
    dms_k = np.asarray(dms_k).transpose(1,0,2,3)
    v1ao = vresp(dms_k).transpose(1,0,2,3)
    mo_e1 = 0.0
    for k in range(nkpt):
        mo_e1 += lib.einsum('xpq,pi,qj->xij', v1ao[k], orbo[k].conj(), orbo[k]) / nkpt
    mo1 = np.asarray(mo1)
    return (mo1[:,:,0], mo1[:,:,1]), mo_e1

def polarizability_with_freq(polobj, freq=None):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    h1 = get_h1(polobj)
    mo1 = cphf_with_freq(mf, mo_energy, mo_occ, h1, freq,
                         polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)[0]

    e2 = 0
    for k in range(nkpt):
        e2 += np.einsum('xpi,ypi->xy', h1[k], mo1[0][k].conj()) / nkpt
        e2 += np.einsum('xpi,ypi->xy', h1[k].conj(), mo1[1][k].conj()) / nkpt

    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 *= -2
    if np.linalg.norm(e2.imag) > 1e-8:
        raise RuntimeError("Imaginary polarizability found! Something may be wrong.")
    e2 = e2.real

    log.debug('Polarizability tensor with freq %s', freq)
    log.debug('%s', e2)
    return e2

class Polarizability(mol_polar):
    def __init__(self, mf, kpts=np.zeros((1,3))):
        self.kpts = kpts
        mol_polar.__init__(self, mf)

    def gen_vind(self, mf, mo_coeff, mo_occ):
        '''Induced potential'''
        nkpt, nao, nmo = mo_coeff.shape
        orbo = [mo_coeff[k][:,mo_occ[k]>0] for k in range(nkpt)]
        orbv = [mo_coeff[k][:,mo_occ[k]==0] for k in range(nkpt)]
        nocc = [orbo[k].shape[-1] for k in range(nkpt)]
        nvir = [orbv[k].shape[-1] for k in range(nkpt)]
        vresp = mf.gen_response(hermi=1)
        def vind(mo1):
            dm1 = []
            for k in range(nkpt):
                dm1_k = lib.einsum('xai,pa,qi->xpq', 
                                   mo1[k].reshape(-1, nvir[k], nocc[k]),
                                   orbv[k], orbo[k].conj())
                dm1_k = (dm1_k + dm1_k.transpose(0,2,1).conj()) * 2
                dm1.append(dm1_k)
            dm1 = np.asarray(dm1).transpose(1,0,2,3)
            v1ao = vresp(dm1).transpose(1,0,2,3)
            v1mo = []
            for k in range(nkpt):
                v1mo.append(lib.einsum('xpq,pi,qj->xij', v1ao[k],
                                       orbv[k].conj(), orbo[k]))
            return v1mo
        return vind

    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq
    hyper_polarizability = hyper_polarizability
