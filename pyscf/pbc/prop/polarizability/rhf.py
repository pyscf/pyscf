#!/usr/bin/env python
# Copyright 2020-2021 The PySCF Developers. All Rights Reserved.
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
Non-relativistic static/dynamic polarizability and hyper-polarizability
'''


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

def fft_k_deriv2(cell, fg, kpts, Ls):
    kpts_scaled = cell.get_scaled_kpts(kpts)
    kpts_scaled = np.mod(np.mod(kpts_scaled, 1), 1)
    kpts = cell.get_abs_kpts(kpts_scaled)
    expkL = np.asarray(np.exp(1j*np.dot(kpts, Ls.T)))
    res = -lib.einsum('na,nb,kn,np->abkp', Ls, Ls, expkL, fg)
    return res

def check_k_grids(cell, mat, kpts, fft_tol=1e-6):
    Ls = cell.get_lattice_Ls(discard=False)

    matg = ifft(cell, mat, kpts, Ls)
    matk = fft(cell, matg, kpts, Ls)
    error = np.linalg.norm(mat.flatten() - matk.flatten())
    if error > fft_tol:
        logger.warn(cell, "k-mesh or cell.rcut may be too small: FFT error = %.6e", error)

def get_k_deriv(cell, mat, kpts, deriv=1, check_k = False, fft_tol=1e-6):
    Ls = cell.get_lattice_Ls(discard=False)
    nkpt = len(kpts)
    nao = cell.nao

    if check_k: check_k_grids(cell, mat, kpts, fft_tol)
    matg = ifft(cell, mat, kpts, Ls)
    if deriv == 1:
        mat_dk = fft_k_deriv(cell, matg, kpts, Ls).reshape(3,nkpt,nao,nao)
    elif deriv == 2:
        mat_dk = fft_k_deriv2(cell, matg, kpts, Ls).reshape(3,3,nkpt,nao,nao)
    else:
        raise NotImplementedError
    return mat_dk

def get_z_ao(cell, kpts=np.zeros((1,3)), charge_center=None):
    r'''Computes the periodic version of the dipole matrix in AO basis.

    .. math:: \Omega_{\mu\nu}(k) = (\mu(k)|i e^{ikr} \nabla_{k} e^{-ikr}|\nu(k))

    Args:
        cell : instance of Cell class

    Kwargs:
        kpts : (nkpts, 3) array
        charge_center : list or tuple
            nuclear charge center

    Return:
        (nkpts, 3, nao, nao) array
    '''
    if charge_center is None:
        charges = cell.atom_charges()
        coords  = cell.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
    with cell.with_common_orig(charge_center):
        Zao = cell.pbc_intor('int1e_r', comp=3, kpts=kpts)
    Rao = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=1)
    Omega = Zao + 1j * Rao
    return Omega

def get_z(polobj, Zao=None, Rao=None, Kao=None, charge_center=None):
    r'''Computes the periodic version of the dipole matrix in MO basis, then
    transformed back to AO basis.

    Note this function differs from :func:`get_z_ao` in that it contains the
    k derivative of the MO coefficients.

    .. math::

        \Omega_{pq}(k) = (\psi_{p}(k)|i e^{ikr} \nabla_{k} e^{-ikr}|\psi_{q}(k)) \\
        \Omega_{\mu\nu}(k) = S_{\mu\lambda}(k) C_{\lambda p}(k) \Omega_{pq}(k) C_{\sigma q}(k) S_{\sigma\nu}(k)

    Args:
        polobj : instance of Polarizability class

    Kwargs:
        Zao : list of (3, nao, nao) array
            regular dipole matrix :math:`(\mu(k)| \|r-R\| |\nu(k))`
        Rao : list of (3, nao, nao) array
            1st order k derivative of overlap matrix
        Kao : list of (3, nao, nao) array
            1st order k derivative of Fock matrix
        charge_center : list or tuple (3,)
            nuclear charge center

    Returns:
        Omega : list of (3, nao, nao) array
            :math:`\Omega_{\mu\nu}(k)`
        Qji : list of (3, nmo, nmo) array
            response of the MO coefficients
    '''
    mf = polobj._scf
    kpts = polobj.kpts
    cell = mf.cell
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    nkpt, nao, nmo = mo_coeff.shape

    if charge_center is None:
        charges = cell.atom_charges()
        coords  = cell.atom_coords()
        charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
    if Zao is None:
        with cell.with_common_orig(charge_center):
            Zao = cell.pbc_intor('int1e_r', comp=3, kpts=kpts)
    if Rao is None:
        Rao = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=1)
    if Kao is None:
        fock = mf.get_fock()
        Kao = get_k_deriv(cell, fock, kpts)

    Kji = []
    Rji = []
    for k in range(nkpt):
        Kji.append(lib.einsum('xpq,pi,qj->xij', Kao[:,k],
                              mo_coeff[k].conj(), mo_coeff[k]))
        Rji.append(lib.einsum('xpq,pi,qj->xij', Rao[k],
                              mo_coeff[k].conj(), mo_coeff[k]))

    e_pq = [lib.direct_sum('p-q->pq', mo_energy[k], mo_energy[k]) for k in range(nkpt)]
    mask_pq = []
    #NOTE states with energy difference smaller than TOL_DEG are treated as degenerate states
    #NOTE this is temporary; whether states are degenerate should be determined by symmetry ideally
    TOL_DEG = 1e-6
    for k in range(nkpt):
        mask_pq.append(abs(e_pq[k]) > TOL_DEG)

    Qji = []
    for k in range(nkpt):
        Qji_k = Kji[k] - Rji[k] * mo_energy[k]
        for i in range(3):
            Qji_k[i,mask_pq[k]] *= -1.0 / e_pq[k][mask_pq[k]].flatten()
            Qji_k[i,~mask_pq[k]] = -0.5 * Rji[k][i][~mask_pq[k]].flatten()
        Qji.append(Qji_k)

    S = mf.get_ovlp()
    Qao = []
    for k in range(nkpt):
        Qao_k = lib.einsum('uv,vi,xij,sj,st->xut', S[k], mo_coeff[k], Qji[k], mo_coeff[k].conj(), S[k])
        Qao.append(Qao_k)

    Omega = []
    for k in range(nkpt):
        Omega.append(Zao[k] + 1j * Rao[k] + 1j * Qao[k])
    return Omega, Qji

def get_h1(polobj, Zao=None, Rao=None, Kao=None, charge_center=None, vo_only=False):
    r'''
    Computes h1 for CPHF equations in MO basis, :math:`(\psi_p(k)|h^1|\psi_i(k))`.
    if vo_only is True, h1 has shape (nmo, nocc); otherwise, h1 has shape (nvir, nocc).
    '''
    mf = polobj._scf
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    occidx = []
    viridx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)
        viridx.append(mo_occ[k] == 0)

    orbo = [mo_coeff[k][:,occidx[k]] for k in range(nkpt)]
    if vo_only:
        orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpt)]
    else:
        orbv = mo_coeff

    h1ao = get_z(polobj, Zao, Rao, Kao, charge_center)[0]
    h1 = []
    for k in range(nkpt):
        h1.append(lib.einsum('xpq,pi,qj->xij', h1ao[k],
                              orbv[k].conj(), orbo[k]))
    if vo_only:
        s1 = None
    else:
        s1 = []
        for k in range(nkpt):
            s1.append(np.zeros_like(h1[k]))
    return h1, s1

def dip_moment(polobj):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mo_occ = mf.mo_occ
    mo_coeff = np.asarray(mf.mo_coeff)
    nkpt, nao, nmo = mo_coeff.shape

    occidx = []
    for k in range(nkpt):
        occidx.append(mo_occ[k] > 0)

    h1 = get_h1(polobj)[0]
    dip = 0.0
    for k in range(nkpt):
        dip += h1[k][:,occidx[k]].diagonal(0,1,2).sum(axis=1) * -2.0 / nkpt
    dip = dip.real

    log.debug('Dipole moment\n%s', dip)
    return dip

def polarizability(polobj, with_cphf=True):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = mf.mo_occ
    nkpt, nao, nmo = mo_coeff.shape

    h1, s1 = get_h1(polobj, vo_only=True)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ, vo_only=True)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                         max_cycle=polobj.max_cycle_cphf, tol=polobj.conv_tol,
                         verbose=log)[0]
    else:
        raise NotImplementedError

    e2 = 0
    for k in range(nkpt):
        e2 += np.einsum('xpi,ypi->xy', h1[k].conj(), mo1[k])
    e2 = -2.0 / nkpt * (e2 + e2.T.conj())
    #if np.linalg.norm(e2.imag) > 1e-8:
    #    log.warn("Imaginary polarizability found.")
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
    orbv = [mo_coeff[k][:,viridx[k]] for k in range(nkpt)]

    charges = cell.atom_charges()
    coords  = cell.atom_coords()
    charge_center = np.einsum('i,ix->x', charges, coords) / charges.sum()
    with cell.with_common_orig(charge_center):
        Zao = cell.pbc_intor('int1e_r', comp=3, kpts=kpts)
        Z_dk = cell.pbc_intor('int1e_r', comp=3, kpts=kpts, kderiv=1).reshape(nkpt,3,3,nao,nao)

    Rao = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=1)
    R_dk = cell.pbc_intor("int1e_ovlp", kpts=kpts, kderiv=2).reshape(nkpt,3,3,nao,nao)

    fock = mf.get_fock()
    Kao = get_k_deriv(cell, fock, kpts)
    Kao_dk = get_k_deriv(cell, fock, kpts, deriv=2)

    h1, s1 = get_h1(polobj, Zao=Zao, Rao=Rao, Kao=Kao, charge_center=charge_center)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ, hermi=1)
    if with_cphf:
        mo1, e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                             max_cycle=polobj.max_cycle_cphf, tol=polobj.conv_tol,
                             verbose=log)
    else:
        raise NotImplementedError

    orbo1 = [lib.einsum('xpi,up->xui', mo1[k], mo_coeff[k]) for k in range(nkpt)]
    dm1 = []
    for k in range(nkpt):
        dm1_k = lib.einsum('xui,vi->xuv', orbo1[k], orbo[k].conj())
        dm1_k = (dm1_k + dm1_k.transpose(0,2,1).conj()) * 2
        dm1.append(dm1_k)
    dm1 = np.asarray(dm1).transpose(1,0,2,3)

    vresp = mf.gen_response(hermi=1)
    v1ao = vresp(dm1)
    v1ao_dk = []
    for i in range(3):
        v1ao_dk.append(get_k_deriv(cell, v1ao[i], kpts)) #(z,y,k,u,v)
    v1ao_dk = np.asarray(v1ao_dk).transpose(2,1,0,3,4)

    Omega, Qji = get_z(polobj, Zao=Zao, Rao=Rao, Kao=Kao, charge_center=charge_center)
    G1ao = v1ao.transpose(1,0,2,3) + np.asarray(Omega)
    G1vv = []
    for k in range(nkpt):
        tmp = lib.einsum('xuv,ua,vb->xab', G1ao[k], orbv[k].conj(), orbv[k])
        G1vv.append(tmp)

    ##############
    # dU/dk part #
    ##############
    e_a = [mo_energy[k][viridx[k]] for k in range(nkpt)]
    e_i = [mo_energy[k][occidx[k]] for k in range(nkpt)]
    e_ai = [1 / lib.direct_sum('a-i->ai', e_a[k], e_i[k]) for k in range(nkpt)]

    Kji = []
    Rji = []
    v1zji = []
    for k in range(nkpt):
        Kji.append(lib.einsum('xpq,pi,qj->xij', Kao[:,k],
                              mo_coeff[k].conj(), mo_coeff[k]))
        Rji.append(lib.einsum('xpq,pi,qj->xij', Rao[k],
                              mo_coeff[k].conj(), mo_coeff[k]))
        v1zji.append(lib.einsum('xpq,pi,qj->xij', v1ao[:,k] + Zao[k],
                                mo_coeff[k].conj(), mo_coeff[k]))

    dedk = []
    for k,kpt in enumerate(kpts):
        dedk_k = Kji[k].diagonal(0,1,2) - Rji[k].diagonal(0,1,2) * mo_energy[k]
        dedk.append(dedk_k.real)

    dRdk = []
    for k in range(nkpt):
        tmp  = lib.einsum('yqp,zqi->yzpi', Qji[k].conj(), Rji[k][:,:,occidx[k]])
        tmp += lib.einsum('zpq,yqi->yzpi', Rji[k], Qji[k][:,:,occidx[k]])
        tmp += lib.einsum('up,yzuv,vi->yzpi', mo_coeff[k].conj(), R_dk[k], orbo[k])
        dRdk.append(tmp)

    Kji_d2k = []
    for k in range(nkpt):
        tmp  = lib.einsum('yqp,zqi->yzpi', Qji[k].conj(), Kji[k][:,:,occidx[k]])
        tmp += lib.einsum('zpq,yqi->yzpi', Kji[k], Qji[k][:,:,occidx[k]])
        tmp += lib.einsum('up,yzuv,vi->yzpi', mo_coeff[k].conj(), Kao_dk[:,:,k], orbo[k])
        Kji_d2k.append(tmp)

    dQdk = []
    for k in range(nkpt):
        tmp  = Kji_d2k[k]
        tmp -= lib.einsum('zpi,yi->yzpi', Rji[k][:,:,occidx[k]], dedk[k][:,occidx[k]])
        tmp -= dRdk[k] * mo_energy[k][occidx[k]]

        tmp -= lib.einsum('zpi,yi->yzpi', Qji[k][:,:,occidx[k]], dedk[k][:,occidx[k]])
        tmp += lib.einsum('zpi,yp->yzpi', Qji[k][:,:,occidx[k]], dedk[k])

        tmp[:,:,viridx[k]] *= -e_ai[k]
        tmp[:,:,occidx[k]] = -0.5 * dRdk[k][:,:,occidx[k]]
        dQdk.append(tmp)

    dUdk = []
    for k in range(nkpt):
        v1_dk = v1ao_dk[k] + Z_dk[k]

        tmp  = lib.einsum('yqp,zqi->yzpi', Qji[k][:,:,viridx[k]].conj(), v1zji[k][:,:,occidx[k]])
        tmp += lib.einsum('zpq,yqi->yzpi', v1zji[k][:,viridx[k]], Qji[k][:,:,occidx[k]])
        tmp += lib.einsum('up,yzuv,vi->yzpi', mo_coeff[k][:,viridx[k]].conj(), v1_dk, orbo[k])
        tmp += 1j * (dRdk[k][:,:,viridx[k]] + dQdk[k][:,:,viridx[k]])

        tmp -= lib.einsum('zpi,yi->yzpi', mo1[k][:,viridx[k]], dedk[k][:,occidx[k]])
        tmp += lib.einsum('zpi,yp->yzpi', mo1[k][:,viridx[k]], dedk[k][:,viridx[k]])

        tmp *= -e_ai[k]
        dUdk.append(tmp)
    ##############

    e3 = 0.
    for k in range(nkpt):
        e3_k  = lib.einsum('xai,yab,zbi->xyz', mo1[k][:,viridx[k]].conj(), G1vv[k], mo1[k][:,viridx[k]])
        e3_k -= lib.einsum('xai,yji,zaj->xyz', mo1[k][:,viridx[k]].conj(), e1[k], mo1[k][:,viridx[k]])
        e3_k += 1j * lib.einsum('xai,yzai->xyz', mo1[k][:,viridx[k]].conj(), dUdk[k])
        e3_k = e3_k.real
        e3_k = (e3_k + e3_k.transpose(1,2,0) + e3_k.transpose(2,0,1) +
                e3_k.transpose(0,2,1) + e3_k.transpose(1,0,2) + e3_k.transpose(2,1,0))
        e3 += e3_k
    # *2 for double occupancy
    e3 = -2.0 / nkpt * e3
    log.debug('Static hyper polarizability tensor\n%s', e3)
    return e3

def cphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                   max_cycle=20, tol=1e-6, hermi=False, verbose=logger.WARN):
    # lib.krylov often fails, newton_krylov solver from relatively new scipy
    # library is needed.
    from scipy.optimize import newton_krylov
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

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

    h1, s1 = get_h1(polobj, vo_only=True)
    mo1 = cphf_with_freq(mf, mo_energy, mo_occ, h1, freq,
                         polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)[0]

    e2 = 0
    for k in range(nkpt):
        e2 += np.einsum('xpi,ypi->xy', h1[k], mo1[0][k].conj()) / nkpt
        e2 += np.einsum('xpi,ypi->xy', h1[k].conj(), mo1[1][k].conj()) / nkpt

    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 *= -2
    #if np.linalg.norm(e2.imag) > 1e-8:
    #    log.warn("Imaginary polarizability found.")
    e2 = e2.real

    log.debug('Polarizability tensor with freq %s', freq)
    log.debug('%s', e2)
    return e2

class Polarizability(mol_polar):
    def __init__(self, mf, kpts=np.zeros((1,3))):
        self.kpts = kpts
        mol_polar.__init__(self, mf)
        check_k_grids(mf.cell, mf.get_fock(), kpts)

    def gen_vind(self, mf, mo_coeff, mo_occ, hermi=1, vo_only=False):
        '''Induced potential'''
        nkpt, nao, nmo = mo_coeff.shape
        orbo = [mo_coeff[k][:,mo_occ[k]>0] for k in range(nkpt)]
        if vo_only:
            orbv = [mo_coeff[k][:,mo_occ[k]==0] for k in range(nkpt)]
        else:
            orbv = mo_coeff
        nocc = [orbo[k].shape[-1] for k in range(nkpt)]
        if vo_only:
            nvir = [orbv[k].shape[-1] for k in range(nkpt)]
        else:
            nvir = [nmo,]*nkpt
        vresp = mf.gen_response(hermi=hermi)
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

    dip_moment = dip_moment
    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq
    hyper_polarizability = hyper_polarizability

if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom = """H  0.0 0.0 0.0
                   F  0.9 0.0 0.0
                """
    cell.basis = 'sto-3g'
    cell.a = [[2.82, 0, 0], [0, 2.82, 0], [0, 0, 2.82]]
    cell.dimension = 1
    cell.precision = 1e-10
    cell.build()

    kpts = cell.make_kpts([16,1,1])
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv="ewald").density_fit()
    e0 = kmf.kernel()

    #TODO implement the finite field version
    polar = Polarizability(kmf, kpts)
    dip = polar.dip_moment()
    e2 = polar.polarizability()
    e2_w0 = polar.polarizability_with_freq(freq=0.)
    e2_w1 = polar.polarizability_with_freq(freq=0.1)
    e2_w2 = polar.polarizability_with_freq(freq=-0.1)
    e3 = polar.hyper_polarizability()

    h1 = kmf.get_hcore()
    z = get_z(polar)[0]
    def apply_E(E):
        kmf.get_hcore = lambda *args, **kwargs: h1 + np.einsum('x,kxij->kij', E, z)
        e = kmf.kernel()
        return e

    h = 1e-4
    dip_x = -(apply_E([0.5*h, 0., 0.]) - apply_E([-0.5*h, 0., 0.])) / h
    print(dip)
    print(dip_x)

    e2_xx = -(apply_E([h, 0., 0.]) - 2*e0 + apply_E([-h, 0., 0.])) / h**2
    print(e2)
    print(e2_xx)

    print(e2_w0)
    print(e2_w1)
    print(e2_w2)
    print(e3)
