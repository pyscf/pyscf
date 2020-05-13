#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
UCCSD analytical nuclear gradients
'''

import time
import ctypes
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import uccsd_rdm
from pyscf.scf import ucphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import ccsd as ccsd_grad


#
# Note: only works with canonical orbitals
# Non-canonical formula refers to JCP 95, 2639 (1991); DOI:10.1063/1.460916
#
def grad_elec(cc_grad, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
              d1=None, d2=None, verbose=logger.INFO):
    mycc = cc_grad.base
    if eris is not None:
        if (abs(eris.focka - numpy.diag(eris.focka.diagonal())).max() > 1e-3 or
            abs(eris.fockb - numpy.diag(eris.fockb.diagonal())).max() > 1e-3):
            raise RuntimeError('UCCSD gradients does not support NHF (non-canonical HF)')

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2

    log = logger.new_logger(mycc, verbose)
    time0 = time.clock(), time.time()

    log.debug('Build uccsd rdm1 intermediates')
    if d1 is None:
        d1 = uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    time1 = log.timer_debug1('rdm1 intermediates', *time0)
    log.debug('Build uccsd rdm2 intermediates')
    fdm2 = lib.H5TmpFile()
    if d2 is None:
        d2 = uccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)
    time1 = log.timer_debug1('rdm2 intermediates', *time1)

    mol = cc_grad.mol
    mo_a, mo_b = mycc.mo_coeff
    mo_ea, mo_eb = mycc._scf.mo_energy
    nao, nmoa = mo_a.shape
    nmob = mo_b.shape[1]
    nocca = numpy.count_nonzero(mycc.mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mycc.mo_occ[1] > 0)
    with_frozen = not ((mycc.frozen is None)
                       or (isinstance(mycc.frozen, (int, numpy.integer)) and mycc.frozen == 0)
                       or (len(mycc.frozen) == 0))
    moidx = mycc.get_frozen_mask()
    OA_a, VA_a, OF_a, VF_a = ccsd_grad._index_frozen_active(moidx[0], mycc.mo_occ[0])
    OA_b, VA_b, OF_b, VF_b = ccsd_grad._index_frozen_active(moidx[1], mycc.mo_occ[1])

    log.debug('symmetrized rdm2 and MO->AO transformation')
# Roughly, dm2*2 is computed in _rdm2_mo2ao
    mo_active = (mo_a[:,numpy.hstack((OA_a,VA_a))],
                 mo_b[:,numpy.hstack((OA_b,VA_b))])
    _rdm2_mo2ao(mycc, d2, mo_active, fdm2)  # transform the active orbitals
    time1 = log.timer_debug1('MO->AO transformation', *time1)
    hf_dm1a, hf_dm1b = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
    hf_dm1 = hf_dm1a + hf_dm1b

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    de = numpy.zeros((len(atmlst),3))
    Imata = numpy.zeros((nao,nao))
    Imatb = numpy.zeros((nao,nao))
    vhf1 = fdm2.create_dataset('vhf1', (len(atmlst),2,3,nao,nao), 'f8')

# 2e AO integrals dot 2pdm
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = numpy.zeros((2,3,nao,nao))
        for b0, b1, nf in ccsd_grad._shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2bufa = ccsd_grad._load_block_tril(fdm2['dm2aa+ab'], ip0, ip1, nao)
            dm2bufb = ccsd_grad._load_block_tril(fdm2['dm2bb+ab'], ip0, ip1, nao)
            dm2bufa[:,:,diagidx] *= .5
            dm2bufb[:,:,diagidx] *= .5
            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imata += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2bufa)
            Imatb += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2bufb)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2bufa) * 2
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2bufb) * 2
            dm2bufa = dm2bufb = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao,-1))
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[:,i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[0,i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1a[ip0:ip1])
                vhf[1,i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1b[ip0:ip1])
                vhf[:,i,ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[0,i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1a)
                vhf[1,i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1b)
            eri1 = eri1tmp = None
        vhf1[k] = vhf
        log.debug('2e-part grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

    s0 = mycc._scf.get_ovlp()
    Imata = reduce(numpy.dot, (mo_a.T, Imata, s0, mo_a)) * -1
    Imatb = reduce(numpy.dot, (mo_b.T, Imatb, s0, mo_b)) * -1

    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]
    if with_frozen:
        dco = Imata[OF_a[:,None],OA_a] / (mo_ea[OF_a,None] - mo_ea[OA_a])
        dfv = Imata[VF_a[:,None],VA_a] / (mo_ea[VF_a,None] - mo_ea[VA_a])
        dm1a[OA_a[:,None],OA_a] = (doo + doo.T) * .5
        dm1a[OF_a[:,None],OA_a] = dco
        dm1a[OA_a[:,None],OF_a] = dco.T
        dm1a[VA_a[:,None],VA_a] = (dvv + dvv.T) * .5
        dm1a[VF_a[:,None],VA_a] = dfv
        dm1a[VA_a[:,None],VF_a] = dfv.T
        dco = Imatb[OF_b[:,None],OA_b] / (mo_eb[OF_b,None] - mo_eb[OA_b])
        dfv = Imatb[VF_b[:,None],VA_b] / (mo_eb[VF_b,None] - mo_eb[VA_b])
        dm1b[OA_b[:,None],OA_b] = (dOO + dOO.T) * .5
        dm1b[OF_b[:,None],OA_b] = dco
        dm1b[OA_b[:,None],OF_b] = dco.T
        dm1b[VA_b[:,None],VA_b] = (dVV + dVV.T) * .5
        dm1b[VF_b[:,None],VA_b] = dfv
        dm1b[VA_b[:,None],VF_b] = dfv.T
    else:
        dm1a[:nocca,:nocca] = (doo + doo.T) * .5
        dm1a[nocca:,nocca:] = (dvv + dvv.T) * .5
        dm1b[:noccb,:noccb] = (dOO + dOO.T) * .5
        dm1b[noccb:,noccb:] = (dVV + dVV.T) * .5

    dm1 = (reduce(numpy.dot, (mo_a, dm1a, mo_a.T)),
           reduce(numpy.dot, (mo_b, dm1b, mo_b.T)))
    vhf = mycc._scf.get_veff(mycc.mol, dm1)
    Xvo = reduce(numpy.dot, (mo_a[:,nocca:].T, vhf[0], mo_a[:,:nocca]))
    XVO = reduce(numpy.dot, (mo_b[:,noccb:].T, vhf[1], mo_b[:,:noccb]))
    Xvo+= Imata[:nocca,nocca:].T - Imata[nocca:,:nocca]
    XVO+= Imatb[:noccb,noccb:].T - Imatb[noccb:,:noccb]

    dm1_resp = _response_dm1(mycc, (Xvo,XVO), eris)
    dm1a += dm1_resp[0]
    dm1b += dm1_resp[1]
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    Imata[nocca:,:nocca] = Imata[:nocca,nocca:].T
    Imatb[noccb:,:noccb] = Imatb[:noccb,noccb:].T
    im1 = reduce(numpy.dot, (mo_a, Imata, mo_a.T))
    im1+= reduce(numpy.dot, (mo_b, Imatb, mo_b.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('h1 and JK1')
    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = cc_grad.base._scf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    zeta = (mo_ea[:,None] + mo_ea) * .5
    zeta[nocca:,:nocca] = mo_ea[:nocca]
    zeta[:nocca,nocca:] = mo_ea[:nocca].reshape(-1,1)
    zeta_a = reduce(numpy.dot, (mo_a, zeta*dm1a, mo_a.T))
    zeta = (mo_eb[:,None] + mo_eb) * .5
    zeta[noccb:,:noccb] = mo_eb[:noccb]
    zeta[:noccb,noccb:] = mo_eb[:noccb].reshape(-1,1)
    zeta_b = reduce(numpy.dot, (mo_b, zeta*dm1b, mo_b.T))

    dm1 = (reduce(numpy.dot, (mo_a, dm1a, mo_a.T)),
           reduce(numpy.dot, (mo_b, dm1b, mo_b.T)))
    vhf_s1occ = mycc._scf.get_veff(mol, (dm1[0]+dm1[0].T, dm1[1]+dm1[1].T))
    p1a = numpy.dot(mo_a[:,:nocca], mo_a[:,:nocca].T)
    p1b = numpy.dot(mo_b[:,:noccb], mo_b[:,:noccb].T)
    vhf_s1occ = (reduce(numpy.dot, (p1a, vhf_s1occ[0], p1a)) +
                 reduce(numpy.dot, (p1b, vhf_s1occ[1], p1b))) * .5
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    dm1pa = hf_dm1a + dm1[0]*2
    dm1pb = hf_dm1b + dm1[1]*2
    dm1 = dm1[0] + dm1[1] + hf_dm1
    zeta_a += rhf_grad.make_rdm1e(mo_ea, mo_a, mycc.mo_occ[0])
    zeta_b += rhf_grad.make_rdm1e(mo_eb, mo_b, mycc.mo_occ[1])
    zeta = zeta_a + zeta_b

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, contribute to f1
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k,0], dm1pa)
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k,1], dm1pb)

    log.timer('%s gradients' % mycc.__class__.__name__, *time0)
    return de

def _response_dm1(mycc, Xvo, eris=None):
    Xvo, XVO = Xvo
    nvira, nocca = Xvo.shape
    nvirb, noccb = XVO.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb
    nova = nocca * nvira
    with_frozen = not ((mycc.frozen is None)
                       or (isinstance(mycc.frozen, (int, numpy.integer)) and mycc.frozen == 0)
                       or (len(mycc.frozen) == 0))
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ
        mo_a, mo_b = mycc.mo_coeff
        def fvind(x):
            x1a = x[0,:nova].reshape(Xvo.shape)
            x1b = x[0,nova:].reshape(XVO.shape)
            dm1a = reduce(numpy.dot, (mo_a[:,nocca:], x1a, mo_a[:,:nocca].T))
            dm1b = reduce(numpy.dot, (mo_b[:,noccb:], x1b, mo_b[:,:noccb].T))
            va, vb = mycc._scf.get_veff(mycc.mol, (dm1a+dm1a.T, dm1b+dm1b.T))
            va = reduce(numpy.dot, (mo_a[:,nocca:].T, va, mo_a[:,:nocca]))
            vb = reduce(numpy.dot, (mo_b[:,noccb:].T, vb, mo_b[:,:noccb]))
            return numpy.hstack((va.ravel(), vb.ravel()))
    else:
        moidx = mycc.get_frozen_mask()
        mo_energy = eris.mo_energy
        mo_occ = (mycc.mo_occ[0][moidx[0]], mycc.mo_occ[1][moidx[1]])
        ovvo = numpy.empty((nocca,nvira,nvira,nocca))
        ovVO = numpy.empty((nocca,nvira,nvirb,noccb))
        OVVO = numpy.empty((noccb,nvirb,nvirb,noccb))
        for i in range(nocca):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 2 - ovvo[i].transpose(1,0,2)
            ovvo[i]-= eris.oovv[i].transpose(2,1,0)
            ovVO[i] = eris.ovVO[i] * 2
        for i in range(noccb):
            OVVO[i] = eris.OVVO[i]
            OVVO[i] = OVVO[i] * 2 - OVVO[i].transpose(1,0,2)
            OVVO[i]-= eris.OOVV[i].transpose(2,1,0)
        def fvind(x):
            x1a = x[0,:nova].reshape(Xvo.shape)
            x1b = x[0,nova:].reshape(XVO.shape)
            va = numpy.einsum('iabj,bj->ai', ovvo, x1a)
            va+= numpy.einsum('iabj,bj->ai', ovVO, x1b)
            vb = numpy.einsum('iabj,bj->ai', OVVO, x1b)
            vb+= numpy.einsum('jbai,bj->ai', ovVO, x1a)
            return numpy.hstack((va.ravel(), vb.ravel()))
    dvo = ucphf.solve(fvind, mo_energy, mo_occ, (Xvo,XVO), max_cycle=30)[0]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1a[nocca:,:nocca] = dvo[0]
    dm1a[:nocca,nocca:] = dvo[0].T
    dm1b = numpy.zeros((nmob,nmob))
    dm1b[noccb:,:noccb] = dvo[1]
    dm1b[:noccb,noccb:] = dvo[1].T
    return dm1a, dm1b

def _rdm2_mo2ao(mycc, d2, mo_coeff, fsave=None):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = time.clock(), time.time()
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    dovov, dovOV, dOVov, dOVOV = d2[0]
    dvvvv, dvvVV, dVVvv, dVVVV = d2[1]
    doooo, dooOO, dOOoo, dOOOO = d2[2]
    doovv, dooVV, dOOvv, dOOVV = d2[3]
    dovvo, dovVO, dOVvo, dOVVO = d2[4]
    dvvov, dvvOV, dVVov, dVVOV = d2[5]
    dovvv, dovVV, dOVvv, dOVVV = d2[6]
    dooov, dooOV, dOOov, dOOOV = d2[7]
    mo_a = numpy.asarray(mo_coeff[0], order='F')
    mo_b = numpy.asarray(mo_coeff[1], order='F')

    nocca, nvira, noccb, nvirb = dovOV.shape
    nao, nmoa = mo_a.shape
    nmob = mo_b.shape[1]
    nao_pair = nao * (nao+1) // 2
    nvira_pair = nvira * (nvira+1) //2
    nvirb_pair = nvirb * (nvirb+1) //2

    fdrv = getattr(_ccsd.libcc, 'AO2MOnr_e2_drv')
    ftrans = _ccsd.libcc.AO2MOtranse2_nr_s1
    fmm = _ccsd.libcc.CCmmm_transpose_sum
    pao_loc = ctypes.POINTER(ctypes.c_void_p)()
    def _trans(vin, mo_coeff, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = numpy.empty((nrow,nao_pair))
        fdrv(ftrans, fmm,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    fswap = lib.H5TmpFile()
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize_a = int(max_memory*.9e6/8/(nao_pair+nmoa**2))
    blksize_a = min(nvira_pair, max(ccsd.BLKMIN, blksize_a))
    chunks_a = (int(min(nao_pair, 4e8/blksize_a)), blksize_a)
    v_aa = fswap.create_dataset('v_aa', (nao_pair,nvira_pair), 'f8',
                                chunks=chunks_a)
    for p0, p1 in lib.prange(0, nvira_pair, blksize_a):
        v_aa[:,p0:p1] = _trans(lib.unpack_tril(dvvvv[p0:p1]*.25), mo_a,
                               (nocca,nmoa,nocca,nmoa)).T

    v_ba = fswap.create_dataset('v_ab', (nao_pair,nvira_pair), 'f8',
                                chunks=chunks_a)
    dvvOP = fswap.create_dataset('dvvOP', (nvira_pair,noccb,nmob), 'f8',
                                 chunks=(int(min(blksize_a,4e8/nmob)),1,nmob))
    for i in range(noccb):
        buf1 = numpy.empty((nmob,nvira,nvira))
        buf1[:noccb] = dOOvv[i] * .5
        buf1[noccb:] = dOVvv[i]
        buf1 = buf1.transpose(1,2,0) + buf1.transpose(2,1,0)
        dvvOP[:,i] = buf1[numpy.tril_indices(nvira)]
    for p0, p1 in lib.prange(0, nvira_pair, blksize_a):
        buf1 = numpy.zeros((p1-p0,nmob,nmob))
        buf1[:,noccb:,noccb:] = lib.unpack_tril(dvvVV[p0:p1] * .5)
        buf1[:,:noccb,:] = dvvOP[p0:p1] * .5
        v_ba[:,p0:p1] = _trans(buf1, mo_b, (0,nmob,0,nmob)).T

    blksize_b = int(max_memory*.9e6/8/(nao_pair+nmob**2))
    blksize_b = min(nvirb_pair, max(ccsd.BLKMIN, blksize_b))
    chunks_b = (int(min(nao_pair, 4e8/blksize_b)), blksize_b)
    v_bb = fswap.create_dataset('v_bb', (nao_pair,nvirb_pair), 'f8',
                                chunks=chunks_b)
    for p0, p1 in lib.prange(0, nvirb_pair, blksize_b):
        v_bb[:,p0:p1] = _trans(lib.unpack_tril(dVVVV[p0:p1]*.25), mo_b,
                               (noccb,nmob,noccb,nmob)).T
    time1 = log.timer_debug1('_rdm2_mo2ao pass 1', *time1)

# transform dm2_ij to get lower triangular (dm2+dm2.transpose(0,1,3,2))
    blksize = int(max_memory*.9e6/8/(nao_pair+nmoa**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    o_aa = fswap.create_dataset('o_aa', (nmoa,nocca,nao_pair), 'f8', chunks=(nocca,nocca,blksize))
    o_ab = fswap.create_dataset('o_ab', (nmoa,nocca,nao_pair), 'f8', chunks=(nocca,nocca,blksize))
    o_bb = fswap.create_dataset('o_bb', (nmob,noccb,nao_pair), 'f8', chunks=(noccb,noccb,blksize))
    buf1 = numpy.zeros((nocca,nocca,nmoa,nmoa))
    buf1[:,:,:nocca,:nocca] = _cp(doooo) * .25
    buf1[:,:,nocca:,nocca:] = _cp(doovv) * .5
    buf1 = _trans(buf1.reshape(nocca**2,-1), mo_a, (0,nmoa,0,nmoa))
    o_aa[:nocca] = buf1.reshape(nocca,nocca,nao_pair)

    buf1 = numpy.zeros((nocca,nocca,nmob,nmob))
    buf1[:,:,:noccb,:noccb] = _cp(dooOO) * .5
    buf1[:,:,:noccb,noccb:] = _cp(dooOV)
    buf1[:,:,noccb:,noccb:] = _cp(dooVV) * .5
    buf1 = _trans(buf1.reshape(nocca**2,-1), mo_b, (0,nmob,0,nmob))
    o_ab[:nocca] = buf1.reshape(nocca,nocca,nao_pair)

    buf1 = numpy.zeros((noccb,noccb,nmob,nmob))
    buf1[:,:,:noccb,:noccb] = _cp(dOOOO) * .25
    buf1[:,:,noccb:,noccb:] = _cp(dOOVV) * .5
    buf1 = _trans(buf1.reshape(noccb**2,-1), mo_b, (0,nmob,0,nmob))
    o_bb[:noccb] = buf1.reshape(noccb,noccb,nao_pair)

    dovoo = numpy.asarray(dooov).transpose(2,3,0,1)
    dovOO = numpy.asarray(dOOov).transpose(2,3,0,1)
    dOVOO = numpy.asarray(dOOOV).transpose(2,3,0,1)
    for p0, p1 in lib.prange(nocca, nmoa, nocca):
        buf1 = numpy.zeros((nocca,p1-p0,nmoa,nmoa))
        buf1[:,:,:nocca,:nocca] = dovoo[:,p0-nocca:p1-nocca]
        buf1[:,:,nocca:,:nocca] = dovvo[:,p0-nocca:p1-nocca] * .5
        buf1[:,:,:nocca,nocca:] = dovov[:,p0-nocca:p1-nocca] * .5
        buf1[:,:,nocca:,nocca:] = dovvv[:,p0-nocca:p1-nocca]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*nocca,-1)
        buf1 = _trans(buf1, mo_a, (0,nmoa,0,nmoa))
        o_aa[p0:p1] = buf1.reshape(p1-p0,nocca,nao_pair)

        buf1 = numpy.zeros((nocca,p1-p0,nmob,nmob))
        buf1[:,:,:noccb,:noccb] = dovOO[:,p0-nocca:p1-nocca]
        buf1[:,:,noccb:,:noccb] = dovVO[:,p0-nocca:p1-nocca]
        buf1[:,:,:noccb,noccb:] = dovOV[:,p0-nocca:p1-nocca]
        buf1[:,:,noccb:,noccb:] = dovVV[:,p0-nocca:p1-nocca]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*nocca,-1)
        buf1 = _trans(buf1, mo_b, (0,nmob,0,nmob))
        o_ab[p0:p1] = buf1.reshape(p1-p0,nocca,nao_pair)

    for p0, p1 in lib.prange(noccb, nmob, noccb):
        buf1 = numpy.zeros((noccb,p1-p0,nmob,nmob))
        buf1[:,:,:noccb,:noccb] = dOVOO[:,p0-noccb:p1-noccb]
        buf1[:,:,noccb:,:noccb] = dOVVO[:,p0-noccb:p1-noccb] * .5
        buf1[:,:,:noccb,noccb:] = dOVOV[:,p0-noccb:p1-noccb] * .5
        buf1[:,:,noccb:,noccb:] = dOVVV[:,p0-noccb:p1-noccb]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*noccb,-1)
        buf1 = _trans(buf1, mo_b, (0,nmob,0,nmob))
        o_bb[p0:p1] = buf1.reshape(p1-p0,noccb,nao_pair)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 2', *time1)
    dovoo = buf1 = None

# transform dm2_kl then dm2 + dm2.transpose(2,3,0,1)
    dm2a = fsave.create_dataset('dm2aa+ab', (nao_pair,nao_pair), 'f8',
                                chunks=(int(min(nao_pair,4e8/blksize)),blksize))
    dm2b = fsave.create_dataset('dm2bb+ab', (nao_pair,nao_pair), 'f8',
                                chunks=(int(min(nao_pair,4e8/blksize)),blksize))
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmoa,nmoa))
        buf1[:,nocca:,nocca:] = lib.unpack_tril(_cp(v_aa[p0:p1]))
        buf1[:,:,:nocca] = o_aa[:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, mo_a, (0,nmoa,0,nmoa))
        if p0 > 0:
            buf1 = _cp(dm2a[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            dm2a[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        dm2a[p0:p1] = buf2
        buf1 = buf2 = None

    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmob,nmob))
        buf1[:,noccb:,noccb:] = lib.unpack_tril(_cp(v_bb[p0:p1]))
        buf1[:,:,:noccb] = o_bb[:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, mo_b, (0,nmob,0,nmob))
        if p0 > 0:
            buf1 = _cp(dm2b[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            dm2b[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        dm2b[p0:p1] = buf2
        buf1 = buf2 = None

    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmoa,nmoa))
        buf1[:,nocca:,nocca:] = lib.unpack_tril(_cp(v_ba[p0:p1]))
        buf1[:,:,:nocca] = o_ab[:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, mo_a, (0,nmoa,0,nmoa))
        dm2a[:,p0:p1] = dm2a[:,p0:p1] + buf2.T
        dm2b[p0:p1] = dm2b[p0:p1] + buf2
        buf1 = buf2 = None

    time1 = log.timer_debug1('_rdm2_mo2ao pass 3', *time1)
    if incore:
        return (fsave['dm2aa+ab'].value, fsave['dm2bb+ab'].value)
    else:
        return fsave

def _cp(a):
    return numpy.array(a, copy=False, order='C')

class Gradients(ccsd_grad.Gradients):
    grad_elec = grad_elec

Grad = Gradients

from pyscf.cc import uccsd
uccsd.UCCSD.Gradients = lib.class_as_method(Gradients)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import uccsd

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g',
        spin = 2,
    )
    mf = scf.UHF(mol).run()
    mycc = uccsd.UCCSD(mf).run()
    g1 = mycc.Gradients().kernel()
# O    -0.0000000000    -0.0000000000     0.1474630318
# H     0.0000000000     0.1118073694    -0.0737315159
# H     0.0000000000    -0.1118073694    -0.0737315159
    print(lib.finger(g1) - -0.22892718069135981)

    myccs = mycc.as_scanner()
    mol.atom[0] = ["O" , (0., 0., 0.001)]
    mol.build(0, 0)
    e1 = myccs(mol)
    mol.atom[0] = ["O" , (0., 0.,-0.001)]
    mol.build(0, 0)
    e2 = myccs(mol)
    print(g1[0,2], (e1-e2)/0.002*lib.param.BOHR)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g',
        spin = 2,
    )
    mf = scf.UHF(mol).run()
    mycc = uccsd.UCCSD(mf)
    mycc.frozen = [0,1,10,11,12]
    mycc.max_memory = 1
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
# O    -0.0000000000    -0.0000000000     0.1544815572
# H     0.0000000000     0.1146948540    -0.0772407786
# H     0.0000000000    -0.1146948540    -0.0772407786
    print(lib.finger(g1) - -0.23639703218041083)

    myccs = mycc.as_scanner()
    mol.atom[0] = ["O" , (0., 0., 0.001)]
    mol.build(0, 0)
    e1 = myccs(mol)
    mol.atom[0] = ["O" , (0., 0.,-0.001)]
    mol.build(0, 0)
    e2 = myccs(mol)
    print(g1[0,2], (e1-e2)/0.002*lib.param.BOHR)

