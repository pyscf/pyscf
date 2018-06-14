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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
G0W0 approximation
'''

import time
import tempfile
import numpy
import numpy as np
import h5py
from scipy.optimize import newton

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import dft
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, td_e, td_xy, eris=None,
           orbs=None, verbose=logger.NOTE):
    '''GW-corrected quasiparticle orbital energies

    Returns:
        A list :  converged, mo_energy, mo_coeff
    '''
    # mf must be DFT; for HF use xc = 'hf'
    mf = gw._scf
    assert(isinstance(mf, (dft.rks.RKS      , dft.uks.UKS,
                           dft.roks.ROKS    , dft.uks.UKS,
                           dft.rks_symm.RKS , dft.uks_symm.UKS,
                           dft.rks_symm.ROKS, dft.uks_symm.UKS)))
    assert(gw.frozen is 0 or gw.frozen is None)

    if eris is None:
        eris = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(gw.nmo)

    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(numpy.dot, (mo_coeff.T, v_mf, mo_coeff))

    nocc = gw.nocc
    nmo = gw.nmo
    nvir = nmo-nocc

    vk_oo = -np.einsum('piiq->pq', eris.oooo)
    vk_ov = -np.einsum('piiq->pq', eris.ooov)
    vk_vv = -np.einsum('ipqi->pq', eris.ovvo).conj()
    vk = np.array(np.bmat([[vk_oo, vk_ov],[vk_ov.T, vk_vv]]))

    nexc = len(td_e)
    # factor of 2 for normalization, see tddft/rhf.py
    td_xy = 2*np.asarray(td_xy) # (nexc, 2, nvir, nocc)
    td_z = np.sum(td_xy, axis=1).reshape(nexc,nvir,nocc)
    tdm_oo = einsum('vai,iapq->vpq', td_z, eris.ovoo)
    tdm_ov = einsum('vai,iapq->vpq', td_z, eris.ovov)
    tdm_vv = einsum('vai,iapq->vpq', td_z, eris.ovvv)
    tdm = []
    for oo,ov,vv in zip(tdm_oo,tdm_ov,tdm_vv):
        tdm.append(np.array(np.bmat([[oo, ov],[ov.T, vv]])))
    tdm = np.asarray(tdm)

    conv = True
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in orbs:
        tdm_p = tdm[:,:,p]
        if gw.linearized:
            de = 1e-6
            ep = gw._scf.mo_energy[p]
            #TODO: analytic sigma derivative
            sigma = get_sigma_element(gw, ep, tdm_p, tdm_p, td_e).real
            dsigma = get_sigma_element(gw, ep+de, tdm_p, tdm_p, td_e).real - sigma
            zn = 1.0/(1-dsigma/de)
            e = ep + zn*(sigma.real + vk[p,p] - v_mf[p,p])
            mo_energy[p] = e
        else:
            def quasiparticle(omega):
                sigma = get_sigma_element(gw, omega, tdm_p, tdm_p, td_e)
                return omega - gw._scf.mo_energy[p] - (sigma.real + vk[p,p] - v_mf[p,p])
            try:
                e = newton(quasiparticle, gw._scf.mo_energy[p], tol=1e-6, maxiter=100)
                mo_energy[p] = e
            except RuntimeError:
                conv = False
    mo_coeff = gw._scf.mo_coeff

    if gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff


def get_sigma_element(gw, omega, tdm_p, tdm_q, td_e, eta=None, vir_sgn=1):
    if eta is None:
        eta = gw.eta

    nocc = gw.nocc
    evi = lib.direct_sum('v-i->vi', td_e, gw._scf.mo_energy[:nocc])
    eva = lib.direct_sum('v+a->va', td_e, gw._scf.mo_energy[nocc:])
    sigma =  np.sum( tdm_p[:,:nocc]*tdm_q[:,:nocc]/(omega + evi - 1j*eta) )
    sigma += np.sum( tdm_p[:,nocc:]*tdm_q[:,nocc:]/(omega - eva + vir_sgn*1j*eta) )
    return sigma


def get_g(omega, mo_energy, mo_occ, eta):
    sgn = mo_occ - 1
    return 1.0/(omega - mo_energy + 1j*eta*sgn)


class GW(lib.StreamObject):
    '''non-relativistic restricted GW

    Saved results

        mo_energy :
            Orbital energies
        mo_coeff
            Orbital coefficients
    '''

    eta = getattr(__config__, 'gw_gw_GW_eta', 1e-3)
    linearized = getattr(__config__, 'gw_gw_GW_linearized', False)

    def __init__(self, mf, tdmf, frozen=0):
        self.mol = mf.mol
        self._scf = mf
        self._tdscf = tdmf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        #TODO: implement frozen orbs
        #self.frozen = frozen
        self.frozen = 0

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ

        keys = set(('eta', 'linearized'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        return self

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def get_g0(self, omega, eta=None):
        if eta is None:
            eta = self.eta
        return get_g(omega, self._scf.mo_energy, self.mo_occ, eta)

    def get_g(self, omega, eta=None):
        if eta is None:
            eta = self.eta
        return get_g(omega, self.mo_energy, self.mo_occ, eta)

    def kernel(self, mo_energy=None, mo_coeff=None, td_e=None, td_xy=None,
               eris=None, orbs=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy

        cput0 = (time.clock(), time.time())
        self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
                kernel(self, mo_energy, mo_coeff, td_e, td_xy,
                       eris=eris, orbs=orbs, verbose=self.verbose)

        logger.timer(self, 'GW', *cput0)
        return self.mo_energy

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)


def _mem_usage(nocc, nvir):
    incore = (nocc+nvir)**4
    # Roughly, factor of two for safety
    incore *= 2
    basic = nocc*nvir**3
    outcore = basic
    return incore*8/1e6, outcore*8/1e6, basic*8/1e6


class _ERIS:
    '''Almost identical to rccsd ERIS except ovvv is dense and no vvvv.
    '''
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.full):
        cput0 = (time.clock(), time.time())
        moidx = get_frozen_mask(cc)
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff[:,moidx]
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            if ao2mofn == ao2mo.full:
                if cc._scf._eri is not None:
                    eri = ao2mo.restore(1, ao2mofn(cc._scf._eri, mo_coeff), nmo)
                else:
                    eri = ao2mo.restore(1, ao2mofn(cc._scf.mol, mo_coeff, compact=0), nmo)
            else:
                eri = ao2mofn(cc._scf.mol, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), compact=0)
                if mo_coeff.dtype == np.float: eri = eri.real
                eri = eri.reshape((nmo,)*4)

            self.dtype = eri.dtype
            self.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovo = eri[:nocc,:nocc,nocc:,:nocc].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
            self.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()

        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            raise NotImplementedError

        else:
            orbo = mo_coeff[:,:nocc]
            self.dtype = mo_coeff.dtype
            ds_type = mo_coeff.dtype.char
            self.feri = lib.H5TmpFile()
            self.oooo = self.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovo = self.feri.create_dataset('oovo', (nocc,nocc,nvir,nocc), ds_type)
            self.ovov = self.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.oovv = self.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovvo = self.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), ds_type)
            self.ovvv = self.feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo.general(cc.mol, (orbo,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmo,nmo,nmo))
                for i in range(nocc):
                    lib.unpack_tril(f['aa'][i*nmo:(i+1)*nmo], out=buf)
                    self.oooo[i] = buf[:nocc,:nocc,:nocc]
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    self.oovo[i] = buf[:nocc,nocc:,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovvo[i] = buf[nocc:,nocc:,:nocc]
                    self.ovvv[i] = buf[nocc:,nocc:,nocc:]
                del(f['aa'])
                buf = None

            cput1 = log.timer_debug1('transforming oopq, ovpq', *cput1)

        log.timer('GW integral transformation', *cput0)


if __name__ == '__main__':
    from pyscf import gto, dft, tddft
    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo-nocc

    td = tddft.dRPA(mf)
    td.nstates = min(100, nocc*nvir)
    td.kernel()

    gw = GW(mf, td)
    gw.kernel()
    print(gw.mo_energy)
# [-20.10555946  -1.2264067   -0.68160939  -0.53066326  -0.44679868
#   0.17291986   0.24457082   0.74758227   0.80045129   1.11748735
#   1.1508353    1.19081928   1.40406947   1.43593681   1.63324734
#   1.81839838   1.86943727   2.37827782   2.48829939   3.26028229
#   3.3247595    3.4958492    3.77735135   4.14572189]

    gw.linearized = True
    gw.kernel(orbs=[nocc-1,nocc])
    print(gw.mo_energy[nocc-1] - -0.44684106)
    print(gw.mo_energy[nocc] - 0.17292032)

