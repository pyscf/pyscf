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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Qiming Sun <osirpt.sun@gmail.com>
#

'''
G0W0 approximation
'''

import time
from functools import reduce
import numpy
import numpy as np
from scipy.optimize import newton

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import dft
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core
from pyscf import __config__

einsum = lib.einsum

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

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
    assert(gw.frozen == 0 or gw.frozen is None)

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
    vk_ov = -np.einsum('iqpi->pq', eris.ovoo)
    vk_vv = -np.einsum('ipqi->pq', eris.ovvo).conj()
    vk = np.array(np.bmat([[vk_oo, vk_ov],[vk_ov.T, vk_vv]]))

    nexc = len(td_e)
    # factor of 2 for normalization, see tdscf/rhf.py
    td_xy = 2*np.asarray(td_xy) # (nexc, 2, nocc, nvir)
    td_z = np.sum(td_xy, axis=1).reshape(nexc,nocc,nvir)
    tdm_oo = einsum('via,iapq->vpq', td_z, eris.ovoo)
    tdm_ov = einsum('via,iapq->vpq', td_z, eris.ovov)
    tdm_vv = einsum('via,iapq->vpq', td_z, eris.ovvv)
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

    def __init__(self, mf, tdmf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self._tdscf = tdmf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ

        keys = set(('eta', 'linearized'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
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

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        self._tdscf.reset(mol)
        return self

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'GW detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'Developer TODO:  Write dfgw.GW for the '
                        'DF-GW calculations')
            raise NotImplementedError
            #return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)


class _ChemistsERIs:
    '''(pq|rs)

    Identical to rccsd _ChemistsERIs except no vvvv.'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.oooo = None
        self.ovoo = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)
# Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.fock.diagonal()
        try:
            gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for GW', gap)
        except ValueError:  # gap.size == 0
            pass
        return self

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    if callable(ao2mofn):
        eri1 = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    logger.timer(mycc, 'GW integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])

    ftmp = lib.H5TmpFile()
    ao2mo.full(mol, mo_coeff, ftmp, max_memory=max_memory, verbose=log)
    eri = ftmp['eri_mo']

    nocc_pair = nocc*(nocc+1)//2
    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    oo = eri[:nocc_pair]
    eris.oooo[:] = ao2mo.restore(1, oo[:,:nocc_pair], nocc)
    oovv = lib.take_2d(oo, tril2sq[:nocc,:nocc].ravel(), tril2sq[nocc:,nocc:].ravel())
    eris.oovv[:] = oovv.reshape(nocc,nocc,nvir,nvir)
    oo = oovv = None

    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    blksize = min(nvir, max(BLKMIN, int(max_memory*1e6/8/nmo**3/2)))
    for p0, p1 in lib.prange(0, nvir, blksize):
        q0, q1 = p0+nocc, p1+nocc
        off0 = q0*(q0+1)//2
        off1 = q1*(q1+1)//2
        buf = lib.unpack_tril(eri[off0:off1])

        tmp = buf[ tril2sq[q0:q1,:nocc] - off0 ]
        eris.ovoo[:,p0:p1] = tmp[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = tmp[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = tmp[:,:,:nocc,nocc:].transpose(1,0,2,3)
        eris.ovvv[:,p0:p1] = tmp[:,:,nocc:,nocc:].transpose(1,0,2,3)

        buf = tmp = None
    log.timer('GW integral transformation', *cput0)
    return eris


if __name__ == '__main__':
    from pyscf import gto, tddft
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

