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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory for
unrestricted references
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.scf import _vhf
from pyscf.agf2 import aux, mpi_helper
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.mp.ump2 import get_nocc, get_nmo, get_frozen_mask
from pyscf.mp.mp2 import _mo_without_core, _mo_energy_without_core
from pyscf.agf2.ragf2 import kernel, _cholesky_build

BLKMIN = getattr(__config__, 'agf2_uagf2_blkmin', 1)


def build_se_part(agf2, eri, gf_occ, gf_vir):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped,
        for a single spin.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : tuple of GreensFunction
            Occupied Green's function for each spin
        gf_vir : tuple of GreensFunction
            Virtual Green's function for each spin

    Returns:
        :class:`SelfEnergy`
    '''
    #TODO: C code

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_occ[1]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction
    assert type(gf_vir[1]) is aux.GreensFunction
    assert type(eri) is _ChemistsERIs

    tol = agf2.weight_tol  #NOTE: tol is unlikely to be met at (None,0)

    eja_a = lib.direct_sum('j,a->ja', gf_occ[0].energy, gf_vir[0].energy)
    eja_a = eja_a.ravel()
    eja_b = lib.direct_sum('j,a->ja', gf_occ[1].energy, gf_vir[1].energy)
    eja_b = eja_b.ravel()
    eja = (eja_a, eja_b)

    def _build_se_part_spin(spin=0):
        ''' Perform the build for a single spin
        
        spin = 0: alpha
        spin = 1: beta
        '''

        a,b = 0,1 if spin == 0 else 1,0
        ab = slice(None) if spin == 0 else slice(None, None, -1)

        nmo = agf2.nmo[a]
        noa, nob = gf_occ[a].naux, gf_occ[b].naux
        nva, nvb = gf_vir[a].naux, gf_vir[b].naux

        vv = np.zeros((nmo, nmo))
        vev = np.zeros((nmo, nmo))

        mem_incore = (nmo*nnoa*(noa*nva+nob*nvb)) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (mem_incore+mem_now < agf2.max_memory):
            eri_qmo = _make_qmo_eris_incore(agf2, eri, gf_occ[ab], 
                                            gf_vir[ab], spin=spin)
        else:
            eri_qmo = _make_qmo_eris_outcore(agf2, eri, gf_occ[ab], 
                                             gf_vir[ab], spin=spin)

        eri_qmo_aa, eri_qmo_ab = eri_qmo

        for i in range(noa):
            xija_aa = eri_qmo_aa[:,i].reshape(nmo, -1)
            xija_ab = eri_qmo_ab[:,i].reshape(nmo, -1)
            xjia_aa = eri_qmo_aa[:,:,i].reshape(nmo, -1)

            eija_aa = eja[a] + gf_occ[a].energy[i]
            eija_ab = eja[b] + gf_occ[a].energy[i]

            vv = lib.dot(xija_aa, xija_aa.T, alpha=1, beta=1, c=vv)
            vv = lib.dot(xija_aa, xjia_aa.T, alpha=-1, beta=1, c=vv)
            vv = lib.dot(xija_ab, xjia_ab.T, alpha=1, beta=1, c=vv)

            exija_aa = xija_aa * eija_aa[None]
            exija_ab = xija_ab * eija_ab[None]

            vev = lib.dot(exija_aa, xija_aa.T, alpha=1, beta=1, c=vev)
            vev = lib.dot(exija_aa, xjia_aa.T, alpha=-1, beta=1, c=vev)
            vev = lib.dot(exija_ab, xjia_ab.T, alpha=1, beta=1, c=vev)

        e, c = _cholesky_build(vv, vev, gf_occ, gf_vir) #FIXME: this won't work for UAGF2!!!
        se = aux.SelfEnergy(e, c, chempot=gf_occ[0].chempot)
        se.remove_uncoupled(tol=tol)
        
        return se

    se_a = _build_se_part_spin(0)

    cput0 = log.timer_debug1('se part (alpha)', *cput0)

    se_b = _build_se_part_spin(1)

    cput0 = log.timer_debug1('se part (beta)', *cput0)

    return (se_a, se_b)


class _ChemistsERIs:
    ''' (pq|rs)

    MO integrals stored in s4 symmetry, we only need QMO integrals
    in low-symmetry tensors and s4 is highest supported by _vhf
    '''

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        mo_coeff = (_mo_without_core(agf2, mo_coeff[0]), 
                    _mo_without_core(agf2, mo_coeff[1]))
        self.mo_coeff = mo_coeff

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        h1e_ao = agf2._scf.get_hcore()
        vhf = agf2._scf.get_veff(agf2.mol, dm)
        fock_ao = agf2.get_fock(vhf=vhf, dm=dm)

        self.h1e = (np.dot(np.dot(mo_coeff[0].conj().T, h1e_ao), mo_coeff[0]),
                    np.dot(np.dot(mo_coeff[1].conj().T, h1e_ao), mo_coeff[1]))
        self.fock = (np.dot(np.dot(mo_coeff[0].conj().T, fock_ao), mo_coeff[0]),
                     np.dot(np.dot(mo_coeff[1].conj().T, fock_ao), mo_coeff[1]))

        self.e_hf = agf2._scf.e_tot

        self.nocc = agf2.nocc
        self.mol = agf2.mol

        mo_e = (self.fock[0].diagonal(), self.fock[1].diagonal())
        gap_a = abs(mo_e[0][:self.nocc,None] - mo_e[0][None,self.nocc:]).min()
        gap_b = abs(mo_e[1][:self.nocc,None] - mo_e[1][None,self.nocc:]).min()
        gap = min(gap_a, gap_b)
        if gap < 1e-5:
            logger.warn(agf2, 'HOMO-LUMO gap %s too small for UAGF2', gap)

        return self

def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    moa, mob = eris.mo_coeff
    nmoa, nmob = moa.shape[1], mob.shape[1]

    eri_aa = ao2mo.incore.full(agf2._scf._eri, moa, verbose=log)
    eri_bb = ao2mo.incore.full(agf2._scf._eri, mob, verbose=log)

    eri_aa = ao2mo.addons.restore('s4', eri_aa, nmoa)
    eri_bb = ao2mo.addons.restore('s4', eri_bb, nmob)
    
    eri_ab = ao2mo.incore.general(agf2._scf._eri, (moa,moa,mob,mob), verbose=log)
    assert eri_ab.shape == (nmoa*(nmob+1)//2, nmob*(nmob+1)//2)
    eri_ba = np.transpose(eri_ab)

    eris.eri_aa = eri_aa
    eris.eri_ab = eri_ab
    eris.eri_ba = eri_ba
    eris.eri_bb = eri_bb
    eris.eri = ((eri_aa, eri_ab), (eri_ba, eri_bb))

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_mo_eris_outcore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''
    #TODO: check all of these are s4 symmetry
    #NOTE: can we just do a bit-by-bit transpose?

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    mol = agf2.mol
    moa = np.asarary(eris.mo_coeff[0], order='F')
    mob = np.asarray(eris.mo_coeff[1], order='F')
    nao, nmoa = moa.shape
    nao, nmob = mob.shape

    eris.feri = lib.H5TmpFile()

    ao2mo.outcore.full(mol, moa, eris.feri, dataname='mo/aa')
    ao2mo.outcore.full(mol, mob, eris.feri, dataname='mo/bb')
    ao2mo.outcore.general(mol, (moa,moa,mob,mob), eris.feri, dataname='mo/ab', verbose=log)
    ao2mo.outcore.general(mol, (mob,mob,moa,moa), eris.feri, dataname='mo/ba', verbose=log)

    eris.eri_aa = eris.feri['mo/aa']
    eris.eri_ab = eris.feri['mo/ab']
    eris.eri_ba = eris.feri['mo/ba']
    eris.eri_bb = eris.feri['mo/bb']

    eris.eri = ((eri_aa, eri_ab), (eri_ba, eri_bb))

    return eris

def _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir, spin=None):
    ''' Returns nested tuple of ndarray

    spin = None: ((aaaa, aabb), (bbaa, bbbb))
    spin = 0: (aaaa, aabb)
    spin = 1: (bbbb, bbaa)
    '''
    #TODO: improve efficiency by storing half-transformed intermediates

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmoa, nmob = agf2.nmo
    npaira, npairb = nmo*(nmo+1)//2, nmob*(nmob+1)//2
    cia = cja = gf_occ[0].coupling
    cib = cjb = gf_occ[1].coupling
    caa, cab = gf_vir[0].coupling, gf_vir[1].coupling
    nia = nja = gf_occ[0].naux
    nib = njb = gf_occ[1].naux
    naa, nab = gf_vir[0].naux, gf_vir[1].naux

    if spin is None or spin == 0:
        c_aa = (np.eye(nmoa), cia, cja, caa)
        c_ab = (np.eye(nmoa), cia, cjb, cab)

        qeri_aa = ao2mo.incore.general(eri.eri_aa, c_aa, compact=False, verbose=log)
        qeri_ab = ao2mo.incore.general(eri.eri_ab, c_ab, compact=False, verbose=log)

        qeri_aa = qeri_aa.reshape(nmoa, nia, nja, naa)
        qeri_ab = qeri_ab.reshape(nmoa, nia, njb, nab)

    if spin is None or spin == 1:
        c_bb = (np.eye(nmob), cib, cjb, cab)
        c_ba = (np.eye(nmob), cib, cja, caa)

        qeri_bb = ao2mo.incore.general(eri.eri_bb, c_bb, compact=False, verbose=log)
        qeri_ba = ao2mo.incore.general(eri.eri_ba, c_ba, compact=False, verbose=log)

        qeri_bb = qeri_bb.reshape(nmob, nib, njb, nab)
        qeri_ba = qeri_ba.reshape(nmob, nib, nja, naa)

    if spin is None:
        qeri = ((qeri_aa, qeri_ab), (qeri_ba, qeri_bb))
    elif spin == 0:
        qeri = (qeri_aa, qeri_ab)
    elif spin == 1:
        qeri = (qeri_bb, qeri_ba)

    log.timer_debug1('QMO integral transformation', *cput0)

    return qeri

def _make_qmo_eris_outcore(agf2, eri, gf_occ, gf_vir):
    ''' Returns nested tuple of H5 dataset

    spin = None: ((aaaa, aabb), (bbaa, bbbb))
    spin = 0: (aaaa, aabb)
    spin = 1: (bbbb, bbaa)
    '''
    #TODO: improve efficiency and check blksize

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmoa, nmob = agf2.nmo
    npaira, npairb = nmo*(nmo+1)//2, nmob*(nmob+1)//2
    cia = cja = gf_occ[0].coupling
    cib = cjb = gf_occ[1].coupling
    caa, cab = gf_vir[0].coupling, gf_vir[1].coupling
    nia = nja = gf_occ[0].naux
    nib = njb = gf_occ[1].naux
    naa, nab = gf_vir[0].naux, gf_vir[1].naux

    # possible to have incore MO, outcore QMO
    if getattr(eri, 'feri', None) is None:
        eri.feri = lib.H5TmpFile()
    else:
        for key in ['aa', 'ab', 'ba', 'bb']:
            if 'qmo/%s'%key in eri.feri:
                del eri.feri['qmo/%s'%key]

    if spin is None or spin == 0:
        eri.feri.create_dataset('qmo/aa', (nmoa, nia, nja, naa), 'f8')
        eri.feri.create_dataset('qmo/ab', (nmoa, nia, njb, nab), 'f8')

        max_memory = agf2.max_memory - lib.current_memory()[0]
        blksize = int((max_memory/8e-6) / max(nmoa**3+nmoa*nja*naa, 
                                              nmoa*nmob**2*njb*nab))
        blksize = min(nmoa, max(BLKMIN, blksize))
        log.debug1('blksize = %d', blksize)

        tril2sq = lib.square_mat_in_trilu_indices(nmoa)
        for p0, p1 in lib.prange(0, nmoa, blksize):
            idx = np.concatenate(tril2sq[p0:p1])

            # aa
            buf = eri.eri_aa[idx] # (blk, nmoa, npaira)
            buf = buf.reshape((p1-p0)*nmoa, -1) # (blk*nmoa, npaira)

            jasym_aa, nja_aa, cja_aa, sja_aa = ao2mo.incore._conc_mos(cja, caa)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_aa, sja_aa, 's2kl', 's1')
            buf = buf.reshape(p1-p0, nmoa, nja, naa)

            buf = lib.einsum('xpja,pi->xija', buf, cia)
            eri.feri['qmo/aa'][p0:p1] = np.asarray(buf, order='C')

            # ab
            buf = eri.eri_ab[idx] # (blk, nmoa, npairb)
            buf = buf.reshape((p1-p0)*nmob, -1) # (blk*nmoa, npairb)

            jasym_ab, nja_ab, cja_ab, sja_ab = ao2mo.incore._conc_mos(cjb, cab)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_ab, sja_ab, 's2kl', 's1')
            buf = buf.reshape(p1-p0, nmoa, njb, nab)

            buf = lib.einsum('xpja,pi->xija', buf, cia)
            eri.feri['qmo/ab'][p0:p1] = np.asarray(buf, order='C')

    if spin is None or spin == 1:
        eri.feri.create_dataset('qmo/ba', (nmob, nib, nja, naa), 'f8')
        eri.feri.create_dataset('qmo/bb', (nmob, nib, njb, nab), 'f8')

        max_memory = agf2.max_memory - lib.current_memory()[0]
        blksize = int((max_memory/8e-6) / max(nmob**3+nmob*njb*nab, 
                                              nmob*nmoa**2*nja*naa))
        blksize = min(nmob, max(BLKMIN, blksize))
        log.debug1('blksize = %d', blksize)

        tril2sq = lib.square_mat_in_trilu_indices(nmob)
        for p0, p1 in lib.prange(0, nmob, blksize):
            idx = np.concatenate(tril2sq[p0:p1])

            # ba
            buf = eri.eri_ba[idx] # (blk, nmob, npaira)
            buf = buf.reshape((p1-p0)*nmob, -1) # (blk*nmob, npaira)

            jasym_ba, nja_ba, cja_ba, sja_ba = ao2mo.incore._conc_mos(cja, caa)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_ba, sja_ba, 's2kl', 's1')
            buf = buf.reshape(p1-p0, nmob, njb, nab)

            buf = lib.einsum('xpja,pi->xija', buf, cib)
            eri.feri['qmo/ba'][p0:p1] = np.asarray(buf, order='C')

            # bb
            buf = eri.eri_bb[idx] # (blk, nmob, npairb)
            buf = buf.reshape((p1-p0)*nmob, -1) # (blk*nmob, npairb)

            jasym_bb, nja_bb, cja_bb, sja_bb = ao2mo.incore._conc_mos(cjb, cab)
            buf = ao2mo._ao2mo.nr_e2(buf, cja_bb, sja_bb, 's2kl', 's1')
            buf = buf.reshape(p1-p0, nmob, njb, nab)

            buf = lib.einsum('xpja,pi->xija', buf, cib)
            eri.feri['qmo/bb'][p0:p1] = np.asarray(buf, order='C')

    if spin is None:
        qeri = ((eri.feri['qmo/aa'], eri.feri['qmo/ab']), 
                (eri.feri['qmo/ba'], eri.feri['qmo/bb']))
    elif spin == 0:
        qeri = (eri.feri['qmo/aa'], eri.feri['qmo/ab'])
    elif spin == 1:
        qeri = (eri.feri['qmo/bb'], eri.feri['qmo/ba'])

    log.timer_debug1('QMO integral transformation', *cput0)

    return qeri


