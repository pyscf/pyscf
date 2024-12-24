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

'''
UMP2 with spatial integrals
'''


import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf.ao2mo import _ao2mo
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_ump2_with_t2', True)


# This is unrestricted (U)MP2, i.e. spin-orbital form.

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    if with_t2:
        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
    else:
        t2 = None

    emp2_ss = emp2_os = 0.0
    for i in range(nocca):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovov[i]
        else:
            eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

        eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2aa[i] = t2i - t2i.transpose(0,2,1)

        if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.ovOV[i]
        else:
            eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
        eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
        emp2_os += numpy.einsum('JaB,JaB', t2i, eris_ovov)
        if with_t2:
            t2ab[i] = t2i

    for i in range(noccb):
        if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
            # When mf._eri is a custom integrals with the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            eris_ovov = eris.OVOV[i]
        else:
            eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
        eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
        t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
        emp2_ss += numpy.einsum('jab,jab', t2i, eris_ovov) * .5
        emp2_ss -= numpy.einsum('jab,jba', t2i, eris_ovov) * .5
        if with_t2:
            t2bb[i] = t2i - t2i.transpose(0,2,1)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2

def energy(mp, t2, eris):
    '''MP2 energy'''
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_ovov = numpy.asarray(eris.ovov).reshape(nocca,nvira,nocca,nvira)
    eris_OVOV = numpy.asarray(eris.OVOV).reshape(noccb,nvirb,noccb,nvirb)
    eris_ovOV = numpy.asarray(eris.ovOV).reshape(nocca,nvira,noccb,nvirb)
    ess  = 0.25 * numpy.einsum('ijab,iajb->', t2aa, eris_ovov)
    ess -= 0.25 * numpy.einsum('ijab,ibja->', t2aa, eris_ovov)
    ess += 0.25 * numpy.einsum('ijab,iajb->', t2bb, eris_OVOV)
    ess -= 0.25 * numpy.einsum('ijab,ibja->', t2bb, eris_OVOV)
    eos  =        numpy.einsum('iJaB,iaJB->', t2ab, eris_ovOV)
    e    = ess + eos
    if abs(e.imag) > 1e-4:
        logger.warn(mp, 'Non-zero imaginary part found in UMP2 energy %s', e)
    e = lib.tag_array(e.real, e_corr_ss=ess.real, e_corr_os=eos.real)
    return e

def update_amps(mp, t2, eris):
    '''Update non-canonical MP2 amplitudes'''
    #assert (isinstance(eris, _ChemistsERIs))
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mp.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mp.level_shift

    focka, fockb = eris.fock
    fooa = focka[:nocca,:nocca] - numpy.diag(mo_ea_o)
    foob = fockb[:noccb,:noccb] - numpy.diag(mo_eb_o)
    fvva = focka[nocca:,nocca:] - numpy.diag(mo_ea_v)
    fvvb = fockb[noccb:,noccb:] - numpy.diag(mo_eb_v)

    u2aa  = lib.einsum('ijae,be->ijab', t2aa, fvva)
    u2bb  = lib.einsum('ijae,be->ijab', t2bb, fvvb)
    u2ab  = lib.einsum('iJaE,BE->iJaB', t2ab, fvvb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, fvva)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, fooa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, foob)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, foob)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, fooa)

    eris_ovov = numpy.asarray(eris.ovov).reshape(nocca,nvira,nocca,nvira).conj() * .5
    eris_OVOV = numpy.asarray(eris.OVOV).reshape(noccb,nvirb,noccb,nvirb).conj() * .5
    eris_ovOV = numpy.asarray(eris.ovOV).reshape(nocca,nvira,noccb,nvirb).conj().copy()
    u2aa += eris_ovov.transpose(0,2,1,3) - eris_ovov.transpose(0,2,3,1)
    u2bb += eris_OVOV.transpose(0,2,1,3) - eris_OVOV.transpose(0,2,3,1)
    u2ab += eris_ovOV.transpose(0,2,1,3)
    u2aa = u2aa + u2aa.transpose(1,0,3,2)
    u2bb = u2bb + u2bb.transpose(1,0,3,2)

    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    u2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    u2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    return u2aa, u2ab, u2bb


def get_nocc(mp):
    frozen = mp.frozen
    if mp._nocc is not None:
        return mp._nocc
    elif frozen is None:
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0)
    elif isinstance(frozen, (int, numpy.integer)):
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0) - frozen
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0) - frozen
        #assert (nocca > 0 and noccb > 0)
    elif hasattr(mp.frozen, '__len__'):
        occidxa = mp.mo_occ[0] > 0
        occidxb = mp.mo_occ[1] > 0
        if len(frozen) > 0:
            if isinstance(frozen[0], (int, numpy.integer)):
                # The same frozen orbital indices for alpha and beta orbitals
                frozen = [frozen, frozen]
            occidxa[list(frozen[0])] = False
            occidxb[list(frozen[1])] = False
        nocca = numpy.count_nonzero(occidxa)
        noccb = numpy.count_nonzero(occidxb)
    else:
        raise NotImplementedError
    return nocca, noccb

def get_nmo(mp):
    frozen = mp.frozen
    if mp._nmo is not None:
        return mp._nmo
    elif frozen is None:
        nmoa = mp.mo_occ[0].size
        nmob = mp.mo_occ[1].size
    elif isinstance(frozen, (int, numpy.integer)):
        nmoa = mp.mo_occ[0].size - frozen
        nmob = mp.mo_occ[1].size - frozen
    elif hasattr(mp.frozen, '__len__'):
        nmoa = mp.mo_occ[0].size
        nmob = mp.mo_occ[1].size
        if len(frozen) > 0:
            if isinstance(frozen[0], (int, numpy.integer)):
                frozen = (frozen, frozen)
            nmoa -= len(set(frozen[0]))
            nmob -= len(set(frozen[1]))
    else:
        raise NotImplementedError
    return nmoa, nmob


def get_frozen_mask(mp):
    '''Get boolean mask for the unrestricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresponds to the frozen orbital.
    '''
    moidxa = numpy.ones(mp.mo_occ[0].size, dtype=bool)
    moidxb = numpy.ones(mp.mo_occ[1].size, dtype=bool)

    frozen = mp.frozen
    if mp._nmo is not None:
        moidxa[mp._nmo[0]:] = False
        moidxb[mp._nmo[1]:] = False
    elif frozen is None:
        pass
    elif isinstance(frozen, (int, numpy.integer)):
        moidxa[:frozen] = False
        moidxb[:frozen] = False
    elif hasattr(mp.frozen, '__len__'):
        if len(frozen) > 0:
            if isinstance(frozen[0], (int, numpy.integer)):
                frozen = (frozen, frozen)
            moidxa[list(frozen[0])] = False
            moidxb[list(frozen[1])] = False
    else:
        raise NotImplementedError
    return moidxa,moidxb

def make_rdm1(mp, t2=None, ao_repr=False, with_frozen=True):
    r'''
    One-particle spin density matrices dm1a, dm1b in MO basis (the
    occupied-virtual blocks due to the orbital response contribution are not
    included).

    dm1a[p,q] = <q_alpha^\dagger p_alpha>
    dm1b[p,q] = <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    '''
    from pyscf.cc import uccsd_rdm
    if t2 is None: t2 = mp.t2
    doo, dvv = _gamma1_intermediates(mp, t2)
    nocca, noccb, nvira, nvirb = t2[1].shape
    dov = numpy.zeros((nocca,nvira))
    dOV = numpy.zeros((noccb,nvirb))
    d1 = (doo, (dov, dOV), (dov.T, dOV.T), dvv)
    return uccsd_rdm._make_rdm1(mp, d1, with_frozen=with_frozen, ao_repr=ao_repr)

def _gamma1_intermediates(mp, t2):
    t2aa, t2ab, t2bb = t2
    dooa  = lib.einsum('imef,jmef->ij', t2aa.conj(), t2aa) *-.5
    dooa -= lib.einsum('imef,jmef->ij', t2ab.conj(), t2ab)
    doob  = lib.einsum('imef,jmef->ij', t2bb.conj(), t2bb) *-.5
    doob -= lib.einsum('mief,mjef->ij', t2ab.conj(), t2ab)

    dvva  = lib.einsum('mnae,mnbe->ba', t2aa.conj(), t2aa) * .5
    dvva += lib.einsum('mnae,mnbe->ba', t2ab.conj(), t2ab)
    dvvb  = lib.einsum('mnae,mnbe->ba', t2bb.conj(), t2bb) * .5
    dvvb += lib.einsum('mnea,mneb->ba', t2ab.conj(), t2ab)
    return ((dooa, doob), (dvva, dvvb))


def _mo_splitter(mp):
    maskact = mp.get_frozen_mask()
    maskocc = [mp.mo_occ[s]>1e-6 for s in [0,1]]
    masks = []
    for s in [0,1]:
        masks.append([
            maskocc[s]  & ~maskact[s],  # frz occ
            maskocc[s]  &  maskact[s],  # act occ
            ~maskocc[s] &  maskact[s],  # act vir
            ~maskocc[s] & ~maskact[s],  # frz vir
        ])
    return masks

def make_fno(mp, thresh=1e-6, pct_occ=None, nvir_act=None, t2=None, eris=None):
    r'''
    Frozen natural orbitals

    Returns:
        frozen : list or ndarray
            Length-2 list of orbitals to freeze
        no_coeff : ndarray
            Length-2 list of semicanonical NO coefficients in the AO basis
    '''
    mf = mp._scf
    dmab = mp.make_rdm1(t2=t2, with_frozen=False)

    masks = _mo_splitter(mp)

    if nvir_act is not None:
        if isinstance(nvir_act, (int, numpy.integer)):
            nvir_act = [nvir_act]*2

    no_frozen = []
    no_coeff = []
    for s,dm in enumerate(dmab):
        nocc = mp.nocc[s]
        nmo = mp.nmo[s]
        nvir = nmo - nocc
        n,v = numpy.linalg.eigh(dm[nocc:,nocc:])
        idx = numpy.argsort(n)[::-1]
        n,v = n[idx], v[:,idx]
        n *= 2  # to match RHF when using same thresh

        if nvir_act is None:
            if pct_occ is None:
                nvir_keep = numpy.count_nonzero(n>thresh)
            else:
                cumsum = numpy.cumsum(n/numpy.sum(n))
                logger.debug(mp, 'Sum(pct_occ): %s', cumsum)
                nvir_keep = numpy.count_nonzero(
                    [c <= pct_occ or numpy.isclose(c, pct_occ) for c in cumsum])
        else:
            nvir_keep = min(nvir, nvir_act[s])

        moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[s][m] for m in masks[s]]
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[s][:,m] for m in masks[s]]

        fvv = numpy.diag(moevir)
        fvv_no = numpy.dot(v.T, numpy.dot(fvv, v))
        _, v_canon = numpy.linalg.eigh(fvv_no[:nvir_keep,:nvir_keep])

        orbviract = numpy.dot(orbvir, numpy.dot(v[:,:nvir_keep], v_canon))
        orbvirfrz = numpy.dot(orbvir, v[:,nvir_keep:])
        no_comp = (orboccfrz0, orbocc, orbviract, orbvirfrz, orbvirfrz0)
        no_coeff.append(numpy.hstack(no_comp))
        nocc_loc = numpy.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen.append(numpy.hstack((numpy.arange(nocc_loc[0], nocc_loc[1]),
                                       numpy.arange(nocc_loc[3], nocc_loc[5]))).astype(int))

    return no_frozen, no_coeff


# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mp, t2=None, ao_repr=False):
    r'''
    Two-particle spin density matrices dm2aa, dm2ab, dm2bb in MO basis

    dm2aa[p,q,r,s] = <q_alpha^\dagger s_alpha^\dagger r_alpha p_alpha>
    dm2ab[p,q,r,s] = <q_alpha^\dagger s_beta^\dagger r_beta p_alpha>
    dm2bb[p,q,r,s] = <q_beta^\dagger s_beta^\dagger r_beta p_beta>

    (p,q correspond to one particle and r,s correspond to another particle)
    Two-particle density matrix should be contracted to integrals with the
    pattern below to compute energy

    E = numpy.einsum('pqrs,pqrs', eri_aa, dm2_aa)
    E+= numpy.einsum('pqrs,pqrs', eri_ab, dm2_ab)
    E+= numpy.einsum('pqrs,rspq', eri_ba, dm2_ab)
    E+= numpy.einsum('pqrs,pqrs', eri_bb, dm2_bb)

    where eri_aa[p,q,r,s] = (p_alpha q_alpha | r_alpha s_alpha )
    eri_ab[p,q,r,s] = ( p_alpha q_alpha | r_beta s_beta )
    eri_ba[p,q,r,s] = ( p_beta q_beta | r_alpha s_alpha )
    eri_bb[p,q,r,s] = ( p_beta q_beta | r_beta s_beta )
    '''
    if t2 is None: t2 = mp.t2
    nmoa, nmob = nmoa0, nmob0 = mp.nmo
    nocca, noccb = nocca0, noccb0 = mp.nocc
    t2aa, t2ab, t2bb = t2

    if mp.frozen is not None:
        nmoa0 = mp.mo_occ[0].size
        nmob0 = mp.mo_occ[1].size
        nocca0 = numpy.count_nonzero(mp.mo_occ[0] > 0)
        noccb0 = numpy.count_nonzero(mp.mo_occ[1] > 0)
        moidxa, moidxb = mp.get_frozen_mask()
        oidxa = numpy.where(moidxa & (mp.mo_occ[0] > 0))[0]
        vidxa = numpy.where(moidxa & (mp.mo_occ[0] ==0))[0]
        oidxb = numpy.where(moidxb & (mp.mo_occ[1] > 0))[0]
        vidxb = numpy.where(moidxb & (mp.mo_occ[1] ==0))[0]

        dm2aa = numpy.zeros((nmoa0,nmoa0,nmoa0,nmoa0), dtype=t2aa.dtype)
        dm2ab = numpy.zeros((nmoa0,nmoa0,nmob0,nmob0), dtype=t2aa.dtype)
        dm2bb = numpy.zeros((nmob0,nmob0,nmob0,nmob0), dtype=t2aa.dtype)

        tmp = t2aa.transpose(0,2,1,3)
        dm2aa[oidxa[:,None,None,None],vidxa[:,None,None],oidxa[:,None],vidxa] = tmp
        dm2aa[vidxa[:,None,None,None],oidxa[:,None,None],vidxa[:,None],oidxa] = tmp.conj().transpose(1,0,3,2)

        tmp = t2bb.transpose(0,2,1,3)
        dm2bb[oidxb[:,None,None,None],vidxb[:,None,None],oidxb[:,None],vidxb] = tmp
        dm2bb[vidxb[:,None,None,None],oidxb[:,None,None],vidxb[:,None],oidxb] = tmp.conj().transpose(1,0,3,2)

        dm2ab[oidxa[:,None,None,None],vidxa[:,None,None],oidxb[:,None],vidxb] = t2ab.transpose(0,2,1,3)
        dm2ab[vidxa[:,None,None,None],oidxa[:,None,None],vidxb[:,None],oidxb] = t2ab.conj().transpose(2,0,3,1)

    else:
        dm2aa = numpy.zeros((nmoa0,nmoa0,nmoa0,nmoa0), dtype=t2aa.dtype)
        dm2ab = numpy.zeros((nmoa0,nmoa0,nmob0,nmob0), dtype=t2aa.dtype)
        dm2bb = numpy.zeros((nmob0,nmob0,nmob0,nmob0), dtype=t2aa.dtype)

#:tmp = (t2aa.transpose(0,2,1,3) - t2aa.transpose(0,3,1,2)) * .5
#: t2aa.transpose(0,2,1,3) == -t2aa.transpose(0,3,1,2)
        tmp = t2aa.transpose(0,2,1,3)
        dm2aa[:nocca0,nocca0:,:nocca0,nocca0:] = tmp
        dm2aa[nocca0:,:nocca0,nocca0:,:nocca0] = tmp.conj().transpose(1,0,3,2)

        tmp = t2bb.transpose(0,2,1,3)
        dm2bb[:noccb0,noccb0:,:noccb0,noccb0:] = tmp
        dm2bb[noccb0:,:noccb0,noccb0:,:noccb0] = tmp.conj().transpose(1,0,3,2)

        dm2ab[:nocca0,nocca0:,:noccb0,noccb0:] = t2ab.transpose(0,2,1,3)
        dm2ab[nocca0:,:nocca0,noccb0:,:noccb0] = t2ab.transpose(2,0,3,1).conj()

    dm1a, dm1b = make_rdm1(mp, t2)
    dm1a[numpy.diag_indices(nocca0)] -= 1
    dm1b[numpy.diag_indices(noccb0)] -= 1

    for i in range(nocca0):
        dm2aa[i,i,:,:] += dm1a.T
        dm2aa[:,:,i,i] += dm1a.T
        dm2aa[:,i,i,:] -= dm1a.T
        dm2aa[i,:,:,i] -= dm1a
        dm2ab[i,i,:,:] += dm1b.T
    for i in range(noccb0):
        dm2bb[i,i,:,:] += dm1b.T
        dm2bb[:,:,i,i] += dm1b.T
        dm2bb[:,i,i,:] -= dm1b.T
        dm2bb[i,:,:,i] -= dm1b
        dm2ab[:,:,i,i] += dm1a.T

    for i in range(nocca0):
        for j in range(nocca0):
            dm2aa[i,i,j,j] += 1
            dm2aa[i,j,j,i] -= 1
    for i in range(noccb0):
        for j in range(noccb0):
            dm2bb[i,i,j,j] += 1
            dm2bb[i,j,j,i] -= 1
    for i in range(nocca0):
        for j in range(noccb0):
            dm2ab[i,i,j,j] += 1

    if ao_repr:
        from pyscf.cc import ccsd_rdm
        from pyscf.cc import uccsd_rdm
        dm2aa = ccsd_rdm._rdm2_mo2ao(dm2aa, mp.mo_coeff[0])
        dm2bb = ccsd_rdm._rdm2_mo2ao(dm2bb, mp.mo_coeff[1])
        dm2ab = uccsd_rdm._dm2ab_mo2ao(dm2ab, mp.mo_coeff[0], mp.mo_coeff[1])
    return dm2aa, dm2ab, dm2bb


class UMP2(mp2.MP2):

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = make_rdm1
    make_fno = make_fno
    make_rdm2 = make_rdm2

    def nuc_grad_method(self):
        from pyscf.grad import ump2
        return ump2.Gradients(self)

    # For non-canonical MP2
    energy = energy
    update_amps = update_amps
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    to_gpu = lib.to_gpu

MP2 = UMP2

from pyscf import scf
scf.uhf.UHF.MP2 = lib.class_as_method(MP2)


#TODO: Merge this _ChemistsERIs class with uccsd._ChemistsERIs class
class _ChemistsERIs(mp2._ChemistsERIs):
    def __init__(self, mol=None):
        mp2._ChemistsERIs.__init__(self, mol)
        self.OVOV = None
        self.ovOV = None

    def _common_init_(self, mp, mo_coeff=None):
        self.mol = mp.mol
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')

        mo_idx = mp.get_frozen_mask()
        mo_a = mo_coeff[0][:,mo_idx[0]]
        mo_b = mo_coeff[1][:,mo_idx[1]]
        self.mo_coeff = (mo_a, mo_b)

        if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            self.mo_energy = (mp._scf.mo_energy[0][mo_idx[0]],
                              mp._scf.mo_energy[1][mo_idx[1]])
            self.fock = (numpy.diag(self.mo_energy[0]),
                         numpy.diag(self.mo_energy[1]))
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            focka = mo_a.conj().T.dot(fockao[0]).dot(mo_a)
            fockb = mo_b.conj().T.dot(fockao[1]).dot(mo_b)
            self.fock = (focka, fockb)
            nocca, noccb = self.nocc = mp.nocc
            self.mo_energy = (focka.diagonal().real, fockb.diagonal().real)
        return self

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nao = eris.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (nao_pair**2 + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory-mem_now)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]

    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore+mem_now < mp.max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((orboa,orbva,orboa,orbva)).reshape(nocca*nvira,nocca*nvira)
            eris.ovOV = ao2mofn((orboa,orbva,orbob,orbvb)).reshape(nocca*nvira,noccb*nvirb)
            eris.OVOV = ao2mofn((orbob,orbvb,orbob,orbvb)).reshape(noccb*nvirb,noccb*nvirb)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (orboa,orbva,orboa,orbva))
            eris.ovOV = ao2mo.general(mp._scf._eri, (orboa,orbva,orbob,orbvb))
            eris.OVOV = ao2mo.general(mp._scf._eri, (orbob,orbvb,orbob,orbvb))

    elif getattr(mp._scf, 'with_df', None):
        logger.warn(mp, 'UMP2 detected DF being used in the HF object. '
                    'MO integrals are computed based on the DF 3-index tensors.\n'
                    'It\'s recommended to use DF-UMP2 module.')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((orboa,orbva,orboa,orbva))
        eris.ovOV = mp._scf.with_df.ao2mo((orboa,orbva,orbob,orbvb))
        eris.OVOV = mp._scf.with_df.ao2mo((orbob,orbvb,orbob,orbvb))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        _ao2mo_ovov(mp, (orboa,orbva,orbob,orbvb), eris.feri,
                    max(2000, max_memory), log)
        if nocca*nvira > 0:
            eris.ovov = eris.feri['ovov']
        else:
            eris.ovov = numpy.zeros((nocca*nvira,nocca*nvira))
        if nocca*nvira*noccb*nvirb > 0:
            eris.ovOV = eris.feri['ovOV']
        else:
            eris.ovOV = numpy.zeros((nocca*nvira,noccb*nvirb))
        if noccb*nvirb > 0:
            eris.OVOV = eris.feri['OVOV']
        else:
            eris.OVOV = numpy.zeros((noccb*nvirb,noccb*nvirb))

    log.timer('Integral transformation', *time0)
    return eris

def _ao2mo_ovov(mp, orbs, feri, max_memory=2000, verbose=None):
    from pyscf.scf.uhf import UHF
    assert isinstance(mp._scf, UHF)
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    orboa = numpy.asarray(orbs[0], order='F')
    orbva = numpy.asarray(orbs[1], order='F')
    orbob = numpy.asarray(orbs[2], order='F')
    orbvb = numpy.asarray(orbs[3], order='F')
    nao, nocca = orboa.shape
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nbas = mol.nbas
    assert (nvira <= nao)
    assert (nvirb <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocca)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    disk = (nocca**2*(nao*(nao+dmax)/2+nvira**2) +
            noccb**2*(nao*(nao+dmax)/2+nvirb**2) +
            nocca*noccb*(nao**2+nvira*nvirb))
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, disk*8/1e6)

    fint = gto.moleintor.getints4c
    aa_blk_slices = []
    ab_blk_slices = []
    count_ab = 0
    count_aa = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ish0, ish1, ni in sh_ranges:
            for jsh0, jsh1, nj in sh_ranges:
                i0, i1 = int(ao_loc[ish0]), int(ao_loc[ish1])
                j0, j1 = int(ao_loc[jsh0]), int(ao_loc[jsh1])

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = lib.ddot(orboa.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                tmp_li = lib.ddot(orbob.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                tmp_li = tmp_li.reshape(noccb,nocca,(i1-i0),(j1-j0))
                save('ab/%d'%count_ab, tmp_li.transpose(1,0,2,3))
                ab_blk_slices.append((i0,i1,j0,j1))
                count_ab += 1

                if ish0 >= jsh0:
                    tmp_li = lib.ddot(orboa.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(nocca,nocca,(i1-i0),(j1-j0))
                    save('aa/%d'%count_aa, tmp_li.transpose(1,0,2,3))

                    tmp_i = lib.ddot(orbob.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                    tmp_li = lib.ddot(orbob.T, tmp_i.reshape(noccb*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(noccb,noccb,(i1-i0),(j1-j0))
                    save('bb/%d'%count_aa, tmp_li.transpose(1,0,2,3))
                    aa_blk_slices.append((i0,i1,j0,j1))
                    count_aa += 1

                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = None

    if nocca*nvira > 0:
        fovov = feri.create_dataset('ovov', (nocca*nvira,nocca*nvira), 'f8',
                                    chunks=(nvira,nvira))
    if nocca*nvira*noccb*nvirb > 0:
        fovOV = feri.create_dataset('ovOV', (nocca*nvira,noccb*nvirb), 'f8',
                                    chunks=(nvira,nvirb))
    if noccb*nvirb > 0:
        fOVOV = feri.create_dataset('OVOV', (noccb*nvirb,noccb*nvirb), 'f8',
                                    chunks=(nvirb,nvirb))
    occblk = int(min(max(nocca,noccb),
                     max(4, 250/max(1,nocca), max_memory*.9e6/8/(nao**2*max(1,nocca))/5)))

    def load_aa(h5g, nocc, i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(aa_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(h5g[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def load_ab(h5g, nocca, i0, eri):
        if i0 < nocca:
            i1 = min(i0+occblk, nocca)
            for k, (p0,p1,q0,q1) in enumerate(ab_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]

    def save(h5dat, nvir, i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,-1)

    with lib.call_in_background(save) as bsave:
        with lib.call_in_background(load_aa) as prefetch:
            if nocca*nvira > 0:
                buf_prefecth = numpy.empty((occblk,nocca,nao,nao))
                buf = numpy.empty_like(buf_prefecth)
                load_aa(ftmp['aa'], nocca, 0, buf_prefecth)
                for i0, i1 in lib.prange(0, nocca, occblk):
                    buf, buf_prefecth = buf_prefecth, buf
                    prefetch(ftmp['aa'], nocca, i1, buf_prefecth)
                    eri = buf[:i1-i0].reshape((i1-i0)*nocca,nao,nao)
                    dat = _ao2mo.nr_e2(eri, orbva, (0,nvira,0,nvira), 's1', 's1')
                    bsave(fovov, nvira, i0, i1,
                          dat.reshape(i1-i0,nocca,nvira,nvira).transpose(0,2,1,3))
                    time1 = log.timer_debug1('pass2 ao2mo for aa [%d:%d]' % (i0,i1), *time1)

            if noccb*nvirb > 0:
                buf_prefecth = numpy.empty((occblk,noccb,nao,nao))
                buf = numpy.empty_like(buf_prefecth)
                load_aa(ftmp['bb'], noccb, 0, buf_prefecth)
                for i0, i1 in lib.prange(0, noccb, occblk):
                    buf, buf_prefecth = buf_prefecth, buf
                    prefetch(ftmp['bb'], noccb, i1, buf_prefecth)
                    eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                    dat = _ao2mo.nr_e2(eri, orbvb, (0,nvirb,0,nvirb), 's1', 's1')
                    bsave(fOVOV, nvirb, i0, i1,
                          dat.reshape(i1-i0,noccb,nvirb,nvirb).transpose(0,2,1,3))
                    time1 = log.timer_debug1('pass2 ao2mo for bb [%d:%d]' % (i0,i1), *time1)

        if nocca*nvira*noccb*nvirb > 0:
            orbvab = numpy.asarray(numpy.hstack((orbva, orbvb)), order='F')
            with lib.call_in_background(load_ab) as prefetch:
                load_ab(ftmp['ab'], nocca, 0, buf_prefecth)
                for i0, i1 in lib.prange(0, nocca, occblk):
                    buf, buf_prefecth = buf_prefecth, buf
                    prefetch(ftmp['ab'], nocca, i1, buf_prefecth)
                    eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                    dat = _ao2mo.nr_e2(eri, orbvab, (0,nvira,nvira,nvira+nvirb), 's1', 's1')
                    bsave(fovOV, nvira, i0, i1,
                          dat.reshape(i1-i0,noccb,nvira,nvirb).transpose(0,2,1,3))
                    time1 = log.timer_debug1('pass2 ao2mo for ab [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)

del (WITH_T2)


if __name__ == '__main__':
    from functools import reduce
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    frozen = [[0,1],[0,1]]
    pt = UMP2(mf, frozen=frozen)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    pt.max_memory = 1
    emp2, t2 = pt.kernel()
    print(emp2 - -0.345306881488508)

    dm1a,dm1b = pt.make_rdm1()
    dm2aa,dm2ab,dm2bb = pt.make_rdm2()
    mo_a = mf.mo_coeff[0]
    mo_b = mf.mo_coeff[1]
    nmoa = mo_a.shape[1]
    nmob = mo_b.shape[1]
    eriaa = ao2mo.kernel(mf._eri, mo_a, compact=False).reshape([nmoa]*4)
    eribb = ao2mo.kernel(mf._eri, mo_b, compact=False).reshape([nmob]*4)
    eriab = ao2mo.kernel(mf._eri, (mo_a,mo_a,mo_b,mo_b), compact=False)
    eriab = eriab.reshape([nmoa,nmoa,nmob,nmob])
    hcore = mf.get_hcore()
    h1a = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1b = reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1a, dm1a)
    e1+= numpy.einsum('ij,ji', h1b, dm1b)
    e1+= numpy.einsum('ijkl,ijkl', eriaa, dm2aa) * .5
    e1+= numpy.einsum('ijkl,ijkl', eriab, dm2ab)
    e1+= numpy.einsum('ijkl,ijkl', eribb, dm2bb) * .5
    e1+= mol.energy_nuc()
    print(e1 - pt.e_tot)

    pt = UMP2(scf.density_fit(mf, 'weigend'))
    print(pt.kernel()[0] - -0.3503781525098727)

    mf = scf.UHF(mol).run(max_cycle=1)
    pt = UMP2(mf)
    print(pt.kernel()[0] - -0.117601521171095)
