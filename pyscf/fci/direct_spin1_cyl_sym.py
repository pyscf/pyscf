#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
Cylindrical symmetry

This module is much slower than direct_spin1_symm.

In this implementation, complex orbitals is used to construct the Hamiltonian.
FCI wavefunction (called complex wavefunction here) is solved using the complex
Hamiltonian. For 2D irreps, the real part and the imaginary part of the complex
FCI wavefunction are identical to the Ex and Ey wavefunction obtained from
direct_spin1_symm module. However, any observables from the complex FCI
wavefunction should have an indentical one from either Ex or Ey wavefunction
of direct_spin1_symm.
'''

import functools
import ctypes
import numpy
import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import symm
from pyscf.scf.hf_symm import map_degeneracy
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import direct_spin1_symm
from pyscf.fci.direct_spin1_symm import (_sv_associated_det,
                                         _strs_angular_momentum,
                                         _cyl_sym_orbital_rotation)
from pyscf.fci import direct_nosym
from pyscf.fci import addons
from pyscf import __config__

libfci = direct_spin1.libfci

def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    if orbsym is None:
        return direct_nosym.contract_2e(eri, fcivec, norb, nelec, link_index)

    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    link_indexa, link_indexb = direct_nosym._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]

    wfn_momentum = symm.basis.linearmole_irrep2momentum(wfnsym)
    wfnsym_in_d2h = wfnsym % 10
    wfn_ungerade = wfnsym_in_d2h >= 4
    orbsym_d2h = orbsym % 10
    orb_ungerade = orbsym_d2h >= 4
    if np.any(orb_ungerade) or wfn_ungerade:
        max_gerades = 2
    else:
        max_gerades = 1

    orb_l = _get_orb_l(orbsym)
    max_eri_l = abs(orb_l).max() * 2

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    strsa_l = _strs_angular_momentum(strsa, orbsym)
    max_stra_l = strsa_l.max()
    if neleca == nelecb:
        strsb_l = strsa_l
        max_strb_l = max_stra_l
    else:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        strsb_l = _strs_angular_momentum(strsb, orbsym)
        max_strb_l = strsb_l.max()
    max_momentum = max(max_stra_l, max_strb_l, max_eri_l)

    eri, rank_eri, irrep_eri = reorder_eri(eri, norb, orbsym, max_momentum,
                                           max_gerades)
    eri_ir_dims = eri.ir_dims
    aidx, link_indexa = gen_str_irrep(strsa, orbsym, link_indexa, rank_eri,
                                      irrep_eri, max_momentum, max_gerades)
    nas = nbs = np.array([x.size for x in aidx], dtype=np.int32)
    if neleca == nelecb:
        bidx, link_indexb = aidx, link_indexa
    else:
        bidx, link_indexb = gen_str_irrep(strsb, orbsym, link_indexb, rank_eri,
                                          irrep_eri, max_momentum, max_gerades)
        nbs = np.array([x.size for x in bidx], dtype=np.int32)

    nirreps = (max_momentum * 2 + 1) * max_gerades
    ug_offsets = max_momentum * 2 + 1
    ab_idx = [np.zeros(0, dtype=int)] * nirreps
    for ag in range(max_gerades):
        bg = wfn_ungerade ^ ag
        # abs(al) < max_stra_l and abs(bl := wfn_momentum-al) < max_strb_l
        for al in range(max(-max_stra_l, wfn_momentum-max_strb_l),
                        min( max_stra_l, wfn_momentum+max_strb_l)+1):
            bl = wfn_momentum - al
            stra_ir = al + max_momentum + ag * ug_offsets
            strb_ir = bl + max_momentum + bg * ug_offsets
            ab_idx[stra_ir] = (aidx[stra_ir][:,None] * nb + bidx[strb_ir]).ravel()
    ci_size = np.array([x.size for x in ab_idx], dtype=np.int32)

    if fcivec.size == na * nb:
        ab_idx = np.hstack(ab_idx)
        ci0 = fcivec.ravel()[ab_idx]
    else:
        ci0 = fcivec
    ci1 = np.zeros_like(ci0)

    libfci.FCIcontract_2e_cyl_sym(
        eri.ctypes.data_as(ctypes.c_void_p),
        ci0.ctypes.data_as(ctypes.c_void_p),
        ci1.ctypes.data_as(ctypes.c_void_p),
        eri_ir_dims.ctypes.data_as(ctypes.c_void_p),
        ci_size.ctypes.data_as(ctypes.c_void_p),
        nas.ctypes.data_as(ctypes.c_void_p),
        nbs.ctypes.data_as(ctypes.c_void_p),
        link_indexa.ctypes.data_as(ctypes.c_void_p),
        link_indexb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb), ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
        ctypes.c_int(max_momentum), ctypes.c_int(max_gerades),
        ctypes.c_int(wfn_momentum), ctypes.c_int(wfn_ungerade))

    if fcivec.size == na * nb:
        ci1new = np.zeros(fcivec.size, dtype=fcivec.dtype)
        ci1new[ab_idx] = ci1
        ci1 = ci1new.reshape(fcivec.shape)
    return ci1.view(direct_spin1.FCIvector)

def _get_orb_l(orbsym):
    '''angular momentum for each orbital'''
    orb_l = (orbsym // 10) * 2
    orbsym_d2h = orbsym % 10
    e1_mask = np.isin(orbsym_d2h, (2, 3, 6, 7))
    orb_l[e1_mask] += 1
    ey_mask = np.isin(orbsym_d2h, (1, 3, 4, 6))
    orb_l[ey_mask] *= -1
    return orb_l

def reorder_eri(eri, norb, orbsym, max_momentum, max_gerades):
    eri = eri.reshape(norb,norb,norb,norb)
    # Swap last two indices because they are contracted to the t1 intermediates
    # in FCIcontract_2e_cyl_sym. t1 is generated with swapped orbital indices (a*norb+i).
    eri = eri.transpose(0,1,3,2).reshape(norb**2, norb**2)

    # % 10 to map irrep IDs of Dooh or Coov, etc. to irreps of D2h, C2v
    orbsym_d2h = orbsym % 10
    orb_ungerade = orbsym_d2h >= 4
    nirreps = (max_momentum * 2 + 1) * max_gerades

    # irrep of (ij| pair
    orb_l = _get_orb_l(orbsym)
    ll_prod = (orb_l[:,None] - orb_l).ravel()
    maxll = abs(orb_l).max() * 2

    old_eri_irrep = np.asarray(ll_prod+max_momentum, dtype=np.int32)
    rank_in_irrep = np.empty_like(old_eri_irrep)
    ir_idx_pairs = [None] * nirreps

    if max_gerades == 2:
        ug_offsets = max_momentum * 2 + 1
        ug_prod = (orb_ungerade[:,None] ^ orb_ungerade).ravel()
        old_eri_irrep[ug_prod] += ug_offsets

        # gerade
        idx = np.asarray(np.where((ll_prod == 0) & ~ug_prod)[0], dtype=np.int32)
        ir_idx_pairs[max_momentum] = (idx, idx)
        rank_in_irrep[idx] = np.arange(idx.size, dtype=np.int32)
        # ungerade
        idx = np.asarray(np.where((ll_prod == 0) & ug_prod)[0], dtype=np.int32)
        ir_idx_pairs[max_momentum+ug_offsets] = (idx, idx)
        rank_in_irrep[idx] = np.arange(idx.size, dtype=np.int32)

        for ll in range(1, maxll+1):
            # gerade
            idx_p = np.asarray(np.where((ll_prod == ll) & ~ug_prod)[0], dtype=np.int32)
            idx_m = np.asarray(np.where((ll_prod ==-ll) & ~ug_prod)[0], dtype=np.int32)
            assert idx_p.size == idx_m.size
            if idx_p.size > 0:
                ir_idx_pairs[max_momentum+ll] = (idx_p, idx_p)
                ir_idx_pairs[max_momentum-ll] = (idx_m, idx_m)
                rank_in_irrep[idx_p] = np.arange(idx_p.size, dtype=np.int32)
                rank_in_irrep[idx_m] = np.arange(idx_m.size, dtype=np.int32)
            # ungerade
            idx_p = np.asarray(np.where((ll_prod == ll) & ug_prod)[0], dtype=np.int32)
            idx_m = np.asarray(np.where((ll_prod ==-ll) & ug_prod)[0], dtype=np.int32)
            assert idx_p.size == idx_m.size
            if idx_p.size > 0:
                ir_idx_pairs[max_momentum+ll+ug_offsets] = (idx_p, idx_p)
                ir_idx_pairs[max_momentum-ll+ug_offsets] = (idx_m, idx_m)
                rank_in_irrep[idx_p] = np.arange(idx_p.size, dtype=np.int32)
                rank_in_irrep[idx_m] = np.arange(idx_m.size, dtype=np.int32)
    else:
        idx = np.asarray(np.where(ll_prod == 0)[0], dtype=np.int32)
        ir_idx_pairs[max_momentum] = (idx, idx)
        rank_in_irrep[idx] = np.arange(idx.size, dtype=np.int32)

        for ll in range(1, maxll+1):
            idx_p = np.asarray(np.where(ll_prod == ll)[0], dtype=np.int32)
            idx_m = np.asarray(np.where(ll_prod ==-ll)[0], dtype=np.int32)
            assert idx_p.size == idx_m.size
            if idx_p.size > 0:
                ir_idx_pairs[max_momentum+ll] = (idx_p, idx_p)
                ir_idx_pairs[max_momentum-ll] = (idx_m, idx_m)
                rank_in_irrep[idx_p] = np.arange(idx_p.size, dtype=np.int32)
                rank_in_irrep[idx_m] = np.arange(idx_m.size, dtype=np.int32)

    ir_dims = np.hstack([0 if x is None else x[0].size for x in ir_idx_pairs])
    eri_irs = np.empty((ir_dims**2).sum())
    p1 = 0
    for idx in ir_idx_pairs:
        if idx is not None:
            p0, p1 = p1, p1 + idx[0].size**2
            lib.take_2d(eri, idx[0], idx[1], out=eri_irs[p0:p1])
    eri_irs = lib.tag_array(eri_irs, ir_dims=np.asarray(ir_dims, dtype=np.int32))
    return eri_irs, rank_in_irrep, old_eri_irrep

def argsort_strs_by_irrep(strs, orbsym, max_momentum, max_gerades):
    strs_ls = _strs_angular_momentum(strs, orbsym)
    maxl = abs(strs_ls).max()
    nirreps = (max_momentum * 2 + 1) * max_gerades
    aidx = [np.zeros(0, dtype=np.int32)] * nirreps

    if max_gerades == 2:
        ug_offsets = max_momentum * 2 + 1
        irreps_d2h = direct_spin1_symm._gen_strs_irrep(strs, orbsym)
        strs_ug = irreps_d2h >= 4
        for l in range(-maxl, maxl+1):
            idx = np.where((strs_ls == l) & ~strs_ug)[0]
            aidx[max_momentum+l] = idx
            idx = np.where((strs_ls == l) & strs_ug)[0]
            aidx[max_momentum+l+ug_offsets] = idx
    else:
        for l in range(-maxl, maxl+1):
            idx = np.where(strs_ls == l)[0]
            aidx[max_momentum+l] = idx
    return aidx

def gen_str_irrep(strs, orbsym, link_index, rank_eri, irrep_eri, max_momentum,
                  max_gerades):
    aidx = argsort_strs_by_irrep(strs, orbsym, max_momentum, max_gerades)
    na = len(strs)
    rank = np.zeros(na, dtype=np.int32)
    for idx in aidx:
        if idx.size > 0:
            rank[idx] = np.arange(idx.size, dtype=np.int32)

    link_index = link_index.copy()
    norb = orbsym.size
    ai_addr = link_index[:,:,0] * norb + link_index[:,:,1]
    link_index[:,:,0] = rank_eri[ai_addr]
    link_index[:,:,1] = irrep_eri[ai_addr]
    link_index[:,:,2] = rank[link_index[:,:,2]]

    link_index = link_index.take(np.hstack(aidx), axis=0)
    return aidx, link_index

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0,
                   sym_allowed_idx=None):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
    na = len(strsa)
    nb = len(strsb)
    degen = orbsym.degen_mapping

    if sym_allowed_idx is None:
        sym_allowed_idx = sym_allowed_indices(nelec, orbsym, wfnsym)
    s_idx = np.hstack(sym_allowed_idx)
    idx_a, idx_b = divmod(s_idx, nb)
    if hdiag.size == na*nb:
        hdiag = hdiag[s_idx]
    civec_size = hdiag.size

    if neleca == nelecb and na == nb:
        idx = np.arange(idx_a.size)[idx_a >= idx_b]
        idx_a = idx_a[idx]
        idx_b = idx_b[idx]
        hdiag = hdiag[idx]

    ci0 = []
    for k in hdiag.argsort():
        addra, addrb = idx_a[k], idx_b[k]
        x = np.zeros(civec_size)
        x[s_idx==addra*nb+addrb] = 1.
        if wfnsym in (0, 1, 4, 5):
            addra1, sign_a = _sv_associated_det(strsa[addra], degen)
            addrb1, sign_b = _sv_associated_det(strsb[addrb], degen)
            # If (E+) and (E-) are associated determinants
            # (E+)(E-') + (E-)(E+') => A1
            # (E+)(E-') - (E-)(E+') => A2
            if wfnsym in (0, 5):  # A1g, A1u
                # ensure <|sigma_v|> = 1
                x[s_idx==addra1*nb+addrb1] += sign_a * sign_b
            elif wfnsym in (1, 4):  # A2g, A2u
                # ensure <|sigma_v|> = -1
                x[s_idx==addra1*nb+addrb1] -= sign_a * sign_b

        norm = np.linalg.norm(x)
        if norm < 1e-3:
            continue
        x *= 1./norm
        ci0.append(x.view(direct_spin1.FCIvector))
        if len(ci0) >= nroots:
            break

    if len(ci0) == 0:
        raise RuntimeError(f'Initial guess for symmetry {wfnsym} not found')
    return ci0


def _guess_wfnsym(civec, strsa, strsb, orbsym):
    degen_mapping = orbsym.degen_mapping
    idx = abs(civec).argmax()
    na = strsa.size
    nb = strsb.size
    civec = civec.reshape(na,nb)
    addra = idx // nb
    addrb = idx % nb
    addra1, sign_a = _sv_associated_det(strsa[addra], degen_mapping)
    addrb1, sign_b = _sv_associated_det(strsb[addrb], degen_mapping)

    airreps_d2h = direct_spin1_symm._gen_strs_irrep(strsa[[addra]], orbsym)
    birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsb[[addrb]], orbsym)
    a_ls = _strs_angular_momentum(strsa[[addra]], orbsym)
    b_ls = _strs_angular_momentum(strsb[[addrb]], orbsym)
    a_ungerade = airreps_d2h >= 4
    b_ungerade = birreps_d2h >= 4
    wfn_ungerade = a_ungerade[0] ^ b_ungerade[0]
    wfn_momentum = a_ls[0] + b_ls[0]

    if wfn_momentum == 0:
        # For A1g and A1u, CI coefficient and its sigma_v associated one have
        # the same sign
        if sign_a*sign_b * civec[addra,addrb] * civec[addra1,addrb1] > 1e-6: # A1
            if wfn_ungerade:
                wfnsym = 5
            else:
                wfnsym = 0
        else:
            # For A2g and A2u, CI coefficient and its sigma_v associated one
            # have opposite signs
            if wfn_ungerade:
                wfnsym = 4
            else:
                wfnsym = 1

    elif wfn_momentum % 2 == 1:
        if wfn_momentum > 0:  # Ex
            if wfn_ungerade:
                wfnsym = 7
            else:
                wfnsym = 2
        else:  # Ey
            if wfn_ungerade:
                wfnsym = 6
            else:
                wfnsym = 3
    else:
        if wfn_momentum > 0:  # Ex
            if wfn_ungerade:
                wfnsym = 5
            else:
                wfnsym = 0
        else:  # Ey
            if wfn_ungerade:
                wfnsym = 4
            else:
                wfnsym = 1

    wfnsym += (abs(wfn_momentum) // 2) * 10
    return wfnsym

def guess_wfnsym(solver, norb, nelec, fcivec=None, orbsym=None, wfnsym=None, **kwargs):
    '''
    Guess point group symmetry of the FCI wavefunction.  If fcivec is
    given, the symmetry of fcivec is used.  Otherwise the symmetry is
    same to the HF determinant.
    '''
    if orbsym is None:
        orbsym = solver.orbsym

    verbose = kwargs.get('verbose', None)
    log = logger.new_logger(solver, verbose)

    neleca, nelecb = nelec = direct_spin1._unpack_nelec(nelec, solver.spin)
    if fcivec is None or not hasattr(orbsym, 'degen_mapping'):
        # guess wfnsym if initial guess is not given
        wfnsym = direct_spin1_symm._id_wfnsym(solver, norb, nelec, orbsym, wfnsym)
        log.debug('Guessing CI wfn symmetry = %s', wfnsym)

    else:
        strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
        if neleca != nelecb:
            strsb = cistring.gen_strings4orblist(range(norb), nelecb)

        if not isinstance(fcivec, np.ndarray) or fcivec.ndim > 2:
            fcivec = fcivec[0]
        wfnsym1 = _guess_wfnsym(fcivec, strsa, strsb, orbsym)

        if (wfnsym is not None and
            wfnsym1 != direct_spin1_symm._id_wfnsym(solver, norb, nelec, orbsym, wfnsym)):
            raise RuntimeError(f'Input wfnsym {wfnsym} is not consistent with '
                               f'fcivec symmetry {wfnsym1}')
        wfnsym = wfnsym1
    return wfnsym

def sym_allowed_indices(nelec, orbsym, wfnsym):
    '''Indices of symmetry allowed determinants for each irrep'''
    norb = orbsym.size
    neleca, nelecb = nelec
    strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
    strsa_l = _strs_angular_momentum(strsa, orbsym)
    max_stra_l = max_strb_l = strsa_l.max()
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        strsb_l = _strs_angular_momentum(strsb, orbsym)
        max_strb_l = strsb_l.max()
    nb = len(strsb)

    wfn_momentum = symm.basis.linearmole_irrep2momentum(wfnsym)
    wfnsym_in_d2h = wfnsym % 10
    wfn_ungerade = wfnsym_in_d2h >= 4
    orbsym_d2h = orbsym % 10
    orb_ungerade = orbsym_d2h >= 4
    if np.any(orb_ungerade) or wfn_ungerade:
        max_gerades = 2
    else:
        max_gerades = 1
    orb_l = _get_orb_l(orbsym)
    max_eri_l = abs(orb_l).max() * 2
    max_momentum = max(max_stra_l, max_strb_l, max_eri_l)

    aidx = bidx = argsort_strs_by_irrep(strsa, orbsym, max_momentum, max_gerades)
    if neleca != nelecb:
        bidx = argsort_strs_by_irrep(strsb, orbsym, max_momentum, max_gerades)

    nirreps = (max_momentum * 2 + 1) * max_gerades
    ug_offsets = max_momentum * 2 + 1
    ab_idx = [np.zeros(0, dtype=int)] * nirreps
    for ag in range(max_gerades):
        bg = wfn_ungerade ^ ag
        # abs(al) < max_stra_l and abs(bl := wfn_momentum-al) < max_strb_l
        for al in range(max(-max_stra_l, wfn_momentum-max_strb_l),
                        min( max_stra_l, wfn_momentum+max_strb_l)+1):
            bl = wfn_momentum - al
            stra_ir = al + max_momentum + ag * ug_offsets
            strb_ir = bl + max_momentum + bg * ug_offsets
            ab_idx[stra_ir] = (aidx[stra_ir][:,None] * nb + bidx[strb_ir]).ravel()
    return ab_idx

def _dm_wrapper(fn_rdm):
    def transform(dm, u):
        if dm.ndim == 2:
            dm = u.conj().T.dot(dm).dot(u)
        else:
            dm = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', dm, u.conj(), u, u.conj(), u)
        return dm.real.copy()

    @functools.wraps(fn_rdm)
    def make_dm(self, fcivec, norb, nelec, *args, **kwargs):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        dms = fn_rdm(fcivec, norb, nelec, *args, **kwargs)
        orbsym = self.orbsym
        degen_mapping = self.orbsym.degen_mapping
        u = _cyl_sym_orbital_rotation(orbsym, degen_mapping)
        if isinstance(dms, np.ndarray):
            return transform(dms, u)
        else:
            return [transform(dm, u) for dm in dms]
    return make_dm


class FCISolver(direct_spin1_symm.FCISolver):

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        raise NotImplementedError

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None:
            wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym, self.wfnsym)
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym)

    def get_init_guess(self, norb, nelec, nroots, hdiag, orbsym=None, wfnsym=None):
        if orbsym is None:
            orbsym = self.orbsym
        if wfnsym is None:
            wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, orbsym, self.wfnsym)
        return get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym,
                              self.sym_allowed_idx)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        na = cistring.num_strings(norb, nelec[0])
        nb = cistring.num_strings(norb, nelec[1])
        s_idx = numpy.hstack(self.sym_allowed_idx)
        # Screen symmetry forbidden elements
        hdiag, hdiag0 = numpy.empty(na*nb), hdiag.ravel()
        hdiag[:] = 1e99
        if hdiag0.size == s_idx.size:
            hdiag[s_idx] = hdiag0
        else:
            hdiag[s_idx] = hdiag0[s_idx]

        np = min(np, s_idx.size)
        addr0, h = direct_spin1.pspace(h1e, eri, norb, nelec, hdiag, np)

        # mapping the address in (na,nb) civec to address in sym-allowed civec
        s_idx_allowed = numpy.where(numpy.isin(s_idx, addr0))[0]
        addr1 = s_idx[s_idx_allowed]
        new_idx = numpy.empty_like(s_idx_allowed)
        new_idx[addr0.argsort()] = addr1.argsort()
        addr = s_idx_allowed[new_idx]
        return addr, h

    absorb_h1e = direct_nosym.FCISolver.absorb_h1e
    make_hdiag = direct_spin1_symm.FCISolver.make_hdiag
    guess_wfnsym = guess_wfnsym

    make_rdm1 = _dm_wrapper(direct_spin1.make_rdm1)
    make_rdm1s = _dm_wrapper(direct_spin1.make_rdm1s)
    make_rdm12 = _dm_wrapper(direct_spin1.make_rdm12)
    make_rdm12s = _dm_wrapper(direct_spin1.make_rdm12s)
    trans_rdm1 = _dm_wrapper(direct_spin1.trans_rdm1)
    trans_rdm1s = _dm_wrapper(direct_spin1.trans_rdm1s)
    trans_rdm12 = _dm_wrapper(direct_spin1.trans_rdm12)
    trans_rdm12s = _dm_wrapper(direct_spin1.trans_rdm12s)

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            if 'verbose' not in kwargs:
                kwargs['verbose'] = self.verbose
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec = direct_spin1._unpack_nelec(nelec, self.spin)

        if not hasattr(orbsym, 'degen_mapping'):
            degen_mapping = map_degeneracy(h1e.diagonal(), orbsym)
            orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)

        u = _cyl_sym_orbital_rotation(orbsym, orbsym.degen_mapping)
        h1e = u.dot(h1e).dot(u.conj().T)
        eri = ao2mo.restore(1, eri, norb)
        eri = lib.einsum('pqrs,ip,jq,kr,ls->ijkl', eri, u, u.conj(), u, u.conj())
        assert abs(h1e.imag).max() < 1e-12, 'Cylindrical symmetry broken'
        assert abs(eri.imag).max() < 1e-12, 'Cylindrical symmetry broken'
        h1e = h1e.real.copy()
        # Note: eri is real but it does not have the permutation relation
        # (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # The nosym version fci contraction is required
        eri = eri.real.copy()

        wfnsym_ir = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)
        if wfnsym_ir in (1, 4):
            # sym_allowed_idx does not distinguish A2g and A2u
            davidson_only = True
        self.sym_allowed_idx = sym_allowed_indices(nelec, orbsym, wfnsym_ir)
        self.orbsym = orbsym
        logger.debug(self, 'Num symmetry allowed elements %d',
                     sum([x.size for x in self.sym_allowed_idx]))

        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca == nelecb:
            link_indexb = link_indexa
        else:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

        with lib.temporary_env(self, wfnsym=wfnsym_ir):
            e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0,
                                           (link_indexa,link_indexb),
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)

        na = link_indexa.shape[0]
        nb = link_indexb.shape[0]
        s_idx = np.hstack(self.sym_allowed_idx)
        if nroots > 1:
            c, c_raw = [], c
            for vec in c_raw:
                c1 = np.zeros(na*nb)
                c1[s_idx] = vec.T
                c.append(c1)
        else:
            c1 = np.zeros(na*nb)
            c1[s_idx] = c
            c = c1.reshape(na, nb).view(direct_spin1.FCIvector)

        self.eci, self.ci = e, c
        return e, c

FCI = FCISolver
