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
Different FCI solvers are implemented to support different type of symmetry.
                    Symmetry
File                Point group   Spin singlet   Real hermitian*    Alpha/beta degeneracy
direct_spin0_symm   Yes           Yes            Yes                Yes
direct_spin1_symm   Yes           No             Yes                Yes
direct_spin0        No            Yes            Yes                Yes
direct_spin1        No            No             Yes                Yes
direct_uhf          No            No             Yes                No
direct_nosym        No            No             No**               Yes

*  Real hermitian Hamiltonian implies (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
** Hamiltonian is real but not hermitian, (ij|kl) != (ji|kl) ...
'''

import sys
import ctypes
import numpy
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.scf.hf_symm import map_degeneracy
from pyscf.fci import cistring
from pyscf.fci import direct_spin1
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf.fci.addons import _unpack_nelec
from pyscf import __config__

libfci = lib.load_library('libfci')

TOTIRREPS = 8

def contract_1e(f1e, fcivec, norb, nelec, link_index=None, orbsym=None):
    return direct_spin1.contract_1e(f1e, fcivec, norb, nelec, link_index)

# Note eri is NOT the 2e hamiltonian matrix, the 2e hamiltonian is
# h2e = eri_{pq,rs} p^+ q r^+ s
#     = (pq|rs) p^+ r^+ s q - (pq|rs) \delta_{qr} p^+ s
# so eri is defined as
#       eri_{pq,rs} = (pq|rs) - (1/Nelec) \sum_q (pq|qs)
# to restore the symmetry between pq and rs,
#       eri_{pq,rs} = (pq|rs) - (.5/Nelec) [\sum_q (pq|qs) + \sum_p (pq|rp)]
# Please refer to the treatment in direct_spin1.absorb_h1e
def contract_2e(eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    if orbsym is None:
        return direct_spin1.contract_2e(eri, fcivec, norb, nelec, link_index)

    eri = ao2mo.restore(4, eri, norb)
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    eri_irs, rank_eri, irrep_eri = reorder_eri(eri, norb, orbsym)

    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    aidx, link_indexa = gen_str_irrep(strsa, orbsym, link_indexa, rank_eri, irrep_eri)
    if neleca == nelecb:
        bidx, link_indexb = aidx, link_indexa
    else:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        bidx, link_indexb = gen_str_irrep(strsb, orbsym, link_indexb, rank_eri, irrep_eri)

    Tirrep = ctypes.c_void_p*TOTIRREPS
    linka_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexa])
    linkb_ptr = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in link_indexb])
    eri_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in eri_irs])
    dimirrep = (ctypes.c_int*TOTIRREPS)(*[x.shape[0] for x in eri_irs])
    fcivec_shape = fcivec.shape
    fcivec = fcivec.reshape((na,nb), order='C')
    ci1new = numpy.zeros_like(fcivec)
    nas = (ctypes.c_int*TOTIRREPS)(*[x.size for x in aidx])
    nbs = (ctypes.c_int*TOTIRREPS)(*[x.size for x in bidx])

    # aa, ab
    ci0 = []
    ci1 = []
    wfnsym_in_d2h = wfnsym % 10
    for ir in range(TOTIRREPS):
        ma, mb = aidx[ir].size, bidx[wfnsym_in_d2h ^ ir].size
        ci0.append(numpy.zeros((ma,mb)))
        ci1.append(numpy.zeros((ma,mb)))
        if ma > 0 and mb > 0:
            lib.take_2d(fcivec, aidx[ir], bidx[wfnsym_in_d2h ^ ir], out=ci0[ir])
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nas, nbs,
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                linka_ptr, linkb_ptr, dimirrep,
                                ctypes.c_int(wfnsym_in_d2h))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, ci1[ir], aidx[ir], bidx[wfnsym_in_d2h ^ ir])

    # bb, ba
    ci0T = []
    for ir in range(TOTIRREPS):
        mb, ma = bidx[ir].size, aidx[wfnsym_in_d2h ^ ir].size
        ci0T.append(numpy.zeros((mb,ma)))
        if ma > 0 and mb > 0:
            lib.transpose(ci0[wfnsym_in_d2h ^ ir], out=ci0T[ir])
    ci0, ci0T = ci0T, None
    ci1 = [numpy.zeros_like(x) for x in ci0]
    ci0_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci0])
    ci1_ptrs = Tirrep(*[x.ctypes.data_as(ctypes.c_void_p) for x in ci1])
    libfci.FCIcontract_2e_symm1(eri_ptrs, ci0_ptrs, ci1_ptrs,
                                ctypes.c_int(norb), nbs, nas,
                                ctypes.c_int(nlinkb), ctypes.c_int(nlinka),
                                linkb_ptr, linka_ptr, dimirrep,
                                ctypes.c_int(wfnsym_in_d2h))
    for ir in range(TOTIRREPS):
        if ci0[ir].size > 0:
            lib.takebak_2d(ci1new, lib.transpose(ci1[ir]),
                           aidx[wfnsym_in_d2h ^ ir], bidx[ir])
    return ci1new.reshape(fcivec_shape).view(direct_spin1.FCIvector)


def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    assert (len(orbsym) == norb)
    cis = FCISolver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size
    cis.orbsym = orbsym
    cis.wfnsym = wfnsym

    unknown = {}
    for k, v in kwargs.items():
        if not hasattr(cis, k):
            unknown[k] = v
        setattr(cis, k, v)
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))

    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

make_rdm1 = direct_spin1.make_rdm1
make_rdm1s = direct_spin1.make_rdm1s
make_rdm12 = direct_spin1.make_rdm12

trans_rdm1s = direct_spin1.trans_rdm1s
trans_rdm1 = direct_spin1.trans_rdm1
trans_rdm12 = direct_spin1.trans_rdm12

def energy(h1e, eri, fcivec, norb, nelec, link_index=None, orbsym=None, wfnsym=0):
    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec) * .5
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index, orbsym, wfnsym)
    return numpy.dot(fcivec.ravel(), ci1.ravel())

def _id_wfnsym(cisolver, norb, nelec, orbsym, wfnsym):
    '''Guess wfnsym or convert wfnsym to symmetry ID if it's a symmetry label'''
    gpname = getattr(cisolver.mol, 'groupname', None)
    if wfnsym is None:
        neleca, nelecb = _unpack_nelec(nelec)
        wfnsym = 0  # Ag, A1 or A
        for i in orbsym[nelecb:neleca]:
            wfnsym ^= i % 10
        if gpname in ('Dooh', 'Coov'):
            l = 0
            for i in orbsym[nelecb:neleca]:
                l += symm.basis.linearmole_irrep2momentum(i)
            wfnsym += (l//2) * 10
    elif isinstance(wfnsym, str):
        wfnsym = symm.irrep_name2id(gpname, wfnsym)
    return wfnsym

def _gen_strs_irrep(strs, orbsym):
    # % 10 to convert irrep_ids to irrep of D2h
    orbsym_in_d2h = numpy.asarray(orbsym) % 10
    irreps = numpy.zeros(len(strs), dtype=numpy.int32)
    if isinstance(strs, cistring.OIndexList):
        nocc = strs.shape[1]
        for i in range(nocc):
            irreps ^= orbsym_in_d2h[strs[:,i]]
    else:
        for i, ir in enumerate(orbsym_in_d2h):
            irreps[numpy.bitwise_and(strs, 1 << i) > 0] ^= ir
    return irreps

def _get_init_guess(airreps, birreps, nroots, hdiag, nelec, orbsym, wfnsym=0):
    neleca, nelecb = _unpack_nelec(nelec)
    na = len(airreps)
    nb = len(birreps)
    hdiag = hdiag.reshape(na,nb)
    ci0 = []
    iroot = 0
    sym_allowed = airreps[:,None] == wfnsym ^ birreps
    if neleca == nelecb and na == nb:
        idx = numpy.arange(na)
        sym_allowed[idx[:,None] < idx] = False
    idx_a, idx_b = numpy.where(sym_allowed)
    for k in hdiag[idx_a,idx_b].argsort():
        addra, addrb = idx_a[k], idx_b[k]
        x = numpy.zeros((na, nb))
        x[addra,addrb] = 1
        ci0.append(x.ravel().view(direct_spin1.FCIvector))
        iroot += 1
        if iroot >= nroots:
            break

    if len(ci0) == 0:
        raise RuntimeError(f'Initial guess for symmetry {wfnsym} not found')
    return ci0

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = cistring.gen_strings4orblist(range(norb), neleca)
    airreps = birreps = _gen_strs_irrep(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps = _gen_strs_irrep(strsb, orbsym)
    return _get_init_guess(airreps, birreps, nroots, hdiag, nelec, orbsym, wfnsym)

def get_init_guess_cyl_sym(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
    airreps_d2h = birreps_d2h = _gen_strs_irrep(strsa, orbsym)
    a_ls = b_ls = _strs_angular_momentum(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps_d2h = _gen_strs_irrep(strsb, orbsym)
        b_ls = _strs_angular_momentum(strsb, orbsym)

    wfnsym_in_d2h = wfnsym % 10
    wfn_momentum = symm.basis.linearmole_irrep2momentum(wfnsym)
    na = len(strsa)
    nb = len(strsb)
    hdiag = hdiag.reshape(na,nb)
    degen_mapping = orbsym.degen_mapping
    ci0 = []
    iroot = 0
    wfn_ungerade = wfnsym_in_d2h >= 4
    a_ungerade = airreps_d2h >= 4
    b_ungerade = birreps_d2h >= 4
    sym_allowed = a_ungerade[:,None] == b_ungerade ^ wfn_ungerade
    # total angular momentum == wfn_momentum
    sym_allowed &= a_ls[:,None] == wfn_momentum - b_ls
    if neleca == nelecb and na == nb:
        idx = numpy.arange(na)
        sym_allowed[idx[:,None] < idx] = False
    idx_a, idx_b = numpy.where(sym_allowed)

    for k in hdiag[idx_a,idx_b].argsort():
        addra, addrb = idx_a[k], idx_b[k]
        ca = _cyl_sym_csf2civec(strsa, addra, orbsym, degen_mapping)
        cb = _cyl_sym_csf2civec(strsb, addrb, orbsym, degen_mapping)
        if wfnsym in (0, 1, 4, 5):
            addra1, sign_a = _sv_associated_det(strsa[addra], degen_mapping)
            addrb1, sign_b = _sv_associated_det(strsb[addrb], degen_mapping)
            if wfnsym in (1, 4) and addra == addra1 and addrb == addrb1:
                continue
            x = ca[:,None] * cb

            # If (E+) and (E-) are associated determinants
            # (E+)(E-') + (E-)(E+') => A1
            # (E+)(E-') - (E-)(E+') => A2
            if addra != addra1:
                ca = _cyl_sym_csf2civec(strsa, addra1, orbsym, degen_mapping)
            if addrb != addrb1:
                cb = _cyl_sym_csf2civec(strsb, addrb1, orbsym, degen_mapping)
            if wfnsym in (0, 5):  # A1g, A1u
                x += sign_a * sign_b * ca[:,None] * cb
                #assert (sign_a*sign_b==1 and x.imag==0) or (sign_a*sign_b==-1 and x.real==0)
            elif wfnsym in (1, 4):  # A2g, A2u
                x -= sign_a * sign_b * ca[:,None] * cb
                #assert (sign_a*sign_b==1 and x.real==0) or (sign_a*sign_b==-1 and x.imag==0)
            if numpy.linalg.norm(x.real) > 1e-6:
                x = x.real.copy()
            else:
                x = x.imag.copy()

        elif wfn_momentum > 0:
            x = ca.real[:,None] * cb.real
            x-= ca.imag[:,None] * cb.imag
        else:
            x = ca.imag[:,None] * cb.real
            x+= ca.real[:,None] * cb.imag

        norm = numpy.linalg.norm(x)
        if norm < 1e-3:
            continue
        x *= 1./norm
        ci0.append(x.ravel().view(direct_spin1.FCIvector))
        iroot += 1
        if iroot >= nroots:
            break

    if len(ci0) == 0:
        raise RuntimeError(f'Initial guess for symmetry {wfnsym} not found')
    return ci0

def _cyl_sym_csf2civec(strs, addr, orbsym, degen_mapping):
    '''For orbital basis rotation from E(+/-) basis to Ex/Ey basis, mimic the CI
    transformation  addons.transform_ci(civec, (0, nelec), u)
    '''
    norb = orbsym.size
    one_particle_strs = numpy.asarray([1 << i for i in range(norb)])
    occ_masks = (strs[:,None] & one_particle_strs) != 0
    na = strs.size
    occ_idx_all_strs = numpy.where(occ_masks)[1].reshape(na,-1)

    u = _cyl_sym_orbital_rotation(orbsym, degen_mapping)
    ui = u[occ_masks[addr]].T.copy()
    minors = ui[occ_idx_all_strs]
    civec = numpy.linalg.det(minors)
    return civec

def _cyl_sym_orbital_rotation(orbsym, degen_mapping):
    '''Rotation to transform (E+)/(E-) basis to Ex/Ey basis
    |Ex/Ey> = |E(+/-)> * u
    '''
    norb = orbsym.size
    u = numpy.zeros((norb, norb), dtype=numpy.complex128)
    sqrth = .5**.5
    sqrthi = sqrth * 1j
    for i, j in enumerate(degen_mapping):
        if i == j:  # 1d irrep
            if orbsym[i] in (1, 4):  # A2g, A2u
                u[i,i] = 1j
            else:
                u[i,i] = 1
        elif orbsym[i] % 10 in (0, 2, 5, 7):  # Ex, E(+)
            u[j,j] = sqrthi
            u[i,j] = sqrthi
            u[j,i] = sqrth
            u[i,i] = -sqrth
    return u

def _sv_associated_det(ci_str, degen_mapping):
    '''Associated determinant for the sigma_v operation'''
    ci_str1 = 0
    nelec = 0
    sign = 1
    for i, j in enumerate(degen_mapping):
        if ci_str & (1 << i) > 0:
            if i > j and ci_str & (1 << j) > 0:
                # Ex, Ey orbitals swapped
                sign = -sign
            ci_str1 |= 1 << j
            nelec += 1
    return cistring.str2addr(degen_mapping.size, nelec, ci_str1), sign

def _strs_angular_momentum(strs, orbsym):
    # angular momentum for each orbitals
    orb_l = (orbsym // 10) * 2
    e1_mask = numpy.isin(orbsym % 10, (2, 3, 6, 7))
    orb_l[e1_mask] += 1
    ey_mask = numpy.isin(orbsym % 10, (1, 3, 4, 6))
    orb_l[ey_mask] *= -1

    # total angular for each determinant (CSF)
    ls = numpy.zeros(len(strs), dtype=int)
    if isinstance(strs, cistring.OIndexList):
        nocc = strs.shape[1]
        for i in range(nocc):
            ls += orb_l[strs[:,i]]
    else:
        for i, l in enumerate(orb_l):
            ls[numpy.bitwise_and(strs, 1 << i) > 0] += l
    return ls

def reorder_eri(eri, norb, orbsym):
    if orbsym is None:
        return [eri], numpy.arange(norb), numpy.zeros(norb,dtype=numpy.int32)

    # % 10 to map irrep IDs of Dooh or Coov, etc. to irreps of D2h, C2v
    orbsym = numpy.asarray(orbsym) % 10

    # irrep of (ij| pair
    trilirrep = (orbsym[:,None] ^ orbsym)[numpy.tril_indices(norb)]
    # and the number of occurence for each irrep
    dimirrep = numpy.asarray(numpy.bincount(trilirrep), dtype=numpy.int32)
    # we sort the irreps of (ij| pair, to group the pairs which have same irreps
    # "order" is irrep-id-sorted index. The (ij| paired is ordered that the
    # pair-id given by order[0] comes first in the sorted pair
    # "rank" is a sorted "order". Given nth (ij| pair, it returns the place(rank)
    # of the sorted pair
    old_eri_irrep = numpy.asarray(trilirrep, dtype=numpy.int32)
    rank_in_irrep = numpy.empty_like(old_eri_irrep)
    p0 = 0
    eri_irs = [numpy.zeros((0,0))] * TOTIRREPS
    for ir, nnorb in enumerate(dimirrep):
        idx = numpy.asarray(numpy.where(trilirrep == ir)[0], dtype=numpy.int32)
        rank_in_irrep[idx] = numpy.arange(nnorb, dtype=numpy.int32)
        eri_ir = lib.take_2d(eri, idx, idx)
        # Drop small integrals which may break symmetry?
        #eri_ir[abs(eri_ir) < 1e-13] = 0
        eri_irs[ir] = eri_ir
        p0 += nnorb
    return eri_irs, rank_in_irrep, old_eri_irrep

def gen_str_irrep(strs, orbsym, link_index, rank_eri, irrep_eri):
    airreps = _gen_strs_irrep(strs, orbsym)
    na = len(airreps)
    rank = numpy.zeros(na, dtype=numpy.int32)
    aidx = [numpy.zeros(0,dtype=numpy.int32)] * TOTIRREPS
    for ir in range(TOTIRREPS):
        aidx[ir] = numpy.where(airreps == ir)[0]
        ma = len(aidx[ir])
        if ma > 0:
            rank[aidx[ir]] = numpy.arange(ma, dtype=numpy.int32)
    link_index = link_index.copy()
    link_index[:,:,1] = irrep_eri[link_index[:,:,0]]
    link_index[:,:,0] = rank_eri[link_index[:,:,0]]
    link_index[:,:,2] = rank[link_index[:,:,2]]
    link_index = [link_index.take(aidx[ir], axis=0) for ir in range(TOTIRREPS)]
    return aidx, link_index

def _guess_wfnsym_cyl_sym(civec, strsa, strsb, orbsym):
    degen_mapping = orbsym.degen_mapping
    idx = abs(civec).argmax()
    na = strsa.size
    nb = strsb.size
    addra = idx // nb
    addrb = idx % nb
    addra1, sign_a = _sv_associated_det(strsa[addra], degen_mapping)
    addrb1, sign_b = _sv_associated_det(strsb[addrb], degen_mapping)
    addra, addra1 = min(addra,addra1), max(addra,addra1)
    addrb, addrb1 = min(addrb,addrb1), max(addrb,addrb1)
    ca = ca1 = _cyl_sym_csf2civec(strsa, addra, orbsym, degen_mapping)
    cb = cb1 = _cyl_sym_csf2civec(strsb, addrb, orbsym, degen_mapping)
    if addra != addra1:
        ca1 = _cyl_sym_csf2civec(strsa, addra1, orbsym, degen_mapping)
    if addrb != addrb1:
        cb1 = _cyl_sym_csf2civec(strsb, addrb1, orbsym, degen_mapping)
    ua = numpy.stack([ca, ca1])
    ub = numpy.stack([cb, cb1])
    # civec is in the Ex/Ey basis. Transform the largest coefficient to
    # (E+)/(E-) basis.
    c_max = ua.conj().dot(civec.reshape(na,nb)).dot(ub.conj().T)

    airreps_d2h = _gen_strs_irrep(strsa[[addra,addra1]], orbsym)
    birreps_d2h = _gen_strs_irrep(strsb[[addrb,addrb1]], orbsym)
    a_ls = _strs_angular_momentum(strsa[[addra,addra1]], orbsym)
    b_ls = _strs_angular_momentum(strsb[[addrb,addrb1]], orbsym)
    a_ungerade = airreps_d2h >= 4
    b_ungerade = birreps_d2h >= 4
    idx = abs(c_max).argmax()
    idx_a, idx_b = idx // 2, idx % 2
    wfn_ungerade = a_ungerade[idx_a] ^ b_ungerade[idx_b]
    wfn_momentum = a_ls[idx_a] + b_ls[idx_b]

    if wfn_momentum == 0:
        # For A1g and A1u, CI coefficient and its sigma_v associated one have
        # the same sign
        if (sign_a*sign_b * c_max[0,0].real * c_max[1,1].real > 1e-6 or
            sign_a*sign_b * c_max[0,0].imag * c_max[1,1].imag > 1e-6):  # A1
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
        if abs(c_max[idx_a,idx_b].real) > 1e-6:  # Ex
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
        if abs(c_max[idx_a,idx_b].real) > 1e-6:  # Ex
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
    neleca, nelecb = nelec = _unpack_nelec(nelec, solver.spin)

    groupname = getattr(solver.mol, 'groupname', None)
    if fcivec is None:
        # guess wfnsym if initial guess is not given
        wfnsym = _id_wfnsym(solver, norb, nelec, orbsym, wfnsym)
        log.debug('Guessing CI wfn symmetry = %s', wfnsym)

    elif wfnsym is None:
        if groupname in ('Dooh', 'Coov'):
            strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
            if neleca != nelecb:
                strsb = cistring.gen_strings4orblist(range(norb), nelecb)
            if not isinstance(fcivec, numpy.ndarray) or fcivec.ndim > 2:
                fcivec = fcivec[0]
            wfnsym = _guess_wfnsym_cyl_sym(fcivec, strsa, strsb, orbsym)
        else:
            wfnsym = addons.guess_wfnsym(fcivec, norb, nelec, orbsym)
        log.debug('Guessing CI wfn symmetry = %s', wfnsym)

    else:
        # verify if the input wfnsym is consistent with the symmetry of fcivec
        strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
        if neleca != nelecb:
            strsb = cistring.gen_strings4orblist(range(norb), nelecb)

        if groupname in ('Dooh', 'Coov'):
            if not isinstance(fcivec, numpy.ndarray) or fcivec.ndim > 2:
                fcivec = fcivec[0]
            wfnsym1 = _guess_wfnsym_cyl_sym(fcivec, strsa, strsb, orbsym)
            if wfnsym1 != _id_wfnsym(solver, norb, nelec, orbsym, wfnsym):
                raise RuntimeError(f'Input wfnsym {wfnsym} is not consistent with '
                                   f'fcivec symmetry {wfnsym1}')
            wfnsym = wfnsym1
        else:
            na, nb = strsa.size, strsb.size
            orbsym_in_d2h = numpy.asarray(orbsym) % 10
            airreps = numpy.zeros(na, dtype=numpy.int32)
            birreps = numpy.zeros(nb, dtype=numpy.int32)
            for i, ir in enumerate(orbsym_in_d2h):
                airreps[numpy.bitwise_and(strsa, 1 << i) > 0] ^= ir
                birreps[numpy.bitwise_and(strsb, 1 << i) > 0] ^= ir

            wfnsym = _id_wfnsym(solver, norb, nelec, orbsym, wfnsym)
            groupname = getattr(solver.mol, 'groupname', None)
            mask = airreps[:,None] == (wfnsym % 10) ^ birreps

            if isinstance(fcivec, numpy.ndarray) and fcivec.ndim <= 2:
                fcivec = [fcivec]
            if all(abs(c.reshape(na, nb)[mask]).max() < 1e-5 for c in fcivec):
                raise RuntimeError('Input wfnsym {wfnsym} is not consistent with '
                                   'fcivec coefficients')

    return wfnsym


class FCISolver(direct_spin1.FCISolver):

    davidson_only = getattr(__config__, 'fci_direct_spin1_symm_FCI_davidson_only', True)
    # pspace may break point group symmetry
    pspace_size = getattr(__config__, 'fci_direct_spin1_symm_FCI_pspace_size', 0)

    def __init__(self, mol=None, **kwargs):
        direct_spin1.FCISolver.__init__(self, mol, **kwargs)
        # wfnsym will be guessed based on initial guess if it is None
        self.wfnsym = None

    def dump_flags(self, verbose=None):
        direct_spin1.FCISolver.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if isinstance(self.wfnsym, str):
            log.info('Input CI wfn symmetry = %s', self.wfnsym)
        elif isinstance(self.wfnsym, (int, numpy.number)):
            groupname = getattr(self.mol, 'groupname', None)
            if groupname is not None:
                try:
                    log.info('Input CI wfn symmetry = %s',
                             symm.irrep_id2name(groupname, self.wfnsym))
                except KeyError:
                    raise RuntimeError('FCISolver cannot find mwfnsym Id %s in group %s. '
                                       'This might be caused by the projection from '
                                       'high-symmetry group to D2h symmetry.' %
                                       (self.wfnsym, groupname))
        else:
            log.info('CI wfn symmetry = %s', self.wfnsym)
        return self

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.make_hdiag(h1e, eri, norb, nelec)

    def pspace(self, h1e, eri, norb, nelec, hdiag, np=400):
        nelec = _unpack_nelec(nelec, self.spin)
        return direct_spin1.pspace(h1e, eri, norb, nelec, hdiag, np)

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_1e(f1e, fcivec, norb, nelec, link_index, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        wfnsym = _id_wfnsym(self, norb, nelec, orbsym, wfnsym)
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_2e(eri, fcivec, norb, nelec, link_index, orbsym, wfnsym, **kwargs)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = _id_wfnsym(self, norb, nelec, self.orbsym, self.wfnsym)
        if getattr(self.mol, 'groupname', None) in ('Dooh', 'Coov'):
            return get_init_guess_cyl_sym(
                norb, nelec, nroots, hdiag, self.orbsym, wfnsym)
        else:
            return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    guess_wfnsym = guess_wfnsym

    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        if nroots is None: nroots = self.nroots
        if orbsym is None: orbsym = self.orbsym
        if wfnsym is None: wfnsym = self.wfnsym
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.norb = norb
        self.nelec = nelec

        if (not hasattr(orbsym, 'degen_mapping') and
            getattr(self.mol, 'groupname', None) in ('Dooh', 'Coov')):
            degen_mapping = map_degeneracy(h1e.diagonal(), orbsym)
            orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)

        wfnsym = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)

        if wfnsym > 7:
            # Symmetry broken for Dooh and Coov groups is often observed.
            # A larger max_space is helpful to reduce the error. Also it is
            # hard to converge to high precision.
            if max_space is None and self.max_space == FCISolver.max_space:
                max_space = 20 + 7 * nroots
            if tol is None and self.conv_tol == FCISolver.conv_tol:
                tol = 1e-7

        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0, None,
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c

FCI = FCISolver
