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

import sys
import copy
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf import symm
from pyscf import __config__

LARGE_CI_TOL = getattr(__config__, 'fci_addons_large_ci_tol', 0.1)
RETURN_STRS = getattr(__config__, 'fci_addons_large_ci_return_strs', True)
PENALTY = getattr(__config__, 'fci_addons_fix_spin_shift', 0.2)


def large_ci(ci, norb, nelec, tol=LARGE_CI_TOL, return_strs=RETURN_STRS):
    '''Search for the largest CI coefficients
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    assert(ci.shape == (na, nb))
    addra, addrb = numpy.where(abs(ci) > tol)
    if addra.size == 0:
        # No large CI coefficient > tol, search for the largest coefficient
        addra, addrb = numpy.unravel_index(numpy.argmax(abs(ci)), ci.shape)
        addra = numpy.asarray([addra])
        addrb = numpy.asarray([addrb])
    strsa = cistring.addrs2str(norb, neleca, addra)
    strsb = cistring.addrs2str(norb, nelecb, addrb)
    if return_strs:
        strsa = [bin(x) for x in strsa]
        strsb = [bin(x) for x in strsb]
        return list(zip(ci[addra,addrb], strsa, strsb))
    else:
        occslsta = cistring._strs2occslst(strsa, norb)
        occslstb = cistring._strs2occslst(strsb, norb)
        return list(zip(ci[addra,addrb], occslsta, occslstb))

def initguess_triplet(norb, nelec, binstring):
    '''Generate a triplet initial guess for FCI solver
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    addr = cistring.str2addr(norb, neleca, int(binstring,2))
    ci0 = numpy.zeros((na,nb))
    ci0[addr,0] = numpy.sqrt(.5)
    ci0[0,addr] =-numpy.sqrt(.5)
    return ci0

def symm_initguess(norb, nelec, orbsym, wfnsym=0, irrep_nelec=None):
    '''Generate CI wavefunction initial guess which has the given symmetry.

    Args:
        norb : int
            Number of orbitals.
        nelec : int or 2-item list
            Number of electrons, or 2-item list for (alpha, beta) electrons
        orbsym : list of int
            The irrep ID for each orbital.

    Kwags:
        wfnsym : int
            The irrep ID of target symmetry
        irrep_nelec : dict
            Freeze occupancy for certain irreps

    Returns:
        CI coefficients 2D array which has the target symmetry.
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    orbsym = numpy.asarray(orbsym)
    if not isinstance(orbsym[0], numpy.number):
        raise RuntimeError('TODO: convert irrep symbol to irrep id')

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci1 = numpy.zeros((na,nb))

########################
# pass 1: The fixed occs
    orbleft = numpy.ones(norb, dtype=bool)
    stra = numpy.zeros(norb, dtype=bool)
    strb = numpy.zeros(norb, dtype=bool)
    if irrep_nelec is not None:
        for k,n in irrep_nelec.items():
            orbleft[orbsym==k] = False
            if isinstance(n, (int, numpy.number)):
                idx = numpy.where(orbsym==k)[0][:n//2]
                stra[idx] = True
                strb[idx] = True
            else:
                na, nb = n
                stra[numpy.where(orbsym==k)[0][:na]] = True
                strb[numpy.where(orbsym==k)[0][:nb]] = True
                if (na-nb)%2:
                    wfnsym ^= k

    orbleft = numpy.where(orbleft)[0]
    neleca_left = neleca - stra.sum()
    nelecb_left = nelecb - strb.sum()
    spin = neleca_left - nelecb_left
    assert(neleca_left >= 0)
    assert(nelecb_left >= 0)
    assert(spin >= 0)

########################
# pass 2: search pattern
    def gen_str_iter(orb_list, nelec):
        if nelec == 1:
            for i in orb_list:
                yield [i]
        elif nelec >= len(orb_list):
            yield orb_list
        else:
            restorb = orb_list[1:]
            #yield from gen_str_iter(restorb, nelec)
            for x in gen_str_iter(restorb, nelec):
                yield x
            for x in gen_str_iter(restorb, nelec-1):
                yield [orb_list[0]] + x

# search for alpha and beta pattern which match to the required symmetry
    def query(target, nelec_atmost, spin, orbsym):
        norb = len(orbsym)
        for excite_level in range(1, nelec_atmost+1):
            for beta_only in gen_str_iter(list(range(norb)), excite_level):
                alpha_allow = [i for i in range(norb) if i not in beta_only]
                alpha_orbsym = orbsym[alpha_allow]
                alpha_target = target
                for i in beta_only:
                    alpha_target ^= orbsym[i]
                alpha_only = symm.route(alpha_target, spin+excite_level, alpha_orbsym)
                if alpha_only:
                    alpha_only = [alpha_allow[i] for i in alpha_only]
                    return alpha_only, beta_only
        raise RuntimeError('No pattern found for wfn irrep %s over orbsym %s'
                           % (target, orbsym))

    if spin == 0:
        aonly = bonly = []
        if wfnsym != 0:
            aonly, bonly = query(wfnsym, neleca_left, spin, orbsym[orbleft])
    else:
        # 1. assume "nelecb_left" doubly occupied orbitals
        # search for alpha pattern which match to the required symmetry
        aonly, bonly = orbleft[symm.route(wfnsym, spin, orbsym[orbleft])], []
        # dcompose doubly occupied orbitals, search for alpha and beta pattern
        if len(aonly) != spin:
            aonly, bonly = query(wfnsym, neleca_left, spin, orbsym[orbleft])

    ndocc = neleca_left - len(aonly) # == nelecb_left - len(bonly)
    docc_allow = numpy.ones(len(orbleft), dtype=bool)
    docc_allow[aonly] = False
    docc_allow[bonly] = False
    docclst = orbleft[numpy.where(docc_allow)[0]][:ndocc]
    stra[docclst] = True
    strb[docclst] = True

    def find_addr_(stra, aonly, nelec):
        stra[orbleft[aonly]] = True
        return cistring.str2addr(norb, nelec, ('%i'*norb)%tuple(stra)[::-1])
    if bonly:
        if spin > 0:
            aonly, socc_only = aonly[:-spin], aonly[-spin:]
            stra[orbleft[socc_only]] = True
        stra1 = stra.copy()
        strb1 = strb.copy()

        addra = find_addr_(stra, aonly, neleca)
        addrb = find_addr_(strb, bonly, nelecb)
        addra1 = find_addr_(stra1, bonly, neleca)
        addrb1 = find_addr_(strb1, aonly, nelecb)
        ci1[addra,addrb] = ci1[addra1,addrb1] = numpy.sqrt(.5)
    else:
        addra = find_addr_(stra, aonly, neleca)
        addrb = find_addr_(strb, bonly, nelecb)
        ci1[addra,addrb] = 1

#    target = 0
#    for i,k in enumerate(stra):
#        if k:
#            target ^= orbsym[i]
#    for i,k in enumerate(strb):
#        if k:
#            target ^= orbsym[i]
#    print target
    return ci1


def cylindrical_init_guess(mol, norb, nelec, orbsym, wfnsym=0, singlet=True,
                           nroots=1):
    '''
    FCI initial guess for system of cylindrical symmetry.
    (In testing)

    Examples:

    >>> mol = gto.M(atom='O; O 1 1.2', spin=2, symmetry=True)
    >>> orbsym = [6,7,2,3]
    >>> ci0 = fci.addons.cylindrical_init_guess(mol, 4, (3,3), orbsym, wfnsym=10)[0]
    >>> print(ci0.reshape(4,4))
    >>> ci0 = fci.addons.cylindrical_init_guess(mol, 4, (3,3), orbsym, wfnsym=10, singlet=False)[0]
    >>> print(ci0.reshape(4,4))
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    if isinstance(orbsym[0], str):
        orbsym = [symm.irrep_name2id(mol.groupname, x) for x in orbsym]
    orbsym = numpy.asarray(orbsym)
    if isinstance(wfnsym, str):
        wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)

    if mol.groupname in ('Dooh', 'Coov'):
        def irrep_id2lz(irrep_id):
            # See also symm.basis.DOOH_IRREP_ID_TABLE
            level = irrep_id // 10
            d2h_id = irrep_id % 10
            # irrep_id 0,1,4,5 corresponds to lz = 0,2,4,...
            # irrep_id 2,3,6,7 corresponds to lz = 1,3,5,...
            lz = level * 2 + ((d2h_id==2) | (d2h_id==3) | (d2h_id==6) | (d2h_id==7))

            if isinstance(irrep_id, (int, numpy.number)):
                # irrep_id 1,3,4,6 corresponds to E_y (E_{(-)})
                # irrep_id 0,2,5,7 corresponds to E_x (E_{(+)})
                if (d2h_id==1) | (d2h_id==3) | (d2h_id==4) | (d2h_id==6):
                    lz = -lz
            else:
                lz[(d2h_id==1) | (d2h_id==3) | (d2h_id==4) | (d2h_id==6)] *= -1
            return lz

        orb_lz = irrep_id2lz(orbsym)
        wfn_lz = irrep_id2lz(wfnsym)
        d2h_wfnsym_id = wfnsym % 10
    else:
        raise NotImplementedError
        orb_lz = wfn_lz = d2h_wfnsym_id = None

    occslsta = occslstb = cistring._gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring._gen_occslst(range(norb), nelecb)
    na = len(occslsta)
    nb = len(occslsta)

    gx_mask = orbsym == 2
    gy_mask = orbsym == 3
    ux_mask = orbsym == 7
    uy_mask = orbsym == 6
    all_lz = set(abs(orb_lz))
    def search_open_shell_det(occ_lst):
        occ_mask = numpy.zeros(norb, dtype=bool)
        occ_mask[occ_lst] = True

        # First search Lz of the open-shell orbital
        for lz_open in all_lz:
            if numpy.count_nonzero(orb_lz == lz_open) % 2 == 1:
                break

        n_gx = numpy.count_nonzero(gx_mask & occ_mask & (orb_lz == lz_open))
        n_gy = numpy.count_nonzero(gy_mask & occ_mask & (orb_lz ==-lz_open))
        n_ux = numpy.count_nonzero(ux_mask & occ_mask & (orb_lz == lz_open))
        n_uy = numpy.count_nonzero(uy_mask & occ_mask & (orb_lz ==-lz_open))
        if n_gx > n_gy:
            idx = numpy.where(occ_mask    & (orb_lz == lz_open) & gx_mask)[0][0]
            idy = numpy.where((~occ_mask) & (orb_lz ==-lz_open) & gy_mask)[0][0]
        elif n_gx < n_gy:
            idx = numpy.where((~occ_mask) & (orb_lz == lz_open) & gx_mask)[0][0]
            idy = numpy.where(occ_mask    & (orb_lz ==-lz_open) & gy_mask)[0][0]
        elif n_ux > n_uy:
            idx = numpy.where(occ_mask    & (orb_lz == lz_open) & ux_mask)[0][0]
            idy = numpy.where((~occ_mask) & (orb_lz ==-lz_open) & uy_mask)[0][0]
        elif n_ux < n_uy:
            idx = numpy.where((~occ_mask) & (orb_lz == lz_open) & ux_mask)[0][0]
            idy = numpy.where(occ_mask    & (orb_lz ==-lz_open) & uy_mask)[0][0]
        else:
            raise RuntimeError

        nelec = len(occ_lst)
        det_x = occ_mask.copy()
        det_x[idx] = True
        det_x[idy] = False
        str_x = ''.join(['1' if i else '0' for i in det_x[::-1]])
        addr_x = cistring.str2addr(norb, nelec, str_x)
        det_y = occ_mask.copy()
        det_y[idx] = False
        det_y[idy] = True
        str_y = ''.join(['1' if i else '0' for i in det_y[::-1]])
        addr_y = cistring.str2addr(norb, nelec, str_y)
        return addr_x, addr_y

    ci0 = []
    iroot = 0
    for addr in range(na*nb):
        ci_1 = numpy.zeros((na,nb))
        addra = addr // nb
        addrb = addr % nb
        occa = occslsta[addra]
        occb = occslstb[addrb]
        tot_sym = 0
        for i in occa:
            tot_sym ^= orbsym[i]
        for i in occb:
            tot_sym ^= orbsym[i]
        if tot_sym == d2h_wfnsym_id:
            n_Ex_a = (gx_mask[occa]).sum() + (ux_mask[occa]).sum()
            n_Ey_a = (gy_mask[occa]).sum() + (uy_mask[occa]).sum()
            n_Ex_b = (gx_mask[occb]).sum() + (ux_mask[occb]).sum()
            n_Ey_b = (gy_mask[occb]).sum() + (uy_mask[occb]).sum()
            if abs(n_Ex_a - n_Ey_a) == 1 and abs(n_Ex_b - n_Ey_b) == 1:
                # open-shell for both alpha det and beta det e.g. the
                # valence part of O2 molecule

                addr_x_a, addr_y_a = search_open_shell_det(occa)
                addr_x_b, addr_y_b = search_open_shell_det(occb)
                if singlet:
                    if wfn_lz == 0:
                        ci_1[addr_x_a,addr_x_b] = \
                        ci_1[addr_y_a,addr_y_b] = numpy.sqrt(.5)
                    else:
                        ci_1[addr_x_a,addr_x_b] = numpy.sqrt(.5)
                        ci_1[addr_y_a,addr_y_b] =-numpy.sqrt(.5)
                else:
                    ci_1[addr_x_a,addr_y_b] = numpy.sqrt(.5)
                    ci_1[addr_y_a,addr_x_b] =-numpy.sqrt(.5)
            else:
                # TODO: Other direct-product to direct-sum transofromation
                # which involves CG coefficients.
                ci_1[addra,addrb] = 1
            ci0.append(ci_1.ravel())
            iroot += 1
            if iroot >= nroots:
                break

    return ci0


def _symmetrize_wfn(ci, strsa, strsb, orbsym, wfnsym=0):
    ci = ci.reshape(strsa.size,strsb.size)
    airreps = numpy.zeros(strsa.size, dtype=numpy.int32)
    birreps = numpy.zeros(strsb.size, dtype=numpy.int32)
    for i, ir in enumerate(orbsym):
        airreps[numpy.bitwise_and(strsa, 1<<i) > 0] ^= ir
        birreps[numpy.bitwise_and(strsb, 1<<i) > 0] ^= ir
    mask = (airreps.reshape(-1,1) ^ birreps) == wfnsym
    ci1 = numpy.zeros_like(ci)
    ci1[mask] = ci[mask]
    ci1 *= 1/numpy.linalg.norm(ci1)
    return ci1
def symmetrize_wfn(ci, norb, nelec, orbsym, wfnsym=0):
    '''Symmetrize the CI wavefunction by zeroing out the determinants which
    do not have the right symmetry.

    Args:
        ci : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        nelec : int or 2-item list
            Number of electrons, or 2-item list for (alpha, beta) electrons
        orbsym : list of int
            The irrep ID for each orbital.

    Kwags:
        wfnsym : int
            The irrep ID of target symmetry

    Returns:
        2D array which is the symmetrized CI coefficients
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.make_strings(range(norb), neleca))
    strsb = numpy.asarray(cistring.make_strings(range(norb), nelecb))
    return _symmetrize_wfn(ci, strsa, strsb, orbsym, wfnsym)

def _guess_wfnsym(ci, strsa, strsb, orbsym):
    na = len(strsa)
    nb = len(strsb)
    if isinstance(ci, numpy.ndarray) and ci.ndim <= 2:
        assert(ci.size == na*nb)
        idx = numpy.argmax(abs(ci))
    else:
        assert(ci[0].size == na*nb)
        idx = ci[0].argmax()
    stra = strsa[idx // nb]
    strb = strsb[idx % nb ]

    orbsym = numpy.asarray(orbsym) % 10  # convert to D2h irreps
    airrep = 0
    birrep = 0
    for i, ir in enumerate(orbsym):
        if (stra & (1<<i)):
            airrep ^= ir
        if (strb & (1<<i)):
            birrep ^= ir
    return airrep ^ birrep
def guess_wfnsym(ci, norb, nelec, orbsym):
    '''Guess the wavefunction symmetry based on the non-zero elements in the
    given CI coefficients.

    Args:
        ci : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        nelec : int or 2-item list
            Number of electrons, or 2-item list for (alpha, beta) electrons
        orbsym : list of int
            The irrep ID for each orbital.

    Returns:
        Irrep ID
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.make_strings(range(norb), neleca))
    strsb = numpy.asarray(cistring.make_strings(range(norb), nelecb))
    return _guess_wfnsym(ci, strsa, strsb, orbsym)


def des_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing an alpha electron from
    the N-electron wavefunction.

    ... math::

        |N-1\rangle = \hat{a}_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings.  Note it
        has different number of rows to the input CI coefficients
    '''
    neleca, nelecb = neleca_nelecb
    if neleca <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca-1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]))

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3]
    #print(addr_ci0)
    #print(addr_ci1)
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

def des_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N-1)-electron wavefunction by removing a beta electron from
    N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the annihilation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb <= 0:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    des_index = cistring.gen_des_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb-1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1))

    entry_has_ap = (des_index[:,:,1] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = des_index[entry_has_ap,2]
    sign = des_index[entry_has_ap,3]
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

def cre_a(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding an alpha electron in
    the N-electron wavefunction.

    ... math::

        |N+1\rangle = \hat{a}^+_p |N\rangle

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of rows to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if neleca >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), neleca)
    na_ci1 = cistring.num_strings(norb, neleca+1)
    ci1 = numpy.zeros((na_ci1, ci0.shape[1]))

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3]
    ci1[addr_ci1] = sign.reshape(-1,1) * ci0[addr_ci0]
    return ci1

# construct (N+1)-electron wavefunction by adding a beta electron to
# N-electron wavefunction:
def cre_b(ci0, norb, neleca_nelecb, ap_id):
    r'''Construct (N+1)-electron wavefunction by adding a beta electron in
    the N-electron wavefunction.

    Args:
        ci0 : 2D array
            CI coefficients, row for alpha strings and column for beta strings.
        norb : int
            Number of orbitals.
        (neleca,nelecb) : (int,int)
            Number of (alpha, beta) electrons of the input CI function
        ap_id : int
            Orbital index (0-based), for the creation operator

    Returns:
        2D array, row for alpha strings and column for beta strings. Note it
        has different number of columns to the input CI coefficients.
    '''
    neleca, nelecb = neleca_nelecb
    if nelecb >= norb:
        return numpy.zeros_like(ci0)
    if ci0.ndim == 1:
        ci0 = ci0.reshape(cistring.num_strings(norb, neleca),
                          cistring.num_strings(norb, nelecb))
    cre_index = cistring.gen_cre_str_index(range(norb), nelecb)
    nb_ci1 = cistring.num_strings(norb, nelecb+1)
    ci1 = numpy.zeros((ci0.shape[0], nb_ci1))

    entry_has_ap = (cre_index[:,:,0] == ap_id)
    addr_ci0 = numpy.any(entry_has_ap, axis=1)
    addr_ci1 = cre_index[entry_has_ap,2]
    sign = cre_index[entry_has_ap,3]
    # This sign prefactor accounts for interchange of operators with alpha and beta spins
    if neleca % 2 == 1:
        sign *= -1
    ci1[:,addr_ci1] = ci0[:,addr_ci0] * sign
    return ci1

def det_overlap(string1, string2, norb, s=None):
    '''Determinants overlap on non-orthogonal one-particle basis'''
    if s is None:  # orthogonal basis with s_ij = delta_ij
        return float(string1 == string2)
    else:
        if isinstance(string1, str):
            nelec = string1.count('1')
            string1 = int(string1, 2)
        else:
            nelec = bin(string1).count('1')
        if isinstance(string2, str):
            assert(string2.count('1') == nelec)
            string2 = int(string2, 2)
        else:
            assert(bin(string2).count('1') == nelec)
        idx1 = [i for i in range(norb) if (1<<i & string1)]
        idx2 = [i for i in range(norb) if (1<<i & string2)]
        s1 = lib.take_2d(s, idx1, idx2)
        return numpy.linalg.det(s1)

def overlap(bra, ket, norb, nelec, s=None):
    '''Overlap between two CI wavefunctions

    Args:
        s : 2D array or a list of 2D array
            The overlap matrix of non-orthogonal one-particle basis
    '''
    if s is not None:
        bra = transform_ci_for_orbital_rotation(bra, norb, nelec, s)
    return numpy.dot(bra.ravel().conj(), ket.ravel())

def fix_spin_(fciobj, shift=PENALTY, ss=None, **kwargs):
    r'''If FCI solver cannot stay on spin eigenfunction, this function can
    add a shift to the states which have wrong spin.

    .. math::

        (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

    Args:
        fciobj : An instance of :class:`FCISolver`

    Kwargs:
        shift : float
            Level shift for states which have different spin
        ss : number
            S^2 expection value == s*(s+1)

    Returns
            A modified FCI object based on fciobj.
    '''
    import types
    from pyscf.fci import spin_op
    from pyscf.fci import direct_spin0
    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    if (not isinstance(fciobj, types.ModuleType)
        and 'contract_2e' in getattr(fciobj, '__dict__', {})):
        del fciobj.contract_2e  # To avoid initialize twice
    old_contract_2e = fciobj.contract_2e
    def contract_2e(eri, fcivec, norb, nelec, link_index=None, **kwargs):
        if isinstance(nelec, (int, numpy.number)):
            sz = (nelec % 2) * .5
        else:
            sz = abs(nelec[0]-nelec[1]) * .5
        if ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = ss_value

        if ss < sz*(sz+1)+.1:
# (S^2-ss)|Psi> to shift state other than the lowest state
            ci1 = fciobj.contract_ss(fcivec, norb, nelec).reshape(fcivec.shape)
            ci1 -= ss * fcivec
        else:
# (S^2-ss)^2|Psi> to shift states except the given spin.
# It still relies on the quality of initial guess
            tmp = fciobj.contract_ss(fcivec, norb, nelec).reshape(fcivec.shape)
            tmp -= ss * fcivec
            ci1 = -ss * tmp
            ci1 += fciobj.contract_ss(tmp, norb, nelec).reshape(fcivec.shape)
            tmp = None
        ci1 *= shift

        ci0 = old_contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)
        ci1 += ci0.reshape(fcivec.shape)
        return ci1

    fciobj.davidson_only = True
    fciobj.contract_2e = contract_2e
    return fciobj
def fix_spin(fciobj, shift=.1, ss=None):
    return fix_spin_(copy.copy(fciobj), shift, ss)

def transform_ci_for_orbital_rotation(ci, norb, nelec, u):
    '''Transform CI coefficients to the representation in new one-particle basis.
    Solving CI problem for Hamiltonian h1, h2 defined in old basis,
    CI_old = fci.kernel(h1, h2, ...)
    Given orbital rotation u, the CI problem can be either solved by
    transforming the Hamiltonian, or transforming the coefficients.
    CI_new = fci.kernel(u^T*h1*u, ...) = transform_ci_for_orbital_rotation(CI_old, u)

    Args:
        u : 2D array or a list of 2D array
            the orbital rotation to transform the old one-particle basis to new
            one-particle basis
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    strsa = numpy.asarray(cistring.make_strings(range(norb), neleca))
    strsb = numpy.asarray(cistring.make_strings(range(norb), nelecb))
    one_particle_strs = numpy.asarray([1<<i for i in range(norb)])
    na = len(strsa)
    nb = len(strsb)
    assert(ci.shape == (na, nb))

    if isinstance(u, numpy.ndarray) and u.ndim == 2:
        ua = ub = u
    else:
        ua, ub = u

    if neleca == 0:
        trans_ci_a = numpy.ones((1,1))
    else:
        # Unitary transformation array trans_ci is the overlap between two sets of CI basis.
        occ_masks = (strsa[:,None] & one_particle_strs) != 0
        trans_ci_a = numpy.zeros((na,na))
        #for i in range(na): # for old basis
        #    for j in range(na):
        #        uij = u[occ_masks[i]][:,occ_masks[j]]
        #        trans_ci_a[i,j] = numpy.linalg.det(uij)
        occ_idx_all_strs = numpy.where(occ_masks)[1].reshape(na,neleca)
        for i in range(na):
            ui = ua[occ_masks[i]].T.copy()
            minors = ui[occ_idx_all_strs]
            trans_ci_a[i,:] = numpy.linalg.det(minors)

    if neleca == nelecb and numpy.allclose(ua, ub):
        trans_ci_b = trans_ci_a
    else:
        if nelecb == 0:
            trans_ci_b = numpy.ones((1,1))
        else:
            occ_masks = (strsb[:,None] & one_particle_strs) != 0
            trans_ci_b = numpy.zeros((nb,nb))
            #for i in range(nb):
            #    for j in range(nb):
            #        uij = u[occ_masks[i]][:,occ_masks[j]]
            #        trans_ci_b[i,j] = numpy.linalg.det(uij)
            occ_idx_all_strs = numpy.where(occ_masks)[1].reshape(nb,nelecb)
            for i in range(nb):
                ui = ub[occ_masks[i]].T.copy()
                minors = ui[occ_idx_all_strs]
                trans_ci_b[i,:] = numpy.linalg.det(minors)

    # Transform old basis to new basis for all alpha-electron excitations
    ci = lib.dot(trans_ci_a.T, ci.reshape(na,nb))
    # Transform old basis to new basis for all beta-electron excitations
    ci = lib.dot(ci.reshape(na,nb), trans_ci_b)
    return ci


def _unpack_nelec(nelec, spin=None):
    if spin is None:
        spin = 0
    else:
        nelec = int(numpy.sum(nelec))
    if isinstance(nelec, (int, numpy.number)):
        nelecb = (nelec-spin)//2
        neleca = nelec - nelecb
        nelec = neleca, nelecb
    return nelec

del(LARGE_CI_TOL, RETURN_STRS, PENALTY)

