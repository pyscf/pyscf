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

import sys
import warnings
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
    assert ci.size == na * nb
    ci = ci.reshape(na, nb)
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
    raise DeprecationWarning


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
    warnings.warn('Initial guess for cylindrical symmetry is under testing')

    neleca, nelecb = _unpack_nelec(nelec)
    if isinstance(orbsym[0], str):
        orbsym = [symm.irrep_name2id(mol.groupname, x) for x in orbsym]
    orbsym = numpy.asarray(orbsym)
    if isinstance(wfnsym, str):
        wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)

    if mol.groupname in ('SO3', 'Dooh', 'Coov'):
        def irrep_id2lz(irrep_id):
            # See also symm.basis.DOOH_IRREP_ID_TABLE
            level = irrep_id // 10
            if mol.groupname == 'SO3':
                level = level % 10  # See SO3 irreps in pyscf.symm.basis
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

    occslsta = occslstb = cistring.gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslstb = cistring.gen_occslst(range(norb), nelecb)
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
                        ci_1[addr_x_a,addr_x_b] = numpy.sqrt(.5)
                        ci_1[addr_y_a,addr_y_b] = numpy.sqrt(.5)
                    else:
                        ci_1[addr_x_a,addr_x_b] = numpy.sqrt(.5)
                        ci_1[addr_y_a,addr_y_b] =-numpy.sqrt(.5)
                else:
                    ci_1[addr_x_a,addr_y_b] = numpy.sqrt(.5)
                    ci_1[addr_y_a,addr_x_b] =-numpy.sqrt(.5)
            else:
                # TODO: Other direct-product to direct-sum transformation
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
    orbsym_in_d2h = numpy.asarray(orbsym) % 10
    wfnsym_in_d2h = wfnsym % 10
    for i, ir in enumerate(orbsym_in_d2h):
        airreps[numpy.bitwise_and(strsa, 1 << i) > 0] ^= ir
        birreps[numpy.bitwise_and(strsb, 1 << i) > 0] ^= ir
    mask = (airreps.reshape(-1,1) ^ birreps) == wfnsym_in_d2h
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
    nb = len(strsb)
    idx = abs(ci).argmax()
    stra = strsa[idx // nb]
    strb = strsb[idx % nb ]

    orbsym_in_d2h = numpy.asarray(orbsym) % 10  # convert to D2h irreps
    airrep = 0
    birrep = 0
    for i, ir in enumerate(orbsym_in_d2h):
        if (stra & (1 << i)):
            airrep ^= ir
        if (strb & (1 << i)):
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
    strsa = strsb = numpy.asarray(cistring.make_strings(range(norb), neleca))
    if neleca != nelecb:
        strsb = numpy.asarray(cistring.make_strings(range(norb), nelecb))
    if isinstance(ci, numpy.ndarray) and ci.ndim <= 2:
        wfnsym = _guess_wfnsym(ci, strsa, strsb, orbsym)
    else:
        wfnsym = [_guess_wfnsym(c, strsa, strsb, orbsym) for c in ci]
        if any(wfnsym[0] != x for x in wfnsym):
            warnings.warn('Different wfnsym %s found in different CI vectors' % wfnsym)
        wfnsym = wfnsym[0]
    return wfnsym


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
            assert (string2.count('1') == nelec)
            string2 = int(string2, 2)
        else:
            assert (bin(string2).count('1') == nelec)
        idx1 = [i for i in range(norb) if (1 << i & string1)]
        idx2 = [i for i in range(norb) if (1 << i & string2)]
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

class SpinPenaltyFCISolver:
    __name_mixin__ = 'SpinPenalty'
    _keys = {'ss_value', 'ss_penalty', 'base'}

    def __init__(self, fcibase, shift, ss_value):
        self.base = fcibase.copy()
        self.__dict__.update (fcibase.__dict__)
        self.ss_value = ss_value
        self.ss_penalty = shift
        self.davidson_only = self.base.davidson_only = True

    def undo_fix_spin(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SpinPenaltyFCISolver))
        del obj.base
        del obj.ss_value
        del obj.ss_penalty
        return obj

    def base_contract_2e (self, *args, **kwargs):
        return super().contract_2e (*args, **kwargs)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        if isinstance(nelec, (int, numpy.number)):
            sz = (nelec % 2) * .5
        else:
            sz = abs(nelec[0]-nelec[1]) * .5
        if self.ss_value is None:
            ss = sz*(sz+1)
        else:
            ss = self.ss_value

        if ss < sz*(sz+1)+.1:
            # (S^2-ss)|Psi> to shift state other than the lowest state
            ci1 = self.contract_ss(fcivec, norb, nelec).reshape(fcivec.shape)
            ci1 -= ss * fcivec
        else:
            # (S^2-ss)^2|Psi> to shift states except the given spin.
            # It still relies on the quality of initial guess
            tmp = self.contract_ss(fcivec, norb, nelec).reshape(fcivec.shape)
            tmp -= ss * fcivec
            ci1 = -ss * tmp
            ci1 += self.contract_ss(tmp, norb, nelec).reshape(fcivec.shape)
            tmp = None
        ci1 *= self.ss_penalty

        ci0 = super().contract_2e (eri, fcivec, norb, nelec, link_index, **kwargs)
        ci1 += ci0.reshape(fcivec.shape)
        return ci1

def fix_spin(fciobj, shift=PENALTY, ss=None, **kwargs):
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
    from pyscf.fci import direct_uhf
    if isinstance(fciobj, direct_uhf.FCISolver):
        raise NotImplementedError

    if isinstance (fciobj, types.ModuleType):
        raise DeprecationWarning('fix_spin should be applied on FCI object only')

    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    if isinstance (fciobj, SpinPenaltyFCISolver):
        # recursion avoidance
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj

    return lib.set_class(SpinPenaltyFCISolver(fciobj, shift, ss_value),
                         (SpinPenaltyFCISolver, fciobj.__class__))

def fix_spin_(fciobj, shift=.1, ss=None):
    sp_fci = fix_spin(fciobj, shift, ss)
    fciobj.__class__ = sp_fci.__class__
    fciobj.__dict__ = sp_fci.__dict__
    return fciobj

def transform_ci_for_orbital_rotation(ci, norb, nelec, u):
    '''
    Transform CI coefficients (dimension conserved) to the representation in
    new one-particle basis.  Solving CI problem for Hamiltonian h1, h2 defined
    in old basis,
    CI_old = fci.kernel(h1, h2, ...)
    Given orbital rotation u, the CI problem can be either solved by
    transforming the Hamiltonian, or transforming the coefficients.
    CI_new = fci.kernel(u^T*h1*u, ...) = transform_ci_for_orbital_rotation(CI_old, u)

    Args:
        u : a squared 2D array or a list of 2D array
            the orbital rotation to transform the old one-particle basis to new
            one-particle basis
    '''
    if isinstance(u, numpy.ndarray) and u.ndim == 2:
        assert u.shape == (norb, norb)
    else:
        assert u[0].shape == (norb, norb) and u[1].shape == (norb, norb)
    return transform_ci(ci, nelec, u)

def transform_ci(ci, nelec, u):
    '''Transform CI coefficients to the representation in new one-particle basis.
    Solving CI problem for Hamiltonian h1, h2 defined in old basis,
    CI_old = fci.kernel(h1, h2, ...)
    Given orbital rotation u, the CI problem can be either solved by
    transforming the Hamiltonian, or transforming the coefficients.
    CI_new = fci.kernel(u^T*h1*u, ...) = transform_ci_for_orbital_rotation(CI_old, u)

    Args:
        u : 2D array or a list of 2D array
            the orbital rotation to transform the old one-particle basis to new
            one-particle basis. If u is not a squared matrix, the resultant CI
            coefficients array may have different shape to the input CI
            coefficients.
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    if isinstance(u, numpy.ndarray) and u.ndim == 2:
        ua = ub = u
        assert ua.shape == ub.shape
    else:
        ua, ub = u
    norb_old, norb_new = ua.shape
    na_old = cistring.num_strings(norb_old, neleca)
    nb_old = cistring.num_strings(norb_old, nelecb)
    na_new = cistring.num_strings(norb_new, neleca)
    nb_new = cistring.num_strings(norb_new, nelecb)
    ci = ci.reshape(na_old, nb_old)

    if neleca == 0:
        trans_ci_a = numpy.ones((1, 1))
    else:
        trans_ci_a = numpy.zeros((na_old, na_new), dtype=ua.dtype)
        occ_masks_old = _init_occ_masks(norb_old, neleca, na_old)
        if norb_old == norb_new:
            occ_masks_new = occ_masks_old
        else:
            occ_masks_new = _init_occ_masks(norb_new, neleca, na_new)

        # Perform
        #for i in range(na_old): # old basis
        #    for j in range(na_new): # new basis
        #        uij = u[occ_masks_old[i]][:,occ_masks_new[j]]
        #        trans_ci_a[i,j] = numpy.linalg.det(uij)
        occ_idx_all_strs = numpy.where(occ_masks_new)[1].reshape(na_new,neleca)
        for i in range(na_old):
            ui = ua[occ_masks_old[i]].T.copy()
            minors = ui[occ_idx_all_strs]
            trans_ci_a[i,:] = numpy.linalg.det(minors)

    if neleca == nelecb and numpy.allclose(ua, ub):
        trans_ci_b = trans_ci_a
    elif nelecb == 0:
        trans_ci_b = numpy.ones((1, 1))
    else:
        trans_ci_b = numpy.zeros((nb_old, nb_new), dtype=ub.dtype)
        occ_masks_old = _init_occ_masks(norb_old, nelecb, nb_old)
        if norb_old == norb_new:
            occ_masks_new = occ_masks_old
        else:
            occ_masks_new = _init_occ_masks(norb_new, nelecb, nb_new)

        occ_idx_all_strs = numpy.where(occ_masks_new)[1].reshape(nb_new,nelecb)
        for i in range(nb_old):
            ui = ub[occ_masks_old[i]].T.copy()
            minors = ui[occ_idx_all_strs]
            trans_ci_b[i,:] = numpy.linalg.det(minors)

    # Transform old basis to new basis for all alpha-electron excitations
    ci = lib.dot(trans_ci_a.T, ci)
    # Transform old basis to new basis for all beta-electron excitations
    ci = lib.dot(ci, trans_ci_b)
    return ci

def civec_spinless_repr_generator(ci0_r, norb, nelec_r):
    '''Put CI vectors in the spinless representation; i.e., map
        norb -> 2 * norb
        (neleca, nelecb) -> (neleca+nelecb, 0)
    This permits linear combinations of CI vectors with different
    M == neleca-nelecb at the price of higher memory cost. This function
    does NOT change the datatype.

    Args:
        ci0_r: sequence or generator of ndarray of length nprods
            CAS-CI vectors in the spin-pure representation
        norb: integer
            Number of orbitals
        nelec_r: sequence of tuple of length (2)
            (neleca, nelecb) for each element of ci0_r

    Returns:
        ci1_r_gen: callable that returns a generator of length nprods
            generates spinless CAS-CI vectors
        ss2spinless: callable
            Put a CAS-CI vector in the spinless representation
            Args:
                ci0: ndarray
                    CAS-CI vector
                ne: tuple of length 2
                    neleca, nelecb of target Hilbert space
            Returns:
                ci1: ndarray
                    spinless CAS-CI vector
        spinless2ss: callable
            Perform the reverse operation on a spinless CAS-CI vector
            Args:
                ci2: ndarray
                    spinless CAS-CI vector
                ne: tuple of length 2
                    neleca, nelecb target Hilbert space
            Returns:
                ci3: ndarray
                    CAS-CI vector of ci2 in the (neleca, nelecb) Hilbert space
    '''
    nelec_r_tot = [sum (n) for n in nelec_r]
    if len(set(nelec_r_tot)) > 1:
        raise NotImplementedError("Different particle-number subspaces")
    nelec = nelec_r_tot[0]
    addrs = {}
    ndet_sp = {}
    for ne in set(nelec_r):
        neleca, nelecb = _unpack_nelec(ne)
        ndeta = cistring.num_strings(norb, neleca)
        ndetb = cistring.num_strings(norb, nelecb)
        strsa = cistring.addrs2str(norb, neleca, list(range(ndeta)))
        strsb = cistring.addrs2str(norb, nelecb, list(range(ndetb)))
        strs = numpy.add.outer(strsa, numpy.left_shift(strsb, norb)).ravel()
        addrs[ne] = cistring.strs2addr(2*norb, nelec, strs)
        ndet_sp[ne] = (ndeta,ndetb)
    strs = strsa = strsb = None
    ndet = cistring.num_strings(2*norb, nelec)
    def ss2spinless(ci0, ne, buf=None):
        if buf is None:
            ci1 = numpy.empty(ndet, dtype=ci0.dtype)
        else:
            ci1 = numpy.asarray(buf).flat[:ndet]
        ci1[:] = 0.0
        ci1[addrs[ne]] = ci0[:,:].ravel ()
        neleca, nelecb = _unpack_nelec (ne)
        if abs(neleca*nelecb)%2: ci1[:] *= -1
        # Sign comes from changing representation:
        # ... a2' a1' a0' ... b2' b1' b0' |vac>
        # ->
        # ... b2' b1' b0' .. a2' a1' a0' |vac>
        # i.e., strictly decreasing from left to right
        # (the ordinality of spin-down is conventionally greater than spin-up)
        return ci1[:,None]
    def spinless2ss(ci2, ne):
        ''' Generate the spin-separated CI vector in a particular M
        Hilbert space from a spinless CI vector '''
        ci3 = ci2[addrs[ne]].reshape(ndet_sp[ne])
        neleca, nelecb = _unpack_nelec (ne)
        if abs(neleca*nelecb)%2: ci3[:] *= -1
        return ci3
    def ci1_r_gen(buf=None):
        if callable(ci0_r):
            ci0_r_gen = ci0_r()
        else:
            ci0_r_gen = (c for c in ci0_r)
        for ci0, ne in zip(ci0_r_gen, nelec_r):
            # Doing this in two lines saves memory: ci0 is overwritten
            ci0 = ss2spinless(ci0, ne)
            yield ci0
    return ci1_r_gen, ss2spinless, spinless2ss

def civec_spinless_repr(ci0_r, norb, nelec_r):
    '''Put CI vectors in the spinless representation; i.e., map
        norb -> 2 * norb
        (neleca, nelecb) -> (neleca+nelecb, 0)
    This permits linear combinations of CI vectors with different
    M == neleca-nelecb at the price of higher memory cost. This function
    does NOT change the datatype.

    Args:
        ci0_r: sequence or generator of ndarray of length nprods
            CAS-CI vectors in the spin-pure representation
        norb: integer
            Number of orbitals
        nelec_r: sequence of tuple of length (2)
            (neleca, nelecb) for each element of ci0_r

    Returns:
        ci1_r: ndarray of shape (nprods, ndet_spinless)
            spinless CAS-CI vectors
    '''
    ci1_r_gen, *_ = civec_spinless_repr_generator(ci0_r, norb, nelec_r)
    ci1_r = numpy.stack([x.copy() for x in ci1_r_gen()], axis=0)
    return ci1_r


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

def _init_occ_masks(norb, nelec, nci):
    one_particle_strs = numpy.asarray(cistring.make_strings(range(norb), 1))
    strs = numpy.asarray(cistring.make_strings(range(norb), nelec))
    if norb < 64:
        occ_masks = (strs[:,None] & one_particle_strs) != 0
    else:
        occ_masks = numpy.zeros((nci, norb), dtype=bool)
        for i in range(nci):
            for j in range(norb):
                if one_particle_strs[j][0] in strs[i]:
                    occ_masks[i,j] = True
    return occ_masks

del (LARGE_CI_TOL, RETURN_STRS, PENALTY)
