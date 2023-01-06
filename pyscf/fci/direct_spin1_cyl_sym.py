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

import numpy
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

def get_init_guess(norb, nelec, nroots, hdiag, orbsym, wfnsym=0):
    neleca, nelecb = direct_spin1._unpack_nelec(nelec)
    strsa = strsb = cistring.gen_strings4orblist(range(norb), neleca)
    airreps_d2h = birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsa, orbsym)
    a_ls = b_ls = _strs_angular_momentum(strsa, orbsym)
    if neleca != nelecb:
        strsb = cistring.gen_strings4orblist(range(norb), nelecb)
        birreps_d2h = direct_spin1_symm._gen_strs_irrep(strsb, orbsym)
        b_ls = _strs_angular_momentum(strsb, orbsym)

    wfnsym_in_d2h = wfnsym % 10
    wfn_momentum = symm.basis.linearmole_irrep2momentum(wfnsym)
    na = len(strsa)
    nb = len(strsb)
    hdiag = hdiag.reshape(na,nb)
    degen = orbsym.degen_mapping
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
        x = numpy.zeros((na, nb))
        x[addra, addrb] = 1.
        if wfnsym in (0, 1, 4, 5):
            addra1, sign_a = _sv_associated_det(strsa[addra], degen)
            addrb1, sign_b = _sv_associated_det(strsb[addrb], degen)
            # If (E+) and (E-) are associated determinants
            # (E+)(E-') + (E-)(E+') => A1
            # (E+)(E-') - (E-)(E+') => A2
            if wfnsym in (0, 5):  # A1g, A1u
                # ensure <|sigma_v|> = 1
                x[addra1,addrb1] += sign_a * sign_b
            elif wfnsym in (1, 4):  # A2g, A2u
                # ensure <|sigma_v|> = -1
                x[addra1,addrb1] -= sign_a * sign_b

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

        if not isinstance(fcivec, numpy.ndarray) or fcivec.ndim > 2:
            fcivec = fcivec[0]
        wfnsym1 = _guess_wfnsym(fcivec, strsa, strsb, orbsym)

        if (wfnsym is not None and
            wfnsym1 != direct_spin1_symm._id_wfnsym(solver, norb, nelec, orbsym, wfnsym)):
            raise RuntimeError(f'Input wfnsym {wfnsym} is not consistent with '
                               f'fcivec symmetry {wfnsym1}')
        wfnsym = wfnsym1
    return wfnsym


class FCISolver(direct_spin1_symm.FCISolver):

    def contract_1e(self, f1e, fcivec, norb, nelec, link_index=None, **kwargs):
        return direct_nosym.contract_1e(f1e, fcivec, norb, nelec, link_index)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None,
                    orbsym=None, wfnsym=None, **kwargs):
        return direct_nosym.contract_2e(eri, fcivec, norb, nelec, link_index)

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        wfnsym = direct_spin1_symm._id_wfnsym(self, norb, nelec, self.orbsym, self.wfnsym)
        return get_init_guess(norb, nelec, nroots, hdiag, self.orbsym, wfnsym)

    absorb_h1e = direct_nosym.FCISolver.absorb_h1e
    make_hdiag = direct_nosym.FCISolver.make_hdiag
    pspace = direct_spin1.FCISolver.pspace
    guess_wfnsym = guess_wfnsym

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, reorder=True):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        dm1, dm2 = direct_spin1.make_rdm12(fcivec, norb, nelec, link_index, reorder)
        orbsym = self.orbsym
        degen_mapping = self.orbsym.degen_mapping
        u = _cyl_sym_orbital_rotation(orbsym, degen_mapping)
        dm1 = u.conj().T.dot(dm1).dot(u)
        dm2 = lib.einsum('pqrs,pi,qj,rk,sl->ijkl', dm2, u.conj(), u, u.conj(), u)
        return dm1.real.copy(), dm2.real.copy()

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

        if not hasattr(orbsym, 'degen_mapping'):
            degen_mapping = map_degeneracy(h1e.diagonal(), orbsym)
            orbsym = lib.tag_array(orbsym, degen_mapping=degen_mapping)
        self.orbsym = orbsym
        u = _cyl_sym_orbital_rotation(orbsym, orbsym.degen_mapping)
        wfnsym = self.guess_wfnsym(norb, nelec, ci0, orbsym, wfnsym, **kwargs)

        h1e = u.dot(h1e).dot(u.conj().T)
        eri = ao2mo.restore(1, eri, norb)
        eri = lib.einsum('pqrs,ip,jq,kr,ls->ijkl', eri, u, u.conj(), u, u.conj())
        assert abs(h1e.imag).max() < 1e-12, 'Cylindrical symmetry broken'
        assert abs(eri.imag).max() < 1e-12, 'Cylindrical symmetry broken'
        h1e = h1e.real.copy()
        # Note: although eri is real, it does not have the permutation relation
        # (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # The nosym version fci contraction is required
        eri = eri.real.copy()

        neleca, nelecb = direct_spin1._unpack_nelec(nelec)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        with lib.temporary_env(self, orbsym=orbsym, wfnsym=wfnsym):
            e, c = direct_spin1.kernel_ms1(self, h1e, eri, norb, nelec, ci0,
                                           (link_indexa,link_indexb),
                                           tol, lindep, max_cycle, max_space,
                                           nroots, davidson_only, pspace_size,
                                           ecore=ecore, **kwargs)

        self.eci, self.ci = e, c
        return e, c

FCI = FCISolver
