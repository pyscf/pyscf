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
ROMP2 for ROHF/ROKS references.

This module provides the common semi-canonical ROMP2 machinery (used by
both the conventional 4-center ROMP2 and the density-fitted DF-ROMP2 in
pyscf.mp.dfromp2):

- semi_canonicalize: semi-canonicalization of the ROHF orbitals
  (alpha Fock diagonalized in the doubly+singly occupied and virtual blocks,
  beta Fock in the doubly occupied and singly+virtual blocks), plus the
  spin Fock matrices in the semi-canonical basis.
- e_singles: the second-order singles correction
  E_singles = sum_s sum_{i in occ_s, a in vir_s} -F^s_ai^2 / (e_a - e_i)
  where the beta virtual space includes the singly occupied orbitals.
- ROMP2: the class using conventional 4-center integrals (doubles from the
  UMP2 kernel evaluated with the semi-canonical orbitals).

See P. J. Knowles, J. S. Andrews, R. D. Amos, N. C. Handy and J. A. Pople,
Chem. Phys. Lett. 186, 130 (1991).
'''

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp import ump2
from pyscf.mp import mp2 as mp2_base

WITH_T2 = getattr(__config__, 'mp_romp2_with_t2', True)


def semi_canonicalize(mf, verbose=None):
    '''Semi-canonicalize the ROHF/ROKS orbitals (Knowles et al., CPL 186, 130).

    The alpha spin Fock matrix is diagonalized within the doubly+singly
    occupied and the virtual blocks; the beta spin Fock matrix is
    diagonalized within the doubly occupied and the singly+virtual blocks.

    Args:
        mf : an ROHF (or ROKS) object with converged orbitals

    Returns:
        A tuple of (mo_coeff, mo_energy, mo_occ, fock) in UHF format.
        mo_coeff and mo_energy are the semi-canonical orbitals and the
        corresponding orbital energies (diagonal of the semi-canonical Fock
        matrices).  fock are the spin Fock matrices in the semi-canonical
        basis, in which the occupied-virtual blocks are retained for the
        singles correction.
    '''
    log = logger.new_logger(mf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    hcore = mf.get_hcore()
    fock = [np.asarray(hcore + vhf[0]), np.asarray(hcore + vhf[1])]

    occidx_a = mo_occ > 0  # doubly + singly occupied
    occidx_b = mo_occ == 2  # doubly occupied
    mo_coeff_semi = []
    mo_energy_semi = []
    fock_semi = []
    for s, occidx in enumerate([occidx_a, occidx_b]):
        viridx = ~occidx
        f = fock[s]
        c_occ = mo_coeff[:, occidx]
        c_vir = mo_coeff[:, viridx]
        f_oo = reduce(lib.dot, (c_occ.conj().T, f, c_occ))
        f_vv = reduce(lib.dot, (c_vir.conj().T, f, c_vir))
        # eigh on an empty block is not portable across LAPACK versions
        if c_occ.shape[1] > 0:
            _, uo = scipy.linalg.eigh(f_oo)
            c_occ = np.dot(c_occ, uo)
        if c_vir.shape[1] > 0:
            _, uv = scipy.linalg.eigh(f_vv)
            c_vir = np.dot(c_vir, uv)
        c_semi = np.hstack((c_occ, c_vir))
        f_semi = reduce(lib.dot, (c_semi.conj().T, f, c_semi))
        mo_coeff_semi.append(c_semi)
        mo_energy_semi.append(np.ascontiguousarray(f_semi.diagonal().real))
        fock_semi.append(f_semi)
        log.debug('semi-canonicalization for spin %d diagonalized blocks '
                  '%d/%d (occ) and %d/%d (vir)', s, occidx.sum(), mo_occ.size,
                  viridx.sum(), mo_occ.size)

    mo_coeff_semi = lib.tag_array(np.asarray(mo_coeff_semi), mo_ea=mo_energy_semi[0],
                                  mo_eb=mo_energy_semi[1])
    mo_occ_semi = np.asarray([np.where(occidx_a, 1., 0.),
                              np.where(occidx_b, 1., 0.)])
    mo_energy_semi = np.asarray(mo_energy_semi)
    log.timer('semi-canonicalization', *cput0)
    return mo_coeff_semi, mo_energy_semi, mo_occ_semi, fock_semi


def singles_amps(mp, fock_semi=None):
    '''First-order singles amplitudes of the semi-canonical ROMP2,

        t1_ia = F^s_ai / (e_i - e_a)      (i in occ_s, a in vir_s)

    per spin, in UHF format and restricted to the non-frozen active space
    (the beta virtual space includes the singly occupied orbitals).'''
    if fock_semi is None:
        fock_semi = mp.fock_semi
    mask_frozen = mp.get_frozen_mask()
    t1 = []
    for s in [0, 1]:
        occidx = (mp.mo_occ[s] > 1e-6) & mask_frozen[s]
        viridx = (mp.mo_occ[s] <= 1e-6) & mask_frozen[s]
        f_ai = fock_semi[s][np.ix_(viridx, occidx)]
        denom = mp.mo_energy[s][viridx][:, None] - mp.mo_energy[s][occidx]
        t1.append((-f_ai / denom).T)
    return tuple(t1)


def e_singles(mp, fock_semi=None):
    '''Second-order singles correction of the semi-canonical ROMP2.

    E_singles = sum_s sum_{i in occ_s, a in vir_s} -F^s_ai^2 / (e_a - e_i)

    where occ_s/vir_s are the occupied/virtual spaces of the spin-resolved
    Fock matrices (the beta virtual space includes the singly occupied
    orbitals).  Frozen orbitals are excluded.
    '''
    if fock_semi is None:
        fock_semi = mp.fock_semi
    mask_frozen = mp.get_frozen_mask()
    e = 0.
    for s, t1 in enumerate(singles_amps(mp, fock_semi)):
        occidx = (mp.mo_occ[s] > 1e-6) & mask_frozen[s]
        viridx = (mp.mo_occ[s] <= 1e-6) & mask_frozen[s]
        f_ai = fock_semi[s][np.ix_(viridx, occidx)]
        e += np.einsum('ai,ia->', f_ai, t1).real
    return e


def add_singles(mp, e_corr):
    '''Add the second-order singles correction to the doubles correlation
    energy and tag the result with the correlation energy components.  The T1
    amplitudes are stored on ``mp.t1`` so that they enter the RDMs.'''
    if mp.include_singles:
        mp.e_corr_singles = e_singles(mp, mp.fock_semi)
        mp.t1 = singles_amps(mp, mp.fock_semi)
    else:
        mp.e_corr_singles = 0.
        mp.t1 = None
    return lib.tag_array(e_corr + mp.e_corr_singles,
                         e_corr_ss=e_corr.e_corr_ss,
                         e_corr_os=e_corr.e_corr_os,
                         e_corr_singles=mp.e_corr_singles)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    '''ROMP2 kernel.  The doubles contribution is computed by the UMP2 kernel
    in the semi-canonical basis; the singles correction is added on top.'''
    e_corr, t2 = ump2.kernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)
    return add_singles(mp, e_corr), t2


class ROMP2(ump2.UMP2):
    '''ROMP2 with semi-canonicalized ROHF orbitals and 4-center integrals

    Saved results

        e_corr : float
            ROMP2 correlation correction (doubles + singles)
        e_corr_ss/os : float
            Same-spin and opposite-spin component of the doubles correlation
            energy (the singles correction is not included in either component)
        e_corr_singles : float
            Second-order singles correction of the semi-canonical formalism
        e_tot : float
            Total ROMP2 energy (HF + correlation)
        t2 :
            T amplitudes t2[i,j,a,b]  (i,j in occ, a,b in virt)

    Attributes:
        include_singles : bool
            Whether to include the second-order singles correction.
            Default value is True.
    '''

    _keys = ump2.UMP2._keys | {'e_corr_singles', 'include_singles', 'fock_semi',
                               'mo_energy'}

    include_singles = getattr(__config__, 'mp_romp2_include_singles', True)

    def __init__(self, mf, frozen=None):
        if not mf.istype('ROHF'):
            raise RuntimeError('ROMP2 requires an ROHF (or ROKS) reference. '
                               'For UHF references, use UMP2 (mp.UMP2).')
        mp2_base.MP2Base.__init__(self, mf, frozen)
        self.mo_coeff, self.mo_energy, self.mo_occ, self.fock_semi = \
            semi_canonicalize(self._scf)

    def get_e_hf(self, mo_coeff=None):
        '''HF energy of the ROHF reference. The semi-canonicalization only
        rotates orbitals within the occupied/virtual blocks, which does not
        change the HF energy.'''
        return self._scf.e_tot

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        # ump2._make_eris/_ChemistsERIs expects a UHF-like mean-field object
        # for the UHF-format semi-canonical orbitals. A temporary view of the
        # ROHF object as UHF is used here; the original object is untouched.
        mf_uhf = self._scf.view(scf.uhf.UHF)
        mf_uhf.mo_coeff = mo_coeff
        mf_uhf.mo_energy = self.mo_energy
        mf_uhf.mo_occ = self.mo_occ
        scf_saved = self._scf
        self._scf = mf_uhf
        try:
            eris = ump2._make_eris(self, mo_coeff, verbose=self.verbose)
        finally:
            self._scf = scf_saved
        return eris

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None,
               with_t2=WITH_T2):
        if mo_coeff is not None or mo_energy is not None:
            logger.warn(self, 'ROMP2 ignores the given mo_coeff/mo_energy. '
                              'The semi-canonicalized ROHF orbitals are used.')
        self.mo_coeff, self.mo_energy, self.mo_occ, self.fock_semi = \
            semi_canonicalize(self._scf)
        return mp2_base.MP2Base.kernel(self, eris=eris, with_t2=with_t2)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None,
                  with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfromp2
        mymp = dfromp2.DFROMP2(self._scf, self.frozen)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = mymp.with_df.copy()
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError


from pyscf import scf  # noqa: E402
scf.rohf.ROHF.MP2 = lib.class_as_method(ROMP2)
