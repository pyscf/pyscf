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
density fitting ROMP2 for ROHF/ROKS references, 3-center integrals incore.

Semi-canonical ROMP2: the ROHF canonical orbitals are semi-canonicalized by
diagonalizing the alpha/beta Fock matrices within the occupied/virtual
subspaces (alpha spin in the doubly+singly occupied and virtual blocks, beta
spin in the doubly occupied and singly+virtual blocks).  The MP2 correlation
energy is the UMP2 doubles energy evaluated in this semi-canonical basis plus
a second-order singles correction:

    E_singles = sum_s sum_{i in occ_s, a in vir_s} -F^s_ai^2 / (e_a - e_i)

where the beta virtual space includes the singly occupied orbitals.
See P. J. Knowles, J. S. Andrews, R. D. Amos, N. C. Handy and J. A. Pople,
Chem. Phys. Lett. 186, 130 (1991).
'''

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp import dfump2
from pyscf.mp import mp2 as mp2_base
from pyscf.mp.ump2 import _mo_splitter  # noqa: F401

WITH_T2 = getattr(__config__, 'mp_dfromp2_with_t2', True)


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
        _, uo = scipy.linalg.eigh(f_oo)
        _, uv = scipy.linalg.eigh(f_vv)
        c_semi = np.hstack((np.dot(c_occ, uo), np.dot(c_vir, uv)))
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
    for s in [0, 1]:
        occidx = (mp.mo_occ[s] > 1e-6) & mask_frozen[s]
        viridx = (mp.mo_occ[s] <= 1e-6) & mask_frozen[s]
        f_ai = fock_semi[s][viridx][:, occidx]
        denom = (mp.mo_energy[s][viridx][:, None] - mp.mo_energy[s][occidx])
        e += np.einsum('ai,ai->', f_ai.conj() * f_ai, -1. / denom).real
    return e


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    '''DF-ROMP2 kernel.  The doubles contribution is computed by the DF-UMP2
    kernel in the semi-canonical basis; the singles correction is added on
    top.'''
    e_corr, t2 = dfump2.kernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)
    if mp.include_singles:
        mp.e_corr_singles = e_singles(mp, mp.fock_semi)
    else:
        mp.e_corr_singles = 0.
    e_corr = lib.tag_array(e_corr + mp.e_corr_singles,
                           e_corr_ss=e_corr.e_corr_ss,
                           e_corr_os=e_corr.e_corr_os,
                           e_corr_singles=mp.e_corr_singles)
    return e_corr, t2


class DFROMP2(dfump2.DFUMP2):
    '''density-fitted ROMP2 with semi-canonicalized ROHF orbitals

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

    _keys = dfump2.DFUMP2._keys | {'e_corr_singles', 'include_singles', 'fock_semi'}

    include_singles = getattr(__config__, 'mp_dfromp2_include_singles', True)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if not mf.istype('ROHF'):
            raise RuntimeError('DF-ROMP2 requires an ROHF (or ROKS) reference. '
                               'For UHF references, use DF-UMP2 (mp.DFUMP2).')
        if mo_coeff is not None or mo_occ is not None:
            raise NotImplementedError('DF-ROMP2 does not support custom '
                                      'mo_coeff/mo_occ. The ROHF orbitals are '
                                      'always semi-canonicalized.')
        dfump2.DFUMP2.__init__(self, mf, frozen)
        self.mo_coeff, self.mo_energy, self.mo_occ, self.fock_semi = \
            semi_canonicalize(self._scf)

    def get_e_hf(self, mo_coeff=None):
        '''HF energy of the ROHF reference. The semi-canonicalization only
        rotates orbitals within the occupied/virtual blocks, which does not
        change the HF energy.'''
        return self._scf.e_tot

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None,
               with_t2=WITH_T2):
        if mo_coeff is not None or mo_energy is not None:
            logger.warn(self, 'DF-ROMP2 ignores the given mo_coeff/mo_energy. '
                              'The semi-canonicalized ROHF orbitals are used.')
        self.mo_coeff, self.mo_energy, self.mo_occ, self.fock_semi = \
            semi_canonicalize(self._scf)
        return mp2_base.MP2Base.kernel(self, eris=eris, with_t2=with_t2)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None,
                  with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError


from pyscf import scf  # noqa: E402
scf.rohf.ROHF.DFROMP2 = lib.class_as_method(DFROMP2)
scf.rohf.ROHF.DFMP2 = lib.class_as_method(DFROMP2)
