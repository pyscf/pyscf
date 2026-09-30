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
ROMP2 for ROHF/ROKS references with conventional 4-center integrals.

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

from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.mp import ump2
from pyscf.mp import mp2 as mp2_base
from pyscf.mp.dfromp2 import semi_canonicalize, e_singles

WITH_T2 = getattr(__config__, 'mp_romp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    '''ROMP2 kernel.  The doubles contribution is computed by the UMP2 kernel
    in the semi-canonical basis; the singles correction is added on top.'''
    e_corr, t2 = ump2.kernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)
    if mp.include_singles:
        mp.e_corr_singles = e_singles(mp, mp.fock_semi)
    else:
        mp.e_corr_singles = 0.
    e_corr = lib.tag_array(e_corr + mp.e_corr_singles,
                           e_corr_ss=e_corr.e_corr_ss,
                           e_corr_os=e_corr.e_corr_os,
                           e_corr_singles=mp.e_corr_singles)
    return e_corr, t2


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

    _keys = ump2.UMP2._keys | {'e_corr_singles', 'include_singles', 'fock_semi'}

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

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    def update_amps(self, t2, eris):
        raise NotImplementedError


from pyscf import scf  # noqa: E402
scf.rohf.ROHF.MP2 = lib.class_as_method(ROMP2)
