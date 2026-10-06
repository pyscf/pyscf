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

DF-ROMP2 is the density-fitted counterpart of the conventional 4-center
semi-canonical ROMP2 (pyscf.mp.romp2).  The doubles contribution is computed
by the DF-UMP2 kernel evaluated with the semi-canonical orbitals; the
second-order singles correction is added on top.  See pyscf.mp.romp2 and
P. J. Knowles, J. S. Andrews, R. D. Amos, N. C. Handy and J. A. Pople,
Chem. Phys. Lett. 186, 130 (1991).
'''

from pyscf import lib
from pyscf import df
from pyscf.mp import dfump2
from pyscf.mp import dfmp2
from pyscf.mp.romp2 import ROMP2, add_singles, WITH_T2


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=None):
    '''DF-ROMP2 kernel.  The doubles contribution is computed by the DF-UMP2
    kernel in the semi-canonical basis; the singles correction is added on
    top.'''
    e_corr, t2 = dfump2.kernel(mp, mo_energy, mo_coeff, eris, with_t2, verbose)
    return add_singles(mp, e_corr), t2


class DFROMP2(ROMP2):
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
        with_df : DF object
            Density fitting object used for the 3-center integrals.
    '''

    _keys = ROMP2._keys | {'with_df', 'force_outcore'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mo_coeff is not None or mo_occ is not None:
            raise NotImplementedError('DF-ROMP2 does not support custom '
                                      'mo_coeff/mo_occ. The ROHF orbitals are '
                                      'always semi-canonicalized.')
        ROMP2.__init__(self, mf, frozen)

        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)

        # DEBUG:
        self.force_outcore = False

    split_mo_coeff = dfump2.DFUMP2.split_mo_coeff
    split_mo_energy = dfump2.DFUMP2.split_mo_energy
    split_mo_occ = dfump2.DFUMP2.split_mo_occ

    reset = dfmp2.DFRMP2.reset

    def ao2mo(self, mo_coeff=None, ovL=None, ovL_to_save=None):
        return dfump2._make_df_eris(self, mo_coeff, ovL, ovL_to_save)

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None,
                  with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)


from pyscf import scf  # noqa: E402
scf.rohf.ROHF.DFROMP2 = lib.class_as_method(DFROMP2)
scf.rohf.ROHF.DFMP2 = lib.class_as_method(DFROMP2)
