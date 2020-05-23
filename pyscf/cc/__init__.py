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
Coupled Cluster
===============

Simple usage::

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> cc.CCSD(mf).run()

:func:`cc.CCSD` returns an instance of CCSD class.  Followings are parameters
to control CCSD calculation.

    verbose : int
        Print level.  Default value equals to :class:`Mole.verbose`
    max_memory : float or int
        Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
    conv_tol : float
        converge threshold.  Default is 1e-7.
    conv_tol_normt : float
        converge threshold for norm(t1,t2).  Default is 1e-5.
    max_cycle : int
        max number of iterations.  Default is 50.
    diis_space : int
        DIIS space size.  Default is 6.
    diis_start_cycle : int
        The step to start DIIS.  Default is 0.
    direct : bool
        AO-direct CCSD. Default is False.
    async_io : bool
        Allow for asynchronous function execution. Default is True.
    incore_complete : bool
        Avoid all I/O. Default is False.
    frozen : int or list
        If integer is given, the inner-most orbitals are frozen from CC
        amplitudes.  Given the orbital indices (0-based) in a list, both
        occupied and virtual orbitals can be frozen in CC calculation.


Saved results

    converged : bool
        CCSD converged or not
    e_tot : float
        Total CCSD energy (HF + correlation)
    t1, t2 :
        t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    l1, l2 :
        Lambda amplitudes l1[i,a], l2[i,j,a,b]  (i,j in occ, a,b in virt)
'''

from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm
from pyscf.cc import addons
from pyscf.cc import rccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.cc import eom_rccsd
from pyscf.cc import eom_uccsd
from pyscf.cc import eom_gccsd
from pyscf import scf

def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UCCSD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.ghf.GHF):
        return GCCSD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCCSD(mf, frozen, mo_coeff, mo_occ)
CCSD.__doc__ = ccsd.CCSD.__doc__

scf.hf.SCF.CCSD = CCSD


def RCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf.cc import dfccsd

    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError('RCCSD cannot be used with UHF method.')
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RCCSD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCCSD method is called.')
        mf = scf.addons.convert_to_uhf(mf)
        return UCCSD(mf, frozen, mo_coeff, mo_occ)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, 'with_df', None):
        return dfccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

    elif numpy.iscomplexobj(mo_coeff) or numpy.iscomplexobj(mf.mo_coeff):
        return rccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

    else:
        return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)
RCCSD.__doc__ = ccsd.CCSD.__doc__


def UCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-UCCSD')
    else:
        return uccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)
UCCSD.__doc__ = uccsd.UCCSD.__doc__


def GCCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.ghf.GHF):
        mf = scf.addons.convert_to_ghf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-GCCSD')
    else:
        return gccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)
GCCSD.__doc__ = gccsd.GCCSD.__doc__


def FNOCCSD(mf, thresh=1e-6, pct_occ=None, nvir_act=None):
    """Frozen natural orbital CCSD

    Attributes:
        thresh : float
            Threshold on NO occupation numbers.  Default is 1e-6.
        pct_occ : float
            Percentage of total occupation number.  Default is None.  If present, overrides `thresh`.
    """
    #from pyscf import mp
    #pt = mp.MP2(mf).set(verbose=0).run()
    from pyscf.mp.mp2 import MP2
    pt = MP2(mf).set(verbose=0).run()
    frozen, no_coeff = pt.make_fno(thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)
    #pt_no = mp.MP2(mf, frozen=frozen, mo_coeff=no_coeff).set(verbose=0).run() #avoid DF
    pt_no = MP2(mf, frozen=frozen, mo_coeff=no_coeff).set(verbose=0).run()
    mycc = ccsd.CCSD(mf, frozen=frozen, mo_coeff=no_coeff) #avoid DF
    mycc.delta_emp2 = pt.e_corr - pt_no.e_corr
    from pyscf.lib import logger
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if self.converged:
            logger.info(self, 'FNO-%s converged', self.__class__.__name__)
        else:
            logger.note(self, 'FNO-%s not converged', self.__class__.__name__)
        logger.note(self, 'E(FNO-%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'E(FNO-%s+delta-MP2) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot+self.delta_emp2,
                    self.e_corr+self.delta_emp2)
        return self
    mycc._finalize = _finalize.__get__(mycc, mycc.__class__)
    return mycc
