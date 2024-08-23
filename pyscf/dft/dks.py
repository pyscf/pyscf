#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
4-component Dirac-Kohn-Sham
'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import dhf
from pyscf.dft import rks
from pyscf.dft import gks
from pyscf.dft import r_numint


@lib.with_doc(gks.get_veff.__doc__)
def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if ks.do_nlc():
        raise NotImplementedError(f'NLC functional {ks.xc} + {ks.nlc}')
    return gks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    r'''Electronic part of DKS energy.

    Note this function has side effects which cause mf.scf_summary updated.

    Args:
        ks : an instance of DFT class

        dm : 2D ndarray
            one-particle density matrix
        h1e : 2D ndarray
            Core hamiltonian

    Returns:
        DKS electronic energy and the 2-electron contribution
    '''
    e1, e2 = rks.energy_elec(ks, dm, h1e, vhf)
    if not ks.with_ssss and ks.ssss_approx == 'Visscher':
        e2 += dhf._vischer_ssss_correction(ks, dm)
        ks.scf_summary['e2'] = e2
    return e1, e2


class KohnShamDFT(rks.KohnShamDFT):
    def __init__(self, xc='LDA,VWN'):
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = r_numint.RNumInt()

    def dump_flags(self, verbose=None):
        rks.KohnShamDFT.dump_flags(self, verbose)
        logger.info(self, 'collinear = %s', self._numint.collinear)
        if self._numint.collinear[0] == 'm':
            logger.info(self, 'mcfun spin_samples = %s', self._numint.spin_samples)
            logger.info(self, 'mcfun collinear_thrd = %s', self._numint.collinear_thrd)
            logger.info(self, 'mcfun collinear_samples = %s', self._numint.collinear_samples)
        return self

    get_veff = gks.get_veff
    energy_elec = gks.energy_elec

    @property
    def collinear(self):
        return self._numint.collinear
    @collinear.setter
    def collinear(self, val):
        self._numint.collinear = val

    @property
    def spin_samples(self):
        return self._numint.spin_samples
    @spin_samples.setter
    def spin_samples(self, val):
        self._numint.spin_samples = val

    def to_rhf(self):
        raise RuntimeError

    def to_uhf(self):
        raise RuntimeError

    def to_ghf(self):
        raise RuntimeError

    def to_rks(self, xc=None):
        raise RuntimeError

    def to_uks(self, xc=None):
        raise RuntimeError

    def to_gks(self, xc=None):
        raise RuntimeError

    def to_dhf(self):
        '''Convert the input mean-field object to a DHF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(dhf.DHF)
        mf.converged = False
        return mf

    to_hf = to_dhf

    def to_dks(self, xc=None):
        if xc is not None and xc != self.xc:
            mf = self.copy()
            mf.xc = xc
            mf.converged = False
        return self

    to_ks = to_dks


class DKS(KohnShamDFT, dhf.DHF):
    '''Kramers unrestricted Dirac-Kohn-Sham'''
    def __init__(self, mol, xc='LDA,VWN'):
        dhf.DHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        dhf.DHF.dump_flags(self, verbose)
        KohnShamDFT.dump_flags(self, verbose)
        return self

    def x2c1e(self):
        from pyscf.x2c import dft
        x2chf = dft.UKS(self.mol)
        x2chf.__dict__.update(self.__dict__)
        return x2chf
    x2c = x2c1e

    to_gpu = lib.to_gpu

UKS = UDKS = DKS

class RDKS(DKS, dhf.RDHF):
    '''Kramers restricted Dirac-Kohn-Sham'''
    def __init__(self, mol, xc='LDA,VWN'):
        dhf.RDHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)

    def x2c1e(self):
        from pyscf.x2c import dft
        x2chf = dft.RKS(self.mol)
        x2chf.__dict__.update(self.__dict__)
        return x2chf
    x2c = x2c1e

    def to_dhf(self):
        '''Convert the input mean-field object to a DHF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = self.view(dhf.RDHF)
        mf.converged = False
        return mf

RKS = RDKS
