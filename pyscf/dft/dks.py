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


@lib.with_doc(gks.get_veff.__doc__)
def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    if ks.nlc != '':
        raise NotImplementedError(ks.nlc)
    return gks.get_veff(ks, mol, dm, dm_last, vhf_last, hermi)


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    r'''Electronic part of DKS energy.

    Note this function has side effects which cause mf.scf_summary updated.

    Args:
        ks : an instance of DFT class

        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian

    Returns:
        DKS electronic energy and the 2-electron contribution
    '''
    e1, e2 = rks.energy_elec(ks, dm, h1e, vhf)
    if not ks.with_ssss and ks.ssss_approx == 'Visscher':
        e2 += _vischer_ssss_correction(mf, dm)
        mf.scf_summary['e2'] = e2
    return e1, e2


class DKS(rks.KohnShamDFT, dhf.DHF):
    '''Kramers unrestricted Dirac-Kohn-Sham'''
    def __init__(self, mol, xc='LDA,VWN'):
        from pyscf.dft import r_numint
        dhf.DHF.__init__(self, mol)
        rks.KohnShamDFT.__init__(self, xc)
        self._numint = r_numint.RNumInt()

    def dump_flags(self, verbose=None):
        dhf.DHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = energy_elec

    def x2c1e(self):
        from pyscf.x2c import x2c
        x2chf = x2c.UKS(self.mol)
        x2c_keys = x2chf._keys
        x2chf.__dict__.update(self.__dict__)
        x2chf._keys = self._keys.union(x2c_keys)
        return x2chf
    x2c = x2c1e

    @property
    def collinear(self):
        return self._numint.collinear
    @collinear.setter
    def collinear(self, val):
        self._numint.collinear = val

UKS = UDKS = DKS

class RDKS(DKS, dhf.RDHF):
    '''Kramers restricted Dirac-Kohn-Sham'''
    _eigh = dhf.RDHF._eigh

    def x2c1e(self):
        from pyscf.x2c import x2c
        x2chf = x2c.RKS(self.mol)
        x2c_keys = x2chf._keys
        x2chf.__dict__.update(self.__dict__)
        x2chf._keys = self._keys.union(x2c_keys)
        return x2chf
    x2c = x2c1e

    @property
    def collinear(self):
        return self._numint.collinear
    @collinear.setter
    def collinear(self, val):
        self._numint.collinear = val

RKS = RDKS
