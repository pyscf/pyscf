#!/usr/bin/env python
# Copyright 2021 The PySCF Developers. All Rights Reserved.
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
Numerical integration functions for (2-component) GKS with real AO basis
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.dft import numint


class NumInt2C(numint._NumIntMixin):
    '''Numerical integration methods for 2-component basis (used by GKS)'''

    def __init__(self):
        self.omega = None  # RSH paramter
        self.collinear = True

    def eval_rho(self, mol, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        nao = ao.shape[-1]
        assert dm.ndim == 2 and nao * 2 == dm.shape[0]
        dm_a = dm[:nao,:nao].real.copy()
        dm_b = dm[nao:,nao:].real.copy()
        rho = numint.eval_rho(mol, ao, dm_a, non0tab, xctype, hermi, verbose)
        rho += numint.eval_rho(mol, ao, dm_b, non0tab, xctype, hermi, verbose)
        if dm.dtype == np.complex128:
            dm_a = dm[:nao,:nao].imag
            dm_b = dm[nao:,nao:].imag
            if abs(dm_a).max() > 1e-8 or abs(dm_b).max() > 1e-8:
                rhoI = numint.eval_rho(mol, ao, dm_a, non0tab, xctype, hermi, verbose)
                rhoI += numint.eval_rho(mol, ao, dm_b, non0tab, xctype, hermi, verbose)
                rho = rho + rhoI * 1j
        return rho

    def eval_rho2(self, mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
                  verbose=None):
        '''Calculate the electron density for LDA functional and the density
        derivatives for GGA functional in the framework of 2-component basis.
        '''
        if mo_coeff.dtype == np.double:
            nao = ao.shape[-1]
            assert nao * 2 == mo_coeff.shape[0]
            mo_aR = mo_coeff[:nao]
            mo_bR = mo_coeff[nao:]
            hermi = 1
            rho  = numint.eval_rho2(mol, ao, mo_aR, mo_occ, non0tab, xctype, hermi, verbose)
            rho += numint.eval_rho2(mol, ao, mo_bR, mo_occ, non0tab, xctype, hermi, verbose)
        else:
            dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
            hermi = 1
            rho = self.eval_rho(mol, dm, ao, dm, non0tab, xctype, hermi, verbose)
        return rho

    def cache_xc_kernel(self, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                        max_memory=2000):
        '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
        DFT hessian module etc.
        '''
        xctype = self._xc_type(xc_code)
        if xctype == 'MGGA':
            ao_deriv = 2
        elif xctype == 'GGA':
            ao_deriv = 1
        elif xctype == 'NLC':
            raise NotImplementedError('NLC')
        else:
            ao_deriv = 0

        nao = mo_coeff.shape[0] // 2
        dm = np.dot(mo_coeff * mo_occ, mo_coeff.conj().T)
        dm_a = dm[:nao,:nao].real.copy()
        dm_b = dm[nao:,nao:].real.copy()
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in self.block_loop(mol, grids, nao, ao_deriv, max_memory):
            # rhoa and rhob have to be real
            rhoa.append(numint.eval_rho(mol, ao, dm_a, mask, xctype))
            rhob.append(numint.eval_rho(mol, ao, dm_b, mask, xctype))
        rho = (np.hstack(rhoa), np.hstack(rhob))
        vxc, fxc = self.eval_xc(xc_code, rho, spin=1, relativity=0, deriv=2,
                                verbose=0)[1:3]
        return rho, vxc, fxc

    def get_rho(self, mol, dm, grids, max_memory=2000):
        '''Density in real space
        '''
        nao = dm.shape[-1] // 2
        dm_a = dm[:nao,:nao].real
        dm_b = dm[nao:,nao:].real
        ni = self.view(numint.NumInt)
        return numint.get_rho(ni, mol, dm_a+dm_b, grids, max_memory)

    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        dms = np.asarray(dms)
        nao = dms.shape[-1] // 2
        # ground state density is real
        dm_a = dms[...,:nao,:nao].real.copy()
        dm_b = dms[...,nao:,nao:].real.copy()
        ni = self.view(numint.NumInt)
        n, exc, vxc = numint.nr_uks(ni, mol, grids, xc_code, (dm_a, dm_b),
                                    max_memory=max_memory)
        vmat = np.zeros_like(dms)
        vmat[...,:nao,:nao] = vxc[0]
        vmat[...,nao:,nao:] = vxc[1]
        return n, exc, vmat
    get_vxc = nr_gks_vxc = nr_vxc

    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        dms = np.asarray(dms)
        nao = dms.shape[-1] // 2
        dm0a = dm0[:nao,:nao].real.copy()
        dm0b = dm0[nao:,nao:].real.copy()
        # dms_a and dms_b may be complex if they are TDDFT amplitudes
        dms_a = dms[...,:nao,:nao].copy()
        dms_b = dms[...,nao:,nao:].copy()
        ni = self.view(numint.NumInt)
        vmat = numint.nr_uks_fxc(
            ni, mol, grids, xc_code, (dm0a, dm0b), (dms_a, dms_b), relativity=0,
            hermi=0, rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None)
        fxcmat = np.zeros_like(dms)
        fxcmat[...,:nao,:nao] = vmat[0]
        fxcmat[...,nao:,nao:] = vmat[1]
        return fxcmat
    get_fxc = nr_gks_fxc = nr_fxc
