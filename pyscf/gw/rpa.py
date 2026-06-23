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
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

"""
Spin-restricted random phase approximation (direct RPA/dRPA in chemistry)
with N^4 scaling

Method:
    Main routines are based on GW-AC method described in:
    T. Zhu and G.K.-L. Chan, J. Chem. Theory. Comput. 17, 727-741 (2021)
    X. Ren et al., New J. Phys. 14, 053020 (2012)
"""

import numpy as np, scipy

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp import dfmp2

einsum = lib.einsum

# ****************************************************************************
# core routines kernel
# ****************************************************************************

def kernel(rpa, eris=None, nw=40, x0=0.5, verbose=None):
    '''
    RPA correlation and total energy

    Args:
        cderi_ov:
            Array-like object, Cholesky decomposed ERI in OV subspace.
        nw:
            number of frequency point on imaginary axis.
        x0:
            scaling factor for frequency grid.

    Returns:
        e_tot:
            RPA total energy
        e_hf:
            EXX energy
        e_corr:
            RPA correlation energy
    '''
    log = logger.new_logger(rpa, verbose)

    if eris is None:
        eris = rpa.ao2mo()
    naux = eris.naux

    cput1 = (logger.process_clock(), logger.perf_counter())

    # Compute exact exchange energy (EXX)
    e_hf = rpa.get_e_hf()

    cput2 = cput1 = log.timer('EXX energy', *cput1)

    # Compute RPA correlation energy
    e_corr = 0
    e_ov = rpa.make_e_ov()
    f_ov = rpa.make_f_ov()
    for igrid,(omega,weigh) in enumerate(zip(*_get_scaled_legendre_roots(nw, x0))):
        diel = rpa.make_dielectric_matrix(omega, e_ov, f_ov, eris)
        factor = weigh / (2.0 * np.pi)
        e_corr += factor * np.log(np.linalg.det(np.eye(naux) - diel))
        e_corr += factor * np.trace(diel)
        diel = None

        cput2 = log.timer_debug1('RPA corr grid %d/%d'%(igrid+1,nw), *cput2)
    log.timer('RPA corr', *cput1)

    if abs(e_corr.imag) > 1e-4:
        log.warn('Non-zero imaginary part found in %s energy %s', rpa.__class__.__name__, e_corr)
    e_corr = e_corr.real

    return e_hf, e_corr

# ****************************************************************************
# frequency integral quadrature, legendre, clenshaw_curtis
# ****************************************************************************

def make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=None):
    '''
    Compute dielectric matrix at a given frequency omega

    Args:
        omega : float, frequency
        e_ov : 1D array (nocc * nvir), orbital energy differences
        eris : DF ERI object

    Returns:
        diel : 2D array (naux, naux), dielectric matrix
    '''
    assert blksize is not None

    nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux

    isreal = eris.dtype == np.float64

    chi0 = (2.0 * e_ov * f_ov / (omega ** 2 + e_ov ** 2)).ravel()
    diel = np.zeros((naux, naux), dtype=eris.dtype)

    for p0,p1 in lib.prange(0, nocc*nvir, blksize):
        ovL = eris.get_ov_blk(p0,p1)
        ovL_chi = (ovL.T * chi0[p0:p1]).T
        if isreal:
            lib.ddot(ovL_chi.T, ovL, c=diel, beta=1)
        else:
            lib.dot(ovL_chi.T, ovL.conj(), c=diel, beta=1)
        ovL = ovL_chi = None

    return diel

def _get_scaled_legendre_roots(nw, x0=0.5):
    '''
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D array
        wts : 1D array
    '''
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    freqs_new = x0 * (1.0 + freqs) / (1.0 - freqs)
    wts = wts * 2.0 * x0 / (1.0 - freqs)**2
    return freqs_new, wts

def _get_clenshaw_curtis_roots(nw):
    '''
    Clenshaw-Curtis quadrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)

    Returns:
        freqs : 1D array
        wts : 1D array
    '''
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w + 1.0) / nw * np.pi * 0.5
        freqs[w] = a / np.tan(t)
        if w != nw - 1:
            wts[w] = a * np.pi * 0.50 / nw / (np.sin(t)**2)
        else:
            wts[w] = a * np.pi * 0.25 / nw / (np.sin(t)**2)
    return freqs[::-1], wts[::-1]


class RPA(dfmp2.DFMP2):

    _keys = {
        'mol', 'frozen',
        'with_df', 'mo_energy',
        'mo_coeff', 'mo_occ',
        'e_corr', 'e_hf', 'e_tot',
    }

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        log.info('nocc = %d  nmo = %d', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', self.frozen)
        return self

    def kernel(self, eris=None, nw=40, x0=0.5):
        '''
        The kernel function for direct RPA
        '''
        if np.iscomplexobj(self.mo_coeff):
            ''' The code runs for complex-valued orbitals but the results ain't quite right
                (very close though...). Throw an exception for now.
            '''
            raise NotImplementedError

        log = logger.new_logger(self)
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        res = kernel(
            self, eris=eris, nw=nw, x0=x0, verbose=self.verbose)
        self.e_hf, self.e_corr = res

        log.timer(self.__class__.__name__, *cput0)

        self._finalize()

        return self.e_corr

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('E(%s) = %.15g  E_corr = %.15g  E_hf = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr, self.e_hf)

    def get_e_hf(self):
        if self.e_hf is None:
            mf = self._scf
            if isinstance(mf, scf.hf.KohnShamDFT):
                mf = mf.to_hf()
            if getattr(mf, 'with_df', None) is None:
                mf = mf.density_fit(with_df=self.with_df)
            dm = mf.make_rdm1()
            self.e_hf = mf.energy_elec(dm=dm)[0] + mf.energy_nuc()
        return self.e_hf

    def make_e_ov(self):
        '''
        Compute orbital energy differences
        '''
        log = logger.new_logger(self)
        moeocc, moevir = self.split_mo_energy()[1:3]
        e_ov = (moeocc[:,None] - moevir).ravel()

        gap = (-e_ov.max(), )
        log.info('Lowest orbital energy difference: % 6.4e', np.min(gap))

        if (np.min(gap) < 1e-3):
            log.warn('RPA code is not well-defined for degenerate systems!')
            log.warn('Lowest orbital energy difference: % 6.4e', np.min(gap))

        return e_ov

    def make_f_ov(self):
        '''
        Compute orbital occupation number differences
        '''
        focc, fvir = self.split_mo_occ()[1:3]
        return (focc[:,None] - fvir).ravel()

    def make_dielectric_matrix(self, omega, e_ov=None, f_ov=None, eris=None,
                               max_memory=None, blksize=None):
        '''
        Args:
            omega : float, frequency
            e_ov : 1D array (nocc * nvir), orbital energy differences
            mo_coeff :  (nao, nmo), mean-field mo coefficient
            cderi_ov :  (naux, nocc, nvir), Cholesky decomposed ERI in OV subspace.

        Returns:
            diel : 2D array (naux, naux), dielectric matrix
        '''
        if e_ov is None: e_ov = self.make_e_ov()
        if f_ov is None: f_ov = self.make_f_ov()
        if eris is None: eris = self.ao2mo()
        if max_memory is None: max_memory = self.max_memory

        if blksize is None:
            mem_avail = max_memory - lib.current_memory()[0]
            nocc, nvir, naux = eris.nocc, eris.nvir, eris.naux
            dsize = eris.dsize
            mem_blk = 2*naux * dsize/1e6    # ovL and ovL*chi0
            blksize = max(1, min(nocc*nvir, int(np.floor(mem_avail*0.7 / mem_blk))))
        else:
            blksize = min(blksize, e_ov.size)

        diel = make_dielectric_matrix(omega, e_ov, f_ov, eris, blksize=blksize)

        return diel


dRPA = DirectRPA = RPA


if __name__ == '__main__':
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2svp'
    mol.build()
    mol.verbose = 4

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    rpa = RPA(mf)
    rpa.verbose = 4

    eris = rpa.ao2mo()
    e_corr_0 = rpa.kernel(eris=eris)

    assert (abs(rpa.e_corr - -0.307830040357800) < 1e-6)
    assert (abs(rpa.e_tot  - -76.26651423730257) < 1e-6)

    # Another implementation of direct RPA N^6
    e_ov = -rpa.make_e_ov()
    nov = e_ov.size
    v_ov = np.asarray(eris.get_ov_blk(0,nov).T, order='C')
    a = e_ov * np.eye(nov) + 2 * np.dot(v_ov.T, v_ov)
    b = 2 * np.dot(v_ov.T, v_ov)
    cmat = np.block([[a,b],[-b.conj(),-a.conj()]])
    ev = scipy.linalg.eig(cmat)[0]
    ev = ev.real
    ev = ev[ev>0]
    e_corr_1 = 0.5 * (np.sum(ev) - np.trace(a))

    assert abs(e_corr_0 - e_corr_1) < 1e-8
