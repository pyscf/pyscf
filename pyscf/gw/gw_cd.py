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
#
# Author: Tianyu Zhu <zhutianyu1991@gmail.com>
#

'''
Spin-restricted G0W0 approximation with contour deformation

This implementation has the same scaling (N^4) as GW-AC, more robust but slower.
GW-CD is particularly recommended for accurate core and high-energy states.

Method:
    See T. Zhu and G.K.-L. Chan, arxiv:2007.03148 (2020) for details
    Compute Sigma directly on real axis with density fitting
    through a contour deformation method

Useful References:
    J. Chem. Theory Comput. 14, 4856-4869 (2018)
'''

from functools import reduce
import numpy
import numpy as np
import h5py
from scipy.optimize import newton, least_squares

from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import df, scf
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__

einsum = lib.einsum

def kernel(gw, mo_energy, mo_coeff, Lpq=None, orbs=None,
           nw=None, vhf_df=False, verbose=logger.NOTE):
    '''GW-corrected quasiparticle orbital energies

    Returns:
        A list :  converged, mo_energy, mo_coeff
    '''
    mf = gw._scf
    if gw.frozen is None:
        frozen = 0
    else:
        frozen = gw.frozen
    assert frozen == 0

    if Lpq is None:
        Lpq = gw.ao2mo(mo_coeff)
    if orbs is None:
        orbs = range(gw.nmo)

    # v_xc
    v_mf = mf.get_veff() - mf.get_j()
    v_mf = reduce(numpy.dot, (mo_coeff.T, v_mf, mo_coeff))

    nocc = gw.nocc
    nmo = gw.nmo

    # v_hf from DFT/HF density
    if vhf_df: # and frozen == 0:
        # density fitting for vk
        vk = -einsum('Lni,Lim->nm',Lpq[:,:,:nocc],Lpq[:,:nocc,:])
    else:
        # exact vk without density fitting
        dm = mf.make_rdm1()
        rhf = scf.RHF(gw.mol)
        vk = rhf.get_veff(gw.mol,dm) - rhf.get_j(gw.mol,dm)
        vk = reduce(numpy.dot, (mo_coeff.T, vk, mo_coeff))

    # Grids for integration on imaginary axis
    freqs,wts = _get_scaled_legendre_roots(nw)

    # Compute Wmn(iw) on imaginary axis
    logger.debug(gw, "Computing the imaginary part")
    Wmn = get_WmnI_diag(gw, orbs, Lpq, freqs)

    conv = True
    mo_energy = np.zeros_like(gw._scf.mo_energy)
    for p in orbs:
        if gw.linearized:
            # FIXME
            logger.warn(gw,'linearization with CD leads to wrong quasiparticle energy')
            raise NotImplementedError
        else:
            def quasiparticle(omega):
                sigma = get_sigma_diag(gw, omega, p, Lpq, Wmn[:,p-orbs[0],:], freqs, wts).real
                return omega - gw._scf.mo_energy[p] - (sigma.real + vk[p,p] - v_mf[p,p])
            try:
                if p < nocc:
                    delta = -1e-2
                else:
                    delta = 1e-2
                e = newton(quasiparticle, gw._scf.mo_energy[p]+delta, tol=1e-6, maxiter=50)
                logger.debug(gw, "Computing poles for QP (orb: %s)"%(p))
                mo_energy[p] = e
            except RuntimeError:
                conv = False
    mo_coeff = gw._scf.mo_coeff

    if gw.verbose >= logger.DEBUG:
        numpy.set_printoptions(threshold=nmo)
        logger.debug(gw, '  GW mo_energy =\n%s', mo_energy)
        numpy.set_printoptions(threshold=1000)

    return conv, mo_energy, mo_coeff

def get_sigma_diag(gw, ep, p, Lpq, Wmn, freqs, wts):
    '''
    Compute self-energy on real axis using contour deformation
    '''
    nocc = gw.nocc
    ef = (gw._scf.mo_energy[nocc-1] + gw._scf.mo_energy[nocc])/2.
    sign = np.sign(ef-gw._scf.mo_energy)
    sigmaI = get_sigmaI_diag(gw, ep, Wmn, sign, freqs, wts)
    sigmaR = get_sigmaR_diag(gw, ep, p, ef, Lpq)
    return sigmaI + sigmaR


def get_rho_response(omega, mo_energy, Lpq):
    '''
    Compute density response function in auxiliary basis at freq iw
    '''
    naux, nocc, nvir = Lpq.shape
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    eia = eia/(omega**2+eia*eia)
    Pia = einsum('Pia,ia->Pia',Lpq,eia)
    # Response from both spin-up and spin-down density
    Pi = 4. * einsum('Pia,Qia->PQ',Pia,Lpq)
    return Pi

def get_WmnI_diag(gw, orbs, Lpq, freqs):
    '''
    Compute W_mn(iw) on imarginary axis grids
    Return:
        Wmn: (Nmo, Norbs, Nw)
    '''
    mo_energy = gw._scf.mo_energy
    nocc = gw.nocc
    nmo = gw.nmo
    nw = len(freqs)
    naux = Lpq.shape[0]

    norbs = len(orbs)
    Wmn = np.zeros((nmo,norbs,nw))
    for w in range(nw):
        Pi = get_rho_response(freqs[w], mo_energy, Lpq[:,:nocc,nocc:])
        Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
        Qnm = einsum('Pnm,PQ->Qnm',Lpq[:,orbs,:],Pi_inv)
        Wmn[:,:,w] = einsum('Qnm,Qmn->mn',Qnm,Lpq[:,:,orbs])

    return Wmn

def get_sigmaI_diag(gw, omega, Wmn, sign, freqs, wts):
    '''
    Compute self-energy by integrating on imaginary axis
    '''
    mo_energy = gw._scf.mo_energy
    emo = omega - 1j*gw.eta*sign - mo_energy
    g0 = wts[None,:]*emo[:,None] / ((emo**2)[:,None]+(freqs**2)[None,:])
    sigma = -einsum('mw,mw',g0,Wmn)/np.pi

    return sigma

def get_rho_response_R(gw, omega, Lpq):
    '''
    Compute density response function in auxiliary basis at poles
    '''
    naux, nocc, nvir = Lpq.shape
    mo_energy = gw._scf.mo_energy
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    eia = 1./(omega+eia+2j*gw.eta) + 1./(-omega+eia)
    Pia = einsum('Pia,ia->Pia',Lpq,eia)
    # Response from both spin-up and spin-down density
    Pi = 2. * einsum('Pia,Qia->PQ',Pia,Lpq)
    return Pi

def get_sigmaR_diag(gw, omega, orbp, ef, Lpq):
    '''
    Compute self-energy for poles inside coutour
    (more and more expensive away from Fermi surface)
    '''
    mo_energy = gw._scf.mo_energy
    nocc = gw.nocc
    naux = Lpq.shape[0]

    if omega > ef:
        fm = 1.0
        idx = np.where((mo_energy<omega) & (mo_energy>ef))[0]
    else:
        fm = -1.0
        idx = np.where((mo_energy>omega) & (mo_energy<ef))[0]

    sigmaR = 0j
    if len(idx) > 0:
        for m in idx:
            em = mo_energy[m] - omega
            Pi = get_rho_response_R(gw, abs(em), Lpq[:,:nocc,nocc:])
            Pi_inv = np.linalg.inv(np.eye(naux)-Pi)-np.eye(naux)
            sigmaR += fm * np.dot(np.dot(Lpq[:,orbp,m],Pi_inv), Lpq[:,m,orbp])

    return sigmaR

def _get_scaled_legendre_roots(nw):
    """
    Scale nw Legendre roots, which lie in the
    interval [-1, 1], so that they lie in [0, inf)
    Ref: www.cond-mat.de/events/correl19/manuscripts/ren.pdf

    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs, wts = np.polynomial.legendre.leggauss(nw)
    x0 = 0.5
    freqs_new = x0*(1.+freqs)/(1.-freqs)
    wts = wts*2.*x0/(1.-freqs)**2
    return freqs_new, wts

def _get_clenshaw_curtis_roots(nw):
    """
    Clenshaw-Curtis qaudrature on [0,inf)
    Ref: J. Chem. Phys. 132, 234114 (2010)
    Returns:
        freqs : 1D ndarray
        wts : 1D ndarray
    """
    freqs = np.zeros(nw)
    wts = np.zeros(nw)
    a = 0.2
    for w in range(nw):
        t = (w+1.0)/nw * np.pi/2.
        freqs[w] = a / np.tan(t)
        if w != nw-1:
            wts[w] = a*np.pi/2./nw/(np.sin(t)**2)
        else:
            wts[w] = a*np.pi/4./nw/(np.sin(t)**2)
    return freqs[::-1], wts[::-1]


class GWCD(lib.StreamObject):

    eta = getattr(__config__, 'gw_gw_GW_eta', 1e-3)
    linearized = getattr(__config__, 'gw_gw_GW_linearized', False)

    def __init__(self, mf, frozen=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        #TODO: implement frozen orbs
        if not (self.frozen is None or self.frozen == 0):
            raise NotImplementedError

        # DF-GW must use density fitting integrals
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
        self._keys.update(['with_df'])

##################################################
# don't modify the following attributes, they are not input options
        self._nocc = None
        self._nmo = None
        # self.mo_energy: GW quasiparticle energy, not scf mo_energy
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ
        self.sigma = None

        keys = set(('eta', 'linearized'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        logger.info(self, 'use perturbative linearized QP eqn = %s', self.linearized)
        return self

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def kernel(self, mo_energy=None, mo_coeff=None, Lpq=None, orbs=None, nw=100, vhf_df=False):
        """
        Input:
            orbs: self-energy orbs
            nw: grid number
        Output:
            mo_energy: GW quasiparticle energy
        """
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        cput0 = (logger.process_clock(), logger.perf_counter())
        self.dump_flags()
        self.converged, self.mo_energy, self.mo_coeff = \
                kernel(self, mo_energy, mo_coeff,
                       Lpq=Lpq, orbs=orbs, nw=nw, vhf_df=vhf_df, verbose=self.verbose)

        logger.timer(self, 'GW', *cput0)
        return self.mo_energy

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo
        naux = self.with_df.get_naoaux()
        mem_incore = (2*nmo**2*naux) * 8/1e6
        mem_now = lib.current_memory()[0]

        mo = numpy.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        if (mem_incore + mem_now < 0.99*self.max_memory) or self.mol.incore_anyway:
            Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
            return Lpq.reshape(naux,nmo,nmo)
        else:
            logger.warn(self, 'Memory may not be enough!')
            raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, dft, tddft
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.7571 , 0.5861)],
        [1 , (0. , 0.7571 , 0.5861)]]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()

    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo-nocc

    gw = GWCD(mf)
    gw.kernel(orbs=range(0,nocc+3))
    print(gw.mo_energy)
    assert (abs(gw.mo_energy[nocc-1]--0.41284735)<1e-5)
    assert (abs(gw.mo_energy[nocc]-0.16574524)<1e-5)
    assert (abs(gw.mo_energy[0]--19.53387986)<1e-5)
