# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory
'''

import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.scf import _vhf
from pyscf.agf2 import aux, mpi_helper
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, \
                            _mo_without_core, _mo_energy_without_core

BLKMIN = getattr(__config__, 'agf2_ragf2_blkmin', 1)

#TODO: test outcore and add incore_complete, also async_io??
#TODO: do we want MPI in this?
#TODO: test frozen!!! do we have to rediagonalise Fock matrix?
#TODO: do we want to store RAGF2.qmo_energy, RAGF2.qmo_coeff and RAGF2.qmo_occ at the end?
#TODO: do we really want to store the self-energy? if we do above we can remove both RAGF2.gf and RAGF2.se
#TODO: scs
#TODO: damping
#TODO: should we use conv_tol and max_cycle to automatically assign the _nelec and _rdm1 ones?


def kernel(agf2, eri=None, gf=None, se=None, verbose=None):

    log = logger.new_logger(agf2, verbose)
    cput1 = cput0 = (time.clock(), time.time())
    name = agf2.__class__.__name__

    if eri is None: eri = agf2.ao2mo()
    if gf is None: gf = self.gf
    if se is None: se = self.se
    if verbose is None: verbose = agf2.verbose

    gf = _dict_to_aux(gf, obj=aux.GreensFunction)
    se = _dict_to_aux(se, obj=aux.SelfEnergy)

    if gf is None:
        gf = agf2.init_gf()

    if se is None:
        se = agf2.build_se(eri, gf)

    #NOTE: should we even print/store e_mp2, or name it something else? this is
    # quite a bit off of the real E(mp2) at (None,0)...
    e_mp2 = agf2.energy_mp2(agf2.mo_energy, se)
    log.info('E(MP2) = %.16g  E_corr(MP2) = %.16g', e_mp2+eri.e_hf, e_mp2)

    e_prev = e_1b = e_2b = 0.0
    converged = False
    for niter in range(1, agf2.max_cycle+1):
        # one-body terms
        gf, se, fock_conv = agf2.fock_loop(eri, gf, se)
        e_1b = agf2.energy_1body(eri, gf)

        # two-body terms
        se = agf2.build_se(eri, gf)
        gf = agf2.build_gf(eri, gf, se)
        e_2b = agf2.energy_2body(gf, se)

        e_tot = e_1b + e_2b

        ip = agf2.get_ip(gf, nroots=1)[0][0]
        ea = agf2.get_ea(gf, nroots=1)[0][0]

        log.info('cycle = %d  E(%s) = %.15g  E_corr(%s) = %.15g  dE = %.9g',
                 niter, name, e_tot, name, e_tot-eri.e_hf, e_tot-e_prev)
        log.info('E_1b = %.15g  E_2b = %.15g', e_1b, e_2b)
        log.info('IP = %.15g  EA = %.15g', ip, ea)
        cput1 = log.timer('%s iter'%name, *cput1)

        if abs(e_tot - e_prev) < agf2.conv_tol:
            converged = True
            break

        e_prev = e_tot

    log.timer('%s'%name, *cput0)

    return converged, e_1b, e_2b, gf, se


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped.
    
    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Kwargs:
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Returns:
        :class:`SelfEnergy`
    '''
    #TODO: C code

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction

    nmo = agf2.nmo
    tol = agf2.weight_tol  #NOTE: tol is unlikely to be met at (None,0)

    mem_incore = (gf_occ.nphys*gf_occ.naux**2*gf_vir.naux) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mem_incore+mem_now < agf2.max_memory):
        qeri = _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir)
    else:
        qeri = _make_qmo_eris_outcore(agf2, eri, gf_occ, gf_vir)

    vv = np.zeros((nmo, nmo))
    vev = np.zeros((nmo, nmo))

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)
    eja = eja.ravel()

    for i in range(gf_occ.naux):
        #TODO: should we perform this differently? these are not contiguous and this slicing sucks...
        xija = qeri[:,i].reshape(nmo, -1)
        xjia = qeri[:,:,i].reshape(nmo, -1)

        eija = eja + gf_occ.energy[i]

        vv = lib.dot(xija, xija.T, alpha=fpos, beta=1, c=vv)
        vv = lib.dot(xija, xjia.T, alpha=fneg, beta=1, c=vv)

        exija = xija * eija[None]

        vev = lib.dot(exija, xija.T, alpha=fpos, beta=1, c=vev)
        vev = lib.dot(exija, xjia.T, alpha=fneg, beta=1, c=vev)

    e, c = _cholesky_build(vv, vev, gf_occ, gf_vir)
    se = aux.SelfEnergy(e, c, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    log.timer_debug1('se part', *cput0)

    return se


def _cholesky_build(vv, vev, gf_occ, gf_vir, eps=1e-20):
    ''' Constructs the truncated auxiliaries from :attr:`vv` and :attr:`vev`.
        Performs a Cholesky decomposition via :func:`numpy.linalg.cholesky`,
        for a positive-definite or positive-semidefinite matrix. For the
        latter, the null space is removed.

        The :attr:`vv` matrix of :func:`build_se_part` can be positive-
        semidefinite when :attr:`gf_occ.naux` < :attr:`gf_occ.nphys` for
        the occupied self-energy, or :attr:`gf_vir.naux` < :attr:`gf_vir.nphys`
        for the virtual self-energy.
    '''

    ##NOTE: test this
    ##FIXME: doesn't seem to work when situation arises due to frozen core - might have to rethink this
    ##FIXME: won't work for UAGF2 if we use this solution (need to change naux)
    #if gf_occ.nphys >= (gf_occ.naux**2 * gf_vir.naux):
    #    # remove the null space from vv and vev
    #    zero_rows = np.all(np.absolute(vv) < eps, axis=1)
    #    zero_cols = np.all(np.absolute(vv) < eps, axis=0)
    #    null_space = np.logical_and(zero_rows, zero_cols)

    #    vv = vv[~null_space][:,~null_space]
    #    vev = vev[~null_space][:,~null_space]
    #    np.set_printoptions(precision=4, linewidth=150)

    #NOTE: this seems like an unscientific solution... rescue the above?
    try:
        b = np.linalg.cholesky(vv).T
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(vv)
        w[w < eps] = eps
        vv_posdef = np.dot(np.dot(v, np.diag(w)), v.T.conj())
        b = np.linalg.cholesky(vv_posdef).T

    b_inv = np.linalg.inv(b)

    m = np.dot(np.dot(b_inv.T, vev), b_inv)
    e, c = np.linalg.eigh(m)
    c = np.dot(b.T, c[:gf_occ.nphys])

    if c.shape[0] < gf_occ.nphys:
        c_full = np.zeros((gf_occ.nphys, c.shape[1]), dtype=c.dtype)
        c_full[~null_space] = c
        c = c_full

    return e, c


def get_jk(agf2, eri, rdm1, with_j=True, with_k=True):
    ''' Get the J/K matrices.

    Args:
        eri : ndarray or H5 dataset
            Electronic repulsion integrals (NOT as _ChemistsERIs)
        rdm1 : 2D array
            Reduced density matrix

    Kwargs:
        with_j : bool
            Whether to compute J. Default value is True
        with_k : bool
            Whether to compute K. Default value is True

    Returns:
        tuple of ndarrays corresponding to J and K, if either are
        not requested then they are set to None.
    '''

    if isinstance(eri, np.ndarray):
        vj, vk = _vhf.incore(eri, rdm1, with_j=with_j, with_k=with_k)

    else:
        nmo = rdm1.shape[0]
        npair = nmo*(nmo+1)//2
        vj = vk = None

        if with_j:
            rdm1_tril = lib.pack_tril(rdm1 + np.tril(rdm1, k=-1))
            vj = np.zeros_like(rdm1_tril)

        if with_k:
            vk = np.zeros_like(rdm1)

        blksize = _get_blksize(agf2.max_memory, (nmo*npair, nmo**3))
        blksize = min(nmo, max(BLKMIN, blksize))

        tril2sq = lib.square_mat_in_trilu_indices(nmo)
        for p0, p1 in lib.prange(0, nmo, blksize):
            idx = np.concatenate(tril2sq[p0:p1])
            eri0 = eri[idx]

            # vj built in tril layout with scaled rdm1_tril
            if with_j:
                vj[idx] = np.dot(eri0, rdm1_tril)

            if with_k:
                eri0 = lib.unpack_tril(eri0, axis=-1)
                eri0 = eri0.reshape(p1-p0, nmo, nmo, nmo)

                vk[p0:p1] = lib.einsum('ijkl,jk->il', eri0, rdm1)

        if with_j:
            vj = lib.unpack_tril(vj)

    return vj, vk


def get_fock(agf2, eri, gf=None, rdm1=None):
    ''' Computes the physical space Fock matrix in MO basis. If :attr:`rdm1`
        is not supplied, it is built from :attr:`gf`, which defaults to
        the mean-field Green's function.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals

    Kwargs:
        gf : Greensfunction
            Auxiliaries of the Green's function
        rdm1 : 2D array
            Reduced density matrix.

    Returns:
        ndarray of physical space Fock matrix
    '''

    if rdm1 is None:
        rdm1 = agf2.make_rdm1(gf)

    vj, vk = agf2.get_jk(eri.eri, rdm1)
    fock = eri.h1e + vj - 0.5 * vk

    return fock


def fock_loop(agf2, eri, gf, se):
    ''' Self-consistent loop for the density matrix via the HF self-
        consistent field.

    Args:
        eri : _ChemistERIs
            Electronic repulsion integrals
        gf : GreensFunction
            Auxiliaries of the Green's function
        se : SelfEnergy
            Auxiliaries of the self-energy

    Returns:
        :class:`SelfEnergy`, :class:`GreensFunction` and a boolean 
        indicating wheter convergence was successful. 
    '''

    assert type(gf) is aux.GreensFunction
    assert type(se) is aux.SelfEnergy

    cput0 = cput1 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    diis = lib.diis.DIIS(agf2)
    diis.space = agf2.diis_space
    diis.min_space = agf2.diis_min_space
    fock = agf2.get_fock(eri, gf)

    nelec = agf2.nocc * 2
    nmo = agf2.nmo
    naux = se.naux
    nqmo = nmo + naux
    buf = np.zeros((nqmo, nqmo))
    converged = False
    opts = dict(tol=agf2.conv_tol_nelec, maxiter=agf2.max_cycle_inner)

    for niter1 in range(1, agf2.max_cycle_outer+1):
        se, opt = minimize_chempot(se, fock, nelec, x0=se.chempot, **opts)

        for niter2 in range(1, agf2.max_cycle_inner+1):
            w, v = se.eig(fock, chempot=0.0, out=buf)
            se.chempot, nerr = binsearch_chempot((w, v), nmo, nelec)

            w, v = se.eig(fock, out=buf)
            gf = aux.GreensFunction(w, v[:nmo], chempot=se.chempot)

            fock = agf2.get_fock(eri, gf)
            rdm1 = agf2.make_rdm1(gf)
            fock = diis.update(fock, xerr=None)  #NOTE: should we silence repeated linear dependence warnings since this is nested?

            if niter2 > 1:
                derr = np.max(np.absolute(rdm1 - rdm1_prev))
                if derr < agf2.conv_tol_rdm1:
                    break

            rdm1_prev = rdm1.copy()

        log.debug1('fock loop %d  cycles = %d  dN = %.3g  |ddm| = %.3g',
                   niter1, niter2, nerr, derr)
        cput1 = log.timer_debug1('fock loop %d'%niter1, *cput1)

        if derr < agf2.conv_tol_rdm1 and abs(nerr) < agf2.conv_tol_nelec:
            converged = True
            break

    log.info('fock converged = %s  chempot = %.9g  dN = %.3g  |ddm| = %.3g',
             converged, se.chempot, nerr, derr)

    return gf, se, converged


def energy_1body(agf2, eri, gf):
    ''' Calculates the one-body energy according to the RHF form.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf : GreensFunction
            Auxiliaries of Green's function

    Returns:
        One-body energy
    '''

    assert type(gf) is aux.GreensFunction

    rdm1 = agf2.make_rdm1(gf)
    fock = agf2.get_fock(eri, gf)

    e1b = 0.5 * np.sum(rdm1 * (eri.h1e + fock))
    e1b += agf2.energy_nuc()

    return e1b


def energy_2body(agf2, gf, se):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula. The formula is symmetric and only
        one side needs to be calculated.

    Args:
        gf : GreensFunction
            Auxiliaries of the Green's function
        se : SelfEnergy
            Auxiliaries of the self-energy

    Returns
        Two-body energy
    '''

    assert type(gf) is aux.GreensFunction
    assert type(se) is aux.SelfEnergy

    gf_occ = gf.get_occupied()
    se_vir = se.get_virtual()

    e2b = 0.0

    for l in range(gf_occ.naux):
        vxl = gf_occ.coupling[:,l]
        vxk = se_vir.coupling
        dlk = gf_occ.energy[l] - se_vir.energy

        e2b += lib.einsum('xk,yk,x,y,k->', vxk, vxk.conj(),
                                           vxl, vxl.conj(), 1./dlk)

    e2b *= 2

    return np.ravel(e2b.real)[0] #NOTE: i've had some problems with some einsum implementations not returning scalars... worth doing ravel()[0]?


def energy_mp2(agf2, mo_energy, se):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula for an MP2 self-energy. Per the
        definition of one- and two-body partitioning in the Dyson
        equation, this result is half of :func:`energy_2body`.

    Args:
        gf : GreensFunction
            Auxiliaries of the Green's function
        se : SelfEnergy
            Auxiliaries of the self-energy

    Returns
        MP2 energy
    '''

    assert type(se) is aux.SelfEnergy

    occ = mo_energy < se.chempot
    se_vir = se.get_virtual()

    vxk = se_vir.coupling[occ]
    dxk = lib.direct_sum('x,k->xk', mo_energy[occ], -se_vir.energy)

    emp2 = lib.einsum('xk,xk,xk->', vxk, vxk.conj(), 1./dxk)

    return np.ravel(emp2.real)[0]


class RAGF2(lib.StreamObject):
    ''' Restricted AGF2 with canonical HF reference

    Attributes:
        verbose : int
            Print level. Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB. Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-6.
        conv_tol_nelec : float
            Convergence threshold for the number of electrons. Default 
            value is 1e-6.
        max_cycle : int
            Maximum number of AGF2 iterations. Default value is 50.
        max_cycle_outer : int
            Maximum number of outer Fock loop iterations. Default 
            value is 20.
        max_cycle_inner : int
            Maximum number of inner Fock loop iterations. Default
            value is 50.
        weight_tol : float
            Threshold in spectral weight of auxiliaries to be considered
            zero. Default 1e-11.
        diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        diis_min_space : 
            Minimum space of DIIS. Default value is 1.
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_mp2 : float
            MP2 correlation energy
        converged : bool
            Whether convergence was successful
        se : SelfEnergy
            Auxiliaries of the self-energy
        gf : GreensFunction
            Auxiliaries of the Green's function
    '''

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):

        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.conv_tol = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol', 1e-7)
        self.conv_tol_rdm1 = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol_rdm1', 1e-6)
        self.conv_tol_nelec = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol_nelec', 1e-6)
        self.max_cycle = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle', 50)
        self.max_cycle_outer = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle_outer', 20)
        self.max_cycle_inner = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle_inner', 50)
        self.weight_tol = getattr(__config__, 'agf2_ragf2_RAGF2_weight_tol', 1e-11)
        self.diis_space = getattr(__config__, 'agf2_ragf2_RAGF2_diis_space', 6)
        self.diis_min_space = getattr(__config__, 'agf2_ragf2_RAGF2_diis_min_space', 1)
        self.os_factor = getattr(__config__, 'agf2_os_factor', 1.0)
        self.ss_factor = getattr(__config__, 'agf2_ss_factor', 1.0)

        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.se = None
        self.gf = None
        self.e_1b = mf.e_tot
        self.e_2b = 0.0
        self.e_mp2 = 0.0
        self.frozen = frozen
        self._nmo = None
        self._nocc = None
        self.converged = False
        self.chkfile = mf.chkfile
        self._keys = set(self.__dict__.keys())

    energy_1body = energy_1body
    energy_2body = energy_2body
    fock_loop = fock_loop
    build_se_part = build_se_part
    get_jk = get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the electronic repulsion integrals in MO basis.
        '''

        # happens when e.g. restarting from chkfile
        if self._scf._eri is None and self._scf._is_mem_enough():
            self._scf._eri = self.mol.intor('int2e', aosym='s8')

        mem_incore = ((self.nmo*(self.nmo+1)//2)**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if (self._scf._eri is not None and 
                (mem_incore+mem_now < self.max_memory)):
            eri = _make_mo_eris_incore(self)
        else:
            eri = _make_mo_eris_outcore(self)

        return eri

    def make_rdm1(self, gf=None):
        ''' Computes the one-body reduced density matrix in MO basis.

        Kwargs:
            gf : GreensFunction
                Auxiliaries of the Green's function

        Returns:
            ndarray of density matrix
        '''

        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        return gf.make_rdm1()

    def get_fock(self, eri=None, gf=None, rdm1=None):
        ''' Computes the physical space Fock matrix in MO basis.
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf

        return get_fock(self, eri, gf=gf, rdm1=rdm1)

    def energy_mp2(self, mo_energy=None, se=None):
        if mo_energy is None: mo_energy = self.mo_energy
        if se is None: se = self.build_se(gf=self.gf)
        self.e_mp2 = energy_mp2(self, mo_energy, se)
        return self.e_mp2

    def init_gf(self):
        ''' Builds the Hartree-Fock Green's function.

        Returns:
            :class:`GreensFunction`, :class:`SelfEnergy`
        '''

        mo_energy = _mo_energy_without_core(self, self.mo_energy)
        chempot = binsearch_chempot(np.diag(mo_energy), self.nmo, self.nocc*2)[0]

        gf = aux.GreensFunction(mo_energy, np.eye(self.nmo), chempot=chempot)

        return gf

    def build_gf(self, eri=None, gf=None, se=None):
        ''' Builds the auxiliaries of the Green's function by solving
            the Dyson equation.

        Kwargs:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : GreensFunction
                Auxiliaries of the Green's function
            se : SelfEnergy
                Auxiliaries of the self-energy

        Returns:
            :class:`GreensFunction`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()
        if se is None: se = self.build_se(eri, gf)

        fock = self.get_fock(eri, gf)

        return se.get_greens_function(fock)

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None):
        ''' Builds the auxiliaries of the self-energy.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : GreensFunction
                Auxiliaries of the Green's function

        Kwargs:
            os_factor : float
                Opposite-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            ss_factor : float
                Same-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0

        Returns:
            :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)

        se = aux.combine(se_occ, se_vir)

        return se

    def energy_nuc(self):
        return self._scf.energy_nuc()

        
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('conv_tol = %g' % self.conv_tol)
        log.info('conv_tol_rdm1 = %g' % self.conv_tol_rdm1)
        log.info('conv_tol_nelec = %g' % self.conv_tol_nelec)
        log.info('max_cycle = %g' % self.max_cycle)
        log.info('max_cycle_outer = %g' % self.max_cycle_outer)
        log.info('max_cycle_inner = %g' % self.max_cycle_inner)
        log.info('weight_tol = %g' % self.weight_tol)
        log.info('diis_space = %d' % self.diis_space)
        log.info('diis_min_space = %d', self.diis_min_space)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def _finalize(self):
        ''' Hook for dumping results and clearing up the object.
        '''

        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)

        ip = self.get_ip(self.gf, nroots=1)[0][0]
        ea = self.get_ea(self.gf, nroots=1)[0][0]

        logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)
        logger.note(self, 'IP = %.16g  EA = %.16g', ip, ea)

        return self

    def reset(self, mol=None):
        #NOTE: what is this achieving? attributes mo_coeff, mo_energy, mo_occ will not be reset
        # if this is called...?
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def kernel(self, eri=None, gf=None, se=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if se is None: se = self.se

        gf = _dict_to_aux(gf, obj=aux.GreensFunction)
        se = _dict_to_aux(se, obj=aux.SelfEnergy)

        if gf is None:
            gf = self.init_gf()

        if se is None:
            se = self.build_se(eri, gf)

        self.converged, self.e_1b, self.e_2b, self.gf, self.se = \
                kernel(self, eri=eri, gf=gf, se=se, verbose=self.verbose)

        self._finalize()

        return self.converged, self.e_1b, self.e_2b, self.gf, self.se

    def dump_chk(self, gf=None, se=None, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        if not self.chkfile:
            return self

        if mo_energy is None: mo_energy = self.mo_energy
        if mo_coeff  is None: mo_coeff  = self.mo_coeff
        if mo_occ    is None: mo_occ    = self.mo_occ
        if frozen is None: frozen = self.frozen
        if frozen is None: frozen = 0

        agf2_chk = { 'e_1b': self.e_1b,
                     'e_2b': self.e_2b, 
                     'e_mp2': self.e_mp2,
                     'converged': self.converged,
                     'mo_energy': mo_energy,
                     'mo_coeff': mo_coeff,
                     'mo_occ': mo_occ,
                     'frozen': frozen,
        }

        if gf is None: gf = self.gf
        if se is None: se = self.se

        if gf is not None: agf2_chk['gf'] = _aux_to_dict(gf)
        if se is not None: agf2_chk['se'] = _aux_to_dict(se)

        if self._nmo is not None: agf2_chk['_nmo'] = self._nmo
        if self._nocc is not None: agf2_chk['_nocc'] = self._nocc

        lib.chkfile.dump(self.chkfile, 'agf2', agf2_chk)

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.agf2 import dfragf2
        myagf2 = dfragf2.DFRAGF2(self._scf)
        myagf2.__dict__.update(self.__dict__)

        if with_df is not None:
            myagf2.with_df = with_df

        if auxbasis is not None and myagf2.with_df.auxbasis != auxbasis:
            import copy
            myagf2.with_df = copy.copy(myagf2.with_df)
            myagf2.with_df.auxbasis = auxbasis

        return myagf2


    def get_ip(self, gf, nroots=1):
        gf_occ = gf.get_occupied()
        e_ip = list(-gf_occ.energy[-nroots:])[::-1]
        v_ip = list(gf_occ.coupling[:,-nroots:].T)[::-1]
        return e_ip, v_ip

    def ipagf2(self, nroots=1):
        ''' Find the (N-1)-electron charged excitations, corresponding
            to the largest :attr:`nroots` poles of the occupied
            Green's function.

        Kwargs:
            nroots : int
                Number of roots (poles) requested. Default 1.

        Returns:
            IP and transition moment (float, 1D array) if :attr:`nroots`
            = 1, or array of IPs and moments (1D array, 2D array) if 
            :attr:`nroots` > 1.
        '''

        e_ip, v_ip = self.get_ip(self.gf, nroots=nroots)

        for n, en, vn in zip(range(nroots), e_ip, v_ip):
            qpwt = np.linalg.norm(vn)**2
            logger.note(self, 'IP root %d E = %.16g  qpwt = %0.6g', n, en, qpwt)

        if nroots == 1:
            return e_ip[0], v_ip[0]
        else:
            return e_ip, v_ip

    def get_ea(self, gf, nroots=1):
        gf_vir = gf.get_virtual()
        e_ea = list(gf_vir.energy[:nroots])
        v_ea = list(gf_vir.coupling[:,:nroots].T)
        return e_ea, v_ea

    def eaagf2(self, nroots=1):
        ''' Find the (N+1)-electron charged excitations, corresponding
            to the smallest :attr:`nroots` poles of the virtual
            Green's function.

        Kwargs:
            See ipagf2()
        '''

        e_ea, v_ea = self.get_ea(self.gf, nroots=nroots)

        for n, en, vn in zip(range(nroots), e_ea, v_ea):
            qpwt = np.linalg.norm(vn)**2
            logger.note(self, 'EA root %d E = %.16g  qpwt = %0.6g', n, en, qpwt)

        if nroots == 1:
            return e_ea[0], v_ea[0]
        else:
            return e_ea, v_ea

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask


    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, val):
        self._nmo = val
        
    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, val):
        self._nocc = val

    @property
    def e_tot(self):
        return self.e_1b + self.e_2b

    @property
    def e_corr(self):
        return self.e_tot - self._scf.e_tot


def _dict_to_aux(aux, obj=aux.AuxiliarySpace):
    ''' chkfile stores :attr:`gf` and :attr:`se` as dict, this
        function can be called to ensure that they are converted
        to the proper classes before use.
    '''

    # for compatibility with unrestricted
    if isinstance(aux, tuple):
        return tuple(_dict_to_aux(a, obj=obj) for a in aux)

    if isinstance(aux, obj) or aux is None:
        return aux

    assert isinstance(aux, dict)

    out = obj(aux['energy'], aux['coupling'], chempot=aux['chempot'])

    return out

def _aux_to_dict(aux):
    ''' Inverse of _dict_to_aux
    '''

    # for compatibility with unrestricted
    if isinstance(aux, tuple):
        return tuple(_aux_to_dict(a) for a in aux)

    if isinstance(aux, dict):
        return aux

    out = aux.__dict__

    return out


class _ChemistsERIs:
    ''' (pq|rs)
    
    MO integrals stored in s4 symmetry, we only need QMO integrals
    in low-symmetry tensors and s4 is highest supported by _vhf
    '''
    #NOTE: this is not exactly like rccsd._ChemistsERIs - should we rename it?

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        self.mo_coeff = mo_coeff = _mo_without_core(agf2, mo_coeff)

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        h1e_ao = agf2._scf.get_hcore()
        fock_ao = h1e_ao + agf2._scf.get_veff(agf2.mol, dm)

        self.h1e = np.dot(np.dot(mo_coeff.conj().T, h1e_ao), mo_coeff)
        self.fock = np.dot(np.dot(mo_coeff.conj().T, fock_ao), mo_coeff)

        self.e_hf = agf2._scf.e_tot

        self.nocc = agf2.nocc
        self.mol = agf2.mol

        mo_e = self.fock.diagonal()
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            #TODO: what is a good value for this gap? 1e-5 from CCSD and GW
            logger.warn(agf2, 'HOMO-LUMO gap %s too small for RAGF2', gap)

        return self

def _get_blksize(max_memory_total, *sizes):
    ''' Gets a block size such that the sum of the product of 
        :attr:`sizes` with :attr:`blksize` is less than available
        memory.

        If multiple tuples are provided, then the maximum will
        be used.
    '''
    #NOTE: does pyscf have this already? if not it might be nice

    if isinstance(sizes[0], tuple):
        sum_of_sizes = max([sum(x) for x in sizes])
    else:
        sum_of_sizes = sum(sizes)

    mem_avail = max_memory_total - lib.current_memory()[0] # in MB
    mem_avail = mem_avail * 8e-6 # in bits
    mem_avail = mem_avail / 64 # in 64-bit floats

    return int(mem_avail / sum_of_sizes)

def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    nmo = eris.fock.shape[0]

    eri = ao2mo.incore.full(agf2._scf._eri, eris.mo_coeff, verbose=log)
    eri = ao2mo.addons.restore('s4', eri, agf2.nmo)

    eris.eri = eri

    log.timer('MO integral transformation', *cput0)
    
    return eris

def _make_mo_eris_outcore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    mol = agf2.mol
    mo_coeff = np.asarray(eris.mo_coeff, order='F')
    nao, nmo = mo_coeff.shape

    eris.feri = lib.H5TmpFile()
    #TODO: ioblk_size
    ao2mo.outcore.full(mol, mo_coeff, eris.feri, dataname='mo', 
                       max_memory=agf2.max_memory, verbose=log)
    eris.eri = eris.feri['mo']

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, gf_occ, gf_vir):
    ''' Returns ndarray
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    coeffs = (np.eye(agf2.nmo), gf_occ.coupling,
              gf_occ.coupling, gf_vir.coupling)
    shape = tuple(x.shape[1] for x in coeffs)

    qeri = ao2mo.incore.general(eri.eri, coeffs, compact=False, verbose=log)
    qeri = qeri.reshape(shape)

    log.timer_debug1('QMO integral transformation', *cput0)

    return qeri

def _make_qmo_eris_outcore(agf2, eri, gf_occ, gf_vir):
    ''' Returns H5 dataset
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    ci = cj = gf_occ.coupling
    ca = gf_vir.coupling
    ni = nj = gf_occ.naux
    na = gf_vir.naux
    nmo = agf2.nmo
    npair = nmo*(nmo+1)//2

    # possible to have incore MO, outcore QMO
    if getattr(eri, 'feri', None) is None:
        eri.feri = lib.H5TmpFile()
    elif 'qmo' in eri.feri: 
        del eri.feri['qmo']

    eri.feri.create_dataset('qmo', (nmo, ni, nj, na), 'f8')

    blksize = _get_blksize(agf2.max_memory, (nmo*npair, nj*na, npair), (nmo*ni, nj*na))
    blksize = min(nmo, max(BLKMIN, blksize))
    log.debug1('blksize = %d', blksize)

    #NOTE: are these stored in Fortran layout? if so this is not efficient
    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    for p0, p1 in lib.prange(0, nmo, blksize):
        idx = np.concatenate(tril2sq[p0:p1])

        buf = eri.eri[idx] # (blk, nmo, npair)
        buf = buf.reshape((p1-p0)*nmo, -1) # (blk*nmo, npair)

        jasym, nja, cja, sja = ao2mo.incore._conc_mos(cj, ca, compact=True)
        buf = ao2mo._ao2mo.nr_e2(buf, cja, sja, 's2kl', 's1')
        buf = buf.reshape(p1-p0, nmo, nj, na)

        buf = lib.einsum('xpja,pi->xija', buf, ci)  #NOTE: better way to do this?
        eri.feri['qmo'][p0:p1] = np.asarray(buf, order='C')
        
    log.timer_debug1('QMO integral transformation', *cput0)

    return eri.feri['qmo']



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=3)
    #mol = gto.M(atom='Li 0 0 0; Li 0 0 1', basis='sto6g', verbose=5)
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-11
    rhf.run()

    ragf2 = RAGF2(rhf, frozen=0)

    ragf2.run()
    ragf2.ipagf2(nroots=5)
    ragf2.eaagf2(nroots=5)

    print(mp.MP2(rhf, frozen=ragf2.frozen).run(verbose=0).e_corr)
    print(ragf2.e_mp2)

    #print()
    #import auxgf
    #kwargs = dict(etol=ragf2.conv_tol, dtol=ragf2.conv_tol_rdm1, maxiter=ragf2.max_cycle, fock_maxiter=ragf2.max_cycle_inner, fock_maxruns=ragf2.max_cycle_outer, diis_space=ragf2.diis_space, wtol=ragf2.weight_tol, damping=0)
    #gf2 = auxgf.agf2.RAGF2(auxgf.hf.RHF.from_pyscf(rhf), nmom=(None,0), verbose=False, **kwargs).run()
    #print('E(mp2) = %16.12f' % gf2.e_mp2)
    #print('E(1b)  = %16.12f' % gf2.e_1body)
    #print('E(2b)  = %16.12f' % gf2.e_2body)
    #print('E(cor) = %16.12f' % gf2.e_corr)
    #print('E(tot) = %16.12f' % gf2.e_tot)
    #print('IP     = %16.12f' % gf2.ip[0])
    #print('EA     = %16.12f' % gf2.ea[0])
    #print(gf2._energies['tot'])

    ragf2 = ragf2.density_fit()
    ragf2.run()
