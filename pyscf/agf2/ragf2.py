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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Auxiliary second-order Green's function perturbation theory
'''

import numpy as np
import copy
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.scf import _vhf
from pyscf.agf2 import mpi_helper, _agf2
from pyscf.agf2 import aux_space as aux
from pyscf.agf2 import chkfile as chkutil
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.mp.mp2 import get_frozen_mask as _get_frozen_mask

BLKMIN = getattr(__config__, 'agf2_blkmin', 1)


def kernel(agf2, eri=None, gf=None, se=None, verbose=None, dump_chk=True):

    log = logger.new_logger(agf2, verbose)
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    name = agf2.__class__.__name__

    if eri is None: eri = agf2.ao2mo()
    if gf is None: gf = agf2.gf
    if se is None: se = agf2.se
    if verbose is None: verbose = agf2.verbose

    if gf is None:
        gf = agf2.init_gf()
        gf_froz = agf2.init_gf(frozen=True)
    else:
        gf_froz = gf

    if se is None:
        se = agf2.build_se(eri, gf_froz)

    if dump_chk:
        agf2.dump_chk(gf=gf, se=se)

    if isinstance(agf2.diis, lib.diis.DIIS):
        diis = agf2.diis
    elif agf2.diis:
        diis = lib.diis.DIIS(agf2)
        diis.space = agf2.diis_space
        diis.min_space = agf2.diis_min_space
    else:
        diis = None

    e_init = agf2.energy_mp2(agf2.mo_energy, se)
    log.info('E(init) = %.16g  E_corr(init) = %.16g', e_init+eri.e_hf, e_init)

    e_1b = eri.e_hf
    e_2b = e_init

    e_prev = 0.0
    se_prev = None
    converged = False
    for niter in range(1, agf2.max_cycle+1):
        if agf2.damping != 0.0:
            se_prev = copy.deepcopy(se)

        # one-body terms
        gf, se, fock_conv = agf2.fock_loop(eri, gf, se)
        e_1b = agf2.energy_1body(eri, gf)

        # two-body terms
        se = agf2.build_se(eri, gf, se_prev=se_prev)
        se = agf2.run_diis(se, diis)
        e_2b = agf2.energy_2body(gf, se)

        if dump_chk:
            agf2.dump_chk(gf=gf, se=se)

        e_tot = e_1b + e_2b

        ip = agf2.get_ip(gf, nroots=1)[0][0]
        ea = agf2.get_ea(gf, nroots=1)[0][0]

        log.info('cycle = %3d  E(%s) = %.15g  E_corr(%s) = %.15g  dE = %.9g',
                 niter, name, e_tot, name, e_tot-eri.e_hf, e_tot-e_prev)
        log.info('E_1b = %.15g  E_2b = %.15g', e_1b, e_2b)
        log.info('IP = %.15g  EA = %.15g', ip, ea)
        cput1 = log.timer('%s iter'%name, *cput1)

        if abs(e_tot - e_prev) < agf2.conv_tol:
            converged = True
            break

        e_prev = e_tot

    if dump_chk:
        agf2.dump_chk(gf=gf, se=se)

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

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction

    nmo = eri.nmo
    tol = agf2.weight_tol
    facs = {'os_factor': os_factor, 'ss_factor': ss_factor}

    ci, ei = gf_occ.coupling, gf_occ.energy
    ca, ea = gf_vir.coupling, gf_vir.energy

    mem_incore = (gf_occ.nphys*gf_occ.naux**2*gf_vir.naux) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mem_incore+mem_now < agf2.max_memory) or agf2.incore_complete:
        qeri = _make_qmo_eris_incore(agf2, eri, (ci, ci, ca))
    else:
        qeri = _make_qmo_eris_outcore(agf2, eri, (ci, ci, ca))

    if isinstance(qeri, np.ndarray):
        vv, vev = _agf2.build_mats_ragf2_incore(qeri, ei, ea, **facs)
    else:
        vv, vev = _agf2.build_mats_ragf2_outcore(qeri, ei, ea, **facs)

    e, c = _agf2.cholesky_build(vv, vev)
    se = aux.SelfEnergy(e, c, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = get_frozen_mask(agf2)
        coupling = np.zeros((nmo, se.naux))
        coupling[mask] = se.coupling
        se = aux.SelfEnergy(se.energy, coupling, chempot=se.chempot)

    log.timer('se part', *cput0)

    return se


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

        blksize = _agf2.get_blksize(agf2.max_memory, (nmo*npair, nmo**3))
        blksize = min(1, max(BLKMIN, blksize))
        logger.debug1(agf2, 'blksize (ragf2.get_jk) = %d' % blksize)

        tril2sq = lib.square_mat_in_trilu_indices(nmo)
        for p0, p1 in lib.prange(0, nmo, blksize):
            idx = list(np.concatenate(tril2sq[p0:p1]))
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
        indicating whether convergence was successful.
    '''

    assert type(gf) is aux.GreensFunction
    assert type(se) is aux.SelfEnergy

    cput0 = cput1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    diis = lib.diis.DIIS(agf2)
    diis.space = agf2.fock_diis_space
    diis.min_space = agf2.fock_diis_min_space
    fock = agf2.get_fock(eri, gf)

    nelec = eri.nocc * 2
    nmo = eri.nmo
    naux = se.naux
    nqmo = nmo + naux
    buf = np.zeros((nqmo, nqmo))
    converged = False
    opts = {'tol': agf2.conv_tol_nelec, 'maxiter': agf2.max_cycle_inner}
    rdm1_prev = 0

    for niter1 in range(1, agf2.max_cycle_outer+1):
        se, opt = minimize_chempot(se, fock, nelec, x0=se.chempot, **opts)

        for niter2 in range(1, agf2.max_cycle_inner+1):
            w, v = se.eig(fock, chempot=0.0, out=buf)
            se.chempot, nerr = binsearch_chempot((w, v), nmo, nelec)

            w, v = se.eig(fock, out=buf)
            gf = aux.GreensFunction(w, v[:nmo], chempot=se.chempot)

            fock = agf2.get_fock(eri, gf)
            rdm1 = agf2.make_rdm1(gf)
            fock = diis.update(fock, xerr=None)

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
    log.timer('fock loop', *cput0)

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

    for l in mpi_helper.nrange(gf_occ.naux):
        vxl = gf_occ.coupling[:,l]
        vxk = se_vir.coupling
        dlk = gf_occ.energy[l] - se_vir.energy

        vv = vxk * vxl[:,None]
        e2b += lib.einsum('xk,yk,k->', vv, vv.conj(), 1./dlk)

    e2b *= 2

    mpi_helper.barrier()
    e2b = mpi_helper.allreduce(e2b)

    return np.ravel(e2b.real)[0]


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
        incore_complete : bool
            Avoid all I/O. Default is False.
        conv_tol : float
            Convergence threshold for AGF2 energy. Default value is 1e-7
        conv_tol_rdm1 : float
            Convergence threshold for first-order reduced density matrix.
            Default value is 1e-8.
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
        diis : bool or lib.diis.DIIS
            Whether to use DIIS, can also be a lib.diis.DIIS object. Default
            value is True.
        diis_space : int
            DIIS space size. Default value is 8.
        diis_min_space : int
            Minimum space of DIIS. Default value is 1.
        fock_diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        fock_diis_min_space : int
            Minimum space of DIIS. Default value is 1.
        os_factor : float
            Opposite-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        damping : float
            Damping factor for the self-energy. Default value is 0.0

    Saved results

        e_corr : float
            AGF2 correlation energy
        e_tot : float
            Total energy (HF + correlation)
        e_1b : float
            One-body part of :attr:`e_tot`
        e_2b : float
            Two-body part of :attr:`e_tot`
        e_init : float
            Initial correlation energy (truncated MP2)
        converged : bool
            Whether convergence was successful
        se : SelfEnergy
            Auxiliaries of the self-energy
        gf : GreensFunction
            Auxiliaries of the Green's function
    '''

    async_io = getattr(__config__, 'agf2_async_io', True)
    incore_complete = getattr(__config__, 'agf2_incore_complete', False)

    _keys = {
        'async_io', 'mol', 'incore_complete',
        'conv_tol', 'conv_tol_rdm1', 'conv_tol_nelec', 'max_cycle',
        'max_cycle_outer', 'max_cycle_inner', 'weight_tol', 'fock_diis_space',
        'fock_diis_min_space', 'diis', 'diis_space', 'diis_min_space',
        'os_factor', 'ss_factor', 'damping',
        'mo_energy', 'mo_coeff', 'mo_occ', 'se', 'gf', 'e_1b', 'e_2b', 'e_init',
        'frozen', 'converged', 'chkfile',
    }

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):

        if mo_energy is None: mo_energy = mpi_helper.bcast(mf.mo_energy)
        if mo_coeff is None: mo_coeff = mpi_helper.bcast(mf.mo_coeff)
        if mo_occ is None: mo_occ = mpi_helper.bcast(mf.mo_occ)

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.conv_tol = getattr(__config__, 'agf2_conv_tol', 1e-7)
        self.conv_tol_rdm1 = getattr(__config__, 'agf2_conv_tol_rdm1', 1e-8)
        self.conv_tol_nelec = getattr(__config__, 'agf2_conv_tol_nelec', 1e-6)
        self.max_cycle = getattr(__config__, 'agf2_max_cycle', 50)
        self.max_cycle_outer = getattr(__config__, 'agf2_max_cycle_outer', 20)
        self.max_cycle_inner = getattr(__config__, 'agf2_max_cycle_inner', 50)
        self.weight_tol = getattr(__config__, 'agf2_weight_tol', 1e-11)
        self.fock_diis_space = getattr(__config__, 'agf2_diis_space', 6)
        self.fock_diis_min_space = getattr(__config__, 'agf2_diis_min_space', 1)
        self.diis = getattr(__config__, 'agf2_diis', True)
        self.diis_space = getattr(__config__, 'agf2_diis_space', 8)
        self.diis_min_space = getattr(__config__, 'agf2_diis_min_space', 1)
        self.os_factor = getattr(__config__, 'agf2_os_factor', 1.0)
        self.ss_factor = getattr(__config__, 'agf2_ss_factor', 1.0)
        self.damping = getattr(__config__, 'agf2_damping', 0.0)

        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.se = None
        self.gf = None
        self.e_1b = mf.e_tot
        self.e_2b = 0.0
        self.e_init = 0.0
        self.frozen = frozen
        self._nmo = None
        self._nocc = None
        self.converged = False
        self.chkfile = mf.chkfile

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
                (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            eri = _make_mo_eris_incore(self, mo_coeff)
        else:
            logger.warn(self, 'MO eris are outcore - this may be very '
                              'slow for agf2. increasing max_memory or '
                              'using density fitting is recommended.')
            eri = _make_mo_eris_outcore(self, mo_coeff)

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

        self.e_init = energy_mp2(self, mo_energy, se)

        return self.e_init

    def init_gf(self, frozen=False):
        ''' Builds the Hartree-Fock Green's function.

        Returns:
            :class:`GreensFunction`, :class:`SelfEnergy`
        '''

        energy = self.mo_energy
        coupling = np.eye(self.nmo)

        chempot = binsearch_chempot(np.diag(energy), self.nmo, self.nocc*2)[0]

        if frozen:
            mask = get_frozen_mask(self)
            energy = energy[mask]
            coupling = coupling[:,mask]

        gf = aux.GreensFunction(energy, coupling, chempot=chempot)

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

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
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
            se_prev : SelfEnergy
                Previous self-energy for damping. Default value is None

        Returns:
            :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = {'os_factor': os_factor, 'ss_factor': ss_factor}
        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        if gf_occ.naux == 0 or gf_vir.naux == 0:
            logger.warn(self, 'Attempting to build a self-energy with '
                              'no (i,j,a) or (a,b,i) configurations.')
            se = aux.SelfEnergy([], [[],]*self.nmo, chempot=gf.chempot)
        else:
            se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
            se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)
            se = aux.combine(se_occ, se_vir)

        if se_prev is not None and self.damping != 0.0:
            se.coupling *= np.sqrt(1.0-self.damping)
            se_prev.coupling *= np.sqrt(self.damping)
            se = aux.combine(se, se_prev)
            se = se.compress(n=(None,0))

        return se

    def run_diis(self, se, diis=None):
        ''' Runs the direct inversion of the iterative subspace for the
            self-energy.

        Args:
            se : SelfEnergy
                Auxiliaries of the self-energy
            diis : lib.diis.DIIS
                DIIS object

        Returns:
            :class:`SelfEnergy`
        '''

        if diis is None:
            return se

        se_occ = se.get_occupied()
        se_vir = se.get_virtual()

        vv_occ = np.dot(se_occ.coupling, se_occ.coupling.T)
        vv_vir = np.dot(se_vir.coupling, se_vir.coupling.T)

        vev_occ = np.dot(se_occ.coupling * se_occ.energy[None], se_occ.coupling.T)
        vev_vir = np.dot(se_vir.coupling * se_vir.energy[None], se_vir.coupling.T)

        dat = np.array([vv_occ, vv_vir, vev_occ, vev_vir])
        dat = diis.update(dat)
        vv_occ, vv_vir, vev_occ, vev_vir = dat

        se_occ = aux.SelfEnergy(*_agf2.cholesky_build(vv_occ, vev_occ), chempot=se.chempot)
        se_vir = aux.SelfEnergy(*_agf2.cholesky_build(vv_vir, vev_vir), chempot=se.chempot)
        se = aux.combine(se_occ, se_vir)

        return se

    def energy_nuc(self):
        return self._scf.energy_nuc()


    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_rdm1 = %g', self.conv_tol_rdm1)
        log.info('conv_tol_nelec = %g', self.conv_tol_nelec)
        log.info('max_cycle = %g', self.max_cycle)
        log.info('max_cycle_outer = %g', self.max_cycle_outer)
        log.info('max_cycle_inner = %g', self.max_cycle_inner)
        log.info('weight_tol = %g', self.weight_tol)
        log.info('diis = %d', self.diis)
        log.info('diis_space = %d', self.diis_space)
        log.info('diis_min_space = %d', self.diis_min_space)
        log.info('fock_diis_space = %d', self.fock_diis_space)
        log.info('fock_diis_min_space = %d', self.fock_diis_min_space)
        log.info('os_factor = %g', self.os_factor)
        log.info('ss_factor = %g', self.ss_factor)
        log.info('damping = %g', self.damping)
        log.info('nmo = %s', self.nmo)
        log.info('nocc = %s', self.nocc)
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
        logger.note(self, 'Quasiparticle gap = %.16g', ip+ea)

        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def kernel(self, eri=None, gf=None, se=None, dump_chk=True):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if se is None: se = self.se

        if gf is None:
            gf = self.init_gf()
            gf_froz = self.init_gf(frozen=True)
        else:
            gf_froz = gf

        if se is None:
            se = self.build_se(eri, gf_froz)

        self.converged, self.e_1b, self.e_2b, self.gf, self.se = \
                kernel(self, eri=eri, gf=gf, se=se, verbose=self.verbose, dump_chk=dump_chk)

        self._finalize()

        return self.converged, self.e_1b, self.e_2b, self.gf, self.se

    def dump_chk(self, chkfile=None, key='agf2', gf=None, se=None,
                 frozen=None, nmom=None,
                 mo_energy=None, mo_coeff=None, mo_occ=None):
        if chkfile is None:
            chkfile = self.chkfile

        if not chkfile:
            return self

        chkutil.dump_agf2(self, chkfile, key,
                          gf, se, frozen, None,
                          mo_energy, mo_coeff, mo_occ)
        return self

    def update_from_chk_(self, chkfile=None, key='agf2'):
        if chkfile is None:
            chkfile = self.chkfile

        mol, agf2_dict = chkutil.load_agf2(chkfile, key)
        self.__dict__.update(agf2_dict)

        return self

    update = update_from_chk = update_from_chk_

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.agf2 import dfragf2
        myagf2 = dfragf2.DFRAGF2(self._scf)
        myagf2.__dict__.update(self.__dict__)

        if with_df is not None:
            myagf2.with_df = with_df

        if auxbasis is not None and myagf2.with_df.auxbasis != auxbasis:
            myagf2.with_df = myagf2.with_df.copy()
            myagf2.with_df.auxbasis = auxbasis

        return myagf2

    def get_ip(self, gf, nroots=5):
        gf_occ = gf.get_occupied()
        e_ip = list(-gf_occ.energy[-nroots:])[::-1]
        v_ip = list(gf_occ.coupling[:,-nroots:].T)[::-1]
        return e_ip, v_ip

    def ipagf2(self, nroots=5):
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
            logger.note(self, 'IP energy level %d E = %.16g  QP weight = %0.6g', n, en, qpwt)

        if nroots == 1:
            return e_ip[0], v_ip[0]
        else:
            return e_ip, v_ip

    def get_ea(self, gf, nroots=5):
        gf_vir = gf.get_virtual()
        e_ea = list(gf_vir.energy[:nroots])
        v_ea = list(gf_vir.coupling[:,:nroots].T)
        return e_ea, v_ea

    def eaagf2(self, nroots=5):
        ''' Find the (N+1)-electron charged excitations, corresponding
            to the smallest :attr:`nroots` poles of the virtual
            Green's function.

        Kwargs:
            See ipagf2()
        '''

        e_ea, v_ea = self.get_ea(self.gf, nroots=nroots)

        for n, en, vn in zip(range(nroots), e_ea, v_ea):
            qpwt = np.linalg.norm(vn)**2
            logger.note(self, 'EA energy level %d E = %.16g  QP weight = %0.6g', n, en, qpwt)

        if nroots == 1:
            return e_ea[0], v_ea[0]
        else:
            return e_ea, v_ea

    @property
    def nocc(self):
        if self._nocc is None:
            self._nocc = np.sum(self.mo_occ > 0)
        return self._nocc
    @nocc.setter
    def nocc(self, val):
        self._nocc = val

    @property
    def nmo(self):
        if self._nmo is None:
            self._nmo = self.mo_occ.size
        return self._nmo
    @nmo.setter
    def nmo(self, val):
        self._nmo = val

    @property
    def e_tot(self):
        return self.e_1b + self.e_2b

    @property
    def e_corr(self):
        # TODO Should HF energy be recalculated in case DFT orbitals or so were used?
        e_hf = mpi_helper.bcast(self._scf.e_tot)
        return self.e_tot - e_hf

    @property
    def qmo_energy(self):
        return self.gf.energy

    @property
    def qmo_coeff(self):
        ''' Gives the couplings in AO basis '''
        return np.dot(self.mo_coeff, self.gf.coupling)

    @property
    def qmo_occ(self):
        coeff = self.gf.get_occupied().coupling
        occ = 2.0 * np.linalg.norm(coeff, axis=0) ** 2
        vir = np.zeros_like(self.gf.get_virtual().energy)
        qmo_occ = np.concatenate([occ, vir])
        return qmo_occ


def get_frozen_mask(agf2):
    with lib.temporary_env(agf2, _nocc=None, _nmo=None):
        return _get_frozen_mask(agf2)


class _ChemistsERIs:
    ''' (pq|rs)

    MO integrals stored in s4 symmetry, we only need QMO integrals
    in low-symmetry tensors and s4 is highest supported by _vhf
    '''

    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nmo = None
        self.nocc = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        self.mo_coeff = mo_coeff

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        h1e_ao = agf2._scf.get_hcore()
        fock_ao = h1e_ao + agf2._scf.get_veff(agf2.mol, dm)

        self.h1e = np.dot(np.dot(mo_coeff.conj().T, h1e_ao), mo_coeff)
        self.fock = np.dot(np.dot(mo_coeff.conj().T, fock_ao), mo_coeff)

        self.h1e = mpi_helper.bcast(self.h1e)
        self.fock = mpi_helper.bcast(self.fock)

        self.e_hf = mpi_helper.bcast(agf2._scf.e_tot)

        self.nmo = agf2.nmo
        self.nocc = agf2.nocc
        self.mol = agf2.mol

        mo_e = self.fock.diagonal()
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(agf2, 'HOMO-LUMO gap %s may be too small for AGF2', gap)

        return self

def _make_mo_eris_incore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    eri = ao2mo.incore.full(agf2._scf._eri, eris.mo_coeff, verbose=log)
    eri = ao2mo.addons.restore('s4', eri, eris.nmo)

    eris.eri = eri

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_mo_eris_outcore(agf2, mo_coeff=None):
    ''' Returns _ChemistsERIs
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    mol = agf2.mol
    mo_coeff = np.asarray(eris.mo_coeff, order='F')

    eris.feri = lib.H5TmpFile()
    ao2mo.outcore.full(mol, mo_coeff, eris.feri, dataname='mo',
                       max_memory=agf2.max_memory, verbose=log)
    eris.eri = eris.feri['mo']

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs):
    ''' Returns ndarray
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    cx = np.eye(eri.nmo)
    if not (agf2.frozen is None or agf2.frozen == 0):
        mask = get_frozen_mask(agf2)
        cx = cx[:,mask]

    coeffs = (cx,) + coeffs
    shape = tuple(x.shape[1] for x in coeffs)

    qeri = ao2mo.incore.general(eri.eri, coeffs, compact=False, verbose=log)
    qeri = qeri.reshape(shape)

    log.timer('QMO integral transformation', *cput0)

    return qeri

def _make_qmo_eris_outcore(agf2, eri, coeffs):
    ''' Returns H5 dataset
    '''

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nmo = eri.nmo
    ci, cj, ca = coeffs
    ni = ci.shape[1]
    nj = cj.shape[1]
    na = ca.shape[1]
    npair = nmo*(nmo+1)//2

    mask = get_frozen_mask(agf2)
    frozen = np.sum(~mask)

    # possible to have incore MO, outcore QMO
    if getattr(eri, 'feri', None) is None:
        eri.feri = lib.H5TmpFile()
    elif 'qmo' in eri.feri:
        del eri.feri['qmo']

    eri.feri.create_dataset('qmo', (nmo-frozen, ni, nj, na), 'f8')

    blksize = _agf2.get_blksize(agf2.max_memory, (nmo*npair, nj*na, npair), (nmo*ni, nj*na))
    blksize = min(nmo, max(BLKMIN, blksize))
    log.debug1('blksize (ragf2._make_qmo_eris_outcore) = %d', blksize)

    tril2sq = lib.square_mat_in_trilu_indices(nmo)
    q1 = 0
    for p0, p1 in lib.prange(0, nmo, blksize):
        if not np.any(mask[p0:p1]):
            # block is fully frozen
            continue

        inds = np.arange(p0, p1)[mask[p0:p1]]
        q0, q1 = q1, q1 + len(inds)
        idx = list(np.concatenate(tril2sq[inds]))

        buf = eri.eri[idx] # (blk, nmo, npair)
        buf = buf.reshape((q1-q0)*nmo, -1) # (blk*nmo, npair)

        jasym, nja, cja, sja = ao2mo.incore._conc_mos(cj, ca, compact=True)
        buf = ao2mo._ao2mo.nr_e2(buf, cja, sja, 's2kl', 's1')
        buf = buf.reshape(q1-q0, nmo, nj, na)

        buf = lib.einsum('xpja,pi->xija', buf, ci)
        eri.feri['qmo'][q0:q1] = np.asarray(buf, order='C')

    log.timer('QMO integral transformation', *cput0)

    return eri.feri['qmo']



if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=3)
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-11
    rhf.run()

    ragf2 = RAGF2(rhf, frozen=0)

    ragf2.run()
    ragf2.ipagf2(nroots=5)
    ragf2.eaagf2(nroots=5)

    print(mp.MP2(rhf, frozen=ragf2.frozen).run(verbose=0).e_corr)
    print(ragf2.e_init)

    ragf2 = ragf2.density_fit()
    ragf2.run()
