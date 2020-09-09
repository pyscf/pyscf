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

#TODO max_memory
BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

#TODO warn about HOMO-LUMO gap? CCSD does this for <1e-5
#TODO move mo2qmo to _ChemistsERIs



#NOTE: h1e and eri not stored in agf2 - seems to be how pyscf does it despite being a bit messy
def kernel(agf2, eri=None, verbose=None):

    log = logger.new_logger(agf2, verbose)
    mf = agf2._scf
    mo_energy = agf2.mo_energy
    mo_coeff = agf2.mo_coeff

    cput1 = cput0 = (time.clock(), time.time())

    if getattr(mf, 'with_df', None) is not None:
        raise ValueError('DF integrals not supported in agf2.ragf2.RAGF2')

    if eri is None:
        eri = agf2.ao2mo()

    if verbose is None:
        verbose = agf2.verbose

    e_mp2, gf, se = agf2.init_aux(eri)
    log.info('E(MP2) = %.16g  E_corr(MP2) = %.16g', e_mp2, e_mp2+eri.e_hf)

    e_prev = 0.0
    for niter in range(1, agf2.max_cycle+1):
        gf, se, converged = agf2.fock_loop(eri, gf, se)
        se = agf2.build_se(eri, gf)
        gf = agf2.build_gf(eri, gf, se)

        e_1b = agf2.energy_1body(eri, gf)
        e_2b = agf2.energy_2body(gf, se)
        e_tot = e_1b + e_2b

        ip = agf2.get_ip(gf, nroots=1)[0][0]
        ea = agf2.get_ea(gf, nroots=1)[0][0]

        name = agf2.__class__.__name__
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


def mo2qmo(agf2, eri, gf_occ, gf_vir):
    ''' Transforms the ERIs from MO to QMO basis, either :math:`(xo|ov)` 
        or :math:`(xv|vo)` if :attr:`gf_occ` and :attr:`gf_vir` are swapped.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Returns:
        ndarray of ERIs, either :math:`(xo|ov)` or :math:`(xv|vo)` if 
        :attr:`gf_occ` and :attr:`gf_vir` are swapped.
    '''

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction
    assert type(eri) is _ChemistsERIs

    time0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    cx = np.eye(agf2.nmo)
    ci = cj = gf_occ.coupling
    ca = gf_vir.coupling
    coeffs = (cx, ci, cj, ca)

    if mpi_helper.size == 1:
        eri_qmo = ao2mo.incore.general(eri.eri, coeffs, compact=False)
        eri_qmo = eri_qmo.reshape(tuple(c.shape[1] for c in coeffs))
    else:
        raise NotImplementedError #TODO

    log.timer_debug1('eri_qmo', *time0)

    return eri_qmo


def build_se_part(agf2, eri, gf_occ, gf_vir):
    ''' Builds either the auxiliaries of the occupied self-energy,
        or virtual if :attr:`gf_occ` and :attr:`gf_vir` are swapped.
    
    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf_occ : GreensFunction
            Occupied Green's function
        gf_vir : GreensFunction
            Virtual Green's function

    Returns:
        :class:`SelfEnergy`
    '''
    #TODO: C code
    #TODO: mpi

    time0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ) is aux.GreensFunction
    assert type(gf_vir) is aux.GreensFunction
    assert type(eri) is _ChemistsERIs

    nmo = agf2.nmo  #NOTE: tol is unlikely to be met at (None,0)
    tol = agf2.weight_tol
    eri_qmo = agf2.mo2qmo(eri, gf_occ, gf_vir)

    vv = np.zeros((agf2.nmo, agf2.nmo))
    vev = np.zeros((agf2.nmo, agf2.nmo))

    eja = lib.direct_sum('j,a->ja', gf_occ.energy, -gf_vir.energy)
    eja = eja.ravel()

    for i in range(gf_occ.naux): #TODO: allow for larger slices if BLKMIN is not met?
        xija = eri_qmo[:,i].reshape(nmo, -1)
        xjia = eri_qmo[:,:,i].reshape(nmo, -1)

        eija = eja + gf_occ.energy[i]

        vv = lib.dot(xija, xija.T, alpha=2, beta=1, c=vv)
        vv = lib.dot(xija, xjia.T, alpha=-1, beta=1, c=vv)

        exija = xija * eija[None]

        vev = lib.dot(exija, xija.T, alpha=2, beta=1, c=vev)
        vev = lib.dot(exija, xjia.T, alpha=-1, beta=1, c=vev)

    b = np.linalg.cholesky(vv).T
    b_inv = np.linalg.inv(b)

    m = np.dot(np.dot(b_inv.T, vev), b_inv)

    e, c = np.linalg.eigh(m)
    c = np.dot(b.T, c[:agf2.nmo])

    se = aux.SelfEnergy(e, c, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    log.timer_debug1('se part', *time0)

    return se


def get_fock(agf2, eri, gf=None, rdm1=None):
    ''' Computes the physical space Fock matrix in MO basis. One of
        :attr:`gf` or :attr:`rdm1` must be passed, with the latter
        prioritised if both are passed.

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

    assert type(eri) is _ChemistsERIs

    if gf is not None:
        rdm1 = agf2.make_rdm1(gf)
    assert rdm1 is not None

    if mpi_helper.size == 1:
        vj, vk = _vhf.incore(eri.eri, rdm1)
    else:
        raise NotImplementedError #TODO

    fock = eri.h1e + vj - 0.5 * vk

    return fock


def fock_loop(agf2, eri, gf, se, get_fock=None):
    ''' Self-consistent loop for the density matrix via the HF self-
        consistent field.

    Args:
        eri : _ChemistERIS
            Electronic repulsion integrals
        gf : GreensFunction
            Auxiliaries of the Green's function
        se : SelfEnergy
            Auxiliaries of the self-energy

    Kwargs:
        get_fock : callable
            Function to get the Fock matrix. Should be a callable in
            the format of :func:`get_fock`. Default value is 
            :class:`agf2.get_fock`.

    Returns:
        :class:`SelfEnergy`, :class:`GreensFunction` and a boolean 
        indicating wheter convergence was successful. 
    '''

    assert type(gf) is aux.GreensFunction
    assert type(se) is aux.SelfEnergy
    assert type(eri) is _ChemistsERIs

    time0 = time1 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    if get_fock is None:
        get_fock = agf2.get_fock

    diis = lib.diis.DIIS(agf2)
    diis.space = agf2.diis_space
    diis.min_space = agf2.diis_min_space
    fock = get_fock(eri, gf)

    nelec = agf2.nocc * 2
    nmo = agf2.nmo
    naux = se.naux
    nqmo = nmo + naux

    buf = np.zeros((nqmo, nqmo))
    converged = False

    for niter1 in range(1, agf2.max_cycle_rdm1+1):
        kwargs = dict(tol=agf2.conv_tol_nelec, maxiter=agf2.max_cycle_nelec)
        se, opt = minimize_chempot(se, fock, nelec, x0=se.chempot, **kwargs)

        for niter2 in range(1, agf2.max_cycle_nelec+1):
            w, v = se.eig(fock, chempot=0.0, out=buf)
            se.chempot, nerr = binsearch_chempot((w, v), nmo, nelec)
            gf = aux.GreensFunction(w, v[:nmo], chempot=se.chempot)

            fock = agf2.get_fock(eri, gf)
            rdm1 = agf2.make_rdm1(gf)
            fock = diis.update(fock, xerr=None)  #NOTE: should we silence repeated linear dependence warnings?

            if niter2 > 1:
                derr = np.max(np.absolute(rdm1 - rdm1_prev))
                if derr < agf2.conv_tol_rdm1:
                    break

            rdm1_prev = rdm1.copy()

        log.debug1('fock loop %d  cycles = %d  dN = %.3g  |ddm| = %.3g',
                   niter1, niter2, nerr, derr)
        time1 = log.timer_debug1('fock loop %d'%niter1, *time1)

        if derr < agf2.conv_tol_rdm1 and abs(nerr) < agf2.conv_tol_nelec:
            converged = True
            break

    log.info('fock converged = %s  chempot = %.9g  dN = %.3g  |ddm| = %.3g',
             converged, se.chempot, nerr, derr)

    gf = se.get_greens_function(fock)

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
    assert type(eri) is _ChemistsERIs

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


def energy_mp2(agf2, mo, se):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula for an MP2 self-energy. Per the
        definition of one- and two-body partitioning in the Dyson
        equation, this result is half of :fucn:`energy_2body`.

    Args:
        mo : 1D array
            MO energies
        se : SelfEnergy
            Auxiliaries of the self-energy

    Returns
        MP2 energy
    '''

    assert type(se) is aux.SelfEnergy

    occ = mo < se.chempot
    se_vir = se.get_virtual()

    vxk = se_vir.coupling[occ]
    dxk = lib.direct_sum('x,k->xk', mo[occ], -se_vir.energy)

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
        max_cycle_rdm1 : int
            Maximum number of Fock loop iterations. Default value is 20.
        max_cycle_nelec : int
            Maximum number of electron number loop iterations. Default
            value is 50.
        weight_tol : float
            Threshold in spectral weight of auxiliaries to be considered
            zero. Default 1e-11.
        diis_space : int
            DIIS space size for Fock loop iterations. Default value is 6.
        diis_min_space : 
            Minimum space of DIIS. Default value is 1.

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
        #TODO: allow frozen - use mp.mp2.get_nocc, get_nmo, get_frozen_mask

        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory  #TODO: memory checks

        # NOTE: should we just inherit any of these from CCSD? that seems common in pyscf
        self.conv_tol = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol', 1e-7)
        self.conv_tol_rdm1 = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol_rdm1', 1e-6)
        self.conv_tol_nelec = getattr(__config__, 'agf2_ragf2_RAGF2_conv_tol_nelec', 1e-6)
        self.max_cycle = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle', 50)
        self.max_cycle_rdm1 = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle_rdm1', 20)
        self.max_cycle_nelec = getattr(__config__, 'agf2_ragf2_RAGF2_max_cycle_nelec', 50)
        self.weight_tol = getattr(__config__, 'agf2_ragf2_RAGF2_weight_tol', 1e-11)
        self.diis_space = getattr(__config__, 'agf2_ragf2_RAGF2_diis_space', 6)
        self.diis_min_space = getattr(__config__, 'agf2_ragf2_RAGF2_diis_min_space', 1)

        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.se = None
        self.gf = None
        self.e_1b = mf.e_tot
        self.e_2b = 0.0
        self.e_mp2 = 0.0
        self._nmo = mo_coeff.shape[1] #TODO: from mp.mp2 to support frozen
        self._nocc = np.sum(self.mo_occ > 0)
        self.converged = False
        self._keys = set(self.__dict__.keys())

    mo2qmo = mo2qmo
    energy_1body = energy_1body
    energy_2body = energy_2body
    energy_mp2 = energy_mp2
    fock_loop = fock_loop

    def ao2mo(self, mo_coeff=None):
        ''' Get the electronic repulsion integrals in MO basis.
        '''
        eri = _ChemistsERIs(self, mo_coeff)

    def make_rdm1(self, gf=None):
        ''' Computes the one-body reduced density matrix in MO basis.

        Kwargs:
            gf : GreensFunction
                Auxiliaries of the Green's function

        Returns:
            ndarray of density matrix
        '''

        if gf is None: gf = self.get_init_aux(with_se=False)[0]

        return gf.make_rdm1()

    def get_fock(self, eri=None, gf=None, rdm1=None):
        if eri is None: eri = self.ao2mo()

        return get_fock(self, eri, gf=gf, rdm1=rdm1)

    def get_init_aux(self, eri=None, with_se=True):
        return self.init_aux(eri, with_se=with_se)[1:]
    def init_aux(self, eri=None, with_se=True):
        ''' Builds the Hartree-Fock Green's function.

        Kwargs:
            eri : _ChemistsERIs
                Electronic repulsion integrals

        Returns:
            MP2 energy, :class:`GreensFunction`, :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()

        mo_energy = self.mo_energy
        mo_coeff = self.mo_coeff

        chempot = binsearch_chempot(eri.fock, self.nmo, self.nocc*2)[0]
        gf = aux.GreensFunction(mo_energy, np.eye(self.nmo), chempot=chempot)

        if with_se:
            se = self.build_se(eri, gf)
            self.e_mp2 = self.energy_mp2(mo_energy, se)
        else:
            se = None

        return self.e_mp2, gf, se

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

        if gf is None: return self.get_init_aux(eri, with_se=False)[0]

        if eri is None: eri = self.ao2mo()
        if se is None: se = self.build_se(eri, gf)

        fock = self.get_fock(eri, gf)

        return se.get_greens_function(fock)

    def build_se(self, eri=None, gf=None):
        ''' Builds the auxiliaries of the self-energy.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : GreensFunction
                Auxiliaries of the Green's function

        Returns
            :class:`SelfEnergy`
            '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.get_init_aux(eri, with_se=False)[0]

        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        se_occ = build_se_part(self, eri, gf_occ, gf_vir)
        se_vir = build_se_part(self, eri, gf_vir, gf_occ)

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
        log.info('max_cycle_rdm1 = %g' % self.max_cycle_rdm1)
        log.info('max_cycle_nelec = %g' % self.max_cycle_nelec)
        log.info('weight_tol = %g' % self.weight_tol)
        log.info('diis_space = %d' % self.diis_space)
        log.info('diis_min_space = %d', self.diis_min_space)
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

        #NOTE: i think we can rely on gf always being set already when this function is called?
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

    def kernel(self, eri=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eri is None:
            eri = self.ao2mo()

        self.converged, self.e_1b, self.e_2b, self.gf, self.se = \
                kernel(self, eri=eri, verbose=self.verbose)

        self._finalize()

        return self.converged, self.e_1b, self.e_2b, self.gf, self.se

    def dump_chk(self):
        raise NotImplementedError #TODO

    def get_ip(self, gf, nroots=1):
        gf_occ = gf.get_occupied()
        e_ip = list(gf_occ.energy[-nroots:])[::-1]
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
            logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g', n, en, qpwt)

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
            logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g', n, en, qpwt)

        if nroots == 1:
            return e_ea[0], v_ea[0]
        else:
            return e_ea, v_ea

    ipragf2 = ipagf2
    earagf2 = eaagf2

    @property
    def nmo(self):
        return self._nmo
    @nmo.setter
    def nmo(self, val):
        self._nmo = val
        
    @property
    def nocc(self):
        return self._nocc
    @nocc.setter
    def nocc(self, val):
        self._nocc = val

    @property
    def e_tot(self):
        return self.e_1b + self.e_2b

    @property
    def e_corr(self):
        return self.e_tot - self._scf.e_tot


class _ChemistsERIs:
    def __init__(self, agf2, mo_coeff=None, sym_out='s8'):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = agf2.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        h1e_ao = agf2._scf.get_hcore()
        self.h1e = lib.einsum('pq,pi,qj->ij', h1e_ao, mo_coeff.conj(), mo_coeff)

        fock_ao = h1e_ao + agf2._scf.get_veff()
        self.fock = lib.einsum('pq,pi,qj->ij', fock_ao, mo_coeff.conj(), mo_coeff)

        rdm1_ao = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        ovlp_ao = agf2._scf.get_ovlp()
        sds = lib.einsum('pq,pi,qj->ij', rdm1_ao, ovlp_ao.conj(), ovlp_ao)
        self.rdm1 = lib.einsum('pq,pi,qj->ij', sds, mo_coeff.conj(), mo_coeff)

        eri_ao = agf2._scf._eri
        self.eri = ao2mo.incore.full(eri_ao, mo_coeff, compact=True)
        self.eri = ao2mo.addons.restore(sym_out, self.eri, agf2.nmo)

        self.e_hf = agf2._scf.e_tot


if __name__ == '__main__':
    from pyscf import gto, scf, mp

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='6-31g', verbose=9)
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-11
    rhf.run()

    ragf2 = RAGF2(rhf)

    #eri = ragf2.ao2mo()
    #gf, se = ragf2.get_init_aux(eri)
    #e_mp2 = ragf2.energy_mp2(rhf.mo_energy, se)
    #e_mp2_ref = mp.MP2(rhf).run(verbose=0).e_corr
    #print('MP2 error', abs(e_mp2 - e_mp2_ref))
    #print('Fock error', np.max(abs(eri.fock - ragf2.get_fock(eri, gf=gf))))

    #import auxgf
    #eri = ragf2.ao2mo()
    #se_ref = auxgf.aux.build_mp2(np.diag(eri.fock), ao2mo.addons.restore('s1', eri.eri, ragf2.nmo))
    #se_ref = se_ref.se_compress(eri.fock, nmom=0)
    #print('Aux energy error', np.max(abs(se.energy - se_ref.e)))
    #print('Aux coupling error', np.max(abs(np.dot(se.coupling, se.coupling.T) - np.dot(se_ref.v, se_ref.v.T))))

    #print(e_mp2)
    #print(ragf2.energy_2body(gf, se))
    #gf, se, converged = ragf2.fock_loop(eri, gf, se)
    #print(ragf2.energy_2body(gf, se))

    ragf2.run()
    ragf2.ipragf2(nroots=5)    
    ragf2.earagf2(nroots=5)






