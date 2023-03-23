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
RMP2
'''


import copy
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert (mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2_ss = emp2_os = 0
    for i in range(nocc):
        if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        edi = numpy.einsum('jab,jab', t2i, gi) * 2
        exi = -numpy.einsum('jab,jba', t2i, gi)
        emp2_ss += edi*0.5 + exi
        emp2_os += edi*0.5
        if with_t2:
            t2[i] = t2i

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2.real, t2


# Iteratively solve MP2 if non-canonical HF is provided
def _iterative_kernel(mp, eris, verbose=None):
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    emp2, t2 = mp.init_amps(eris=eris)
    log.info('Init E(MP2) = %.15g', emp2)

    adiis = lib.diis.DIIS(mp)

    conv = False
    for istep in range(mp.max_cycle):
        t2new = mp.update_amps(t2, eris)

        if isinstance(t2new, numpy.ndarray):
            normt = numpy.linalg.norm(t2new - t2)
            t2 = None
            t2new = adiis.update(t2new)
        else: # UMP2
            normt = numpy.linalg.norm([numpy.linalg.norm(t2new[i] - t2[i])
                                       for i in range(3)])
            t2 = None
            t2shape = [x.shape for x in t2new]
            t2new = numpy.hstack([x.ravel() for x in t2new])
            t2new = adiis.update(t2new)
            t2new = lib.split_reshape(t2new, t2shape)

        t2, t2new = t2new, None
        emp2, e_last = mp.energy(t2, eris), emp2
        log.info('cycle = %d  E_corr(MP2) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep+1, emp2, emp2 - e_last, normt)
        cput1 = log.timer('MP2 iter', *cput1)
        if abs(emp2-e_last) < mp.conv_tol and normt < mp.conv_tol_normt:
            conv = True
            break
    log.timer('MP2', *cput0)
    return conv, emp2, t2

def energy(mp, t2, eris):
    '''MP2 energy'''
    nocc, nvir = t2.shape[1:3]
    eris_ovov = numpy.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    ed = numpy.einsum('ijab,iajb', t2, eris_ovov) * 2
    ex = -numpy.einsum('ijab,ibja', t2, eris_ovov)
    emp2_ss = (ed*0.5 + ex).real
    emp2_os = ed.real*0.5
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)
    return emp2

def update_amps(mp, t2, eris):
    '''Update non-canonical MP2 amplitudes'''
    #assert (isinstance(eris, _ChemistsERIs))
    nocc, nvir = t2.shape[1:3]
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mp.level_shift

    foo = fock[:nocc,:nocc] - numpy.diag(mo_e_o)
    fvv = fock[nocc:,nocc:] - numpy.diag(mo_e_v)
    t2new  = lib.einsum('ijac,bc->ijab', t2, fvv)
    t2new -= lib.einsum('ki,kjab->ijab', foo, t2)
    t2new = t2new + t2new.transpose(1,0,3,2)

    eris_ovov = numpy.asarray(eris.ovov).reshape(nocc,nvir,nocc,nvir)
    t2new += eris_ovov.conj().transpose(0,2,1,3)
    eris_ovov = None

    eia = mo_e_o[:,None] - mo_e_v
    t2new /= lib.direct_sum('ia,jb->ijab', eia, eia)
    return t2new


def make_rdm1(mp, t2=None, eris=None, ao_repr=False):
    r'''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Kwargs:
        ao_repr : boolean
            Whether to transfrom 1-particle density matrix to AO
            representation.
    '''
    from pyscf.cc import ccsd_rdm
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = numpy.zeros((nocc,nvir), dtype=doo.dtype)
    dvo = dov.T
    return ccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                               ao_repr=ao_repr)

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo()
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = numpy.zeros((nocc,nocc), dtype=dtype)
    dm1vir = numpy.zeros((nvir,nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += lib.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                - lib.einsum('jca,jbc->ba', l2i, t2i)
        dm1occ += lib.einsum('iab,jab->ij', l2i, t2i) * 2 \
                - lib.einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir


def make_fno(mp, thresh=1e-6, pct_occ=None, nvir_act=None, t2=None):
    r'''
    Frozen natural orbitals

    Attributes:
        thresh : float
            Threshold on NO occupation numbers. Default is 1e-6 (very conservative).
        pct_occ : float
            Percentage of total occupation number. Default is None. If present, overrides `thresh`.
        nvir_act : int
            Number of virtual NOs to keep. Default is None. If present, overrides `thresh` and `pct_occ`.

    Returns:
        frozen : list or ndarray
            List of orbitals to freeze
        no_coeff : ndarray
            Semicanonical NO coefficients in the AO basis
    '''
    mf = mp._scf
    dm = mp.make_rdm1(t2=t2)

    nmo = mp.nmo
    nocc = mp.nocc
    n,v = numpy.linalg.eigh(dm[nocc:,nocc:])
    idx = numpy.argsort(n)[::-1]
    n,v = n[idx], v[:,idx]

    if nvir_act is None:
        if pct_occ is None:
            nvir_act = numpy.count_nonzero(n>thresh)
        else:
            print(numpy.cumsum(n/numpy.sum(n)))
            nvir_act = numpy.count_nonzero(numpy.cumsum(n/numpy.sum(n))<pct_occ)

    fvv = numpy.diag(mf.mo_energy[nocc:])
    fvv_no = numpy.dot(v.T, numpy.dot(fvv, v))
    _, v_canon = numpy.linalg.eigh(fvv_no[:nvir_act,:nvir_act])

    no_coeff_1 = numpy.dot(mf.mo_coeff[:,nocc:], numpy.dot(v[:,:nvir_act], v_canon))
    no_coeff_2 = numpy.dot(mf.mo_coeff[:,nocc:], v[:,nvir_act:])
    no_coeff = numpy.concatenate((mf.mo_coeff[:,:nocc], no_coeff_1, no_coeff_2), axis=1)

    return numpy.arange(nocc+nvir_act,nmo), no_coeff


def make_rdm2(mp, t2=None, eris=None, ao_repr=False):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if t2 is None: t2 = mp.t2
    nmo = nmo0 = mp.nmo
    nocc = nocc0 = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo()
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if mp.frozen is not None:
        nmo0 = mp.mo_occ.size
        nocc0 = numpy.count_nonzero(mp.mo_occ > 0)
        moidx = get_frozen_mask(mp)
        oidx = numpy.where(moidx & (mp.mo_occ > 0))[0]
        vidx = numpy.where(moidx & (mp.mo_occ ==0))[0]
    else:
        moidx = oidx = vidx = None

    dm1 = make_rdm1(mp, t2, eris)
    dm1[numpy.diag_indices(nocc0)] -= 2

    dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=dm1.dtype) # Chemist notation
    #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
    #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
    for i in range(nocc):
        if t2 is None:
            gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        else:
            t2i = t2[i]
        # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
        # above. Transposing it so that it be contracted with ERIs (in Chemist's
        # notation):
        #   E = einsum('pqrs,pqrs', eri, rdm2)
        dovov = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
        dovov *= 2
        if moidx is None:
            dm2[i,nocc:,:nocc,nocc:] = dovov
            dm2[nocc:,i,nocc:,:nocc] = dovov.conj().transpose(0,2,1)
        else:
            dm2[oidx[i],vidx[:,None,None],oidx[:,None],vidx] = dovov
            dm2[vidx[:,None,None],oidx[i],vidx[:,None],oidx] = dovov.conj().transpose(0,2,1)

    # Be careful with convention of dm1 and dm2
    #   dm1[q,p] = <p^\dagger q>
    #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
    #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
    # When adding dm1 contribution, dm1 subscripts need to be flipped
    for i in range(nocc0):
        dm2[i,i,:,:] += dm1.T * 2
        dm2[:,:,i,i] += dm1.T * 2
        dm2[:,i,i,:] -= dm1.T
        dm2[i,:,:,i] -= dm1

    for i in range(nocc0):
        for j in range(nocc0):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2

    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mp.mo_coeff)
    return dm2


def get_nocc(mp):
    if mp._nocc is not None:
        return mp._nocc
    elif mp.frozen is None:
        nocc = numpy.count_nonzero(mp.mo_occ > 0)
        assert (nocc > 0)
        return nocc
    elif isinstance(mp.frozen, (int, numpy.integer)):
        nocc = numpy.count_nonzero(mp.mo_occ > 0) - mp.frozen
        assert (nocc > 0)
        return nocc
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        occ_idx = mp.mo_occ > 0
        occ_idx[list(mp.frozen)] = False
        nocc = numpy.count_nonzero(occ_idx)
        assert (nocc > 0)
        return nocc
    else:
        raise NotImplementedError

def get_nmo(mp):
    if mp._nmo is not None:
        return mp._nmo
    elif mp.frozen is None:
        return len(mp.mo_occ)
    elif isinstance(mp.frozen, (int, numpy.integer)):
        return len(mp.mo_occ) - mp.frozen
    elif isinstance(mp.frozen[0], (int, numpy.integer)):
        return len(mp.mo_occ) - len(set(mp.frozen))
    else:
        raise NotImplementedError

def get_frozen_mask(mp):
    '''Get boolean mask for the restricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidx = numpy.ones(mp.mo_occ.size, dtype=bool)
    if mp._nmo is not None:
        moidx[mp._nmo:] = False
    elif mp.frozen is None:
        pass
    elif isinstance(mp.frozen, (int, numpy.integer)):
        moidx[:mp.frozen] = False
    elif len(mp.frozen) > 0:
        moidx[list(mp.frozen)] = False
    else:
        raise NotImplementedError
    return moidx

def get_e_hf(mp, mo_coeff=None):
    # Get HF energy, which is needed for total MP2 energy.
    if mo_coeff is None:
        mo_coeff = mp.mo_coeff
    dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
    vhf = mp._scf.get_veff(mp._scf.mol, dm)
    return mp._scf.energy_tot(dm=dm, vhf=vhf)


def as_scanner(mp):
    '''Generating a scanner/solver for MP2 PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total MP2 energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    MP2 and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

        >>> from pyscf import gto, scf, mp
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
        >>> mp2_scanner = mp.MP2(scf.RHF(mol)).as_scanner()
        >>> e_tot = mp2_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
        >>> e_tot = mp2_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    if isinstance(mp, lib.SinglePointScanner):
        return mp

    logger.info(mp, 'Set %s as a scanner', mp.__class__)

    class MP2_Scanner(mp.__class__, lib.SinglePointScanner):
        def __init__(self, mp):
            self.__dict__.update(mp.__dict__)
            self._scf = mp._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            self.reset(mol)

            mf_scanner = self._scf
            mf_scanner(mol)
            self.mo_coeff = mf_scanner.mo_coeff
            self.mo_occ = mf_scanner.mo_occ
            self.kernel(**kwargs)
            return self.e_tot
    return MP2_Scanner(mp)


class MP2(lib.StreamObject):
    '''restricted MP2 with canonical HF and non-canonical HF reference

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        conv_tol : float
            For non-canonical MP2, converge threshold for MP2
            correlation energy.  Default value is 1e-7.
        conv_tol_normt : float
            For non-canonical MP2, converge threshold for
            norm(t2).  Default value is 1e-5.
        max_cycle : int
            For non-canonical MP2, max number of MP2
            iterations.  Default value is 50.
        diis_space : int
            For non-canonical MP2, DIIS space size in MP2
            iterations.  Default is 6.
        level_shift : float
            A shift on virtual orbital energies to stablize the MP2 iterations.
        frozen : int or list
            If integer is given, the inner-most orbitals are excluded from MP2
            amplitudes.  Given the orbital indices (0-based) in a list, both
            occupied and virtual orbitals can be frozen in MP2 calculation.

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> # freeze 2 core orbitals
            >>> pt = mp.MP2(mf).set(frozen = 2).run()
            >>> # auto-generate the number of core orbitals to be frozen (1 in this case)
            >>> pt = mp.MP2(mf).set_frozen().run()
            >>> # freeze 2 core orbitals and 3 high lying unoccupied orbitals
            >>> pt.set(frozen = [0,1,16,17,18]).run()

    Saved results

        e_corr : float
            MP2 correlation correction
        e_corr_ss/os : float
            Same-spin and opposite-spin component of the MP2 correlation energy
        e_tot : float
            Total MP2 energy (HF + correlation)
        t2 :
            T amplitudes t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''

    # Use CCSD default settings for the moment
    max_cycle = getattr(__config__, 'cc_ccsd_CCSD_max_cycle', 50)
    conv_tol = getattr(__config__, 'cc_ccsd_CCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_ccsd_CCSD_conv_tol_normt', 1e-5)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

# For iterative MP2
        self.level_shift = 0

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_hf = None
        self.e_corr = None
        self.e_corr_ss = None
        self.e_corr_os = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

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

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_e_hf = get_e_hf

    def set_frozen(self, method='auto', window=(-1000.0, 1000.0)):
        from pyscf import mp
        is_gmp = isinstance(self, mp.gmp2.GMP2)
        from pyscf.cc.ccsd import set_frozen
        return set_frozen(self, method=method, window=window, is_gcc=is_gmp)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    @property
    def emp2(self):
        return self.e_corr

    @property
    def emp2_scs(self):
        # J. Chem. Phys. 118, 9095 (2003)
        return self.e_corr_ss*1./3. + self.e_corr_os*1.2

    @property
    def e_tot(self):
        return self.e_hf + self.e_corr

    @property
    def e_tot_scs(self):
        # J. Chem. Phys. 118, 9095 (2003)
        return self.e_hf + self.emp2_scs

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()

        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo(mo_coeff)

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            self.converged, self.e_corr, self.t2 = _iterative_kernel(self, eris)

        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()
        return self.e_corr, self.t2

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        log = logger.new_logger(self)
        log.note('E(%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot, self.e_corr)
        log.note('E(SCS-%s) = %.15g  E_corr = %.15g',
                 self.__class__.__name__, self.e_tot_scs, self.emp2_scs)
        log.info('E_corr(same-spin) = %.15g', self.e_corr_ss)
        log.info('E_corr(oppo-spin) = %.15g', self.e_corr_os)
        return self

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = make_rdm1
    make_fno = make_fno
    make_rdm2 = make_rdm2

    as_scanner = as_scanner

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfmp2
        mymp = dfmp2.DFMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        from pyscf.grad import mp2
        return mp2.Gradients(self)

    # For non-canonical MP2
    energy = energy
    update_amps = update_amps
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

RMP2 = MP2

from pyscf import scf
scf.hf.RHF.MP2 = lib.class_as_method(MP2)
scf.rohf.ROHF.MP2 = None


def _mo_energy_without_core(mp, mo_energy):
    return mo_energy[get_frozen_mask(mp)]

def _mo_without_core(mp, mo):
    return mo[:,get_frozen_mask(mp)]

def _mem_usage(nocc, nvir):
    nmo = nocc + nvir
    basic = ((nocc*nvir)**2 + nocc*nvir**2*2)*8 / 1e6
    incore = nocc*nvir*nmo**2/2*8 / 1e6 + basic
    outcore = basic
    return incore, outcore, basic

#TODO: Merge this _ChemistsERIs class with ccsd._ChemistsERIs class
class _ChemistsERIs:
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.orbspin = None
        self.ovov = None

    def _common_init_(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')

        self.mo_coeff = _mo_without_core(mp, mo_coeff)
        self.mol = mp.mol

        if mo_coeff is mp._scf.mo_coeff and mp._scf.converged:
            # The canonical MP2 from a converged SCF result. Rebuilding fock
            # can be skipped
            self.mo_energy = _mo_energy_without_core(mp, mp._scf.mo_energy)
            self.fock = numpy.diag(self.mo_energy)
        else:
            dm = mp._scf.make_rdm1(mo_coeff, mp.mo_occ)
            vhf = mp._scf.get_veff(mp.mol, dm)
            fockao = mp._scf.get_fock(vhf=vhf, dm=dm)
            self.fock = self.mo_coeff.conj().T.dot(fockao).dot(self.mo_coeff)
            self.mo_energy = self.fock.diagonal().real
        return self

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = numpy.asarray(mo_coeff[:,:nocc], order='F')
    cv = numpy.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        #ao2mo.outcore.general(mp.mol, (co,cv,co,cv), eris.feri,
        #                      max_memory=max_memory, verbose=log)
        #eris.ovov = eris.feri['eri_mo']
        eris.ovov = _ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    log.timer('Integral transformation', *time0)
    return eris

#
# the MO integral for MP2 is (ov|ov). This is the efficient integral
# (ij|kl) => (ij|ol) => (ol|ij) => (ol|oj) => (ol|ov) => (ov|ov)
#   or    => (ij|ol) => (oj|ol) => (oj|ov) => (ov|ov)
#
def _ao2mo_ovov(mp, orbo, orbv, feri, max_memory=2000, verbose=None):
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nao, nocc = orbo.shape
    nvir = orbv.shape[1]
    nbas = mol.nbas
    assert (nvir <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocc)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc**2*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    buf_i = numpy.empty((nocc*dmax**2*nao))
    buf_li = numpy.empty((nocc**2*dmax**2))
    buf1 = numpy.empty_like(buf_li)

    fint = gto.moleintor.getints4c
    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = numpy.ndarray((nocc,(i1-i0)*(j1-j0)*nao), buffer=buf_i)
                tmp_li = numpy.ndarray((nocc,nocc*(i1-i0)*(j1-j0)), buffer=buf_li)
                lib.ddot(orbo.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao), c=tmp_i)
                lib.ddot(orbo.T, tmp_i.reshape(nocc*(i1-i0)*(j1-j0),nao).T, c=tmp_li)
                tmp_li = tmp_li.reshape(nocc,nocc,(i1-i0),(j1-j0))
                save(str(count), tmp_li.transpose(1,0,2,3))
                buf_li, buf1 = buf1, buf_li
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = buf_i = buf_li = buf1 = None

    h5dat = feri.create_dataset('ovov', (nocc*nvir,nocc*nvir), 'f8',
                                chunks=(nvir,nvir))
    occblk = int(min(nocc, max(4, 250/nocc, max_memory*.9e6/8/(nao**2*nocc)/5)))
    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(jk_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = ftmp[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def save(i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,nocc*nvir)

    orbv = numpy.asarray(orbv, order='F')
    buf_prefecth = numpy.empty((occblk,nocc,nao,nao))
    buf = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocc,nao,nao)

                dat = _ao2mo.nr_e2(eri, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    return h5dat

del (WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()

    pt = MP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204019967288338)
    pt.max_memory = 1
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204019967288338)

    pt = MP2(scf.density_fit(mf, 'weigend'))
    print(pt.kernel()[0] - -0.204254500454)

    mf = scf.RHF(mol).run(max_cycle=1)
    pt = MP2(mf)
    print(pt.kernel()[0] - -0.204479914961218)
