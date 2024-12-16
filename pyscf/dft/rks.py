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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic restricted Kohn-Sham
'''


import textwrap
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import scf
from pyscf.scf import hf
from pyscf.scf import _vhf
from pyscf.scf import jk
from pyscf.scf.dispersion import parse_dft
from pyscf.dft import gen_grid
from pyscf.dft import numint
from pyscf import __config__

def get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + XC functional

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference Vxc potential matrix.
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian

    Returns:
        matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
        dm is a list of density matrices.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    ks.initialize_grids(mol, dm)

    t0 = (logger.process_clock(), logger.perf_counter())

    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)

    ni = ks._numint
    if hermi == 2:  # because rho = 0
        n, exc, vxc = 0, 0, 0
    else:
        max_memory = ks.max_memory - lib.current_memory()[0]
        n, exc, vxc = ni.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        if ks.do_nlc():
            if ni.libxc.is_nlc(ks.xc):
                xc = ks.xc
            else:
                assert ni.libxc.is_nlc(ks.nlc)
                xc = ks.nlc
            n, enlc, vnlc = ni.nr_nlc_vxc(mol, ks.nlcgrids, xc, dm,
                                          max_memory=max_memory)
            exc += enlc
            vxc += vnlc
            logger.debug(ks, 'nelec with nlc grids = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)

    incremental_jk = (ks._eri is None and ks.direct_scf and
                      getattr(vhf_last, 'vj', None) is not None)
    if incremental_jk:
        _dm = numpy.asarray(dm) - numpy.asarray(dm_last)
    else:
        _dm = dm
    if not ni.libxc.is_hybrid_xc(ks.xc):
        vk = None
        vj = ks.get_j(mol, _dm, hermi)
        if incremental_jk:
            vj += vhf_last.vj
        vxc += vj
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
        if omega == 0:
            vj, vk = ks.get_jk(mol, _dm, hermi)
            vk *= hyb
        elif alpha == 0: # LR=0, only SR exchange
            vj = ks.get_j(mol, _dm, hermi)
            vk = ks.get_k(mol, _dm, hermi, omega=-omega)
            vk *= hyb
        elif hyb == 0: # SR=0, only LR exchange
            vj = ks.get_j(mol, _dm, hermi)
            vk = ks.get_k(mol, _dm, hermi, omega=omega)
            vk *= alpha
        else: # SR and LR exchange with different ratios
            vj, vk = ks.get_jk(mol, _dm, hermi)
            vk *= hyb
            vklr = ks.get_k(mol, _dm, hermi, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        if incremental_jk:
            vj += vhf_last.vj
            vk += vhf_last.vk
        vxc += vj - vk * .5

        if ground_state:
            exc -= numpy.einsum('ij,ji', dm, vk).real * .5 * .5

    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
    return vxc

def get_vsap(ks, mol=None):
    '''Superposition of atomic potentials

    S. Lehtola, Assessment of initial guesses for self-consistent
    field calculations. Superposition of Atomic Potentials: simple yet
    efficient, J. Chem. Theory Comput. 15, 1593 (2019). DOI:
    10.1021/acs.jctc.8b01089. arXiv:1810.11659.

    This function evaluates the effective charge of a neutral atom,
    given by exchange-only LDA on top of spherically symmetric
    unrestricted Hartree-Fock calculations as described in

    S. Lehtola, L. Visscher, E. Engel, Efficient implementation of the
    superposition of atomic potentials initial guess for electronic
    structure calculations in Gaussian basis sets, J. Chem. Phys., in
    press (2020).

    The potentials have been calculated for the ground-states of
    spherically symmetric atoms at the non-relativistic level of theory
    as described in

    S. Lehtola, "Fully numerical calculations on atoms with fractional
    occupations and range-separated exchange functionals", Phys. Rev. A
    101, 012516 (2020). DOI: 10.1103/PhysRevA.101.012516

    using accurate finite-element calculations as described in

    S. Lehtola, "Fully numerical Hartree-Fock and density functional
    calculations. I. Atoms", Int. J. Quantum Chem. e25945 (2019).
    DOI: 10.1002/qua.25945

    .. note::
        This function will modify the input ks object.

    Args:
        ks : an instance of :class:`RKS`
            XC functional are controlled by ks.xc attribute.  Attribute
            ks.grids might be initialized.

    Returns:
        matrix Vsap = Vnuc + J + Vxc.
    '''
    if mol is None: mol = ks.mol
    t0 = (logger.process_clock(), logger.perf_counter())

    if ks.grids.coords is None:
        ks.grids.build(with_non0tab=True)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    ni = ks._numint
    max_memory = ks.max_memory - lib.current_memory()[0]
    vsap = ni.nr_sap(mol, ks.grids, max_memory=max_memory)
    return vsap

# The vhfopt of standard Coulomb operator can be used here as an approximate
# opt since long-range part Coulomb is always smaller than standard Coulomb.
# It's safe to prescreen LR integrals with the integral estimation from
# standard Coulomb.
def _get_k_lr(mol, dm, omega=0, hermi=0, vhfopt=None):
    import sys
    sys.stderr.write('This function is deprecated. '
                     'It is replaced by mol.get_k(mol, dm, omege=omega)')
    dm = numpy.asarray(dm)
# Note, ks object caches the ERIs for small systems. The cached eris are
# computed with regular Coulomb operator. ks.get_jk or ks.get_k do not evaluate
# the K matrix with the range separated Coulomb operator.  Here jk.get_jk
# function computes the K matrix with the modified Coulomb operator.
    nao = dm.shape[-1]
    dms = dm.reshape(-1,nao,nao)
    with mol.with_range_coulomb(omega):
        # Compute the long range part of ERIs temporarily with omega. Restore
        # the original omega when the block ends
        if vhfopt is None:
            contents = lambda: None # just a place_holder
        else:
            contents = vhfopt._this.contents
        with lib.temporary_env(contents,
                               fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
            intor = mol._add_suffix('int2e')
            vklr = jk.get_jk(mol, dms, ['ijkl,jk->il']*len(dms), intor=intor,
                             vhfopt=vhfopt)
    return numpy.asarray(vklr).reshape(dm.shape)


def energy_elec(ks, dm=None, h1e=None, vhf=None):
    r'''Electronic part of RKS energy.

    Note this function has side effects which cause mf.scf_summary updated.

    Args:
        ks : an instance of DFT class

        dm : 2D ndarray
            one-particle density matrix
        h1e : 2D ndarray
            Core hamiltonian

    Returns:
        RKS electronic energy and the 2-electron contribution
    '''
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    e1 = numpy.einsum('ij,ji->', h1e, dm).real
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    e2 = ecoul + exc

    ks.scf_summary['e1'] = e1
    ks.scf_summary['coul'] = ecoul
    ks.scf_summary['exc'] = exc
    logger.debug(ks, 'E1 = %s  Ecoul = %s  Exc = %s', e1, ecoul, exc)
    return e1+e2, e2


def prune_small_rho_grids_(ks, mol, dm, grids):
    rho = ks._numint.get_rho(mol, dm, grids, ks.max_memory)
    return grids.prune_by_density_(rho, ks.small_rho_cutoff)

def define_xc_(ks, description, xctype='LDA', hyb=0, rsh=(0,0,0)):
    libxc = ks._numint.libxc
    ks._numint = libxc.define_xc_(ks._numint, description, xctype, hyb, rsh)
    return ks

def _dft_common_init_(mf, xc='LDA,VWN'):
    raise DeprecationWarning

class KohnShamDFT:
    '''
    Attributes for Kohn-Sham DFT:
        xc : str
            'X_name,C_name' for the XC functional.  Default is 'lda,vwn'
        nlc : str
            'NLC_name' for the NLC functional.  Default is '' (i.e., None)
        omega : float
            Omega of the range-separated Coulomb operator e^{-omega r_{12}^2} / r_{12}
        grids : Grids object
            grids.level (0 - 9)  big number for large mesh grids. Default is 3

            radii_adjust
                | radi.treutler_atomic_radii_adjust (default)
                | radi.becke_atomic_radii_adjust
                | None : to switch off atomic radii adjustment

            grids.atomic_radii
                | radi.BRAGG_RADII  (default)
                | radi.COVALENT_RADII
                | None : to switch off atomic radii adjustment

            grids.radi_method  scheme for radial grids
                | radi.treutler  (default)
                | radi.delley
                | radi.mura_knowles
                | radi.gauss_chebyshev

            grids.becke_scheme  weight partition function
                | gen_grid.original_becke  (default)
                | gen_grid.stratmann

            grids.prune  scheme to reduce number of grids
                | gen_grid.nwchem_prune  (default)
                | gen_grid.sg1_prune
                | gen_grid.treutler_prune
                | None : to switch off grids pruning

            grids.symmetry  True/False  to symmetrize mesh grids (TODO)

            grids.atom_grid  Set (radial, angular) grids for particular atoms.
            Eg, grids.atom_grid = {'H': (20,110)} will generate 20 radial
            grids and 110 angular grids for H atom.

        small_rho_cutoff : float
            Drop grids if their contribution to total electrons smaller than
            this cutoff value.  Default is 1e-7.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', verbose=0)
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'b3lyp'
    >>> mf.kernel()
    -76.415443079840458
    '''

    _keys = {'xc', 'nlc', 'grids', 'disp', 'nlcgrids', 'small_rho_cutoff'}

    # Use rho to filter grids
    small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff', 1e-7)

    def __init__(self, xc='LDA,VWN'):
        # By default, self.nlc = '' and self.disp = None
        self.xc = xc
        self.nlc = ''
        self.disp = None
        self.grids = gen_grid.Grids(self.mol)
        self.grids.level = getattr(
            __config__, 'dft_rks_RKS_grids_level', self.grids.level)
        self.nlcgrids = gen_grid.Grids(self.mol)
        self.nlcgrids.level = getattr(
            __config__, 'dft_rks_RKS_nlcgrids_level', self.nlcgrids.level)
##################################################
# don't modify the following attributes, they are not input options
        self._numint = numint.NumInt()

    @property
    def omega(self):
        return self._numint.omega
    @omega.setter
    def omega(self, v):
        self._numint.omega = float(v)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('XC library %s version %s\n    %s',
                 self._numint.libxc.__name__,
                 self._numint.libxc.__version__,
                 self._numint.libxc.__reference__)

        if log.verbose >= logger.INFO:
            log.info('XC functionals = %s', self.xc)
            if hasattr(self._numint.libxc, 'xc_reference'):
                log.info(textwrap.indent('\n'.join(self._numint.libxc.xc_reference(self.xc)), '    '))

        self.grids.dump_flags(verbose)

        if self.do_nlc():
            log.info('** Following is NLC and NLC Grids **')
            if self.nlc:
                log.info('NLC functional = %s', self.nlc)
            else:
                log.info('NLC functional = %s', self.xc)
            self.nlcgrids.dump_flags(verbose)

        log.info('small_rho_cutoff = %g', self.small_rho_cutoff)
        return self

    define_xc_ = define_xc_

    def do_nlc(self):
        '''Check if the object needs to do nlc calculations

        if self.nlc == False (or 0), nlc is disabled regardless the value of self.xc
        if self.nlc == 'vv10', do nlc (vv10) regardless the value of self.xc
        if self.nlc == '', determined by self.xc, certain xc allows the nlc part
        '''
        xc, nlc, _ = parse_dft(self.xc)
        # If nlc is disabled via self.xc
        if nlc == 0:
            if self.nlc == '' or self.nlc == 0:
                return False
            else:
                raise RuntimeError('Conflict found between dispersion and xc.')

        # If nlc is disabled via self.nlc
        if self.nlc == 0:
            return False

        xc_has_nlc = self._numint.libxc.is_nlc(xc)
        return self.nlc == 'vv10' or xc_has_nlc

    def to_rhf(self):
        '''Convert the input mean-field object to a RHF/ROHF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        return self.to_rks().to_hf()

    def to_uhf(self):
        '''Convert the input mean-field object to a UHF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        return self.to_uks().to_hf()

    def to_ghf(self):
        '''Convert the input mean-field object to a GHF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        return self.to_gks().to_hf()

    def to_hf(self):
        '''Convert the input KS object to the associated HF object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        raise NotImplementedError

    def to_rks(self, xc=None):
        '''Convert the input mean-field object to a RKS/ROKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = scf.addons.convert_to_rhf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc and mf.converged
        return mf

    def to_uks(self, xc=None):
        '''Convert the input mean-field object to a UKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        mf = scf.addons.convert_to_uhf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc and mf.converged
        return mf

    def to_gks(self, xc=None):
        '''Convert the input mean-field object to a GKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.dft import numint2c
        mf = scf.addons.convert_to_ghf(self)
        if xc is not None:
            mf.xc = xc
        mf.converged = xc == self.xc and mf.converged
        if not isinstance(mf._numint, numint2c.NumInt2C):
            mf._numint = numint2c.NumInt2C()
        return mf

    def reset(self, mol=None):
        hf.SCF.reset(self, mol)
        self.grids.reset(mol)
        self.nlcgrids.reset(mol)
        return self

    def check_sanity(self):
        out = super().check_sanity()
        if self.do_nlc() and self.do_disp() and self._numint.libxc.is_nlc(self.xc):
            import warnings
            warnings.warn(
                f'nlc-type xc {self.xc} and disp {self.disp} may lead to'
                'double counting in NLC.')
        return out

    def initialize_grids(self, mol=None, dm=None):
        '''Initialize self.grids the first time call get_veff'''
        if mol is None: mol = self.mol

        ground_state = getattr(dm, 'ndim', 0) == 2
        if self.grids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.grids.build(with_non0tab=True)
            if self.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                self.grids = prune_small_rho_grids_(self, self.mol, dm,
                                                    self.grids)
            t0 = logger.timer(self, 'setting up grids', *t0)
        is_nlc = self.do_nlc()
        if is_nlc and self.nlcgrids.coords is None:
            t0 = (logger.process_clock(), logger.perf_counter())
            self.nlcgrids.build(with_non0tab=True)
            if self.small_rho_cutoff > 1e-20 and ground_state:
                # Filter grids the first time setup grids
                self.nlcgrids = prune_small_rho_grids_(self, self.mol, dm,
                                                       self.nlcgrids)
            t0 = logger.timer(self, 'setting up nlc grids', *t0)
        return self

    def to_gpu(self):
        raise NotImplementedError

# Update the KohnShamDFT label in scf.hf module
hf.KohnShamDFT = KohnShamDFT

def init_guess_by_vsap(mf, mol=None):
    '''Form SAP guess'''
    if mol is None: mol = mf.mol

    vsap = mf.get_vsap()
    t = mol.intor_symmetric('int1e_kin')
    s = mf.get_ovlp(mol)
    hsap = t + vsap

    # Form guess orbitals
    mo_energy, mo_coeff = mf.eig(hsap, s)
    logger.debug(mf, 'VSAP mo energies\n{}'.format(mo_energy))

    # and guess density
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    return mf.make_rdm1(mo_coeff, mo_occ)


class RKS(KohnShamDFT, hf.RHF):
    __doc__ = '''Restricted Kohn-Sham\n''' + hf.SCF.__doc__ + KohnShamDFT.__doc__

    def __init__(self, mol, xc='LDA,VWN'):
        hf.RHF.__init__(self, mol)
        KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        hf.RHF.dump_flags(self, verbose)
        return KohnShamDFT.dump_flags(self, verbose)

    get_veff = get_veff
    get_vsap = get_vsap
    energy_elec = energy_elec

    init_guess_by_vsap = init_guess_by_vsap

    def nuc_grad_method(self):
        from pyscf.grad import rks as rks_grad
        return rks_grad.Gradients(self)

    def to_hf(self):
        '''Convert to RHF object.'''
        return self._transfer_attrs_(self.mol.RHF())

    to_gpu = lib.to_gpu
