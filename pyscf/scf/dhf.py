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
Dirac Hartree-Fock
'''


from functools import reduce
import ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import _vhf
from pyscf.scf import chkfile
from pyscf.data import nist
from pyscf import __config__

zquatev = None
if getattr(__config__, 'scf_dhf_SCF_zquatev', True):
    try:
        # Install zquatev with
        # pip install git+https://github.com/sunqm/zquatev
        import zquatev
    except ImportError:
        pass

DEBUG = False


def kernel(mf, conv_tol=1e-9, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True):
    '''the modified SCF kernel for Dirac-Hartree-Fock.  In this kernel, the
    SCF is carried out in three steps.  First the 2-electron part is
    approximated by large component integrals (LL|LL); Next, (SS|LL) the
    interaction between large and small components are added; Finally,
    converge the SCF with the small component contributions (SS|SS)
    '''
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)
    if dm0 is None:
        dm = mf.get_init_guess(mf.mol, mf.init_guess)
    else:
        dm = dm0

    if mf.init_guess != 'chkfile':
        mf._coulomb_level = 'LLLL'
    cycles = 0
    if dm0 is None and mf._coulomb_level.upper() == 'LLLL':
        scf_conv, e_tot, mo_energy, mo_coeff, mo_occ \
                = hf.kernel(mf, 1e-2, 1e-1,
                            dump_chk, dm0=dm, callback=callback,
                            conv_check=False)
        cycles += mf.cycles
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        mf._coulomb_level = 'SSLL'

    if mf.with_ssss:
        if dm0 is None and (mf._coulomb_level.upper() == 'SSLL' or
                            mf._coulomb_level.upper() == 'LLSS'):
            scf_conv, e_tot, mo_energy, mo_coeff, mo_occ \
                    = hf.kernel(mf, 1e-3, 1e-1,
                                dump_chk, dm0=dm, callback=callback,
                                conv_check=False)
            cycles += mf.cycles
            dm = mf.make_rdm1(mo_coeff, mo_occ)
        mf._coulomb_level = 'SSSS'
    else:
        mf._coulomb_level = 'SSLL'

    out = hf.kernel(mf, conv_tol, conv_tol_grad, dump_chk, dm0=dm,
                    callback=callback, conv_check=conv_check)
    mf.cycles = cycles + mf.cycles
    return out

def energy_elec(mf, dm=None, h1e=None, vhf=None):
    r'''Electronic part of Dirac-Hartree-Fock energy

    Args:
        mf : an instance of SCF class

    Kwargs:
        dm : 2D ndarray
            one-particle density matrix
        h1e : 2D ndarray
            Core hamiltonian
        vhf : 2D ndarray
            HF potential

    Returns:
        Hartree-Fock electronic energy and the Coulomb energy
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = numpy.einsum('ij,ji->', h1e, dm).real
    e_coul = numpy.einsum('ij,ji->', vhf, dm).real * .5
    logger.debug(mf, 'E1 = %.14g  E_coul = %.14g', e1, e_coul)

    if not mf.with_ssss and mf.ssss_approx == 'Visscher':
        e_coul += _visscher_ssss_correction(mf, dm)

    mf.scf_summary['e1'] = e1
    mf.scf_summary['e2'] = e_coul
    return e1+e_coul, e_coul

def _visscher_ssss_correction(mf, dm):
    '''
    Visscher point charge corrections for small component, TCA, 98, 68
    Note there is a small difference to Visscher's work. The model
    charges in Visscher's work are obtained from atomic calculations.
    Charges here are Mulliken charges on small components.
    '''
    aoslice = mf.mol.aoslice_2c_by_atom()
    n2c = dm[0].shape[0] // 2
    s = mf.get_ovlp()
    ss_mul_charges = []
    for p0, p1 in aoslice[:,2:] + n2c:
        mul_charge = numpy.einsum('ij,ji->', s[n2c:,p0:p1], dm[p0:p1,n2c:])
        ss_mul_charges.append(mul_charge.real)
    ss_mul_charges = numpy.array(ss_mul_charges)
    e_coul_ss = gto.classical_coulomb_energy(mf.mol, ss_mul_charges)
    mf.scf_summary['e_coul_ss'] = e_coul_ss
    logger.debug(mf, 'Visscher corrections for small component = %.14g', e_coul_ss)
    return e_coul_ss

def get_jk_coulomb(mol, dm, hermi=1, coulomb_allow='SSSS',
                   opt_llll=None, opt_ssll=None, opt_ssss=None,
                   omega=None, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    if hermi == 0 and DEBUG:
        # J matrix is symmetrized in this function which is only true for
        # density matrix with time reversal symmetry
        _ensure_time_reversal_symmetry(mol, dm)

    with mol.with_range_coulomb(omega):
        if coulomb_allow.upper() == 'LLLL':
            log.debug('Coulomb integral: (LL|LL)')
            j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
            log.timer_debug1('LLLL', *t0)
            n2c = j1.shape[1]
            vj = numpy.zeros_like(dm)
            vk = numpy.zeros_like(dm)
            vj[..., :n2c, :n2c] = j1
            vk[..., :n2c, :n2c] = k1
        elif coulomb_allow.upper() == 'SSLL' \
          or coulomb_allow.upper() == 'LLSS':
            log.debug('Coulomb integral: (LL|LL) + (SS|LL)')
            vj, vk = _call_veff_ssll(mol, dm, hermi, opt_ssll)
            t0 = log.timer_debug1('SSLL', *t0)
            j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
            log.timer_debug1('LLLL', *t0)
            n2c = j1.shape[1]
            vj[..., :n2c, :n2c] += j1
            vk[..., :n2c, :n2c] += k1
        else:  # coulomb_allow == 'SSSS'
            log.debug('Coulomb integral: (LL|LL) + (SS|LL) + (SS|SS)')
            vj, vk = _call_veff_ssll(mol, dm, hermi, opt_ssll)
            t0 = log.timer_debug1('SSLL', *t0)
            j1, k1 = _call_veff_llll(mol, dm, hermi, opt_llll)
            t0 = log.timer_debug1('LLLL', *t0)
            n2c = j1.shape[1]
            vj[..., :n2c, :n2c] += j1
            vk[..., :n2c, :n2c] += k1
            j1, k1 = _call_veff_ssss(mol, dm, hermi, opt_ssss)
            log.timer_debug1('SSSS', *t0)
            vj[..., n2c:, n2c:] += j1
            vk[..., n2c:, n2c:] += k1

    return vj, vk
get_jk = get_jk_coulomb


def get_hcore(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    t  = mol.intor_symmetric('int1e_spsp_spinor') * .5
    vn = mol.intor_symmetric('int1e_nuc_spinor')
    wn = mol.intor_symmetric('int1e_spnucsp_spinor')
    h1e = numpy.empty((n4c, n4c), numpy.complex128)
    h1e[:n2c,:n2c] = vn
    h1e[n2c:,:n2c] = t
    h1e[:n2c,n2c:] = t
    h1e[n2c:,n2c:] = wn * (.25/c**2) - t
    return h1e

def get_ovlp(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    s = mol.intor_symmetric('int1e_ovlp_spinor')
    t = mol.intor_symmetric('int1e_spsp_spinor')
    s1e = numpy.zeros((n4c, n4c), numpy.complex128)
    s1e[:n2c,:n2c] = s
    s1e[n2c:,n2c:] = t * (.5/c)**2
    return s1e

make_rdm1 = hf.make_rdm1

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    dm = hf.init_guess_by_minao(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    return UHF(mol).init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_huckel(mol):
    '''Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089.'''
    dm = hf.init_guess_by_huckel(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_mod_huckel(mol):
    '''Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089,
    employing the updated GWH rule from doi:10.1021/ja00480a005.'''
    dm = hf.init_guess_by_mod_huckel(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_sap(mol, mf):
    '''Generate initial guess density matrix from a superposition of
    atomic potentials (SAP), doi:10.1021/acs.jctc.8b01089.
    This is the Gaussian fit implementation, see doi:10.1063/5.0004046.

    Args:
        mol : MoleBase object
            the molecule object for which the initial guess is evaluated
        sap_basis : dict
            SAP basis in internal format (python dictionary)

    Returns:
        dm0 : ndarray
            SAP initial guess density matrix
    '''
    dm = hf.SCF.init_guess_by_sap(mf, mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    '''Read SCF chkfile and make the density matrix for 4C-DHF initial guess.

    Kwargs:
        project : None or bool
            Whether to project chkfile's orbitals to the new basis.  Note when
            the geometry of the chkfile and the given molecule are very
            different, this projection can produce very poor initial guess.
            In PES scanning, it is recommended to switch off project.

            If project is set to None, the projection is only applied when the
            basis sets of the chkfile's molecule are different to the basis
            sets of the given molecule (regardless whether the geometry of
            the two molecules are different).  Note the basis sets are
            considered to be different if the two molecules are derived from
            the same molecule with different ordering of atoms.
    '''
    from pyscf.scf import addons
    chk_mol, scf_rec = chkfile.load_scf(chkfile_name)
    if project is None:
        project = not gto.same_basis_set(chk_mol, mol)

    # Check whether the two molecules are similar
    if abs(mol.inertia_moment() - chk_mol.inertia_moment()).sum() > 0.5:
        logger.warn(mol, "Large deviations found between the input "
                    "molecule and the molecule from chkfile\n"
                    "Initial guess density matrix may have large error.")

    if project:
        s = get_ovlp(mol)

    def fproj(mo):
        #TODO: check if mo is GHF orbital
        if project:
            mo = addons.project_mo_r2r(chk_mol, mo, mol)
            norm = numpy.einsum('pi,pi->i', mo.conj(), s.dot(mo))
            mo /= numpy.sqrt(norm)
        return mo

    mo = scf_rec['mo_coeff']
    mo_occ = scf_rec['mo_occ']
    if numpy.iscomplexobj(mo[0]):  # DHF
        dm = make_rdm1(fproj(mo), mo_occ)
    else:
        if mo[0].ndim == 1: # nr-RHF
            dm = reduce(numpy.dot, (mo*mo_occ, mo.T))
        else: # nr-UHF
            dm = (reduce(numpy.dot, (mo[0]*mo_occ[0], mo[0].T)) +
                  reduce(numpy.dot, (mo[1]*mo_occ[1], mo[1].T)))
        dm = _proj_dmll(chk_mol, dm, mol)
    return dm


def get_init_guess(mol, key='minao', **kwargs):
    '''Generate density matrix for initial guess

    Kwargs:
        key : str
            One of 'minao', 'atom', 'huckel', 'mod_huckel', 'hcore', '1e', 'sap', 'chkfile'.
    '''
    return UHF(mol).get_init_guess(mol, key, **kwargs)

def time_reversal_matrix(mol, mat):
    ''' T(A_ij) = A[T(i),T(j)]^*
    '''
    tao = numpy.asarray(mol.time_reversal_map())
    n2c = tao.size
    # tao(i) = -j  means  T(f_i) = -f_j
    # tao(i) =  j  means  T(f_i) =  f_j
    idx = abs(tao) - 1  # -1 for C indexing convention
    #:signL = [(1 if x>0 else -1) for x in tao]
    #:sign = numpy.hstack((signL, signL))

    #:tmat = numpy.empty_like(mat)
    #:for j in range(mat.__len__()):
    #:    for i in range(mat.__len__()):
    #:        tmat[idx[i],idx[j]] = mat[j,i] * sign[i]*sign[j]
    #:return tmat.conjugate()
    sign_mask = tao < 0
    if mat.shape[0] == n2c * 2:
        idx = numpy.hstack((idx, idx+n2c))
        sign_mask = numpy.hstack((sign_mask, sign_mask))

    tmat = mat[idx[:,None], idx]
    tmat[sign_mask,:] *= -1
    tmat[:,sign_mask] *= -1
    return tmat.conj()

def analyze(mf, verbose=logger.DEBUG, **kwargs):
    from pyscf.tools import dump_mat
    log = logger.new_logger(mf, verbose)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    mf.dump_scf_summary(log)
    log.info('**** MO energy ****')
    for i in range(len(mo_energy)):
        if mo_occ[i] > 0:
            log.info('occupied MO #%d energy= %.15g occ= %g',
                     i+1, mo_energy[i], mo_occ[i])
        else:
            log.info('virtual MO #%d energy= %.15g occ= %g',
                     i+1, mo_energy[i], mo_occ[i])
    mol = mf.mol
    if mf.verbose >= logger.DEBUG1:
        log.debug(' ** MO coefficients of large component of positive state (real part) **')
        label = mol.spinor_labels()
        n2c = mo_coeff.shape[0] // 2
        dump_mat.dump_rec(mf.stdout, mo_coeff[n2c:,:n2c].real, label, start=1)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    pop_chg = mf.mulliken_pop(mol, dm, mf.get_ovlp(), log)
    dip = mf.dip_moment(mol, dm, verbose=log)
    return pop_chg, dip

def mulliken_pop(mol, dm, s=None, verbose=logger.DEBUG):
    r'''Mulliken population analysis

    .. math:: M_{ij} = D_{ij} S_{ji}

    Mulliken charges

    .. math:: \delta_i = \sum_j M_{ij}

    '''
    if s is None: s = get_ovlp(mol)
    log = logger.new_logger(mol, verbose)
    pop = numpy.einsum('ij,ji->i', dm, s).real

    log.info(' ** Mulliken pop  **')
    for i, s in enumerate(mol.spinor_labels()):
        log.info('pop of  %s %10.5f', s, pop[i])

    log.note(' ** Mulliken atomic charges  **')
    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(mol.spinor_labels(fmt=None)):
        chg[s[0]] += pop[i]
    chg = mol.atom_charges() - chg
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('charge of  %d%s =   %10.5f', ia, symb, chg[ia])
    return pop, chg

def dip_moment(mol, dm, unit='Debye', verbose=logger.NOTE, **kwargs):
    r''' Dipole moment calculation

    .. math::

        \mu_x = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|x|\mu) + \sum_A Q_A X_A\\
        \mu_y = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|y|\mu) + \sum_A Q_A Y_A\\
        \mu_z = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|z|\mu) + \sum_A Q_A Z_A

    where :math:`\mu_x, \mu_y, \mu_z` are the x, y and z components of dipole
    moment

    Args:
         mol: an instance of :class:`Mole`
         dm : a 2D ndarrays density matrices

    Return:
        A list: the dipole moment on x, y and z component
    '''
    log = logger.new_logger(mol, verbose)

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / sum(charges)
    with mol.with_common_orig(charge_center):
        ll_dip = mol.intor_symmetric('int1e_r_spinor', comp=3)
        ss_dip = mol.intor_symmetric('int1e_sprsp_spinor', comp=3)

    n2c = mol.nao_2c()
    c = lib.param.LIGHT_SPEED
    dip = numpy.einsum('xij,ji->x', ll_dip, dm[:n2c,:n2c]).real
    dip+= numpy.einsum('xij,ji->x', ss_dip, dm[n2c:,n2c:]).real * (.5/c)**2

    dip *= -1.

    if unit.upper() == 'DEBYE':
        dip *= nist.AU2DEBYE
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *dip)
    return dip

def get_grad(mo_coeff, mo_occ, fock_ao):
    '''DHF Gradients'''
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(numpy.dot, (mo_coeff[:,viridx].T.conj(), fock_ao,
                           mo_coeff[:,occidx]))
    return g.ravel()


# Kramers unrestricted Dirac-Hartree-Fock
class DHF(hf.SCF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for Dirac-Hartree-Fock
        with_ssss : bool or string, for Dirac-Hartree-Fock only
            If False, ignore small component integrals (SS|SS).  Default is True.
        with_gaunt : bool, for Dirac-Hartree-Fock only
            Default is False.
        with_breit : bool, for Dirac-Hartree-Fock only
            Gaunt + gauge term.  Default is False.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> e0 = mf.scf()
    >>> mf = scf.DHF(mol)
    >>> e1 = mf.scf()
    >>> print('Relativistic effects = %.12f' % (e1-e0))
    Relativistic effects = -0.000008854205
    '''

    conv_tol = getattr(__config__, 'scf_dhf_SCF_conv_tol', 1e-8)
    with_ssss = getattr(__config__, 'scf_dhf_SCF_with_ssss', True)
    with_gaunt = getattr(__config__, 'scf_dhf_SCF_with_gaunt', False)
    with_breit = getattr(__config__, 'scf_dhf_SCF_with_breit', False)
    # corrections for small component when with_ssss is set to False
    ssss_approx = getattr(__config__, 'scf_dhf_SCF_ssss_approx', 'Visscher')
    screening = True

    _keys = {'conv_tol', 'with_ssss', 'with_gaunt', 'with_breit', 'ssss_approx'}

    def __init__(self, mol):
        hf.SCF.__init__(self, mol)
        self._coulomb_level = 'SSSS' # 'SSSS' ~ LLLL+LLSS+SSSS

    def dump_flags(self, verbose=None):
        hf.SCF.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('with_ssss %s, with_gaunt %s, with_breit %s',
                 self.with_ssss, self.with_gaunt, self.with_breit)
        if not self.with_ssss:
            log.info('ssss_approx: %s', self.ssss_approx)
        log.info('light speed = %s', lib.param.LIGHT_SPEED)
        return self

    @lib.with_doc(get_hcore.__doc__)
    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_hcore(mol)

    @lib.with_doc(get_ovlp.__doc__)
    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm1)
        return get_grad(mo_coeff, mo_occ, fock)

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    @lib.with_doc(hf.SCF.init_guess_by_huckel.__doc__)
    def init_guess_by_huckel(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, 'Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089.')
        return init_guess_by_huckel(mol)

    @lib.with_doc(hf.SCF.init_guess_by_mod_huckel.__doc__)
    def init_guess_by_mod_huckel(self, mol=None):
        if mol is None: mol = self.mol
        logger.info(self, '''Initial guess from on-the-fly Huckel, doi:10.1021/acs.jctc.8b01089,
employing the updated GWH rule from doi:10.1021/ja00480a005.''')
        return init_guess_by_mod_huckel(mol)

    @lib.with_doc(hf.SCF.init_guess_by_sap.__doc__)
    def init_guess_by_sap(self, mol=None, **kwargs):
        if mol is None: mol = self.mol
        return init_guess_by_sap(mol, self)

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        return self

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        c = lib.param.LIGHT_SPEED
        n4c = len(mo_energy)
        n2c = n4c // 2
        mo_occ = numpy.zeros(n2c * 2)
        nocc = mol.nelectron
        if mo_energy[n2c] > -1.999 * c**2:
            mo_occ[n2c:n2c+nocc] = 1
        else:
            logger.warn(self, 'Variational collapse. PES mo_energy %g < -2c^2',
                        mo_energy[n2c])
            lumo = mo_energy[mo_energy > -1.999 * c**2][nocc]
            mo_occ[mo_energy > -1.999 * c**2] = 1
            mo_occ[mo_energy >= lumo] = 0
        if self.verbose >= logger.INFO:
            if mo_energy[n2c+nocc-1]+1e-3 > mo_energy[n2c+nocc]:
                logger.warn(self, 'HOMO %.15g == LUMO %.15g',
                            mo_energy[n2c+nocc-1], mo_energy[n2c+nocc])
            else:
                logger.info(self, 'HOMO %d = %.12g  LUMO %d = %.12g',
                            nocc, mo_energy[n2c+nocc-1],
                            nocc+1, mo_energy[n2c+nocc])
                logger.debug1(self, 'NES  mo_energy = %s', mo_energy[:n2c])
                logger.debug(self, 'PES  mo_energy = %s', mo_energy[n2c:])
        return mo_occ

    make_rdm1 = lib.module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])
    energy_elec = energy_elec

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        def set_vkscreen(opt, name):
            opt._this.r_vkscreen = _vhf._fpointer(name)

        cpu0 = (logger.process_clock(), logger.perf_counter())
        opt_llll = _VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                           'CVHFrkb_q_cond', 'CVHFrkb_dm_cond',
                           direct_scf_tol=self.direct_scf_tol)
        set_vkscreen(opt_llll, 'CVHFrkbllll_vkscreen')

        c1 = .5 / lib.param.LIGHT_SPEED
        opt_ssss = _VHFOpt(mol, 'int2e_spsp1spsp2_spinor',
                           'CVHFrkbllll_prescreen', 'CVHFrkb_q_cond',
                           'CVHFrkb_dm_cond',
                           direct_scf_tol=self.direct_scf_tol/c1**4)
        opt_ssss.direct_scf_tol = self.direct_scf_tol
        opt_ssss.q_cond *= c1**2
        set_vkscreen(opt_ssss, 'CVHFrkbllll_vkscreen')

        opt_ssll = _VHFOpt(mol, 'int2e_spsp1_spinor',
                           'CVHFrkbssll_prescreen',
                           dmcondname='CVHFrkbssll_dm_cond',
                           direct_scf_tol=self.direct_scf_tol)
        opt_ssll.q_cond = numpy.array([opt_llll.q_cond, opt_ssss.q_cond])
        set_vkscreen(opt_ssll, 'CVHFrkbssll_vkscreen')
        logger.timer(self, 'init_direct_scf_coulomb', *cpu0)

        opt_gaunt_lsls = None
        opt_gaunt_lssl = None

        #TODO: prescreen for gaunt
        if self.with_gaunt:
            if self.with_breit:
                # integral function int2e_breit_ssp1ssp2_spinor evaluates
                # -1/2[alpha1*alpha2/r12 + (alpha1*r12)(alpha2*r12)/r12^3]
                intor_prefix = 'int2e_breit_'
            else:
                # integral function int2e_ssp1ssp2_spinor evaluates only
                # alpha1*alpha2/r12. Minus sign was not included.
                intor_prefix = 'int2e_'
            opt_gaunt_lsls = _VHFOpt(mol, intor_prefix + 'ssp1ssp2_spinor',
                                 'CVHFrkb_gaunt_lsls_prescreen', 'CVHFrkb_asym_q_cond',
                                 'CVHFrkb_dm_cond',
                                 direct_scf_tol=self.direct_scf_tol/c1**2)

            opt_gaunt_lssl = _VHFOpt(mol, intor_prefix + 'ssp1sps2_spinor',
                                 'CVHFrkb_gaunt_lssl_prescreen', 'CVHFrkb_asym_q_cond',
                                 'CVHFrkb_dm_cond',
                                 direct_scf_tol=self.direct_scf_tol/c1**2)

            logger.timer(self, 'init_direct_scf_gaunt_breit', *cpu0)
        #return None, None, None, None, None
        return opt_llll, opt_ssll, opt_ssss, opt_gaunt_lsls, opt_gaunt_lssl

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self)
        if self.direct_scf and self._opt.get(omega) is None:
            with mol.with_range_coulomb(omega):
                self._opt[omega] = self.init_direct_scf(mol)
        vhfopt = self._opt.get(omega)
        if vhfopt is None:
            opt_llll = opt_ssll = opt_ssss = opt_gaunt_lsls = opt_gaunt_lssl = None
        else:
            opt_llll, opt_ssll, opt_ssss, opt_gaunt_lsls, opt_gaunt_lssl = vhfopt
        if self.screening is False:
            opt_llll = opt_ssll = opt_ssss = opt_gaunt_lsls = opt_gaunt_lssl = None

        opt_gaunt = (opt_gaunt_lsls, opt_gaunt_lssl)
        vj, vk = get_jk_coulomb(mol, dm, hermi, self._coulomb_level,
                                opt_llll, opt_ssll, opt_ssss, omega, log)
        t1 = log.timer_debug1('Coulomb', *t0)
        if self.with_breit:
            assert omega is None
            if ('SSSS' in self._coulomb_level.upper() or
                # for the case both with_breit and with_ssss are set
                (not self.with_ssss and 'SSLL' in self._coulomb_level.upper())):
                vj1, vk1 = _call_veff_gaunt_breit(mol, dm, hermi, opt_gaunt, True)
                log.debug('Add Breit term')
                vj += vj1
                vk += vk1
        elif self.with_gaunt and 'SS' in self._coulomb_level.upper():
            assert omega is None
            log.debug('Add Gaunt term')
            vj1, vk1 = _call_veff_gaunt_breit(mol, dm, hermi, opt_gaunt, False)
            vj += vj1
            vk += vk1
        log.timer_debug1('Gaunt and Breit term', *t1)

        log.timer('vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Dirac-Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm) - numpy.array(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

    def scf(self, dm0=None):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.build()
        self.dump_flags()

        if dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = self.make_rdm1()

        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ \
                = kernel(self, self.conv_tol, self.conv_tol_grad,
                         dm0=dm0, callback=self.callback,
                         conv_check=self.conv_check)

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose)

    @lib.with_doc(mulliken_pop.__doc__)
    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if s is None: s = self.get_ovlp(mol)
        return mulliken_pop(mol, dm, s=s, verbose=verbose)

    @lib.with_doc(dip_moment.__doc__)
    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        return dip_moment(mol, dm, unit, verbose=verbose, **kwargs)

    def sfx2c1e(self):
        raise NotImplementedError
    def x2c1e(self):
        from pyscf.x2c import x2c
        x2chf = x2c.UHF(self.mol)
        x2chf.__dict__.update(self.__dict__)
        x2chf.mo_energy = None
        x2chf.mo_coeff = None
        x2chf.mo_occ = None
        return x2chf
    x2c = x2c1e

    def nuc_grad_method(self):
        from pyscf.grad import dhf
        return dhf.Gradients(self)

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self._coulomb_level = 'SSSS' # 'SSSS' ~ LLLL+LLSS+SSSS
        self._opt = {None: None}
        return self

    def stability(self, internal=None, external=None, verbose=None, return_status=False, **kwargs):
        '''
        DHF/DKS stability analysis.

        See also pyscf.scf.stability.rhf_stability function.

        Kwargs:
            return_status: bool
                Whether to return `stable_i` and `stable_e`

        Returns:
            If return_status is False (default), the return value includes
            two set of orbitals, which are more close to the stable condition.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.

            Else, another two boolean variables (indicating current status:
            stable or unstable) are returned.
            The first corresponds to the internal stability
            and the second corresponds to the external stability.
        '''
        from pyscf.scf.stability import dhf_stability
        return dhf_stability(self, verbose, return_status, **kwargs)

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
        return self

    def to_dks(self, xc='HF'):
        '''Convert the input mean-field object to a DKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.dft import dks
        mf = self.view(dks.UDKS)
        mf.xc = xc
        mf.converged = False
        return mf

    to_ks = to_dks

    to_gpu = lib.to_gpu

UHF = UDHF = DHF


class HF1e(DHF):
    scf = hf._hf1e_scf

    def _eigh(self, h, s):
        if zquatev:
            return zquatev.solve_KR_FCSCE(self.mol, h, s)
        else:
            return DHF._eigh(self, h, s)


class RDHF(DHF):
    '''Kramers restricted Dirac-Hartree-Fock'''
    def __init__(self, mol):
        if mol.nelectron.__mod__(2) != 0:
            raise ValueError('Invalid electron number %i.' % mol.nelectron)
        if zquatev is None:
            raise RuntimeError('zquatev library is required to perform Kramers-restricted DHF')
        UHF.__init__(self, mol)

    def _eigh(self, h, s):
        return zquatev.solve_KR_FCSCE(self.mol, h, s)

    def x2c1e(self):
        from pyscf.x2c import x2c
        x2chf = x2c.RHF(self.mol)
        x2chf.__dict__.update(self.__dict__)
        return x2chf
    x2c = x2c1e

    def to_dks(self, xc='HF'):
        '''Convert the input mean-field object to a DKS object.

        Note this conversion only changes the class of the mean-field object.
        The total energy and wave-function are the same as them in the input
        mean-field object.
        '''
        from pyscf.dft import dks
        mf = self.view(dks.RDKS)
        mf.xc = xc
        mf.converged = False
        return mf

RHF = RDHF


def _ensure_time_reversal_symmetry(mol, mat):
    if mat.ndim == 2:
        mat = [mat]
    for m in mat:
        if abs(m - time_reversal_matrix(mol, m)).max() > 1e-9:
            raise RuntimeError('Matrix does have time reversal symmetry')

def _time_reversal_triu_(mol, vj):
    n2c = vj.shape[1]
    idx, idy = numpy.triu_indices(n2c, 1)
    if vj.ndim == 2:
        Tvj = time_reversal_matrix(mol, vj)
        vj[idx,idy] = Tvj[idy,idx].conj()
    else:
        for i in range(vj.shape[0]):
            Tvj = time_reversal_matrix(mol, vj[i])
            vj[i,idx,idy] = Tvj[idy,idx].conj()
    return vj

def _mat_hermi_(vk, hermi):
    if hermi == 1:
        if vk.ndim == 2:
            vk = lib.hermi_triu(vk, hermi)
        else:
            for i in range(vk.shape[0]):
                vk[i] = lib.hermi_triu(vk[i], hermi)
    return vk

def _jk_triu_(mol, vj, vk, hermi):
    if hermi == 0:
        return _time_reversal_triu_(mol, vj), vk
    else:
        return _mat_hermi_(vj, hermi), _mat_hermi_(vk, hermi)


def _call_veff_llll(mol, dm, hermi=1, mf_opt=None):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] // 2
        dms = dm[:n2c,:n2c].copy()
    else:
        n2c = dm.shape[1] // 2
        dms = dm[:,:n2c,:n2c].copy()
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    return _jk_triu_(mol, vj, vk, hermi)

def _call_veff_ssll(mol, dm, hermi=1, mf_opt=None):
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm1 = dm[numpy.newaxis]
    else:
        dm1 = numpy.asarray(dm)
    n_dm = len(dm1)
    n2c = dm1.shape[1] // 2
    dms = numpy.vstack([dm1[:,:n2c,:n2c],
                        dm1[:,n2c:,n2c:],
                        dm1[:,n2c:,:n2c],
                        dm1[:,:n2c,n2c:]])
    if hermi:
        jks = (['lk->s2ij'] * n_dm +
               ['ji->s2kl'] * n_dm +
               ['jk->s1il'] * n_dm)
    else:
        jks = (['lk->s2ij'] * n_dm +
               ['ji->s2kl'] * n_dm +
               ['jk->s1il'] * n_dm +
               ['li->s1kj'] * n_dm)
    c1 = .5 / lib.param.LIGHT_SPEED
    vx = _vhf.rdirect_bindm('int2e_spsp1_spinor', 's4', jks, dms, 1,
                            mol._atm, mol._bas, mol._env, mf_opt)
    vx = vx.reshape(-1,n_dm,n2c,n2c) * c1**2
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)
    if hermi == 0:
        vj[:,n2c:,n2c:] = _time_reversal_triu_(mol, vx[0])
        vj[:,:n2c,:n2c] = _time_reversal_triu_(mol, vx[1])
        vk[:,n2c:,:n2c] = vx[2]
        vk[:,:n2c,n2c:] = vx[3]
    else:
        vj[:,n2c:,n2c:] = _mat_hermi_(vx[0], hermi)
        vj[:,:n2c,:n2c] = _mat_hermi_(vx[1], hermi)
        vk[:,n2c:,:n2c] = vx[2]
        vk[:,:n2c,n2c:] = vx[2].conj().transpose(0,2,1)
    vj = vj.reshape(dm.shape)
    vk = vk.reshape(dm.shape)
    return vj, vk

def _call_veff_ssss(mol, dm, hermi=1, mf_opt=None):
    c1 = .5 / lib.param.LIGHT_SPEED
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n2c = dm.shape[0] // 2
        dms = dm[n2c:,n2c:].copy()
    else:
        n2c = dm[0].shape[0] // 2
        dms = []
        for dmi in dm:
            dms.append(dmi[n2c:,n2c:].copy())
    vj, vk = _vhf.rdirect_mapdm('int2e_spsp1spsp2_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dms, 1,
                                mol._atm, mol._bas, mol._env, mf_opt) * c1**4
    return _jk_triu_(mol, vj, vk, hermi)

def _call_veff_gaunt_breit(mol, dm, hermi=1, mf_opt=None, with_breit=False):
    if mf_opt is not None:
        opt_gaunt_lsls, opt_gaunt_lssl = mf_opt
    else:
        opt_gaunt_lsls = opt_gaunt_lssl = None

    log = logger.new_logger(mol)
    t0 = (logger.process_clock(), logger.perf_counter())

    if with_breit:
        # integral function int2e_breit_ssp1ssp2_spinor evaluates
        # -1/2[alpha1*alpha2/r12 + (alpha1*r12)(alpha2*r12)/r12^3]
        intor_prefix = 'int2e_breit_'
    else:
        # integral function int2e_ssp1ssp2_spinor evaluates only
        # alpha1*alpha2/r12. Minus sign was not included.
        intor_prefix = 'int2e_'

    if hermi == 0 and DEBUG:
        _ensure_time_reversal_symmetry(mol, dm)

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        n_dm = 1
        n2c = dm.shape[0] // 2
        dmls = dm[:n2c,n2c:].copy()
        dmsl = dm[n2c:,:n2c].copy()
        dmll = dm[:n2c,:n2c].copy()
        dmss = dm[n2c:,n2c:].copy()
        dms = [dmsl, dmls, dmll, dmss]
    else:
        n_dm = len(dm)
        n2c = dm[0].shape[0] // 2
        dmll = [dmi[:n2c,:n2c].copy() for dmi in dm]
        dmls = [dmi[:n2c,n2c:].copy() for dmi in dm]
        dmsl = [dmi[n2c:,:n2c].copy() for dmi in dm]
        dmss = [dmi[n2c:,n2c:].copy() for dmi in dm]
        dms = dmsl + dmls + dmll + dmss
    vj = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)
    vk = numpy.zeros((n_dm,n2c*2,n2c*2), dtype=numpy.complex128)

    jks = ('lk->s1ij', 'jk->s1il')
    vj_ls, vk_ls = _vhf.rdirect_mapdm(intor_prefix+'ssp1ssp2_spinor', 's1', jks, dms[:n_dm], 1,
                            mol._atm, mol._bas, mol._env, opt_gaunt_lsls)
    vj[:,:n2c,n2c:] = vj_ls
    vk[:,:n2c,n2c:] = vk_ls
    t0 = log.timer_debug1('LSLS contribution', *t0)

    jks = ('lk->s1ij',) * n_dm \
        + ('li->s1kj',) * n_dm \
        + ('jk->s1il',) * n_dm
    vx = _vhf.rdirect_bindm(intor_prefix+'ssp1sps2_spinor', 's1', jks, dms[n_dm:], 1,
                            mol._atm, mol._bas, mol._env, opt_gaunt_lssl)

    t0 = log.timer_debug1('LSSL contribution', *t0)
    vj[:,:n2c,n2c:]+= vx[      :n_dm  ,:,:]
    vk[:,n2c:,n2c:] = vx[n_dm  :n_dm*2,:,:]
    vk[:,:n2c,:n2c] = vx[n_dm*2:      ,:,:]

    if hermi == 1:
        vj[:,n2c:,:n2c] = vj[:,:n2c,n2c:].transpose(0,2,1).conj()
        vk[:,n2c:,:n2c] = vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    elif hermi == 2:
        vj[:,n2c:,:n2c] = -vj[:,:n2c,n2c:].transpose(0,2,1).conj()
        vk[:,n2c:,:n2c] = -vk[:,:n2c,n2c:].transpose(0,2,1).conj()
    else:
        raise NotImplementedError
    vj = vj.reshape(dm.shape)
    vk = vk.reshape(dm.shape)
    c1 = .5 / lib.param.LIGHT_SPEED
    if with_breit:
        vj *= c1**2
        vk *= c1**2
    else:
        vj *= -c1**2
        vk *= -c1**2
    return vj, vk

def _proj_dmll(mol_nr, dm_nr, mol):
    '''Project non-relativistic atomic density matrix to large component spinor
    representation
    '''
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, numpy.eye(mol_nr.nao_nr()), mol)

    n2c = proj.shape[0]
    n4c = n2c * 2
    dm = numpy.zeros((n4c,n4c), dtype=numpy.complex128)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm[:n2c,:n2c] = (dm_ll + time_reversal_matrix(mol, dm_ll)) * .5
    return dm

class _VHFOpt(_vhf._VHFOpt):
    def set_dm(self, dm, atm, bas, env):
        if self._dmcondname is None:
            return

        mol = self.mol
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            n_dm = 1
        else:
            n_dm = len(dm)
        dm = numpy.asarray(dm, order='C')
        ao_loc = mol.ao_loc_2c()
        if isinstance(self._dmcondname, ctypes._CFuncPtr):
            fdmcond = self._dmcondname
        else:
            fdmcond = getattr(_vhf.libcvhf, self._dmcondname)
        nbas = mol.nbas
        dm_cond = numpy.empty((n_dm*2, nbas, nbas))
        fdmcond(dm_cond.ctypes, dm.ctypes, ctypes.c_int(n_dm),
                ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        self.dm_cond = dm_cond


if __name__ == '__main__':
    import pyscf.gto
    mol = pyscf.gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dhf'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = {
        'He': [(0, 0, (1, 1)),
               (0, 0, (3, 1)),
               (1, 0, (1, 1)), ]}
    mol.build()

    ##############
    # SCF result
    method = UHF(mol)
    energy = method.scf() #-2.38146942868
    print(energy)
    method.with_gaunt = True
    print(method.scf()) # -2.38138339005
    method.with_breit = True
    print(method.scf()) # -2.38138339005
