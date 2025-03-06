#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Junzi Liu <latrix1247@gmail.com>
#          Susi Lehtola <susi.lehtola@gmail.com>

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.lib.scipy_helper import pivoted_cholesky
from pyscf import __config__

LINEAR_DEP_THRESHOLD = getattr(__config__, 'scf_addons_remove_linear_dep_threshold', 1e-8)
CHOLESKY_THRESHOLD = getattr(__config__, 'scf_addons_cholesky_threshold', 1e-10)
FORCE_PIVOTED_CHOLESKY = getattr(__config__, 'scf_addons_force_cholesky', False)
LINEAR_DEP_TRIGGER = getattr(__config__, 'scf_addons_remove_linear_dep_trigger', 1e-10)

SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')

def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    if isinstance(mf, _SmearingSCF):
        mf.sigma = sigma
        mf.method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    assert not mf.istype('KSCF')
    if mf.istype('ROHF'):
        # Roothaan Fock matrix does not make much sense for smearing.
        # Restore the conventional RHF treatment.
        from pyscf import scf
        from pyscf import dft
        known_class = {
            dft.rks_symm.ROKS: dft.rks_symm.RKS,
            dft.roks.ROKS    : dft.rks.RKS     ,
            scf.hf_symm.ROHF : scf.hf_symm.RHF ,
            scf.rohf.ROHF    : scf.hf.RHF      ,
        }
        mf = _object_without_soscf(mf, known_class)
    return lib.set_class(_SmearingSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingSCF, mf.__class__))

def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf

def _get_grad_tril(mo_coeff, mo_occ, fock):
    f_mo = reduce(numpy.dot, (mo_coeff.T.conj(), fock, mo_coeff))
    return f_mo[numpy.tril_indices_from(f_mo, -1)]

def _fermi_smearing_occ(mu, mo_energy, sigma):
    '''Fermi-Dirac smearing'''
    occ = numpy.zeros_like(mo_energy)
    de = (mo_energy - mu) / sigma
    occ[de<40] = 1./(numpy.exp(de[de<40])+1.)
    return occ

def _gaussian_smearing_occ(mu, mo_energy, sigma):
    '''Gaussian smearing'''
    return 0.5 * scipy.special.erfc((mo_energy - mu) / sigma)

def _smearing_optimize(f_occ, mo_es, nocc, sigma):
    def nelec_cost_fn(m):
        mo_occ = f_occ(m, mo_es, sigma)
        return (mo_occ.sum() - nocc)**2

    mu0 = _init_guess_mu(f_occ, mo_es, nocc, sigma)
    res = scipy.optimize.minimize(
        nelec_cost_fn, mu0, method='Powell',
        options={'xtol': 1e-5, 'ftol': 1e-5, 'maxiter': 10000})
    mu = res.x
    mo_occs = f_occ(mu, mo_es, sigma)
    return mu, mo_occs

def _init_guess_mu(f_occ, mo_es, nocc, sigma, box=[-1.,1.], step=1e-3):
    ''' Generate an initial guess for `mu` by manually minimizing the nocc error on a grid
        around `fermi`. The grid is constructed as:

            mus = fermi + numpy.arange(*box, step) / 27.211399

        By default, `box = [-1, 1]` and `step = 1e-3`, which corresponds to searching
        within a +/-1 eV window around `fermi` with a resolution of 1 meV (resulting
        in 2000 grid points).

        Return:
            mu0 : float
                Best initial guess within the searching window
    '''
    fermi = _get_fermi(mo_es, nocc)
    mus = fermi + numpy.arange(*box, step)/27.211399
    noccs = numpy.asarray([f_occ(m, mo_es, sigma).sum() for m in mus])
    return mus[abs(noccs-nocc).argmin()]

def _get_fermi(mo_energy, nocc):
    mo_e_sorted = numpy.sort(mo_energy)
    if isinstance(nocc, int):
        return mo_e_sorted[nocc-1]
    else: # nocc = ?.5 or nocc = ?.0
        return mo_e_sorted[numpy.ceil(nocc).astype(int) - 1]

class _SmearingSCF:

    __name_mixin__ = 'Smearing'

    _keys = {
        'sigma', 'smearing_method', 'mu0', 'fix_spin', 'entropy', 'e_free', 'e_zero'
    }

    def __init__(self, mf, sigma, method, mu0, fix_spin):
        self.__dict__.update(mf.__dict__)
        self.sigma = sigma
        self.smearing_method = method
        self.mu0 = mu0
        self.fix_spin = fix_spin
        self.entropy = None
        self.e_free = None
        self.e_zero = None

    def undo_smearing(self):
        obj = lib.view(self, lib.drop_class(self.__class__, _SmearingSCF))
        del obj.sigma
        del obj.smearing_method
        del obj.fix_spin
        del obj.entropy
        del obj.e_free
        del obj.e_zero
        return obj

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''Label the occupancies for each orbital
        '''
        from pyscf import scf
        from pyscf.pbc.tools import print_mo_energy_occ
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            mo_occ = super().get_occ(mo_energy, mo_coeff)
            return mo_occ

        is_uhf = self.istype('UHF')
        is_rhf = self.istype('RHF')
        if isinstance(self, scf.rohf.ROHF):
            # ROHF leads to two Fock matrices. It's not clear how to define the
            # Roothaan effective Fock matrix from the two.
            raise NotImplementedError('Smearing-ROHF')

        sigma = self.sigma
        if self.smearing_method.lower() == 'fermi':
            f_occ = _fermi_smearing_occ
        else:
            f_occ = _gaussian_smearing_occ

        if self.fix_spin and is_uhf: # spin separated fermi level
            mo_es = mo_energy
            nocc = self.nelec
            if self.mu0 is None:
                mu_a, occa = _smearing_optimize(f_occ, mo_es[0], nocc[0], sigma)
                mu_b, occb = _smearing_optimize(f_occ, mo_es[1], nocc[1], sigma)
            else:
                if numpy.isscalar(self.mu0):
                    mu_a = mu_b = self.mu0
                elif len(self.mu0) == 2:
                    mu_a, mu_b = self.mu0
                else:
                    raise TypeError(f'Unsupported mu0: {self.mu0}')
                occa = f_occ(mu_a, mo_es[0], sigma)
                occb = f_occ(mu_b, mo_es[1], sigma)
            mu = [mu_a, mu_b]
            mo_occs = [occa, occb]
            self.entropy  = self._get_entropy(mo_es[0], mo_occs[0], mu[0])
            self.entropy += self._get_entropy(mo_es[1], mo_occs[1], mu[1])
            fermi = (_get_fermi(mo_es[0], nocc[0]), _get_fermi(mo_es[1], nocc[1]))

            logger.debug(self, '    Alpha-spin Fermi level %g  Sum mo_occ = %s  should equal nelec = %s',
                         fermi[0], mo_occs[0].sum(), nocc[0])
            logger.debug(self, '    Beta-spin  Fermi level %g  Sum mo_occ = %s  should equal nelec = %s',
                         fermi[1], mo_occs[1].sum(), nocc[1])
            logger.info(self, '    sigma = %g  Optimized mu_alpha = %.12g  entropy = %.12g',
                        sigma, mu[0], self.entropy)
            logger.info(self, '    sigma = %g  Optimized mu_beta  = %.12g  entropy = %.12g',
                        sigma, mu[1], self.entropy)
            if self.verbose >= logger.DEBUG:
                print_mo_energy_occ(self, mo_energy, mo_occs, True)
        else: # all orbitals treated with the same fermi level
            nelectron = self.mol.nelectron
            if is_uhf:
                mo_es = numpy.hstack(mo_energy)
            else:
                mo_es = mo_energy
            if is_rhf:
                nelectron = nelectron / 2

            if self.mu0 is None:
                mu, mo_occs = _smearing_optimize(f_occ, mo_es, nelectron, sigma)
            else:
                # If mu0 is given, fix mu instead of electron number. XXX -Chong Sun
                mu = self.mu0
                assert numpy.isscalar(mu)
                mo_occs = f_occ(mu, mo_es, sigma)
            self.entropy = self._get_entropy(mo_es, mo_occs, mu)
            if is_rhf:
                mo_occs *= 2
                self.entropy *= 2

            fermi = _get_fermi(mo_es, nelectron)
            logger.debug(self, '    Fermi level %g  Sum mo_occ = %s  should equal nelec = %s',
                         fermi, mo_occs.sum(), nelectron)
            logger.info(self, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                        sigma, mu, self.entropy)
            if is_uhf:
                mo_occs = mo_occs.reshape(2, -1)
            if self.verbose >= logger.DEBUG:
                print_mo_energy_occ(self, mo_energy, mo_occs, is_uhf)
        return mo_occs

    # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
    def _get_entropy(self, mo_energy, mo_occ, mu):
        if self.smearing_method.lower() == 'fermi':
            f = mo_occ
            f = f[(f>0) & (f<1)]
            entropy = -(f*numpy.log(f) + (1-f)*numpy.log(1-f)).sum()
        else:
            entropy = (numpy.exp(-((mo_energy-mu)/self.sigma)**2).sum()
                       / (2*numpy.sqrt(numpy.pi)))
        return entropy

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            return super().get_grad(mo_coeff, mo_occ, fock)

        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(self.mol, dm1)
        if self.istype('UHF'):
            ga = _get_grad_tril(mo_coeff[0], mo_occ[0], fock[0])
            gb = _get_grad_tril(mo_coeff[1], mo_occ[1], fock[1])
            return numpy.hstack((ga,gb))
        else: # rhf and ghf
            return _get_grad_tril(mo_coeff, mo_occ, fock)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        e_tot = self.energy_elec(dm, h1e, vhf)[0] + self.energy_nuc()
        if self.sigma and self.smearing_method and self.entropy is not None:
            self.e_free = e_tot - self.sigma * self.entropy
            self.e_zero = e_tot - self.sigma * self.entropy * .5
            logger.info(self, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, self.e_free, self.e_zero)
        return e_tot

def frac_occ_(mf, tol=1e-3):
    '''
    Addons for SCF methods to assign fractional occupancy for degenerated
    occupied HOMOs.

    Examples::

        >>> mf = gto.M(atom='O 0 0 0; O 0 0 1', verbose=4).RHF()
        >>> mf = scf.addons.frac_occ(mf)
        >>> mf.run()
    '''
    from pyscf.scf import uhf, rohf
    old_get_occ = mf.get_occ
    mol = mf.mol

    def guess_occ(mo_energy, nocc):
        mo_occ = numpy.zeros_like(mo_energy)
        if nocc:
            sorted_idx = numpy.argsort(mo_energy)
            homo = mo_energy[sorted_idx[nocc-1]]
            lumo = mo_energy[sorted_idx[nocc]]
            frac_occ_lst = abs(mo_energy - homo) < tol
            integer_occ_lst = (mo_energy <= homo) & (~frac_occ_lst)
            mo_occ[integer_occ_lst] = 1
            degen = numpy.count_nonzero(frac_occ_lst)
            frac = nocc - numpy.count_nonzero(integer_occ_lst)
            mo_occ[frac_occ_lst] = float(frac) / degen
        else:
            homo = 0.0
            lumo = 0.0
            frac_occ_lst = numpy.zeros_like(mo_energy, dtype=bool)
        return mo_occ, numpy.where(frac_occ_lst)[0], homo, lumo

    if mf.istype('UHF'):
        def get_occ(mo_energy, mo_coeff=None):
            nocca, noccb = mol.nelec
            mo_occa, frac_lsta, homoa, lumoa = guess_occ(mo_energy[0], nocca)
            mo_occb, frac_lstb, homob, lumob = guess_occ(mo_energy[1], noccb)

            if abs(homoa - lumoa) < tol or abs(homob - lumob) < tol:
                mo_occ = numpy.array([mo_occa, mo_occb])
                if len(frac_lstb):
                    logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  '
                                '%6g for beta orbitals %s',
                                mo_occa[frac_lsta[0]], frac_lsta,
                                mo_occb[frac_lstb[0]], frac_lstb)
                    logger.info(mf, '  alpha HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                    logger.info(mf, '  beta  HOMO = %.12g  LUMO = %.12g', homob, lumob)
                    logger.debug(mf, '  alpha mo_energy = %s', mo_energy[0])
                    logger.debug(mf, '  beta  mo_energy = %s', mo_energy[1])
                else:
                    logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  ',
                                mo_occa[frac_lsta[0]], frac_lsta)
                    logger.info(mf, '  alpha HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                    logger.debug(mf, '  alpha mo_energy = %s', mo_energy[0])
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

    elif mf.istype('ROHF'):
        def get_occ(mo_energy, mo_coeff=None):
            nocca, noccb = mol.nelec
            mo_occa, frac_lsta, homoa, lumoa = guess_occ(mo_energy, nocca)
            mo_occb, frac_lstb, homob, lumob = guess_occ(mo_energy, noccb)

            if abs(homoa - lumoa) < tol or abs(homob - lumob) < tol:
                mo_occ = mo_occa + mo_occb
                if len(frac_lstb):
                    logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  '
                                '%6g for beta orbitals %s',
                                mo_occa[frac_lsta[0]], frac_lsta,
                                mo_occb[frac_lstb[0]], frac_lstb)
                    logger.info(mf, '  HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                    logger.debug(mf, '  mo_energy = %s', mo_energy)
                else:
                    logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  ',
                                mo_occa[frac_lsta[0]], frac_lsta)
                    logger.info(mf, '  HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                    logger.debug(mf, '  mo_energy = %s', mo_energy)
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

        def get_grad(mo_coeff, mo_occ, fock):
            occidxa = mo_occ > 0
            occidxb = mo_occ > 1
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa.reshape(-1,1) & occidxa
            uniq_var_b = viridxb.reshape(-1,1) & occidxb

            if getattr(fock, 'focka', None) is not None:
                focka = fock.focka
                fockb = fock.fockb
            elif getattr(fock, 'ndim', None) == 3:
                focka, fockb = fock
            else:
                focka = fockb = fock
            focka = reduce(numpy.dot, (mo_coeff.T.conj(), focka, mo_coeff))
            fockb = reduce(numpy.dot, (mo_coeff.T.conj(), fockb, mo_coeff))

            g = numpy.zeros_like(focka)
            g[uniq_var_a]  = focka[uniq_var_a]
            g[uniq_var_b] += fockb[uniq_var_b]
            return g[uniq_var_a | uniq_var_b]

    elif mf.istype('RHF'):
        def get_occ(mo_energy, mo_coeff=None):
            nocc = (mol.nelectron+1) // 2  # n_docc + n_socc
            mo_occ, frac_lst, homo, lumo = guess_occ(mo_energy, nocc)
            n_docc = mol.nelectron // 2
            n_socc = nocc - n_docc
            if abs(homo - lumo) < tol or n_socc:
                mo_occ *= 2
                degen = len(frac_lst)
                mo_occ[frac_lst] -= float(n_socc) / degen
                logger.warn(mf, 'fraction occ = %6g  for orbitals %s',
                            mo_occ[frac_lst[0]], frac_lst)
                logger.info(mf, 'HOMO = %.12g  LUMO = %.12g', homo, lumo)
                logger.debug(mf, '  mo_energy = %s', mo_energy)
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ
    else:
        def get_occ(mo_energy, mo_coeff=None):
            nocc = mol.nelectron
            mo_occ, frac_lst, homo, lumo = guess_occ(mo_energy, nocc)
            if abs(homo - lumo) < tol:
                logger.warn(mf, 'fraction occ = %6g  for orbitals %s',
                            mo_occ[frac_lst[0]], frac_lst)
                logger.info(mf, 'HOMO = %.12g  LUMO = %.12g', homo, lumo)
                logger.debug(mf, '  mo_energy = %s', mo_energy)
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

    mf.get_occ = get_occ
    try:
        mf.get_grad = get_grad
    except NameError:
        pass
    return mf
frac_occ = frac_occ_


def dynamic_occ_(mf, tol=1e-3):
    '''
    Dynamically adjust the occupancy to avoid degeneracy between HOMO and LUMO
    '''
    assert mf.istype('RHF')
    old_get_occ = mf.get_occ
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        nocc = mol.nelectron // 2
        sort_mo_energy = numpy.sort(mo_energy)
        lumo = sort_mo_energy[nocc]
        if abs(sort_mo_energy[nocc-1] - lumo) < tol:
            mo_occ = numpy.zeros_like(mo_energy)
            mo_occ[mo_energy<lumo] = 2
            lst = abs(mo_energy - lumo) < tol
            mo_occ[lst] = 0
            logger.warn(mf, 'set charge = %d', mol.charge+int(lst.sum())*2)
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        sort_mo_energy[nocc-1], sort_mo_energy[nocc])
            logger.debug(mf, '  mo_energy = %s', sort_mo_energy)
        else:
            mo_occ = old_get_occ(mo_energy, mo_coeff)
        return mo_occ
    mf.get_occ = get_occ
    return mf
dynamic_occ = dynamic_occ_

def dynamic_level_shift_(mf, factor=1.):
    '''Dynamically change the level shift in each SCF cycle.  The level shift
    value is set to (HF energy change * factor)
    '''
    old_get_fock = mf.get_fock
    mf._last_e = None
    def get_fock(h1e, s1e, vhf, dm, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        if cycle > 0 or diis is not None:
            if 'exc' in mf.scf_summary:  # DFT
                e_tot = mf.scf_summary['e1'] + mf.scf_summary['coul'] + mf.scf_summary['exc']
            else:
                e_tot = mf.scf_summary['e1'] + mf.scf_summary['e2']
            if mf._last_e is not None:
                level_shift_factor = abs(e_tot-mf._last_e) * factor
                logger.info(mf, 'Set level shift to %g', level_shift_factor)
            mf._last_e = e_tot
        return old_get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle,
                            level_shift_factor, damp_factor, fock_last=fock_last)
    mf.get_fock = get_fock
    return mf
dynamic_level_shift = dynamic_level_shift_

def float_occ_(mf):
    '''
    For UHF, allowing the Sz value being changed during SCF iteration.
    Determine occupation of alpha and beta electrons based on energy spectrum
    '''
    from pyscf.scf import uhf
    assert mf.istype('UHF')
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        ee = numpy.sort(numpy.hstack(mo_energy))
        n_a = numpy.count_nonzero(mo_energy[0]<(ee[mol.nelectron-1]+1e-3))
        n_b = mol.nelectron - n_a
        if mf.nelec is None:
            nelec = mf.mol.nelec
        else:
            nelec = mf.nelec
        if n_a != nelec[0]:
            logger.info(mf, 'change num. alpha/beta electrons '
                        ' %d / %d -> %d / %d',
                        nelec[0], nelec[1], n_a, n_b)
            mf.nelec = (n_a, n_b)
        return uhf.UHF.get_occ(mf, mo_energy, mo_coeff)
    mf.get_occ = get_occ
    return mf
dynamic_sz_ = float_occ = float_occ_

def follow_state_(mf, occorb=None):
    occstat = [occorb]
    old_get_occ = mf.get_occ
    def get_occ(mo_energy, mo_coeff=None):
        if occstat[0] is None:
            mo_occ = old_get_occ(mo_energy, mo_coeff)
        else:
            mo_occ = numpy.zeros_like(mo_energy)
            s = reduce(numpy.dot, (occstat[0].T, mf.get_ovlp(), mo_coeff))
            nocc = mf.mol.nelectron // 2
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx = numpy.argsort(numpy.einsum('ij,ij->j', s, s))
            mo_occ[idx[-nocc:]] = 2
            logger.debug(mf, '  mo_occ = %s', mo_occ)
            logger.debug(mf, '  mo_energy = %s', mo_energy)
        occstat[0] = mo_coeff[:,mo_occ>0]
        return mo_occ
    mf.get_occ = get_occ
    return mf
follow_state = follow_state_

def mom_occ_(mf, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration.'''
    from pyscf.scf import uhf, rohf
    log = logger.Logger(mf.stdout, mf.verbose)
    if mf.istype('UHF'):
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]
    elif mf.istype('ROHF'):
        if mf.mol.spin != (numpy.sum(setocc[0]) - numpy.sum(setocc[1])):
            raise ValueError('Wrong occupation setting for restricted open-shell calculation.')
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    else: # GHF, and DHF
        assert setocc.ndim == 1

    if mf.istype('UHF') or mf.istype('ROHF'):
        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_coeff is None: mo_coeff = mf.mo_coeff
            if mf.istype('ROHF'):
                mo_coeff = numpy.array([mo_coeff, mo_coeff])
            mo_occ = numpy.zeros_like(setocc)
            nocc_a = int(numpy.sum(setocc[0]))
            nocc_b = int(numpy.sum(setocc[1]))
            s_a = reduce(numpy.dot, (coef_occ_a.conj().T, mf.get_ovlp(), mo_coeff[0]))
            s_b = reduce(numpy.dot, (coef_occ_b.conj().T, mf.get_ovlp(), mo_coeff[1]))
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a))[::-1]
            idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b))[::-1]
            mo_occ[0,idx_a[:nocc_a]] = 1.
            mo_occ[1,idx_b[:nocc_b]] = 1.

            log.debug(' New alpha occ pattern: %s', mo_occ[0])
            log.debug(' New beta occ pattern: %s', mo_occ[1])
            if isinstance(mf.mo_energy, numpy.ndarray) and mf.mo_energy.ndim == 1:
                log.debug1(' Current mo_energy(sorted) = %s', mo_energy)
            else:
                log.debug1(' Current alpha mo_energy(sorted) = %s', mo_energy[0])
                log.debug1(' Current beta mo_energy(sorted) = %s', mo_energy[1])

            if (int(numpy.sum(mo_occ[0])) != nocc_a):
                log.error('mom alpha electron occupation numbers do not match: %d, %d',
                          nocc_a, int(numpy.sum(mo_occ[0])))
            if (int(numpy.sum(mo_occ[1])) != nocc_b):
                log.error('mom beta electron occupation numbers do not match: %d, %d',
                          nocc_b, int(numpy.sum(mo_occ[1])))

            #output 1-dimension occupation number for restricted open-shell
            if mf.istype('ROHF'): mo_occ = mo_occ[0, :] + mo_occ[1, :]
            return mo_occ
    else:
        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_coeff is None: mo_coeff = mf.mo_coeff
            mo_occ = numpy.zeros_like(setocc)
            nocc = int(setocc.sum())
            s = occorb[:,setocc>0].conj().T.dot(mf.get_ovlp()).dot(mo_coeff)
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx = numpy.argsort(numpy.einsum('ij,ij->j', s, s))[::-1]
            mo_occ[idx[:nocc]] = 1.
            return mo_occ

    mf.get_occ = get_occ
    return mf

mom_occ = mom_occ_

def project_mo_nr2nr(mol1, mo1, mol2):
    r''' Project orbital coefficients from basis set 1 (C1 for mol1) to basis
    set 2 (C2 for mol2).

    .. math::

        |\psi1\rangle = |AO1\rangle C1

        |\psi2\rangle = P |\psi1\rangle = |AO2\rangle S^{-1}\langle AO2| AO1\rangle> C1 = |AO2\rangle> C2

        C2 = S^{-1}\langle AO2|AO1\rangle C1

    There are three relevant functions:
    :func:`project_mo_nr2nr` is the projection for non-relativistic (scalar) basis.
    :func:`project_mo_nr2r` projects from non-relativistic to relativistic basis.
    :func:`project_mo_r2r`  is the projection between relativistic (spinor) basis.
    '''
    s22 = mol2.intor_symmetric('int1e_ovlp')
    s21 = mole.intor_cross('int1e_ovlp', mol2, mol1)
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        return lib.cho_solve(s22, numpy.dot(s21, mo1), strict_sym_pos=False)
    else:
        return [lib.cho_solve(s22, numpy.dot(s21, x), strict_sym_pos=False)
                for x in mo1]

@lib.with_doc(project_mo_nr2nr.__doc__)
def project_mo_nr2r(mol1, mo1, mol2):
    assert (not mol1.cart)
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_sph', mol2, mol1)

    ua, ub = mol2.sph2spinor_coeff()
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        mo2 = numpy.dot(s21, mo1)
        return lib.cho_solve(s22, mo2, strict_sym_pos=False)
    else:
        return [lib.cho_solve(s22, numpy.dot(s21, x), strict_sym_pos=False)
                for x in mo1]

@lib.with_doc(project_mo_nr2nr.__doc__)
def project_mo_r2r(mol1, mo1, mol2):
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    t22 = mol2.intor_symmetric('int1e_spsp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_spinor', mol2, mol1)
    t21 = mole.intor_cross('int1e_spsp_spinor', mol2, mol1)
    n2c = s21.shape[1]
    pl = lib.cho_solve(s22, s21, strict_sym_pos=False)
    ps = lib.cho_solve(t22, t21, strict_sym_pos=False)
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        return numpy.vstack((numpy.dot(pl, mo1[:n2c]),
                             numpy.dot(ps, mo1[n2c:])))
    else:
        return [numpy.vstack((numpy.dot(pl, x[:n2c]),
                              numpy.dot(ps, x[n2c:]))) for x in mo1]

def project_dm_nr2nr(mol1, dm1, mol2):
    r''' Project density matrix representation from basis set 1 (mol1) to basis
    set 2 (mol2).

    .. math::

        |AO2\rangle DM_AO2 \langle AO2|

        = |AO2\rangle P DM_AO1 P \langle AO2|

        DM_AO2 = P DM_AO1 P

        P = S_{AO2}^{-1}\langle AO2|AO1\rangle

    There are three relevant functions:
    :func:`project_dm_nr2nr` is the projection for non-relativistic (scalar) basis.
    :func:`project_dm_nr2r` projects from non-relativistic to relativistic basis.
    :func:`project_dm_r2r`  is the projection between relativistic (spinor) basis.
    '''
    s22 = mol2.intor_symmetric('int1e_ovlp')
    s21 = mole.intor_cross('int1e_ovlp', mol2, mol1)
    p21 = lib.cho_solve(s22, s21, strict_sym_pos=False)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())

@lib.with_doc(project_dm_nr2nr.__doc__)
def project_dm_nr2r(mol1, dm1, mol2):
    assert (not mol1.cart)
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_sph', mol2, mol1)

    ua, ub = mol2.sph2spinor_coeff()
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    p21 = lib.cho_solve(s22, s21, strict_sym_pos=False)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())

@lib.with_doc(project_dm_nr2nr.__doc__)
def project_dm_r2r(mol1, dm1, mol2):
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    t22 = mol2.intor_symmetric('int1e_spsp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_spinor', mol2, mol1)
    t21 = mole.intor_cross('int1e_spsp_spinor', mol2, mol1)
    pl = lib.cho_solve(s22, s21, strict_sym_pos=False)
    ps = lib.cho_solve(t22, t21, strict_sym_pos=False)
    p21 = scipy.linalg.block_diag(pl, ps)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())

def canonical_orth_(S, thr=1e-7):
    '''Löwdin's canonical orthogonalization'''
    # Ensure the basis functions are normalized (symmetry-adapted ones are not!)
    normlz = numpy.power(numpy.diag(S), -0.5)
    Snorm = numpy.dot(numpy.diag(normlz), numpy.dot(S, numpy.diag(normlz)))
    # Form vectors for normalized overlap matrix
    Sval, Svec = numpy.linalg.eigh(Snorm)
    X = Svec[:,Sval>=thr] / numpy.sqrt(Sval[Sval>=thr])
    # Plug normalization back in
    X = numpy.dot(numpy.diag(normlz), X)
    return X

def partial_cholesky_orth_(S, canthr=1e-7, cholthr=1e-9):
    '''Partial Cholesky orthogonalization for curing overcompleteness.

    References:

    Susi Lehtola, Curing basis set overcompleteness with pivoted
    Cholesky decompositions, J. Chem. Phys. 151, 241102 (2019),
    doi:10.1063/1.5139948.

    Susi Lehtola, Accurate reproduction of strongly repulsive
    interatomic potentials, Phys. Rev. A 101, 032504 (2020),
    doi:10.1103/PhysRevA.101.032504.
    '''
    # Ensure the basis functions are normalized
    normlz = numpy.power(numpy.diag(S), -0.5)
    Snorm = numpy.dot(numpy.diag(normlz), numpy.dot(S, numpy.diag(normlz)))

    # Sort the basis functions according to the Gershgorin circle
    # theorem so that the Cholesky routine is well-initialized
    odS = numpy.abs(Snorm)
    numpy.fill_diagonal(odS, 0.0)
    odSs = numpy.sum(odS, axis=0)
    sortidx = numpy.argsort(odSs, kind='stable')

    # Run the pivoted Cholesky decomposition
    Ssort = Snorm[numpy.ix_(sortidx, sortidx)].copy()
    c, piv, r_c = pivoted_cholesky(Ssort, tol=cholthr)
    # The functions we're going to use are given by the pivot as
    idx = sortidx[piv[:r_c]]

    # Get the (un-normalized) sub-basis
    Ssub = S[numpy.ix_(idx, idx)].copy()
    # Orthogonalize sub-basis
    Xsub = canonical_orth_(Ssub, thr=canthr)

    # Full X
    X = numpy.zeros((S.shape[0], Xsub.shape[1]), dtype=Xsub.dtype)
    X[idx,:] = Xsub

    return X

def remove_linear_dep_(mf, threshold=LINEAR_DEP_THRESHOLD,
                       lindep=LINEAR_DEP_TRIGGER,
                       cholesky_threshold=CHOLESKY_THRESHOLD,
                       force_pivoted_cholesky=FORCE_PIVOTED_CHOLESKY):
    '''
    Args:
        threshold : float
            The threshold under which the eigenvalues of the overlap matrix are
            discarded to avoid numerical instability.
        lindep : float
            The threshold that triggers the special treatment of the linear
            dependence issue.
    '''
    s = mf.get_ovlp()
    cond = numpy.max(lib.cond(s))
    if cond < 1./lindep and not force_pivoted_cholesky:
        return mf

    logger.info(mf, 'Applying remove_linear_dep_ on SCF object.')
    logger.debug(mf, 'Overlap condition number %g', cond)
    if (cond < 1./numpy.finfo(s.dtype).eps and not force_pivoted_cholesky):
        logger.info(mf, 'Using canonical orthogonalization with threshold {}'.format(threshold))
        mf._eigh = _eigh_with_canonical_orth(threshold)
    else:
        logger.info(mf, 'Using partial Cholesky orthogonalization '
                    '(doi:10.1063/1.5139948, doi:10.1103/PhysRevA.101.032504)')
        logger.info(mf, 'Using threshold {} for pivoted Cholesky'.format(cholesky_threshold))
        logger.info(mf, 'Using threshold {} to orthogonalize the subbasis'.format(threshold))
        mf._eigh = _eigh_with_pivot_cholesky(threshold, cholesky_threshold)
    return mf
remove_linear_dep = remove_linear_dep_

def _eigh_with_canonical_orth(threshold=LINEAR_DEP_THRESHOLD):
    def eigh(h, s):
        x = canonical_orth_(s, threshold)
        xhx = reduce(lib.dot, (x.conj().T, h, x))
        e, c = scipy.linalg.eigh(xhx)
        c = numpy.dot(x, c)
        return e, c
    return eigh

def _eigh_with_pivot_cholesky(threshold=LINEAR_DEP_THRESHOLD,
                              cholesky_threshold=CHOLESKY_THRESHOLD):
    def eigh(h, s):
        x = partial_cholesky_orth_(s, canthr=threshold, cholthr=cholesky_threshold)
        xhx = reduce(lib.dot, (x.conj().T, h, x))
        e, c = scipy.linalg.eigh(xhx)
        c = numpy.dot(x, c)
        return e, c
    return eigh

def convert_to_uhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the unrestricted HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is an second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An unrestricted SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    assert (isinstance(mf, scf.hf.SCF))

    logger.debug(mf, 'Converting %s to UHF', mf.__class__)

    if mf.istype('GHF'):
        raise NotImplementedError

    elif out is not None:
        assert out.istype('UHF')
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif mf.istype('UHF'):
        # Remove with_df for SOSCF method because the post-HF code checks the
        # attribute .with_df to identify whether an SCF object is DF-SCF method.
        # with_df in SOSCF is used in orbital hessian approximation only.  For the
        # returned SCF object, whether with_df exists in SOSCF has no effects on the
        # mean-field energy and other properties.
        if getattr(mf, '_scf', None):
            return _update_mf_without_soscf(mf, mf._scf.copy(), remove_df)
        else:
            return mf.copy()

    else:
        known_cls = {
            dft.rks_symm.ROKS : dft.uks_symm.UKS,
            dft.rks_symm.RKS  : dft.uks_symm.UKS,
            dft.roks.ROKS     : dft.uks.UKS,
            dft.rks.RKS       : dft.uks.UKS,
            scf.hf_symm.ROHF  : scf.uhf_symm.UHF,
            scf.hf_symm.RHF   : scf.uhf_symm.UHF,
            scf.rohf.ROHF     : scf.uhf.UHF,
            scf.hf.RHF        : scf.uhf.UHF,
        }
        out = _object_without_soscf(mf, known_cls, remove_df)

    return _update_mo_to_uhf_(mf, out)

def _object_without_soscf(mf, known_class, remove_df=False):
    '''Create a new SCF object, excluding the SOSCF base class'''
    from pyscf.soscf import newton_ah
    from pyscf.df.df_jk import _DFHF
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        mf = mf.undo_soscf()

    if remove_df and isinstance(mf, _DFHF):
        mf = mf.undo_df()

    for old_cls in mf.__class__.__mro__:
        if old_cls in known_class:
            break
    else:
        raise NotImplementedError(
            "Incompatible object types. Mean-field `mf` class not found in "
            "`known_class` type.\n\nmf = '%s'\n\nknown_class = '%s'" %
            (mf.__class__, known_class))

    new_cls = known_class[old_cls]
    out = new_cls(mf.mol)
    out.__dict__.update(mf.__dict__)
    out.__class__ = lib.replace_class(mf.__class__, old_cls, new_cls)
    return out

def _update_mf_without_soscf(mf, out, remove_df=False):
    '''Update an SCF object, excluding the SOSCF base class'''
    from pyscf.soscf import newton_ah
    mf_dic = dict(mf.__dict__)

    # if mf is SOSCF object, avoid to overwrite the with_df method
    # FIXME: it causes bug when converting pbc-SOSCF.
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        mf_dic.pop('_scf')
        if not hasattr(mf._scf, 'with_df'):
            mf_dic.pop('with_df', None)
    if remove_df:
        mf_dic.pop('with_df', None)

    out.__dict__.update(mf_dic)
    return out

def convert_to_rhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the restricted HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is an second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An unrestricted SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    assert (isinstance(mf, scf.hf.SCF))

    logger.debug(mf, 'Converting %s to RHF', mf.__class__)

    if getattr(mf, 'nelec', None) is None:
        nelec = mf.mol.nelec
    else:
        nelec = mf.nelec

    if mf.istype('GHF'):
        raise NotImplementedError

    elif out is not None:
        assert out.istype('RHF')
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif (mf.istype('RHF') or
          (nelec[0] != nelec[1] and mf.istype('ROHF'))):
        if getattr(mf, '_scf', None):
            return _update_mf_without_soscf(mf, mf._scf.copy(), remove_df)
        else:
            return mf.copy()

    else:
        if nelec[0] == nelec[1]:
            known_cls = {
                dft.rks_symm.ROKS: dft.rks_symm.RKS,
                dft.roks.ROKS    : dft.rks.RKS     ,
                dft.uks_symm.UKS : dft.rks_symm.RKS,
                dft.uks.UKS      : dft.rks.RKS     ,
                scf.hf_symm.ROHF : scf.hf_symm.RHF ,
                scf.rohf.ROHF    : scf.hf.RHF      ,
                scf.uhf_symm.UHF : scf.hf_symm.RHF ,
                scf.uhf.UHF      : scf.hf.RHF      ,
            }
        else:
            known_cls = {
                dft.uks_symm.UKS : dft.rks_symm.ROKS,
                dft.uks.UKS      : dft.roks.ROKS    ,
                scf.uhf_symm.UHF : scf.hf_symm.ROHF ,
                scf.uhf.UHF      : scf.rohf.ROHF    ,
            }
        out = _object_without_soscf(mf, known_cls, remove_df)

    return _update_mo_to_rhf_(mf, out)

def convert_to_ghf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the generalized HF/KS object

    Note this conversion only changes the class of the mean-field object.
    The total energy and wave-function are the same as them in the input
    mf object. If mf is an second order SCF (SOSCF) object, the SOSCF layer
    will be discarded. Its underlying SCF object mf._scf will be converted.

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An generalized SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    assert (isinstance(mf, scf.hf.SCF))

    logger.debug(mf, 'Converting %s to GHF', mf.__class__)

    if out is not None:
        assert out.istype('GHF')
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif mf.istype('GHF'):
        if getattr(mf, '_scf', None):
            return _update_mf_without_soscf(mf, mf._scf.copy(), remove_df)
        else:
            return mf.copy()

    else:
        known_cls = {
            dft.rks_symm.ROKS : dft.gks_symm.GKS,
            dft.rks_symm.RKS  : dft.gks_symm.GKS,
            dft.uks_symm.UKS  : dft.gks_symm.GKS,
            dft.roks.ROKS     : dft.gks.GKS,
            dft.rks.RKS       : dft.gks.GKS,
            dft.uks.UKS       : dft.gks.GKS,
            scf.hf_symm.ROHF  : scf.ghf_symm.GHF,
            scf.hf_symm.RHF   : scf.ghf_symm.GHF,
            scf.uhf_symm.UHF  : scf.ghf_symm.GHF,
            scf.rohf.ROHF     : scf.ghf.GHF,
            scf.hf.RHF        : scf.ghf.GHF,
            scf.uhf.UHF       : scf.ghf.GHF,
        }
        out = _object_without_soscf(mf, known_cls, remove_df)

    return _update_mo_to_ghf_(mf, out)

def _update_mo_to_uhf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('UHF') or mf.istype('KUHF'):
        mf1.mo_occ = mf.mo_occ
        mf1.mo_coeff = mf.mo_coeff
        mf1.mo_energy = mf.mo_energy
    elif getattr(mf, 'kpts', None) is None:  # RHF/ROHF
        mf1.mo_occ = numpy.array((mf.mo_occ>0, mf.mo_occ==2), dtype=numpy.double)
        # ROHF orbital energies, not canonical UHF orbital energies
        mo_ea = getattr(mf.mo_energy, 'mo_ea', mf.mo_energy)
        mo_eb = getattr(mf.mo_energy, 'mo_eb', mf.mo_energy)
        mf1.mo_energy = numpy.array((mo_ea, mo_eb))
        mf1.mo_coeff = numpy.array((mf.mo_coeff, mf.mo_coeff))
    else:  # This to handle KRHF object
        mf1.mo_occ = numpy.array([
            [numpy.asarray(occ> 0, dtype=numpy.double) for occ in mf.mo_occ],
            [numpy.asarray(occ==2, dtype=numpy.double) for occ in mf.mo_occ]])
        mo_ea = getattr(mf.mo_energy, 'mo_ea', mf.mo_energy)
        mo_eb = getattr(mf.mo_energy, 'mo_eb', mf.mo_energy)
        mf1.mo_energy = numpy.array((mo_ea, mo_eb))
        mf1.mo_coeff = numpy.array((mf.mo_coeff, mf.mo_coeff))
    return mf1

def _update_mo_to_rhf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('RHF') or mf.istype('KRHF'): # RHF/ROHF/KRHF/KROHF
        mf1.mo_occ = mf.mo_occ
        mf1.mo_coeff = mf.mo_coeff
        mf1.mo_energy = mf.mo_energy
    elif getattr(mf, 'kpts', None) is None:  # UHF
        mf1.mo_occ = mf.mo_occ[0] + mf.mo_occ[1]
        mf1.mo_energy = mf.mo_energy[0]
        mf1.mo_coeff =  mf.mo_coeff[0]
        if getattr(mf.mo_coeff[0], 'orbsym', None) is not None:
            mf1.mo_coeff = lib.tag_array(mf1.mo_coeff, orbsym=mf.mo_coeff[0].orbsym)
        mf1.converged = False
    else:  # KUHF
        mf1.mo_occ = [occa+occb for occa, occb in zip(*mf.mo_occ)]
        mf1.mo_energy = mf.mo_energy[0]
        mf1.mo_coeff =  mf.mo_coeff[0]
        mf1.converged = False
    return mf1

def _update_mo_to_ghf_(mf, mf1):
    if mf.mo_energy is None:
        return mf1

    if mf.istype('KSCF'):
        raise NotImplementedError('KSCF')
    elif mf.istype('RHF'):
        nao, nmo = mf.mo_coeff.shape
        orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)

        mf1.mo_energy = numpy.empty(nmo*2)
        mf1.mo_energy[orbspin==0] = mf.mo_energy
        mf1.mo_energy[orbspin==1] = mf.mo_energy
        mf1.mo_occ = numpy.empty(nmo*2)
        mf1.mo_occ[orbspin==0] = mf.mo_occ > 0
        mf1.mo_occ[orbspin==1] = mf.mo_occ == 2

        mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mf.mo_coeff.dtype)
        mo_coeff[:nao,orbspin==0] = mf.mo_coeff
        mo_coeff[nao:,orbspin==1] = mf.mo_coeff
        if getattr(mf.mo_coeff, 'orbsym', None) is not None:
            orbsym = numpy.zeros_like(orbspin)
            orbsym[orbspin==0] = mf.mo_coeff.orbsym
            orbsym[orbspin==1] = mf.mo_coeff.orbsym
            mo_coeff = lib.tag_array(mo_coeff, orbsym=orbsym)
        mf1.mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)

    else: # UHF
        nao, nmo = mf.mo_coeff[0].shape
        orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, False)

        mf1.mo_energy = numpy.empty(nmo*2)
        mf1.mo_energy[orbspin==0] = mf.mo_energy[0]
        mf1.mo_energy[orbspin==1] = mf.mo_energy[1]
        mf1.mo_occ = numpy.empty(nmo*2)
        mf1.mo_occ[orbspin==0] = mf.mo_occ[0]
        mf1.mo_occ[orbspin==1] = mf.mo_occ[1]

        mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mf.mo_coeff[0].dtype)
        mo_coeff[:nao,orbspin==0] = mf.mo_coeff[0]
        mo_coeff[nao:,orbspin==1] = mf.mo_coeff[1]
        if getattr(mf.mo_coeff[0], 'orbsym', None) is not None:
            orbsym = numpy.zeros_like(orbspin)
            orbsym[orbspin==0] = mf.mo_coeff[0].orbsym
            orbsym[orbspin==1] = mf.mo_coeff[1].orbsym
            mo_coeff = lib.tag_array(mo_coeff, orbsym=orbsym)
        mf1.mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)
    return mf1

def get_ghf_orbspin(mo_energy, mo_occ, is_rhf=None):
    '''Spin of each GHF orbital when the GHF orbitals are converted from
    RHF/UHF orbitals

    For RHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.  In the occupied orbital space, if degenerated, first
    alpha then beta, last the (open-shell) singly occupied (alpha) orbitals. In
    the unoccupied orbital space, first the (open-shell) unoccupied (beta)
    orbitals if applicable, then alpha and beta orbitals

    For UHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.
    '''
    if is_rhf is None:  # guess whether the orbitals are RHF orbitals
        is_rhf = mo_energy[0].ndim == 0

    if is_rhf:
        nmo = mo_energy.size
        nocc = numpy.count_nonzero(mo_occ >0)
        nvir = nmo - nocc
        ndocc = numpy.count_nonzero(mo_occ==2)
        nsocc = nocc - ndocc
        orbspin = numpy.array([0,1]*ndocc + [0]*nsocc + [1]*nsocc + [0,1]*nvir)
    else:
        nmo = mo_energy[0].size
        nocca = numpy.count_nonzero(mo_occ[0]>0)
        nvira = nmo - nocca
        noccb = numpy.count_nonzero(mo_occ[1]>0)
        nvirb = nmo - noccb
        # round(6) to avoid numerical uncertainty in degeneracy
        es = numpy.append(mo_energy[0][mo_occ[0] >0],
                          mo_energy[1][mo_occ[1] >0])
        oidx = numpy.argsort(es.round(6), kind='stable')
        es = numpy.append(mo_energy[0][mo_occ[0]==0],
                          mo_energy[1][mo_occ[1]==0])
        vidx = numpy.argsort(es.round(6), kind='stable')
        orbspin = numpy.append(numpy.array([0]*nocca+[1]*noccb)[oidx],
                               numpy.array([0]*nvira+[1]*nvirb)[vidx])
    return orbspin

del (LINEAR_DEP_THRESHOLD, LINEAR_DEP_TRIGGER)

def fast_newton(mf, mo_coeff=None, mo_occ=None, dm0=None,
                auxbasis=None, dual_basis=None, **newton_kwargs):
    '''This is a wrap function which combines several operations. This
    function first setup the initial guess
    from density fitting calculation then use  for
    Newton solver and call Newton solver.

    Newton solver attributes [max_cycle_inner, max_stepsize, ah_start_tol,
    ah_conv_tol, ah_grad_trust_region, ...] can be passed through ``**newton_kwargs``.
    '''
    from pyscf.lib import logger
    from pyscf import df
    from pyscf.soscf import newton_ah
    if auxbasis is None:
        auxbasis = df.addons.aug_etb_for_dfbasis(mf.mol, 'ahlrichs', beta=2.5)
    if dual_basis:
        mf1 = mf.newton()
        pmol = mf1.mol = newton_ah.project_mol(mf.mol, dual_basis)
        mf1 = mf1.density_fit(auxbasis)
    else:
        mf1 = mf.newton().density_fit(auxbasis)
    mf1.with_df._compatible_format = False
    mf1.direct_scf_tol = 1e-7

    if getattr(mf, 'grids', None):
        from pyscf.dft import gen_grid
        approx_grids = gen_grid.Grids(mf.mol)
        approx_grids.verbose = 0
        approx_grids.level = max(0, mf.grids.level-3)
        mf1.grids = approx_grids

        approx_numint = mf._numint.copy()
        mf1._numint = approx_numint
    for key in newton_kwargs:
        setattr(mf1, key, newton_kwargs[key])

    if mo_coeff is None or mo_occ is None:
        mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ

    if dm0 is not None:
        mo_coeff, mo_occ = mf1.from_dm(dm0)
    elif mo_coeff is None or mo_occ is None:
        logger.note(mf, '========================================================')
        logger.note(mf, 'Generating initial guess with DIIS-SCF for newton solver')
        logger.note(mf, '========================================================')
        if dual_basis:
            mf0 = mf.copy()
            mf0.mol = pmol
            mf0 = mf0.density_fit(auxbasis)
        else:
            mf0 = mf.density_fit(auxbasis)
        mf0.direct_scf_tol = 1e-7
        mf0.conv_tol = 3.
        mf0.conv_tol_grad = 1.
        if mf0.level_shift == 0:
            mf0.level_shift = .2
        if getattr(mf, 'grids', None):
            mf0.grids = approx_grids
            mf0._numint = approx_numint
# Note: by setting small_rho_cutoff, dft.get_veff function may overwrite
# approx_grids and approx_numint.  It will further changes the corresponding
# mf1 grids and _numint.  If initial guess dm0 or mo_coeff/mo_occ were given,
# dft.get_veff are not executed so that more grid points may be found in
# approx_grids.
            mf0.small_rho_cutoff = mf.small_rho_cutoff * 10
        mf0.kernel()
        mf1.with_df = mf0.with_df
        mo_coeff, mo_occ = mf0.mo_coeff, mf0.mo_occ

        if dual_basis:
            if mo_occ.ndim == 2:
                mo_coeff =(project_mo_nr2nr(pmol, mo_coeff[0], mf.mol),
                           project_mo_nr2nr(pmol, mo_coeff[1], mf.mol))
            else:
                mo_coeff = project_mo_nr2nr(pmol, mo_coeff, mf.mol)
            mo_coeff, mo_occ = mf1.from_dm(mf.make_rdm1(mo_coeff,mo_occ))
        mf0 = None

        logger.note(mf, '============================')
        logger.note(mf, 'Generating initial guess end')
        logger.note(mf, '============================')

    mf1.kernel(mo_coeff, mo_occ)
    mf.converged = mf1.converged
    mf.e_tot     = mf1.e_tot
    mf.mo_energy = mf1.mo_energy
    mf.mo_coeff  = mf1.mo_coeff
    mf.mo_occ    = mf1.mo_occ

#    mf = copy.copy(mf)
#    def mf_kernel(*args, **kwargs):
#        logger.warn(mf, "fast_newton is a wrap function to quickly setup and call Newton solver. "
#                    "There's no need to call kernel function again for fast_newton.")
#        del (mf.kernel)  # warn once and remove circular dependence
#        return mf.e_tot
#    mf.kernel = mf_kernel
    return mf
