#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
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

import numpy
import scipy.optimize
from scipy.special import erfc
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__

SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')

def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Fermi-Dirac or Gaussian smearing'''
    if isinstance(mf, _SmearingSCF):
        mf.sigma = sigma
        mf.method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    if mf.istype('_CIAH_SOSCF'):
        raise NotImplementedError('Smearing with second order SCF is not supported')

    if mf.istype('KSCF'):
        from pyscf.pbc.scf.smearing import smearing
        return smearing(mf, sigma, method, mu0, fix_spin)

    if mf.istype('ROHF'):
        if fix_spin:
            raise RuntimeError('Smearing-ROHF with fix_spin not supported. Use UHF instead.')
        # Roothaan Fock matrix for ROHF is not supported by the smearing method.
        # The single occupancy can be handled using the regular RHF class
        from pyscf.scf.addons import _object_without_soscf
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
    f_mo = mo_coeff.conj().T.dot(fock).dot(mo_coeff)
    return f_mo[numpy.tril_indices_from(f_mo, -1)]

def _fermi_smearing_occ(mu, mo_energy, sigma):
    '''Fermi-Dirac smearing'''
    occ = numpy.zeros_like(mo_energy)
    de = (mo_energy - mu) / sigma
    occ[de<40] = 1./(numpy.exp(de[de<40])+1.)
    return occ

def _gaussian_smearing_occ(mu, mo_energy, sigma):
    '''Gaussian smearing'''
    return 0.5 * erfc((mo_energy - mu) / sigma)

def _smearing_optimize(f_occ, mo_es, nocc, sigma):
    def rootfn(m):
        mo_occ = f_occ(m, mo_es, sigma)
        return mo_occ.sum() - nocc

    # it's okay to set small xtol according to the docs.
    mu = scipy.optimize.bisect(rootfn, mo_es.min()-10., mo_es.max()+10.,
                               xtol=1e-16, maxiter=10000)

    cur_err = abs(rootfn(mu))

    # Check if we can further improve mu by moving it up/down
    # by the minimum machine-representable amount.
    # In many cases with Fermi-type smearing and sigma~1e-6,
    # the minimum possible error is still >1e-11 because the
    # smearing function is just so sharp. Because xtol is set to 1e-16 above,
    # this should not take too many iterations.

    iters, maxiter = 0, 1000

    while abs(rootfn(numpy.nextafter(mu, numpy.inf))) < cur_err and iters < maxiter:
        mu = numpy.nextafter(mu, numpy.inf)
        cur_err = abs(rootfn(mu))
        iters += 1

    while abs(rootfn(numpy.nextafter(mu, -numpy.inf))) < cur_err and iters < maxiter:
        mu = numpy.nextafter(mu, -numpy.inf)
        cur_err = abs(rootfn(mu))
        iters += 1

    return mu, f_occ(mu, mo_es, sigma)

def _get_fermi(mo_energy, nocc):
    mo_e_sorted = numpy.sort(mo_energy)
    if isinstance(nocc, (int, numpy.integer)):
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
        from pyscf.pbc.tools import print_mo_energy_occ
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            mo_occ = super().get_occ(mo_energy, mo_coeff)
            return mo_occ

        if self.istype('ROHF'):
            # ROHF leads to two Fock matrices. It's not clear how to define the
            # Roothaan effective Fock matrix from the two.
            raise NotImplementedError('Smearing-ROHF')
        is_uhf = self.istype('UHF')
        is_rhf = not is_uhf

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

    def to_gpu(self):
        obj = self.undo_smearing().to_gpu().smearing()
        obj.sigma = self.sigma
        obj.smearing_method = self.smearing_method
        obj.fix_spin = self.fix_spin
        obj.entropy = self.entropy
        obj.e_free = self.e_free
        obj.e_zero = self.e_zero
        return obj
