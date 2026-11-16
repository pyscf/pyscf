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
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import addons as mol_addons
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints

__all__ = ['smearing', 'smearing_', 'KPoints']

SMEARING_METHOD = mol_addons.SMEARING_METHOD

def _get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
    grad_kpts = []
    for k, mo in enumerate(mo_coeff_kpts):
        f_mo = mo.conj().T.dot(fock[k]).dot(mo)
        nmo = f_mo.shape[0]
        grad_kpts.append(f_mo[numpy.tril_indices(nmo, -1)])
    return numpy.hstack(grad_kpts)

def _partition_occ(mo_occ, mo_energy_kpts):
    mo_occ_kpts = []
    p1 = 0
    for e in mo_energy_kpts:
        p0, p1 = p1, p1 + e.size
        occ = mo_occ[p0:p1]
        mo_occ_kpts.append(occ)
    return mo_occ_kpts

class _SmearingKSCF(mol_addons._SmearingSCF):
    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        from pyscf.pbc import scf
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            mo_occ_kpts = super().get_occ(mo_energy_kpts, mo_coeff_kpts)
            return mo_occ_kpts

        is_uhf = self.istype('KUHF')
        is_rhf = self.istype('KRHF')

        sigma = self.sigma
        if self.smearing_method.lower() == 'fermi':
            f_occ = mol_addons._fermi_smearing_occ
        else:
            f_occ = mol_addons._gaussian_smearing_occ

        kpts = getattr(self, 'kpts', None)
        if isinstance(kpts, KPoints):
            nkpts = kpts.nkpts
            mo_energy_kpts = kpts.transform_mo_energy(mo_energy_kpts)
        else:
            nkpts = len(kpts)

        if self.fix_spin and is_uhf: # spin separated fermi level
            mo_es = [numpy.hstack(mo_energy_kpts[0]),
                     numpy.hstack(mo_energy_kpts[1])]
            nocc = self.nelec
            if self.mu0 is None:
                mu_a, occa = mol_addons._smearing_optimize(f_occ, mo_es[0], nocc[0], sigma)
                mu_b, occb = mol_addons._smearing_optimize(f_occ, mo_es[1], nocc[1], sigma)
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
            self.entropy /= nkpts

            fermi = (mol_addons._get_fermi(mo_es[0], nocc[0]),
                     mol_addons._get_fermi(mo_es[1], nocc[1]))
            logger.debug(self, '    Alpha-spin Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[0], mo_occs[0].sum(), nocc[0])
            logger.debug(self, '    Beta-spin  Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi[1], mo_occs[1].sum(), nocc[1])
            logger.info(self, '    sigma = %g  Optimized mu_alpha = %.12g  entropy = %.12g',
                        sigma, mu[0], self.entropy)
            logger.info(self, '    sigma = %g  Optimized mu_beta  = %.12g  entropy = %.12g',
                        sigma, mu[1], self.entropy)

            mo_occ_kpts =(_partition_occ(mo_occs[0], mo_energy_kpts[0]),
                          _partition_occ(mo_occs[1], mo_energy_kpts[1]))
            tools.print_mo_energy_occ_kpts(self, mo_energy_kpts, mo_occ_kpts, True)
        else:
            nocc = nelectron = self.mol.tot_electrons(nkpts)
            if is_uhf:
                mo_es_a = numpy.hstack(mo_energy_kpts[0])
                mo_es_b = numpy.hstack(mo_energy_kpts[1])
                mo_es = numpy.append(mo_es_a, mo_es_b)
            else:
                mo_es = numpy.hstack(mo_energy_kpts)
            if is_rhf:
                nocc = (nelectron + 1) // 2

            if self.mu0 is None:
                mu, mo_occs = mol_addons._smearing_optimize(f_occ, mo_es, nocc, sigma)
            else:
                # If mu0 is given, fix mu instead of electron number. XXX -Chong Sun
                mu = self.mu0
                assert numpy.isscalar(mu)
                mo_occs = f_occ(mu, mo_es, sigma)
            self.entropy = self._get_entropy(mo_es, mo_occs, mu) / nkpts
            if is_rhf:
                mo_occs *= 2
                self.entropy *= 2

            fermi = mol_addons._get_fermi(mo_es, nocc)
            logger.debug(self, '    Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                         fermi, mo_occs.sum(), nelectron)
            logger.info(self, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                        sigma, mu, self.entropy)

            if is_uhf:
                # mo_es_a and mo_es_b may have different dimensions for
                # different k-points
                nmo_a = mo_es_a.size
                mo_occ_kpts =(_partition_occ(mo_occs[:nmo_a], mo_energy_kpts[0]),
                              _partition_occ(mo_occs[nmo_a:], mo_energy_kpts[1]))
            else:
                mo_occ_kpts = _partition_occ(mo_occs, mo_energy_kpts)
            tools.print_mo_energy_occ_kpts(self, mo_energy_kpts, mo_occ_kpts, is_uhf)

        if isinstance(kpts, KPoints):
            if is_uhf:
                mo_occ_kpts = (kpts.check_mo_occ_symmetry(mo_occ_kpts[0]),
                               kpts.check_mo_occ_symmetry(mo_occ_kpts[1]))
            else:
                mo_occ_kpts = kpts.check_mo_occ_symmetry(mo_occ_kpts)
        return mo_occ_kpts

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            return super().get_grad(mo_coeff_kpts, mo_occ_kpts, fock)

        if fock is None:
            dm1 = self.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = self.get_hcore() + self.get_veff(self.mol, dm1)
        if self.istype('KUHF'):
            ga = _get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = _get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return numpy.hstack((ga,gb))
        else: # rhf and ghf
            return _get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

def smearing(mf, sigma=None, method=SMEARING_METHOD, mu0=None, fix_spin=False):
    '''Apply fractional occupations across the sampled k-points.

    Args:
        mf : SCF object
            Periodic HF or Kohn-Sham object. KROHF/KROKS is not supported.
            Objects without k-point sampling are passed to
            :func:`pyscf.scf.smearing.smearing`.

    Kwargs:
        sigma : float or None
            Positive smearing width in Hartree; for Fermi-Dirac smearing,
            ``sigma = k_B T``. None (default) or zero disables smearing.
        method : str
            ``'fermi'`` for Fermi-Dirac occupations or ``'gauss'`` (also
            ``'gaussian'``) for Gaussian occupations. Defaults to ``'fermi'``
            unless configured by ``pbc_scf_addons_smearing_method``. Matching
            is case-insensitive; any other nonempty string selects Gaussian
            smearing. None or an empty string disables smearing.
        mu0 : float or pair of floats or None
            Fixed chemical potential in Hartree. With None (default), optimize
            the potential over all sampled k-points to preserve the electron
            number. A fixed value allows the electron number to change.
            For KUHF/KUKS with ``fix_spin=True``, a pair sets alpha and beta
            potentials separately; a scalar sets both to the same value.
        fix_spin : bool
            For KUHF/KUKS, use separate alpha and beta chemical potentials
            across the k-points. With ``mu0=None``, preserve the spin
            populations in ``mf.nelec``. Default is False, allowing spin
            polarization to change at a common chemical potential.

    Returns:
        An SCF object configured to use smearing. If mf already uses smearing,
        returns mf itself.

    Notes:
        Call ``run`` or ``kernel`` to start SCF. Attributes are copied by
        reference from ``mf`` and may be shared. If ``mf`` already uses smearing,
        its settings are updated.

        After a calculation with active smearing, ``entropy`` is normalized per
        unit cell, including spin degeneracy. The energy attributes (Hartree per
        cell) satisfy ``e_free = e_tot - sigma * entropy`` and
        ``e_zero = e_tot - sigma * entropy / 2``. ``kernel`` returns ``e_tot``.
        See :func:`pyscf.scf.smearing.smearing` for the entropy convention and
        behavior when smearing is disabled.

        With symmetry-adapted KPoints, occupations are determined over the full
        Brillouin-zone mesh and checked for consistency with the k-point symmetry.
        See ``examples/pbc/23-smearing.py`` for a complete calculation.
    '''
    from pyscf.pbc.scf import khf
    assert not mf.istype('KROHF'), 'Smearing for KROHF/KROKS is not supported'
    if not isinstance(mf, khf.KSCF):
        return mol_addons.smearing(mf, sigma, method, mu0, fix_spin)

    if isinstance(mf, mol_addons._SmearingSCF):
        mf.sigma = sigma
        mf.smearing_method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    if mf.istype('ROHF'):
        # ROHF leads to two Fock matrices. It's not clear how to define the
        # Roothaan effective Fock matrix from the two.
        raise NotImplementedError('Smearing-ROHF')

    return lib.set_class(_SmearingKSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingKSCF, mf.__class__))

def smearing_(mf, *args, **kwargs):
    '''Apply periodic smearing in place and return the original SCF object.

    ``mf`` is the SCF object to modify. Additional positional and keyword
    arguments are passed to :func:`smearing`: ``sigma``, ``method``, ``mu0``,
    and ``fix_spin``. See that function for units, defaults, and limitations.

    This replaces ``mf.__class__`` and ``mf.__dict__``, so all references to
    ``mf`` see the change. No SCF calculation is run.
    '''
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf
