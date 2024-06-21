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
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
import scipy.special
import scipy.optimize
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools
from pyscf.lib import logger
from pyscf.scf import addons as mol_addons
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.tools import k2gamma
from pyscf import __config__

SMEARING_METHOD = mol_addons.SMEARING_METHOD


def project_mo_nr2nr(cell1, mo1, cell2, kpts=None):
    r''' Project orbital coefficients

    .. math::

        |\psi1> = |AO1> C1

        |\psi2> = P |\psi1> = |AO2>S^{-1}<AO2| AO1> C1 = |AO2> C2

        C2 = S^{-1}<AO2|AO1> C1
    '''
    s22 = cell2.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    s21 = pbcgto.intor_cross('int1e_ovlp', cell2, cell1, kpts=kpts)
    if kpts is None or numpy.shape(kpts) == (3,):  # A single k-point
        return scipy.linalg.solve(s22, s21.dot(mo1), assume_a='pos')
    else:
        assert (len(kpts) == len(mo1))
        return [scipy.linalg.solve(s22[k], s21[k].dot(mo1[k]), assume_a='pos')
                for k, kpt in enumerate(kpts)]

def project_dm_k2k(cell, dm, kpts1, kpts2):
    '''Project density matrix from k-point mesh 1 to k-point mesh 2'''
    bvk_mesh = k2gamma.kpts_to_kmesh(cell, kpts1)
    Ls = k2gamma.translation_vectors_for_kmesh(cell, bvk_mesh, True)
    c = _k2k_projection(kpts1, kpts2, Ls)
    return lib.einsum('km,kuv->muv', c, dm)

def _k2k_projection(kpts1, kpts2, Ls):
    weight = 1. / len(Ls)
    expRk1 = numpy.exp(1j*numpy.dot(Ls, kpts1.T))
    expRk2 = numpy.exp(-1j*numpy.dot(Ls, kpts2.T))
    c = expRk1.T.dot(expRk2) * weight
    return (c*c.conj()).real.copy()

def _partition_occ(mo_occ, mo_energy_kpts):
    mo_occ_kpts = []
    p1 = 0
    for e in mo_energy_kpts:
        p0, p1 = p1, p1 + e.size
        occ = mo_occ[p0:p1]
        mo_occ_kpts.append(occ)
    return mo_occ_kpts

def _get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
    grad_kpts = []
    for k, mo in enumerate(mo_coeff_kpts):
        f_mo = reduce(numpy.dot, (mo.T.conj(), fock[k], mo))
        nmo = f_mo.shape[0]
        grad_kpts.append(f_mo[numpy.tril_indices(nmo, -1)])
    return numpy.hstack(grad_kpts)

class _SmearingKSCF(mol_addons._SmearingSCF):
    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        if (self.sigma == 0) or (not self.sigma) or (not self.smearing_method):
            mo_occ_kpts = super().get_occ(mo_energy_kpts, mo_coeff_kpts)
            return mo_occ_kpts

        is_uhf = self.istype('KUHF')
        is_rhf = self.istype('KRHF')
        is_rohf = self.istype('KROHF')

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

        if self.fix_spin and (is_uhf or is_rohf): # spin separated fermi level
            if is_rohf: # treat rohf as uhf
                mo_energy_kpts = (mo_energy_kpts, mo_energy_kpts)
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
            if is_rohf:
                mo_occ_kpts = numpy.array(mo_occ_kpts, dtype=numpy.float64).sum(axis=0)
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
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.pbc.scf import khf
    if not isinstance(mf, khf.KSCF):
        return mol_addons.smearing(mf, sigma, method, mu0, fix_spin)

    if isinstance(mf, mol_addons._SmearingSCF):
        mf.sigma = sigma
        mf.smearing_method = method
        mf.mu0 = mu0
        mf.fix_spin = fix_spin
        return mf

    return lib.set_class(_SmearingKSCF(mf, sigma, method, mu0, fix_spin),
                         (_SmearingKSCF, mf.__class__))

def smearing_(mf, *args, **kwargs):
    mf1 = smearing(mf, *args, **kwargs)
    mf.__class__ = mf1.__class__
    mf.__dict__ = mf1.__dict__
    return mf

def canonical_occ_(mf, nelec=None):
    '''Label the occupancies for each orbital for sampled k-points.
    This is for KUHF objects.
    Each k-point has a fixed number of up and down electrons in this,
    which results in a finite size error for metallic systems
    but can accelerate convergence.
    '''
    from pyscf.pbc.scf import kuhf
    assert (isinstance(mf, kuhf.KUHF))

    def get_occ(mo_energy_kpts=None, mo_coeff=None):
        if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

        if nelec is None:
            cell_nelec = mf.cell.nelec
        else:
            cell_nelec = nelec

        homo=[-1e8,-1e8]
        lumo=[1e8,1e8]
        mo_occ_kpts = [[], []]
        for s in [0,1]:
            for k, mo_energy in enumerate(mo_energy_kpts[s]):
                e_idx = numpy.argsort(mo_energy)
                e_sort = mo_energy[e_idx]
                n = cell_nelec[s]
                mo_occ = numpy.zeros_like(mo_energy)
                mo_occ[e_idx[:n]] = 1
                homo[s] = max(homo[s], e_sort[n-1])
                lumo[s] = min(lumo[s], e_sort[n])
                mo_occ_kpts[s].append(mo_occ)

        for nm,s in zip(['alpha','beta'],[0,1]):
            logger.info(mf, nm+' HOMO = %.12g  LUMO = %.12g', homo[s], lumo[s])
            if homo[s] > lumo[s]:
                logger.warn(mf, "WARNING! HOMO is greater than LUMO! "
                            "This may lead to incorrect canonical occupation.")

        return mo_occ_kpts

    mf.get_occ = get_occ
    return mf
canonical_occ = canonical_occ_


def convert_to_uhf(mf, out=None):
    '''Convert the given mean-field object to the corresponding unrestricted
    HF/KS object
    '''
    from pyscf.pbc import scf
    from pyscf.pbc import dft

    if out is None:
        if isinstance(mf, (scf.uhf.UHF, scf.kuhf.KUHF)):
            return mf.copy()
        else:
            if isinstance(mf, (scf.ghf.GHF, scf.kghf.KGHF)):
                raise NotImplementedError(
                    f'No conversion from {mf.__class__} to uhf object')

            known_cls = {
                dft.krks_ksymm.KRKS  : dft.kuks_ksymm.KUKS,
                dft.kroks.KROKS: dft.kuks.KUKS,
                dft.krks.KRKS  : dft.kuks.KUKS,
                dft.roks.ROKS  : dft.uks.UKS  ,
                dft.rks.RKS    : dft.uks.UKS  ,
                scf.khf_ksymm.KRHF : scf.kuhf_ksymm.KUHF,
                scf.krohf.KROHF: scf.kuhf.KUHF,
                scf.khf.KRHF   : scf.kuhf.KUHF,
                scf.rohf.ROHF  : scf.uhf.UHF  ,
                scf.hf.RHF     : scf.uhf.UHF  ,
            }
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, False)
    else:
        assert (isinstance(out, (scf.uhf.UHF, scf.kuhf.KUHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))

    if out is not None:
        out = mol_addons._update_mf_without_soscf(mf, out, False)

    return mol_addons._update_mo_to_uhf_(mf, out)

def convert_to_rhf(mf, out=None):
    '''Convert the given mean-field object to the corresponding restricted
    HF/KS object
    '''
    from pyscf.pbc import scf
    from pyscf.pbc import dft

    if getattr(mf, 'nelec', None) is None:
        nelec = mf.cell.nelec
    else:
        nelec = mf.nelec

    if out is not None:
        assert (isinstance(out, (scf.hf.RHF, scf.khf.KRHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))
        out = mol_addons._update_mf_without_soscf(mf, out, False)

    elif nelec[0] != nelec[1] and isinstance(mf, (scf.rohf.ROHF, scf.krohf.KROHF)):
        if getattr(mf, '_scf', None):
            return mol_addons._update_mf_without_soscf(mf, mf._scf.copy(), False)
        else:
            return mf.copy()

    else:
        if isinstance(mf, (scf.hf.RHF, scf.khf.KRHF)):
            return mf.copy()
        else:
            if isinstance(mf, (scf.ghf.GHF, scf.kghf.KGHF)):
                raise NotImplementedError(
                    f'No conversion from {mf.__class__} to rhf object')

            if nelec[0] == nelec[1]:
                known_cls = {
                    dft.kuks_ksymm.KUKS : dft.krks_ksymm.KRKS,
                    dft.kuks.KUKS       : dft.krks.KRKS      ,
                    dft.kroks.KROKS     : dft.krks.KRKS      ,
                    dft.uks.UKS         : dft.rks.RKS        ,
                    dft.roks.ROKS       : dft.rks.RKS        ,
                    scf.kuhf_ksymm.KUHF : scf.khf_ksymm.KRHF ,
                    scf.kuhf.KUHF       : scf.khf.KRHF       ,
                    scf.krohf.KROHF     : scf.khf.KRHF       ,
                    scf.uhf.UHF         : scf.hf.RHF         ,
                    scf.rohf.ROHF       : scf.hf.RHF         ,
                }
            else:
                known_cls = {
                    dft.kuks.KUKS : dft.kroks.KROKS,
                    dft.uks.UKS   : dft.roks.ROKS  ,
                    scf.kuhf.KUHF : scf.krohf.KROHF,
                    scf.uhf.UHF   : scf.rohf.ROHF  ,
                }
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, False)

    return mol_addons._update_mo_to_rhf_(mf, out)

def convert_to_ghf(mf, out=None):
    '''Convert the given mean-field object to the generalized HF/KS object

    Args:
        mf : SCF object

    Returns:
        An generalized SCF object
    '''
    from pyscf.pbc import scf

    if out is not None:
        assert (isinstance(out, (scf.ghf.GHF, scf.kghf.KGHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert (isinstance(out, scf.khf.KSCF))
        else:
            assert (not isinstance(out, scf.khf.KSCF))

    if isinstance(mf, (scf.ghf.GHF, scf.kghf.KGHF)):
        if out is None:
            return mf.copy()
        else:
            out.__dict__.update(mf.__dict__)
            return out

    elif isinstance(mf, scf.khf.KSCF):

        def update_mo_(mf, mf1):
            mf1.__dict__.update(mf.__dict__)
            if mf.mo_energy is not None:
                mf1.mo_energy = []
                mf1.mo_occ = []
                mf1.mo_coeff = []
                if hasattr(mf.kpts, 'nkpts_ibz'):
                    nkpts = mf.kpts.nkpts_ibz
                else:
                    nkpts = len(mf.kpts)
                is_rhf = mf.istype('KRHF')
                for k in range(nkpts):
                    if is_rhf:
                        mo_a = mo_b = mf.mo_coeff[k]
                        ea = getattr(mf.mo_energy[k], 'mo_ea', mf.mo_energy[k])
                        eb = getattr(mf.mo_energy[k], 'mo_eb', mf.mo_energy[k])
                        occa = mf.mo_occ[k] > 0
                        occb = mf.mo_occ[k] == 2
                        orbspin = mol_addons.get_ghf_orbspin(ea, mf.mo_occ[k], True)
                    else:
                        mo_a = mf.mo_coeff[0][k]
                        mo_b = mf.mo_coeff[1][k]
                        ea = mf.mo_energy[0][k]
                        eb = mf.mo_energy[1][k]
                        occa = mf.mo_occ[0][k]
                        occb = mf.mo_occ[1][k]
                        orbspin = mol_addons.get_ghf_orbspin((ea, eb), (occa, occb), False)

                    nao, nmo = mo_a.shape

                    mo_energy = numpy.empty(nmo*2)
                    mo_energy[orbspin==0] = ea
                    mo_energy[orbspin==1] = eb
                    mo_occ = numpy.empty(nmo*2)
                    mo_occ[orbspin==0] = occa
                    mo_occ[orbspin==1] = occb

                    mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mo_a.dtype)
                    mo_coeff[:nao,orbspin==0] = mo_a
                    mo_coeff[nao:,orbspin==1] = mo_b
                    mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)

                    mf1.mo_energy.append(mo_energy)
                    mf1.mo_occ.append(mo_occ)
                    mf1.mo_coeff.append(mo_coeff)

            return mf1

        if out is None:
            out = scf.kghf.KGHF(mf.cell)
        return update_mo_(mf, out)

    else:
        if out is None:
            out = scf.ghf.GHF(mf.cell)
        out = mol_addons.convert_to_ghf(mf, out, remove_df=False)
        # Manually update .with_df because this attribute may not be passed to the
        # output object correctly in molecular convert function
        out.with_df = mf.with_df
        return out

def convert_to_kscf(mf, out=None):
    '''Convert gamma point SCF object to k-point SCF object
    '''
    from pyscf.pbc import scf, dft
    if not isinstance(mf, scf.khf.KSCF):
        assert isinstance(mf, scf.hf.SCF)
        known_cls = {
            dft.uks.UKS   : dft.kuks.KUKS  ,
            dft.roks.ROKS : dft.kroks.KROKS,
            dft.rks.RKS   : dft.krks.KRKS  ,
            dft.gks.GKS   : dft.kgks.KGKS  ,
            scf.uhf.UHF   : scf.kuhf.KUHF  ,
            scf.rohf.ROHF : scf.krohf.KROHF,
            scf.hf.RHF    : scf.khf.KRHF   ,
            scf.ghf.GHF   : scf.kghf.KGHF  ,
        }
        mf = mol_addons._object_without_soscf(mf, known_cls, False)
        if mf.mo_energy is not None:
            if mf.istype('UHF'):
                mf.mo_occ = mf.mo_occ[:,numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[:,numpy.newaxis]
                mf.mo_energy = mf.mo_energy[:,numpy.newaxis]
            else:
                mf.mo_occ = mf.mo_occ[numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[numpy.newaxis]
                mf.mo_energy = mf.mo_energy[numpy.newaxis]

    if out is None:
        return mf

    return mol_addons._update_mf_without_soscf(mf, out, False)

convert_to_khf = convert_to_kscf

def mo_energy_with_exxdiv_none(mf, mo_coeff=None):
    ''' compute mo energy from the diagonal of fock matrix with exxdiv=None
    '''
    from pyscf.pbc import scf, dft
    from pyscf.lib import logger
    log = logger.new_logger(mf)

    if mo_coeff is None: mo_coeff = mf.mo_coeff

    if mf.exxdiv is None and mf.mo_coeff is mo_coeff:
        return mf.mo_energy

    with lib.temporary_env(mf, exxdiv=None):
        dm = mf.make_rdm1(mo_coeff)
        vhf = mf.get_veff(mf.mol, dm)
        fockao = mf.get_fock(vhf=vhf, dm=dm)

    def _get_moe1(C, fao):
        return lib.einsum('pi,pq,qi->i', C.conj(), fao, C)
    def _get_moek(kC, kfao):
        return [_get_moe1(C, fao) for C,fao in zip(kC, kfao)]

    # avoid using isinstance as some are other's derived class
    if mf.__class__ in [scf.rhf.RHF, scf.ghf.GHF, dft.rks.RKS, dft.gks.GKS]:
        return _get_moe1(mo_coeff, fockao)
    elif mf.__class__ in [scf.uhf.UHF, dft.uks.UKS]:
        return _get_moek(mo_coeff, fockao)
    elif mf.__class__ in [scf.krhf.KRHF, scf.kghf.KGHF, dft.krks.KRKS, dft.kgks.KGKS]:
        return _get_moek(mo_coeff, fockao)
    elif mf.__class__ in [scf.kuhf.KUHF, dft.kuks.KUKS]:
        return [_get_moek(kC, kfao) for kC,kfao in zip(mo_coeff,fockao)]
    else:
        log.error(f'Unknown SCF type {mf.__class__.__name__}')
        raise NotImplementedError

######## ADDED by Ning to deal with occupation number in KUHF ########

def _time_reversal_pairs(cell, kpts:numpy.ndarray):
    
    assert kpts.ndim <= 2 
    
    kpts = kpts.reshape(-1, 3)  
    kmesh = k2gamma.kpts_to_kmesh(cell, kpts) 
    
    assert numpy.prod(kmesh) == kpts.shape[0]
    
    nkpts = kpts.shape[0]
    
    time_reversal_pairs = [None] * nkpts
    
    for ix in range(kmesh[0]):
        for iy in range(kmesh[1]):
            for iz in range(kmesh[2]):
                loc1 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
                loc2 = ((kmesh[0]-ix) % kmesh[0]) * kmesh[1] * kmesh[2] + ((kmesh[1]-iy) % kmesh[1]) * kmesh[2] + ((kmesh[2]-iz) % kmesh[2])
                if time_reversal_pairs[loc1] is not None:
                    assert time_reversal_pairs[loc1] == loc2
                else:
                    time_reversal_pairs[loc1] = loc2
                if time_reversal_pairs[loc2] is not None:
                    assert time_reversal_pairs[loc2] == loc1
                else:
                    time_reversal_pairs[loc2] = loc1
    
    return numpy.array(time_reversal_pairs, dtype=numpy.int32).reshape(nkpts)

def _get_k_occ(mo_energy_kpts:numpy.ndarray, fermi:float):
    '''
    given fermi energy, get the number of occupied orbitals within each k-point sector
    '''
    
    nkpts = len(mo_energy_kpts)
    res = numpy.zeros(nkpts, dtype=numpy.int32)

    for k in range(nkpts):
        res[k] = numpy.sum(mo_energy_kpts[k] <= fermi)

    return res

def _get_k_occ_preserve_time_reversal_symmetry(mo_energy_kpts, nocc, time_reversal_pairing, tol=1e-4):

    
    mo_energy = numpy.sort(numpy.hstack(mo_energy_kpts))
    fermi     = mo_energy[nocc-1]
    print("Fermi energy = ", fermi)
    print("nocc         = ", nocc)
    nkpts     = len(mo_energy_kpts)
    k_mo_occ  = _get_k_occ(mo_energy_kpts, fermi)
    print("k_mo_occ = ", k_mo_occ)
    
    ##### check whether the current
    
    tr_preserved = True
    
    for k, k_occ in enumerate(k_mo_occ):
        k2 = time_reversal_pairing[k]
        if k_occ != k_mo_occ[k2]:
            tr_preserved = False
            break
    
    if tr_preserved:  ### nothing to be done 
        
        #mo_occ_kpts = numpy.zeros_like(mo_energy_kpts, dtype=numpy.float64)
        mo_occ_kpts = []
        for ikpt in range(nkpts):
            mo_occ_kpts.append(numpy.zeros_like(mo_energy_kpts[ikpt], dtype=numpy.float64))
            mo_occ_kpts[ikpt][:k_mo_occ[ikpt]] = 1.0
        return mo_occ_kpts

    print("Time reversal symmetry is not preserved, trying to fix it ...")

    ###### find the fermi energy/nocc that preserves time reversal symmetry ######
    
    nocc_tr_preserved = None
    kocc_tr_preserved = None
    
    for nocc_now in range(nocc - 1, -1, -1):
        fermi = mo_energy[nocc_now-1]
        k_occ_tmp = _get_k_occ(mo_energy_kpts, fermi)
        if numpy.sum(k_occ_tmp) != nocc_now:
            continue
        
        tr_preserved = True
        
        for k, k_occ in enumerate(k_occ_tmp):
            k2 = time_reversal_pairing[k]
            if k_occ != k_occ_tmp[k2]:
                tr_preserved = False
                break
        
        if tr_preserved:
            nocc_tr_preserved = nocc_now
            kocc_tr_preserved = k_occ_tmp
            break
    
    assert nocc_tr_preserved is not None
    
    print("Time reversal symmetry is preserved at nocc = ", nocc_tr_preserved)
    
    k_mo_occ = kocc_tr_preserved
    print("kocc_tr_preserved = ", kocc_tr_preserved)
    
    #mo_occ_kpts = numpy.zeros_like(mo_energy_kpts, dtype=numpy.float64)
    mo_occ_kpts = []
    for ikpt in range(nkpts):
        mo_occ_kpts.append(numpy.zeros_like(mo_energy_kpts[ikpt], dtype=numpy.float64))
        mo_occ_kpts[ikpt][:k_mo_occ[ikpt]] = 1.0
        #mo_occ_kpts[:k_mo_occ[ikpt]] = 1.0
        
    k_mo_occ = numpy.asarray(kocc_tr_preserved, dtype=numpy.int32)

    #print("mo_occ_kpts = ", mo_occ_kpts)

    nocc_left = nocc - nocc_tr_preserved
    nocc_now  = nocc_tr_preserved

    while True:
        
        ##### find the set of degenerated orbitals #####
        
        ndegenerate = 1
        
        for i in range(nocc_now+1, len(mo_energy)):
            if abs(mo_energy[i] - mo_energy[nocc_now]) < tol:
                ndegenerate += 1
            else:
                break
        
        ##### assign fractional occupation to the degenerated orbitals #####
        
        if ndegenerate == 1:
            kpt_id = None
            for ikpt in range(nkpts):
                if abs(mo_energy_kpts[ikpt][k_mo_occ[ikpt]] - mo_energy[nocc_now]) < 1e-8:
                    kpt_id = ikpt
                    break
            assert kpt_id is not None
            if time_reversal_pairing[kpt_id] != kpt_id:
                print("kpt_id = ", kpt_id, " time_reversal_pairing = ", time_reversal_pairing[kpt_id])
                print("force TR symmetry by setting ndegenerate = 2")
                ndegenerate = 2

        occ_assigned = min(nocc_left, ndegenerate) / ndegenerate
        print("Assigning ", occ_assigned, " to ", ndegenerate, " degenerated orbitals")
        print("nocc_left = ", nocc_left, " nocc_now = ", nocc_now, " ndegenerate = ", ndegenerate)
        print("mo_energy = ", mo_energy[nocc_now:nocc_now+ndegenerate])
        
        # find the k-point to assgin via loop over all k-points
        
        k_added = numpy.zeros(nkpts, dtype=numpy.int32)
        
        for _id_ in range(ndegenerate):
            for ikpt in range(nkpts):
                if abs(mo_energy_kpts[ikpt][k_mo_occ[ikpt]] - mo_energy[nocc_now+_id_]) < 1e-8:
                    mo_occ_kpts[ikpt][k_mo_occ[ikpt]] = occ_assigned
                    k_mo_occ[ikpt] += 1
                    k_added[ikpt]  += 1
                    break
        
        for ikpt in range(nkpts):
            assert k_added[ikpt] == k_added[time_reversal_pairing[ikpt]]
        
        print("k_added = ", k_added)
        
        # update info #
         
        nocc_left -= min(nocc_left, ndegenerate)
        nocc_now  += ndegenerate
        
        if nocc_left == 0:
            break    

    return mo_occ_kpts

######## forcing time reversal symmetries ########

def pbc_frac_occ_(mf, tol=1e-3):
    '''
    Addons for SCF methods to assign fractional occupancy for degenerated
    occupied HOMOs.

    Examples::

        >>> mf = gto.M(atom='O 0 0 0; O 0 0 1', verbose=4).RHF()
        >>> mf = scf.addons.frac_occ(mf)
        >>> mf.run()
    '''

    #### if not SCF with k-point, then call the original one ####

    from pyscf.scf.addons import frac_occ_

    if mf.istype("UHF"):
        return frac_occ_(mf, tol)
    elif mf.istype("ROHF"):
        return frac_occ_(mf, tol)
    elif mf.istype("RHF"):
        return frac_occ_(mf, tol)
    
    ########### deal with k-point symmetry case ###########
    
    time_reversal_pairing = _time_reversal_pairs(mf.cell, mf.kpts)
    
    old_get_occ = mf.get_occ
    cell        = mf.cell
    
    if mf.istype("KUHF"):
        
        print("fix occ problem")
        
        def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
            '''Label the occupancies for each orbital for sampled k-points.

            This is a k-point version of scf.hf.SCF.get_occ
            '''

            if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

            nocc_a, nocc_b = mf.nelec
            print("nocc_a = ", nocc_a, " nocc_b = ", nocc_b)
            mo_energy = numpy.sort(numpy.hstack(mo_energy_kpts[0]))
            fermi_a = mo_energy[nocc_a-1]
            mo_occ_kpts = [[], []]
            
            # for mo_e in mo_energy_kpts[0]:
            #     mo_occ_kpts[0].append((mo_e <= fermi_a).astype(numpy.double))
            # if nocc_a < len(mo_energy):
            #     logger.info(mf, 'alpha HOMO = %.12g  LUMO = %.12g', fermi_a, mo_energy[nocc_a])
            # else:
            #     logger.info(mf, 'alpha HOMO = %.12g  (no LUMO because of small basis) ', fermi_a) 
            
            mo_occ_kpts[0] = _get_k_occ_preserve_time_reversal_symmetry(mo_energy_kpts[0], nocc_a, time_reversal_pairing, tol=tol)

            if nocc_b > 0:
                #mo_energy = numpy.sort(numpy.hstack(mo_energy_kpts[1]))
                #fermi_b = mo_energy[nocc_b-1]
                #for mo_e in mo_energy_kpts[1]:
                #    mo_occ_kpts[1].append((mo_e <= fermi_b).astype(numpy.double))
                #if nocc_b < len(mo_energy):
                #    logger.info(mf, 'beta HOMO = %.12g  LUMO = %.12g', fermi_b, mo_energy[nocc_b])
                #else:
                #    logger.info(mf, 'beta HOMO = %.12g  (no LUMO because of small basis) ', fermi_b)
                mo_occ_kpts[1] = _get_k_occ_preserve_time_reversal_symmetry(mo_energy_kpts[1], nocc_b, time_reversal_pairing, tol=tol)
            else:
                for mo_e in mo_energy_kpts[1]:
                    mo_occ_kpts[1].append(numpy.zeros_like(mo_e))

            if mf.verbose >= logger.DEBUG:
                numpy.set_printoptions(threshold=len(mo_energy))
                logger.debug(mf, '     k-point                  alpha mo_energy')
                for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
                    if (numpy.count_nonzero(mo_occ_kpts[0][k]) > 0 and
                        numpy.count_nonzero(mo_occ_kpts[0][k] == 0) > 0):
                        logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                                     k, kpt[0], kpt[1], kpt[2],
                                     mo_energy_kpts[0][k][mo_occ_kpts[0][k]> 0],
                                     mo_energy_kpts[0][k][mo_occ_kpts[0][k]==0])
                    else:
                        logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                                     k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[0][k])
                logger.debug(mf, '     k-point                  beta  mo_energy')
                for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
                    if (numpy.count_nonzero(mo_occ_kpts[1][k]) > 0 and
                        numpy.count_nonzero(mo_occ_kpts[1][k] == 0) > 0):
                        logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                                     k, kpt[0], kpt[1], kpt[2],
                                     mo_energy_kpts[1][k][mo_occ_kpts[1][k]> 0],
                                     mo_energy_kpts[1][k][mo_occ_kpts[1][k]==0])
                    else:
                        logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                                     k, kpt[0], kpt[1], kpt[2], mo_energy_kpts[1][k])
                numpy.set_printoptions(threshold=1000)

            print("mo_occ_kpts = ", mo_occ_kpts)
            #print("mo_occ_kpts.shape = ", mo_occ_kpts[0].shape, mo_occ_kpts[1].shape)
            
            return mo_occ_kpts
        
    elif mf.istype("KRHF"):
        
        def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
            '''Label the occupancies for each orbital for sampled k-points.
            This is a k-point version of scf.hf.SCF.get_occ
            '''
            if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

            nkpts = len(mo_energy_kpts)
            nocc = cell.tot_electrons(nkpts) // 2

            mo_occ_kpts = _get_k_occ_preserve_time_reversal_symmetry(mo_energy_kpts, nocc, time_reversal_pairing, tol)
            mo_occ_kpts*= 2.0

            if nocc < mo_energy.size:
                logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
                if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                    logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                                mo_energy[nocc-1], mo_energy[nocc])
            else:
                logger.info(mf, 'HOMO = %.12g', mo_energy[nocc-1])

            if mf.verbose >= logger.DEBUG:
                numpy.set_printoptions(threshold=len(mo_energy))
                logger.debug(mf, '     k-point                  mo_energy')
                for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
                    logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                                 k, kpt[0], kpt[1], kpt[2],
                                 numpy.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                                 numpy.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
                numpy.set_printoptions(threshold=1000)

            return mo_occ_kpts

    else:
        raise NotImplementedError("pbc_frac_occ for %s is not implemented"%(mf.__class__))
    
    mf.get_occ = get_occ

    return mf

pbc_frac_occ = pbc_frac_occ_