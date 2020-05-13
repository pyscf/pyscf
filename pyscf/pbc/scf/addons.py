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

import copy
from functools import reduce
import numpy
import scipy.linalg
import scipy.special
import scipy.optimize
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.lib import logger
from pyscf.scf import addons as mol_addons
from pyscf import __config__

SMEARING_METHOD = getattr(__config__, 'pbc_scf_addons_smearing_method', 'fermi')


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
        return scipy.linalg.solve(s22, s21.dot(mo1), sym_pos=True)
    else:
        assert(len(kpts) == len(mo1))
        return [scipy.linalg.solve(s22[k], s21[k].dot(mo1[k]), sym_pos=True)
                for k, kpt in enumerate(kpts)]


def smearing_(mf, sigma=None, method=SMEARING_METHOD, mu0=None):
    '''Fermi-Dirac or Gaussian smearing'''
    from pyscf.scf import uhf
    from pyscf.scf import ghf
    from pyscf.pbc.scf import khf
    mf_class = mf.__class__
    is_uhf = isinstance(mf, uhf.UHF)
    is_ghf = isinstance(mf, ghf.GHF)
    is_rhf = (not is_uhf) and (not is_ghf)
    is_khf = isinstance(mf, khf.KSCF)

    def fermi_smearing_occ(m, mo_energy_kpts, sigma):
        occ = numpy.zeros_like(mo_energy_kpts)
        de = (mo_energy_kpts - m) / sigma
        occ[de<40] = 1./(numpy.exp(de[de<40])+1.)
        return occ
    def gaussian_smearing_occ(m, mo_energy_kpts, sigma):
        return 0.5 * scipy.special.erfc((mo_energy_kpts - m) / sigma)

    def partition_occ(mo_occ, mo_energy_kpts):
        mo_occ_kpts = []
        p1 = 0
        for e in mo_energy_kpts:
            p0, p1 = p1, p1 + e.size
            occ = mo_occ[p0:p1]
            mo_occ_kpts.append(occ)
        return mo_occ_kpts

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mo_occ_kpts

        if is_khf:
            nkpts = len(mf.kpts)
        else:
            nkpts = 1
        nelectron = mf.cell.tot_electrons(nkpts)
        if is_uhf:
            nocc = nelectron
            mo_es = numpy.append(numpy.hstack(mo_energy_kpts[0]),
                                 numpy.hstack(mo_energy_kpts[1]))
        elif is_ghf:
            nocc = nelectron
            mo_es = numpy.hstack(mo_energy_kpts)
        else:
            nocc = nelectron // 2
            mo_es = numpy.hstack(mo_energy_kpts)

        if mf.smearing_method.lower() == 'fermi':  # Fermi-Dirac smearing
            f_occ = fermi_smearing_occ
        else:  # Gaussian smearing
            f_occ = gaussian_smearing_occ

        mo_energy = numpy.sort(mo_es.ravel())

        # If mu0 is given, fix mu instead of electron number. XXX -Chong Sun
        sigma = mf.sigma
        fermi = mo_energy[nocc-1]
        if mu0 is None:
            def nelec_cost_fn(m):
                mo_occ_kpts = f_occ(m, mo_es, sigma)
                if is_rhf:
                    mo_occ_kpts *= 2
                return (mo_occ_kpts.sum() - nelectron)**2
            res = scipy.optimize.minimize(nelec_cost_fn, fermi, method='Powell')
            mu = res.x
            mo_occs = f = f_occ(mu, mo_es, sigma)
        else:
            mu = mu0
            mo_occs = f = f_occ(mu, mo_es, sigma)
            

        # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
        if mf.smearing_method.lower() == 'fermi':
            f = f[(f>0) & (f<1)]
            mf.entropy = -(f*numpy.log(f) + (1-f)*numpy.log(1-f)).sum() / nkpts
        else:
            mf.entropy = (numpy.exp(-((mo_es-mu)/mf.sigma)**2).sum()
                          / (2*numpy.sqrt(numpy.pi)) / nkpts)
        if is_rhf:
            mo_occs *= 2
            mf.entropy *= 2

        # DO NOT use numpy.array for mo_occ_kpts and mo_energy_kpts, they may
        # have different dimensions for different k-points
        if is_uhf:
            if is_khf:
                nao_tot = mo_occs.size//2
                mo_occ_kpts =(partition_occ(mo_occs[:nao_tot], mo_energy_kpts[0]),
                              partition_occ(mo_occs[nao_tot:], mo_energy_kpts[1]))
            else:
                mo_occ_kpts = partition_occ(mo_occs, mo_energy_kpts)
        else: # rhf and ghf
            if is_khf:
                mo_occ_kpts = partition_occ(mo_occs, mo_energy_kpts)
            else:
                mo_occ_kpts = mo_occs

        logger.debug(mf, '    Fermi level %g  Sum mo_occ_kpts = %s  should equal nelec = %s',
                     fermi, mo_occs.sum(), nelectron)
        logger.info(mf, '    sigma = %g  Optimized mu = %.12g  entropy = %.12g',
                    mf.sigma, mu, mf.entropy)

        return mo_occ_kpts

    def get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
        if is_khf:
            grad_kpts = []
            for k, mo in enumerate(mo_coeff_kpts):
                f_mo = reduce(numpy.dot, (mo.T.conj(), fock[k], mo))
                nmo = f_mo.shape[0]
                grad_kpts.append(f_mo[numpy.tril_indices(nmo, -1)])
            return numpy.hstack(grad_kpts)
        else:
            f_mo = reduce(numpy.dot, (mo_coeff_kpts.T.conj(), fock, mo_coeff_kpts))
            nmo = f_mo.shape[0]
            return f_mo[numpy.tril_indices(nmo, -1)]

    def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mf_class.get_grad(mf, mo_coeff_kpts, mo_occ_kpts, fock)
        if fock is None:
            dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = mf.get_hcore() + mf.get_veff(mf.cell, dm1)
        if is_uhf:
            ga = get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return numpy.hstack((ga,gb))
        else: # rhf and ghf
            return get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

    def energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        e_tot = mf.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)[0] + mf.energy_nuc()
        if (mf.sigma and mf.smearing_method and
            mf.entropy is not None):
            mf.e_free = e_tot - mf.sigma * mf.entropy
            mf.e_zero = e_tot - mf.sigma * mf.entropy * .5
            logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                        e_tot, mf.e_free, mf.e_zero)
        return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf.e_free = None
    mf.e_zero = None
    mf._keys = mf._keys.union(['sigma', 'smearing_method',
                               'entropy', 'e_free', 'e_zero'])

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf


def canonical_occ_(mf, nelec=None):
    '''Label the occupancies for each orbital for sampled k-points.
    This is for KUHF objects.
    Each k-point has a fixed number of up and down electrons in this,
    which results in a finite size error for metallic systems
    but can accelerate convergence.
    '''
    from pyscf.pbc.scf import kuhf
    assert(isinstance(mf, kuhf.KUHF))

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
            return copy.copy(mf)
        else:
            unknown_cls = [scf.kghf.KGHF]
            for i, cls in enumerate(mf.__class__.__mro__):
                if cls in unknown_cls:
                    raise NotImplementedError(
                        "No conversion from %s to uhf object" % cls)

            known_cls = {dft.krks.KRKS  : dft.kuks.KUKS,
                         dft.kroks.KROKS: dft.kuks.KUKS,
                         scf.khf.KRHF   : scf.kuhf.KUHF,
                         scf.krohf.KROHF: scf.kuhf.KUHF,
                         dft.rks.RKS    : dft.uks.UKS  ,
                         dft.roks.ROKS  : dft.uks.UKS  ,
                         scf.hf.RHF     : scf.uhf.UHF  ,
                         scf.rohf.ROHF  : scf.uhf.UHF  ,}
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, remove_df=False)
    else:
        assert(isinstance(out, (scf.uhf.UHF, scf.kuhf.KUHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert(isinstance(out, scf.khf.KSCF))
        else:
            assert(not isinstance(out, scf.khf.KSCF))

    out = mol_addons.convert_to_uhf(mf, out, False)
    # Manually update .with_df because this attribute may not be passed to the
    # output object correctly in molecular convert function
    out.with_df = mf.with_df
    return out

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
        assert(isinstance(out, (scf.hf.RHF, scf.khf.KRHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert(isinstance(out, scf.khf.KSCF))
        else:
            assert(not isinstance(out, scf.khf.KSCF))

    elif nelec[0] != nelec[1] and isinstance(mf, scf.rohf.ROHF):
        if getattr(mf, '_scf', None):
            return mol_addons._update_mf_without_soscf(mf, copy.copy(mf._scf), False)
        else:
            return copy.copy(mf)

    else:
        if isinstance(mf, (scf.hf.RHF, scf.khf.KRHF)):
            return copy.copy(mf)
        else:
            unknown_cls = [scf.kghf.KGHF]
            for i, cls in enumerate(mf.__class__.__mro__):
                if cls in unknown_cls:
                    raise NotImplementedError(
                        "No conversion from %s to rhf object" % cls)

            if nelec[0] == nelec[1]:
                known_cls = {dft.kuks.KUKS : dft.krks.KRKS,
                             scf.kuhf.KUHF : scf.khf.KRHF ,
                             dft.uks.UKS   : dft.rks.RKS  ,
                             scf.uhf.UHF   : scf.hf.RHF   ,
                             dft.kroks.KROKS : dft.krks.KRKS,
                             scf.krohf.KROHF : scf.khf.KRHF ,
                             dft.roks.ROKS   : dft.rks.RKS  ,
                             scf.rohf.ROHF   : scf.hf.RHF   }
            else:
                known_cls = {dft.kuks.KUKS : dft.krks.KROKS,
                             scf.kuhf.KUHF : scf.khf.KROHF ,
                             dft.uks.UKS   : dft.rks.ROKS  ,
                             scf.uhf.UHF   : scf.hf.ROHF   }
            # .with_df should never be removed or changed during the conversion.
            # It is needed to compute JK matrix in all pbc SCF objects
            out = mol_addons._object_without_soscf(mf, known_cls, remove_df=False)

    out = mol_addons.convert_to_rhf(mf, out, False)
    # Manually update .with_df because this attribute may not be passed to the
    # output object correctly in molecular convert function
    out.with_df = mf.with_df
    return out

def convert_to_ghf(mf, out=None):
    '''Convert the given mean-field object to the generalized HF/KS object

    Args:
        mf : SCF object

    Returns:
        An generalized SCF object
    '''
    from pyscf.pbc import scf

    if out is not None:
        assert(isinstance(out, (scf.ghf.GHF, scf.kghf.KGHF)))
        if isinstance(mf, scf.khf.KSCF):
            assert(isinstance(out, scf.khf.KSCF))
        else:
            assert(not isinstance(out, scf.khf.KSCF))

    if isinstance(mf, scf.ghf.GHF):
        if out is None:
            return copy.copy(mf)
        else:
            out.__dict__.update(mf.__dict__)
            return out

    elif isinstance(mf, scf.khf.KSCF):

        def update_mo_(mf, mf1):
            _keys = mf._keys.union(mf1._keys)
            mf1.__dict__.update(mf.__dict__)
            mf1._keys = _keys
            if mf.mo_energy is not None:
                mf1.mo_energy = []
                mf1.mo_occ = []
                mf1.mo_coeff = []
                nkpts = len(mf.kpts)
                is_rhf = isinstance(mf, scf.hf.RHF)
                for k in range(nkpts):
                    if is_rhf:
                        mo_a = mo_b = mf.mo_coeff[k]
                        ea = eb = mf.mo_energy[k]
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

def convert_to_khf(mf, out=None):
    '''Convert gamma point SCF object to k-point SCF object
    '''
    from pyscf.pbc import scf, dft
    if not isinstance(mf, scf.khf.KSCF):
        known_cls = {dft.uks.UKS   : dft.kuks.KUKS,
                     scf.uhf.UHF   : scf.kuhf.KUHF,
                     dft.rks.RKS   : dft.krks.KRKS,
                     scf.hf.RHF    : scf.khf.KRHF,
                     dft.roks.ROKS : dft.kroks.KROKS,
                     scf.rohf.ROHF : scf.krohf.KROHF,
                     #TODO: dft.gks.GKS   : dft.kgks.KGKS,
                     scf.ghf.GHF   : scf.kghf.KGHF}
        # Keep the attribute with_df
        with_df = mf.with_df
        mf = mol_addons._object_without_soscf(mf, known_cls, remove_df=False)
        mf.with_df = with_df

    if out is None:
        return mf
    else:
        out.__dict__.update(mf.__dict__)
        return out

del(SMEARING_METHOD)


if __name__ == '__main__':
    import pyscf.pbc.scf as pscf
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = 'ccpvdz'
    cell.a = numpy.eye(3) * 4
    cell.mesh = [17] * 3
    cell.verbose = 4
    cell.build()
    nks = [2,1,1]
    mf = pscf.KUHF(cell, cell.make_kpts(nks))
    #mf = smearing_(mf, .1) # -5.86052594663696
    mf = smearing_(mf, .1, mu0=0.280911009667) # -5.86052594663696
    #mf = smearing_(mf, .1, method='gauss')
    mf.kernel()
