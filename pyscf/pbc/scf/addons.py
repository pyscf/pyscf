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

import numpy
import scipy.linalg
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.lib import logger
from pyscf.scf import addons as mol_addons
from pyscf.pbc.tools import k2gamma
from pyscf import __config__
# The smearing utilities were moved to a separate module. They were implemented
# in the addons module. Import them into this namespace for backward compatibility
from pyscf.pbc.scf.smearing import (  # noqa
    SMEARING_METHOD,
    smearing,
    smearing_,
    _get_grad_tril,
    _partition_occ,
    _SmearingKSCF
)

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
            if isinstance(mf, scf.kuhf.KUHF):
                mf.mo_occ = mf.mo_occ[:, numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[:, numpy.newaxis]
                mf.mo_energy = mf.mo_energy[:, numpy.newaxis]
            else:
                mf.mo_occ = mf.mo_occ[numpy.newaxis]
                mf.mo_coeff = mf.mo_coeff[numpy.newaxis]
                mf.mo_energy = mf.mo_energy[numpy.newaxis]

        if hasattr(mf, '_numint'):
            kpts = getattr(mf.kpts, 'kpts', mf.kpts)
            mf._numint = dft.numint.KNumInt(kpts)

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
        return lib.einsum('pi,pq,qi->i', C.conj(), fao, C).real
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
