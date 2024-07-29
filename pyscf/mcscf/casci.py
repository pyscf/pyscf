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

import sys

from functools import reduce
import warnings
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.mcscf import addons
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'mcscf_analyze_with_meta_lowdin', True)
LARGE_CI_TOL = getattr(__config__, 'mcscf_analyze_large_ci_tol', 0.1)
PENALTY = getattr(__config__, 'mcscf_casci_CASCI_fix_spin_shift', 0.2)
FRAC_OCC_THRESHOLD = 1e-6

if sys.version_info < (3,):
    RANGE_TYPE = list
else:
    RANGE_TYPE = range


def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''CAS sapce one-electron hamiltonian

    Args:
        casci : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    hcore = casci.get_hcore()
    energy_core = casci.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
        corevhf = casci.get_veff(casci.mol, core_dm)
        energy_core += numpy.einsum('ij,ji', core_dm, hcore).real
        energy_core += numpy.einsum('ij,ji', core_dm, corevhf).real * .5
    h1eff = reduce(numpy.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    return h1eff, energy_core

def analyze(casscf, mo_coeff=None, ci=None, verbose=None,
            large_ci_tol=LARGE_CI_TOL, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    from pyscf.mcscf import addons
    log = logger.new_logger(casscf, verbose)

    if mo_coeff is None: mo_coeff = casscf.mo_coeff
    if ci is None: ci = casscf.ci
    nelecas = casscf.nelecas
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = ncore + ncas
    mocore = mo_coeff[:,:ncore]
    mocas = mo_coeff[:,ncore:nocc]

    label = casscf.mol.ao_labels()
    if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
        not isinstance(casscf.fcisolver, addons.StateAverageFCISolver)):
        log.warn('Mulitple states found in CASCI/CASSCF solver. Density '
                 'matrix of the first state is generated in .analyze() function.')
        civec = ci[0]
    else:
        civec = ci
    if getattr(casscf.fcisolver, 'make_rdm1s', None):
        casdm1a, casdm1b = casscf.fcisolver.make_rdm1s(civec, ncas, nelecas)
        casdm1 = casdm1a + casdm1b
        dm1b = numpy.dot(mocore, mocore.conj().T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.conj().T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.conj().T))
        dm1 = dm1a + dm1b
        spin_dm1 = dm1a - dm1b
        if log.verbose >= logger.DEBUG2:
            log.info('alpha density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1a, label, **kwargs)
            log.info('beta density matrix (on AO)')
            dump_mat.dump_tri(log.stdout, dm1b, label, **kwargs)
    else:
        casdm1 = casscf.fcisolver.make_rdm1(civec, ncas, nelecas)
        dm1a = (numpy.dot(mocore, mocore.conj().T) * 2 +
                reduce(numpy.dot, (mocas, casdm1, mocas.conj().T)))
        dm1b = None
        dm1 = dm1a
        spin_dm1 = None

    if log.verbose >= logger.INFO:
        ovlp_ao = casscf._scf.get_ovlp()
        # note the last two args of ._eig for mc1step_symm
        occ, ucas = casscf._eig(-casdm1, ncore, nocc)
        log.info('Natural occ %s', str(-occ))
        mocas = numpy.dot(mocas, ucas)
        if with_meta_lowdin:
            log.info('Natural orbital (expansion on meta-Lowdin AOs) in CAS space')
            orth_coeff = orth.orth_ao(casscf.mol, 'meta_lowdin', s=ovlp_ao)
            mocas = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mocas))
        else:
            log.info('Natural orbital (expansion on AOs) in CAS space')
        dump_mat.dump_rec(log.stdout, mocas, label, start=1, **kwargs)
        if log.verbose >= logger.DEBUG2:
            if not casscf.natorb:
                log.debug2('NOTE: mc.mo_coeff in active space is different to '
                           'the natural orbital coefficients printed in above.')
            if with_meta_lowdin:
                c = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff))
                log.debug2('MCSCF orbital (expansion on meta-Lowdin AOs)')
            else:
                c = mo_coeff
                log.debug2('MCSCF orbital (expansion on AOs)')
            dump_mat.dump_rec(log.stdout, c, label, start=1, **kwargs)

        if casscf._scf.mo_coeff is not None:
            addons.map2hf(casscf, casscf._scf.mo_coeff)

        if (ci is not None and
            (getattr(casscf.fcisolver, 'large_ci', None) or
             getattr(casscf.fcisolver, 'states_large_ci', None))):
            log.info('** Largest CI components **')
            if isinstance(ci, (list, tuple, RANGE_TYPE)):
                if hasattr(casscf.fcisolver, 'states_large_ci'):
                    # defined in state_average_mix_ mcscf object
                    res = casscf.fcisolver.states_large_ci(ci, casscf.ncas, casscf.nelecas,
                                                           large_ci_tol, return_strs=False)
                else:
                    res = [casscf.fcisolver.large_ci(civec, casscf.ncas, casscf.nelecas,
                                                     large_ci_tol, return_strs=False)
                           for civec in ci]
                for i, civec in enumerate(ci):
                    log.info('  [alpha occ-orbitals] [beta occ-orbitals]  state %-3d CI coefficient', i)
                    for c,ia,ib in res[i]:
                        log.info('  %-20s %-30s % .12f', ia, ib, c)
            else:
                log.info('  [alpha occ-orbitals] [beta occ-orbitals]            CI coefficient')
                res = casscf.fcisolver.large_ci(ci, casscf.ncas, casscf.nelecas,
                                                large_ci_tol, return_strs=False)
                for c,ia,ib in res:
                    log.info('  %-20s %-30s % .12f', ia, ib, c)

        if with_meta_lowdin:
            casscf._scf.mulliken_meta(casscf.mol, dm1, s=ovlp_ao, verbose=log)
        else:
            casscf._scf.mulliken_pop(casscf.mol, dm1, s=ovlp_ao, verbose=log)
        if spin_dm1 is not None:
            if with_meta_lowdin:
                log.info('Mulliken spin population analysis on meta-Lowdin AOs:')
                spin_pop, spin_chg = casscf._scf.mulliken_meta(casscf.mol, spin_dm1, s=ovlp_ao, verbose=log)
            else:
                log.info('Mulliken spin population analysis on AOs:')
                spin_pop, spin_chg = casscf._scf.mulliken_pop(casscf.mol, spin_dm1, s=ovlp_ao, verbose=log)
            for i, s in enumerate(label):
                log.info('spop of %-12s %10.5f', s, spin_pop[i])
            spin_chg = casscf.mol.atom_charges() - spin_chg
            log.note('Mulliken atomic spins:')
            for ia in range(casscf.mol.natm):
                symb = casscf.mol.atom_symbol(ia)
                log.note('spin of  %d%s =   %10.5f', ia, symb, spin_chg[ia])
    return dm1a, dm1b

def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    r'''
    Effective one-electron Fock matrix in AO representation
    f = \sum_{pq} E_{pq} F_{pq}
    F_{pq} = h_{pq} + \sum_{rs} [(pq|rs)-(ps|rq)] DM_{sr}

    Ref.
    Theor. Chim. Acta., 91, 31
    Chem. Phys. 48, 157

    For state-average CASCI/CASSCF object, the effective fock matrix is based
    on the state-average density matrix.  To obtain Fock matrix of a specific
    state in the state-average calculations, you can pass "casdm1" of the
    specific state to this function.

    Args:
        mc: a CASSCF/CASCI object or RHF object

    Kwargs:
        mo_coeff (ndarray): orbitals that span the core, active and external
            space.
        ci (ndarray): CI coefficients (or objects to represent the CI
            wavefunctions in DMRG/QMC-MCSCF calculations).
        eris: Integrals for the MCSCF object. Input this object to reduce the
            overhead of computing integrals. It can be generated by
            :func:`mc.ao2mo` method.
        casdm1 (ndarray): 1-particle density matrix in active space. Without
            input casdm1, the density matrix is computed with the input ci
            coefficients/object. If neither ci nor casdm1 were given, density
            matrix is computed by :func:`mc.fcisolver.make_rdm1` method. For
            state-average CASCI/CASCF calculation, this results in the
            effective Fock matrix based on the state-average density matrix.
            To obtain the effective Fock matrix for one particular state, you
            can assign the density matrix of that state to the kwarg casdm1.

    Returns:
        Fock matrix
    '''

    if ci is None: ci = mc.ci
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    if getattr(eris, 'ppaa', None) is not None:
        vj = numpy.empty((nmo,nmo))
        vk = numpy.empty((nmo,nmo))
        for i in range(nmo):
            vj[i] = numpy.einsum('ij,qij->q', casdm1, eris.ppaa[i])
            vk[i] = numpy.einsum('ij,iqj->q', casdm1, eris.papa[i])
        mo_inv = numpy.dot(mo_coeff.conj().T, mc._scf.get_ovlp())
        fock = (mc.get_hcore() +
                reduce(numpy.dot, (mo_inv.conj().T, eris.vhf_c+vj-vk*.5, mo_inv)))
    else:
        dm_core = numpy.dot(mo_coeff[:,:ncore]*2, mo_coeff[:,:ncore].conj().T)
        mocas = mo_coeff[:,ncore:nocc]
        dm = dm_core + reduce(numpy.dot, (mocas, casdm1, mocas.conj().T))
        vj, vk = mc._scf.get_jk(mc.mol, dm)
        fock = mc.get_hcore() + vj-vk*.5
    return fock

def cas_natorb(mc, mo_coeff=None, ci=None, eris=None, sort=False,
               casdm1=None, verbose=None, with_meta_lowdin=WITH_META_LOWDIN):
    '''Transform active orbitals to natural orbitals, and update the CI wfn
    accordingly

    Args:
        mc : a CASSCF/CASCI object or RHF object

    Kwargs:
        sort : bool
            Sort natural orbitals wrt the occupancy.

    Returns:
        A tuple, the first item is natural orbitals, the second is updated CI
        coefficients, the third is the natural occupancy associated to the
        natural orbitals.
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    from pyscf.tools.mo_mapping import mo_1to1map
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    log = logger.new_logger(mc, verbose)
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nmo = mo_coeff.shape[1]
    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    if getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = numpy.copy(mo_coeff.orbsym)
    else:
        orbsym = numpy.zeros(mo_coeff.shape[1], dtype=int)
    if getattr(mc, 'extrasym', None) is not None:
        orbsym_extra = numpy.asarray([str(i1) + str(i2)
                                      for i1, i2 in zip(orbsym, mc.extrasym)])
    else:
        orbsym_extra = orbsym
    # orbital symmetry is reserved in this _eig call
    cas_occ, ucas = mc._eig(-casdm1, ncore, nocc, orbsym_extra[ncore:nocc])
    if sort:
        casorb_idx = numpy.argsort(cas_occ.round(9), kind='mergesort')
        cas_occ = cas_occ[casorb_idx]
        ucas = ucas[:,casorb_idx]

    cas_occ = -cas_occ
    mo_occ = numpy.zeros(mo_coeff.shape[1])
    mo_occ[:ncore] = 2
    mo_occ[ncore:nocc] = cas_occ

    mo_coeff1 = mo_coeff.copy()
    mo_coeff1[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
    if getattr(mo_coeff, 'orbsym', None) is not None:
        if sort:
            orbsym[ncore:nocc] = orbsym[ncore:nocc][casorb_idx]
        mo_coeff1 = lib.tag_array(mo_coeff1, orbsym=orbsym)
    else:
        orbsym = numpy.zeros(nmo, dtype=int)

    # When occupancies of active orbitals equal to 2 or 0, these orbitals
    # need to be canonicalized along with inactive(core or virtual) orbitals
    # using general Fock matrix. Because they are strongly coupled with
    # inactive orbitals, the 0th order Hamiltonian of MRPT methods can be
    # strongly affected. Numerical uncertainty may be found in the perturbed
    # correlation energy.
    # See issue https://github.com/pyscf/pyscf/issues/1041
    occ2_idx = numpy.where(2 - cas_occ < FRAC_OCC_THRESHOLD)[0]
    occ0_idx = numpy.where(cas_occ < FRAC_OCC_THRESHOLD)[0]
    if occ2_idx.size > 0 or occ0_idx.size > 0:
        fock_ao = mc.get_fock(mo_coeff, ci, eris, casdm1, verbose)

        def _diag_subfock_(idx):
            c = mo_coeff1[:,idx]
            fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
            w, c = mc._eig(fock, None, None, orbsym[idx])
            mo_coeff1[:,idx] = mo_coeff1[:,idx].dot(c)

        if occ2_idx.size > 0:
            log.warn('Active orbitals %s (occs = %s) are canonicalized with core orbitals',
                     occ2_idx, cas_occ[occ2_idx])
            full_occ2_idx = numpy.append(numpy.arange(ncore), ncore + occ2_idx)
            _diag_subfock_(full_occ2_idx)
        if occ0_idx.size > 0:
            log.warn('Active orbitals %s (occs = %s) are canonicalized with external orbitals',
                     occ0_idx, cas_occ[occ0_idx])
            full_occ0_idx = numpy.append(ncore + occ0_idx, numpy.arange(nocc, nmo))
            _diag_subfock_(full_occ0_idx)

    # Rotate CI according to the unitary coefficients ucas if applicable
    fcivec = None
    if getattr(mc.fcisolver, 'transform_ci_for_orbital_rotation', None):
        if isinstance(ci, (fci.FCIvector, fci.SCIvector, numpy.ndarray)):
            fcivec = mc.fcisolver.transform_ci_for_orbital_rotation(ci, ncas, nelecas, ucas)
        elif (isinstance(ci, (list, tuple)) and
              all(isinstance(x[0], (fci.FCIvector, fci.SCIvector, numpy.ndarray)) for x in ci)):
            fcivec = [mc.fcisolver.transform_ci_for_orbital_rotation(x, ncas, nelecas, ucas)
                      for x in ci]
    elif getattr(mc.fcisolver, 'states_transform_ci_for_orbital_rotation', None):
        fcivec = mc.fcisolver.states_transform_ci_for_orbital_rotation(ci, ncas, nelecas, ucas)

    # Rerun fcisolver to get wavefunction if it cannot be transformed from
    # existed one.
    if fcivec is None:
        log.info('FCI vector not available, call CASCI to update wavefunction')
        mocas = mo_coeff1[:,ncore:nocc]
        hcore = mc.get_hcore()
        dm_core = numpy.dot(mo_coeff1[:,:ncore]*2, mo_coeff1[:,:ncore].conj().T)
        ecore = mc.energy_nuc()
        ecore+= numpy.einsum('ij,ji', hcore, dm_core)
        h1eff = reduce(numpy.dot, (mocas.conj().T, hcore, mocas))
        if getattr(eris, 'ppaa', None) is not None:
            ecore += eris.vhf_c[:ncore,:ncore].trace()
            h1eff += reduce(numpy.dot, (ucas.conj().T, eris.vhf_c[ncore:nocc,ncore:nocc], ucas))
            aaaa = ao2mo.restore(4, eris.ppaa[ncore:nocc,ncore:nocc,:,:], ncas)
            aaaa = ao2mo.incore.full(aaaa, ucas)
        else:
            if getattr(mc, 'with_df', None):
                aaaa = mc.with_df.ao2mo(mocas)
            else:
                aaaa = ao2mo.kernel(mc.mol, mocas)
            corevhf = mc.get_veff(mc.mol, dm_core)
            ecore += numpy.einsum('ij,ji', dm_core, corevhf) * .5
            h1eff += reduce(numpy.dot, (mocas.conj().T, corevhf, mocas))


        # See label_symmetry_ function in casci_symm.py which initialize the
        # orbital symmetry information in fcisolver.  This orbital symmetry
        # labels should be reordered to match the sorted active space orbitals.
        if sort and getattr(mo_coeff1, 'orbsym', None) is not None:
            mc.fcisolver.orbsym = mo_coeff1.orbsym[ncore:nocc]

        max_memory = max(400, mc.max_memory-lib.current_memory()[0])
        e, fcivec = mc.fcisolver.kernel(h1eff, aaaa, ncas, nelecas, ecore=ecore,
                                        max_memory=max_memory, verbose=log)
        log.debug('In Natural orbital, CASCI energy = %s', e)

    if log.verbose >= logger.INFO:
        ovlp_ao = mc._scf.get_ovlp()
        # where_natorb gives the new locations of the natural orbitals
        where_natorb = mo_1to1map(ucas)
        log.debug('where_natorb %s', str(where_natorb))
        log.info('Natural occ %s', str(cas_occ))
        if with_meta_lowdin:
            log.info('Natural orbital (expansion on meta-Lowdin AOs) in CAS space')
            label = mc.mol.ao_labels()
            orth_coeff = orth.orth_ao(mc.mol, 'meta_lowdin', s=ovlp_ao)
            mo_cas = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff1[:,ncore:nocc]))
        else:
            log.info('Natural orbital (expansion on AOs) in CAS space')
            label = mc.mol.ao_labels()
            mo_cas = mo_coeff1[:,ncore:nocc]
        dump_mat.dump_rec(log.stdout, mo_cas, label, start=1)

        if mc._scf.mo_coeff is not None:
            s = reduce(numpy.dot, (mo_coeff1[:,ncore:nocc].conj().T,
                                   mc._scf.get_ovlp(), mc._scf.mo_coeff))
            idx = numpy.argwhere(abs(s)>.4)
            for i,j in idx:
                log.info('<CAS-nat-orb|mo-hf>  %-5d  %-5d  % 12.8f',
                         ncore+i+1, j+1, s[i,j])
    return mo_coeff1, fcivec, mo_occ

def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE,
                 with_meta_lowdin=WITH_META_LOWDIN, stav_dm1=False):
    '''Canonicalized CASCI/CASSCF orbitals of effecitive Fock matrix and
    update CI coefficients accordingly.

    Effective Fock matrix is built with one-particle density matrix (see
    also :func:`mcscf.casci.get_fock`). For state-average CASCI/CASSCF object,
    the canonicalized orbitals are based on the state-average density matrix.
    To obtain canonicalized orbitals for an individual state, you need to pass
    "casdm1" of the specific state to this function.

    Args:
        mc: a CASSCF/CASCI object or RHF object

    Kwargs:
        mo_coeff (ndarray): orbitals that span the core, active and external
            space.
        ci (ndarray): CI coefficients (or objects to represent the CI
            wavefunctions in DMRG/QMC-MCSCF calculations).
        eris: Integrals for the MCSCF object. Input this object to reduce the
            overhead of computing integrals. It can be generated by
            :func:`mc.ao2mo` method.
        sort (bool): Whether the canonicalized orbitals are sorted based on
            the orbital energy (diagonal part of the effective Fock matrix)
            within each subspace (core, active, external). If point group
            symmetry is not available in the system, orbitals are always
            sorted. When point group symmetry is available, sort=False will
            preserve the symmetry label of input orbitals and only sort the
            orbitals in each symmetry sector. sort=True will reorder all
            orbitals over all symmetry sectors in each subspace and the
            symmetry labels may be changed.
        cas_natorb (bool): Whether to transform active orbitals to natual
            orbitals. If enabled, the output orbitals in active space are
            transformed to natural orbitals and CI coefficients are updated
            accordingly.
        casdm1 (ndarray): 1-particle density matrix in active space. This
            density matrix is used to build effective fock matrix. Without
            input casdm1, the density matrix is computed with the input ci
            coefficients/object. If neither ci nor casdm1 were given, density
            matrix is computed by :func:`mc.fcisolver.make_rdm1` method. For
            state-average CASCI/CASCF calculation, this results in a set of
            canonicalized orbitals of state-average effective Fock matrix.
            To canonicalize the orbitals for one particular state, you can
            assign the density matrix of that state to the kwarg casdm1.
        stav_dm1 (bool): Use state-average 1-particle density matrix for
            computing Fock matrices and natural orbitals

    Returns:
        A tuple, (natural orbitals, CI coefficients, orbital energies)
        The orbital energies are the diagonal terms of effective Fock matrix.
    '''
    from pyscf.mcscf import addons
    log = logger.new_logger(mc, verbose)

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if casdm1 is None:
        if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
                not isinstance(mc.fcisolver, addons.StateAverageFCISolver)):
            if stav_dm1:
                log.warn('Mulitple states found in CASCI solver. '
                         'Use state-average 1RDM  to compute the Fock matrix'
                         ' and natural orbitals in the active space.')
                casdm1 = mc.fcisolver.make_rdm1(ci[0], mc.ncas, mc.nelecas)
                for root in range(1, len(ci)):
                    casdm1 += mc.fcisolver.make_rdm1(ci[root], mc.ncas,
                                                     mc.nelecas)
                casdm1 /= len(ci)
            else:
                log.warn('Mulitple states found in CASCI solver. '
                         'First state is used to compute the Fock matrix'
                         ' and natural orbitals in active space.')
                casdm1 = mc.fcisolver.make_rdm1(ci[0], mc.ncas, mc.nelecas)
        else:
            casdm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)

    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_coeff.shape[1]
    fock_ao = mc.get_fock(mo_coeff, ci, eris, casdm1, verbose)

    if cas_natorb:
        mo_coeff1, ci, mc.mo_occ = mc.cas_natorb(mo_coeff, ci, eris, sort, casdm1,
                                                 verbose, with_meta_lowdin)
    else:
        # Keep the active space unchanged by default.  The rotation in active space
        # may cause problem for external CI solver eg DMRG.
        mo_coeff1 = mo_coeff.copy()
        log.info('Density matrix diagonal elements %s', casdm1.diagonal())

    mo_energy = numpy.einsum('pi,pi->i', mo_coeff1.conj(), fock_ao.dot(mo_coeff1))

    if getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = mo_coeff.orbsym
    else:
        orbsym = numpy.zeros(nmo, dtype=int)
    extrasym = getattr(mc, 'extrasym', None)
    if extrasym is not None:
        orbsym_extra = numpy.asarray([str(i1) + str(i2)
                                      for i1, i2 in zip(orbsym, extrasym)])
    else:
        orbsym_extra = orbsym

    def _diag_subfock_(idx):
        if idx.size > 1:
            c = mo_coeff1[:,idx]
            fock = reduce(numpy.dot, (c.conj().T, fock_ao, c))
            # note the last argument orbysm is needed by mc1step_symm._eig
            w, c = mc._eig(fock, None, None, orbsym_extra[idx])

            if sort:
                sub_order = numpy.argsort(w.round(9), kind='mergesort')
                w = w[sub_order]
                c = c[:,sub_order]
                orbsym[idx] = orbsym[idx][sub_order]

            mo_coeff1[:,idx] = mo_coeff1[:,idx].dot(c)
            mo_energy[idx] = w

    mask = numpy.ones(nmo, dtype=bool)
    frozen = getattr(mc, 'frozen', None)
    if frozen is not None:
        if isinstance(frozen, (int, numpy.integer)):
            mask[:frozen] = False
        else:
            mask[frozen] = False
    core_idx = numpy.where(mask[:ncore])[0]
    vir_idx = numpy.where(mask[nocc:])[0] + nocc
    _diag_subfock_(core_idx)
    _diag_subfock_(vir_idx)

    # orbsym is required only for symmetry-adapted methods. Here to use
    # mo_coeff.orbsym to test if a symmetry-adapted calculation.
    if getattr(mo_coeff, 'orbsym', None) is not None:
        mo_coeff1 = lib.tag_array(mo_coeff1, orbsym=orbsym)

    if log.verbose >= logger.DEBUG:
        for i in range(nmo):
            log.debug('i = %d  <i|F|i> = %12.8f', i+1, mo_energy[i])
# still return ci coefficients, in case the canonicalization funciton changed
# cas orbitals, the ci coefficients should also be updated.
    return mo_coeff1, ci, mo_energy


def kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''CASCI solver

    Args:
        casci: CASCI or CASSCF object

        mo_coeff : ndarray
            orbitals to construct active space Hamiltonian
        ci0 : ndarray or custom types
            FCI sovler initial guess. For external FCI-like solvers, it can be
            overloaded different data type. For example, in the state-average
            FCI solver, ci0 is a list of ndarray. In other solvers such as
            DMRGCI solver, SHCI solver, ci0 are custom types.

    kwargs:
        envs: dict
            The variable envs is created (for PR 807) to passes MCSCF runtime
            environment variables to SHCI solver. For solvers which do not
            need this parameter, a kwargs should be created in kernel method
            and "envs" pop in kernel function
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    log = logger.new_logger(casci, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas

    # 2e
    eri_cas = casci.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = casci.get_h1eff(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t1)

    if h1eff.shape[0] != ncas:
        raise RuntimeError('Active space size error. nmo=%d ncore=%d ncas=%d' %
                           (mo_coeff.shape[1], casci.ncore, ncas))

    # FCI
    max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec


def as_scanner(mc):
    '''Generating a scanner for CASCI PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CASCI energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of MCSCF object
    are automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mf = scf.RHF(gto.Mole().set(verbose=0))
    >>> mc_scanner = mcscf.CASCI(mf, 4, 4).as_scanner()
    >>> mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    if isinstance(mc, lib.SinglePointScanner):
        return mc

    logger.info(mc, 'Create scanner for %s', mc.__class__)
    name = mc.__class__.__name__ + CASCI_Scanner.__name_mixin__
    return lib.set_class(CASCI_Scanner(mc), (CASCI_Scanner, mc.__class__), name)

class CASCI_Scanner(lib.SinglePointScanner):
    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        self._scf = mc._scf.as_scanner()

    def __call__(self, mol_or_geom, mo_coeff=None, ci0=None):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset (mol)

        if mo_coeff is None:
            mf_scanner = self._scf
            mf_scanner(mol)
            mo_coeff = mf_scanner.mo_coeff
        if ci0 is None:
            ci0 = self.ci
        self.mol = mol
        e_tot = self.kernel(mo_coeff, ci0)[0]
        return e_tot


class CASBase(lib.StreamObject):
    '''CASCI/CASSCF

    Args:
        mf_or_mol : SCF object or Mole object
            SCF or Mole to define the problem size.
        ncas : int
            Number of active orbitals.
        nelecas : int or a pair of int
            Number of electrons in active space.

    Kwargs:
        ncore : int
            Number of doubly occupied core orbitals. If not presented, this
            parameter can be automatically determined.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        ncas : int
            Active space size.
        nelecas : tuple of int
            Active (nelec_alpha, nelec_beta)
        ncore : int or tuple of int
            Core electron number.  In UHF-CASSCF, it's a tuple to indicate the different core eletron numbers.
        natorb : bool
            Whether to transform natural orbitals in active space.
            Note: when CASCI/CASSCF are combined with DMRG solver or selected
            CI solver, enabling this parameter may slightly change the total energy.
            False by default.
        canonicalization : bool
            Whether to canonicalize orbitals in core and external space
            against the general Fock matrix.
            The orbitals in active space are NOT transformed by default. To
            get the natural orbitals in active space, the attribute .natorb
            needs to be enabled.
            True by default.
        sorting_mo_energy : bool
            Whether to sort the orbitals based on the diagonal elements of the
            general Fock matrix.  Default is False.
        fcisolver : an instance of :class:`FCISolver`
            The pyscf.fci module provides several FCISolver for different scenario.  Generally,
            fci.direct_spin1.FCISolver can be used for all RHF-CASSCF.  However, a proper FCISolver
            can provide better performance and better numerical stability.  One can either use
            :func:`fci.solver` function to pick the FCISolver by the program or manually assigen
            the FCISolver to this attribute, e.g.

            >>> from pyscf import fci
            >>> mc = mcscf.CASSCF(mf, 4, 4)
            >>> mc.fcisolver = fci.solver(mol, singlet=True)
            >>> mc.fcisolver = fci.direct_spin1.FCISolver(mol)

            You can control FCISolver by setting e.g.::

                >>> mc.fcisolver.max_cycle = 30
                >>> mc.fcisolver.conv_tol = 1e-7

            For more details of the parameter for FCISolver, See :mod:`fci`.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray
            When canonicalization is specified, the orbitals are canonical
            orbitals which make the general Fock matrix (Fock operator on top
            of MCSCF 1-particle density matrix) diagonalized within each
            subspace (core, active, external).  If natorb (natural orbitals in
            active space) is specified, the active segment of the mo_coeff is
            natural orbitls.
        mo_energy : ndarray
            Diagonal elements of general Fock matrix (in mo_coeff
            representation).
        mo_occ : ndarray
            Occupation numbers of natural orbitals if natorb is specified.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASCI(mf, 6, 6)
    >>> mc.kernel()[0]
    -108.980200816243354
    '''

    natorb = getattr(__config__, 'mcscf_casci_CASCI_natorb', False)
    canonicalization = getattr(__config__, 'mcscf_casci_CASCI_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'mcscf_casci_CASCI_sorting_mo_energy', False)

    _keys = set((
        'natorb', 'canonicalization', 'sorting_mo_energy', 'mol', 'max_memory',
        'ncas', 'nelecas', 'ncore', 'fcisolver', 'frozen', 'extrasym',
        'e_tot', 'e_cas', 'ci', 'mo_coeff', 'mo_energy', 'mo_occ', 'converged',
    ))

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        if isinstance(mf_or_mol, gto.Mole):
            mf = scf.RHF(mf_or_mol)
        else:
            mf = mf_or_mol

        mol = mf.mol
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, numpy.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1])
        self.ncore = ncore
        singlet = (getattr(__config__, 'mcscf_casci_CASCI_fcisolver_direct_spin0', False)
                   and self.nelecas[0] == self.nelecas[1])  # leads to direct_spin1
        self.fcisolver = fci.solver(mol, singlet, symm=False)
# CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = getattr(__config__,
                                        'mcscf_casci_CASCI_fcisolver_lindep', 1e-12)
        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_casci_CASCI_fcisolver_max_cycle', 200)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_casci_CASCI_fcisolver_conv_tol', 1e-8)
        self.frozen = None
        self.extrasym = None

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.mo_occ = None
        self.converged = False

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert ncorelec % 2 == 0
            assert ncorelec >= 0
            return ncorelec // 2
        else:
            return self._ncore
    @ncore.setter
    def ncore(self, x):
        assert x is None or isinstance(x, (int, numpy.integer))
        assert x is None or x >= 0
        self._ncore = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        if self.extrasym is not None:
            log.info('extra symmetry labels:\n%s', str(self.extrasym))
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('max_memory %d (MB)', self.max_memory)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(log.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found at SCF level but not applied to the CASCI object.
The SCF solvent model will not be applied to the current CASCI calculation.
To enable the solvent model for CASCI, the following code needs to be called
        from pyscf import solvent
        mc = mcscf.CASCI(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self

    def check_sanity(self):
        super().check_sanity()
        assert self.ncas > 0
        ncore = self.ncore
        nvir = self.mo_coeff.shape[1] - ncore - self.ncas
        assert ncore >= 0
        assert nvir >= 0
        assert ncore * 2 + sum(self.nelecas) == self.mol.nelectron
        assert 0 <= self.nelecas[0] <= self.ncas
        assert 0 <= self.nelecas[1] <= self.ncas
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
            self.fcisolver.mol = mol
        self._scf.reset(mol)
        return self

    def energy_nuc(self):
        return self._scf.energy_nuc()

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    @lib.with_doc(scf.hf.get_jk.__doc__)
    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self._scf.get_jk(mol, dm, hermi,
                                with_j=with_j, with_k=with_k, omega=omega)

    @lib.with_doc(scf.hf.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = numpy.dot(mocore, mocore.conj().T) * 2
# don't call self._scf.get_veff because _scf might be DFT object
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj - vk * .5

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_h2cas(self, mo_coeff=None):
        '''An alias of get_h2eff method'''
        return self.get_h2eff(mo_coeff)

    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        raise NotImplementedError

    def ao2mo(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        raise NotImplementedError

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''An alias of get_h1eff method'''
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = h1e_for_cas

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        raise NotImplementedError

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        raise NotImplementedError

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            if isinstance(self.e_cas, (float, numpy.number)):
                try:
                    ss = self.fcisolver.spin_square(self.ci, self.ncas, self.nelecas)
                    log.note('CASCI E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                             self.e_tot, self.e_cas, ss[0])
                except NotImplementedError:
                    log.note('CASCI E = %#.15g  E(CI) = %#.15g',
                             self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    try:
                        ss = self.fcisolver.spin_square(self.ci[i], self.ncas, self.nelecas)
                        log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                                 i, self.e_tot[i], e, ss[0])
                    except NotImplementedError:
                        log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g',
                                 i, self.e_tot[i], e)

        else:
            if isinstance(self.e_cas, (float, numpy.number)):
                log.note('CASCI E = %#.15g  E(CI) = %#.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI state %3d  E = %#.15g  E(CI) = %#.15g',
                             i, self.e_tot[i], e)
        return self

    @lib.with_doc(cas_natorb.__doc__)
    def cas_natorb(self, mo_coeff=None, ci=None, eris=None, sort=False,
                   casdm1=None, verbose=None, with_meta_lowdin=WITH_META_LOWDIN):
        return cas_natorb(self, mo_coeff, ci, eris, sort, casdm1, verbose,
                          with_meta_lowdin)
    @lib.with_doc(cas_natorb.__doc__)
    def cas_natorb_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                    casdm1=None, verbose=None, with_meta_lowdin=WITH_META_LOWDIN):
        self.mo_coeff, self.ci, self.mo_occ = cas_natorb(self, mo_coeff, ci, eris,
                                                         sort, casdm1, verbose)
        return self.mo_coeff, self.ci, self.mo_occ

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None,
                 verbose=None):
        return get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    canonicalize = canonicalize

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None,
                      with_meta_lowdin=WITH_META_LOWDIN):
        self.mo_coeff, ci, self.mo_energy = \
                canonicalize(self, mo_coeff, ci, eris,
                             sort, cas_natorb, casdm1, verbose, with_meta_lowdin)
        if cas_natorb:  # When active space is changed, the ci solution needs to be updated
            self.ci = ci
        return self.mo_coeff, ci, self.mo_energy

    analyze = analyze

    @lib.with_doc(addons.sort_mo.__doc__)
    def sort_mo(self, caslst, mo_coeff=None, base=1):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return addons.sort_mo(self, mo_coeff, caslst, base)

    @lib.with_doc(addons.state_average.__doc__)
    def state_average_(self, weights=(0.5,0.5), wfnsym=None):
        addons.state_average_(self, weights, wfnsym)
        return self
    @lib.with_doc(addons.state_average.__doc__)
    def state_average(self, weights=(0.5,0.5), wfnsym=None):
        return addons.state_average(self, weights, wfnsym)

    @lib.with_doc(addons.state_specific_.__doc__)
    def state_specific_(self, state=1):
        addons.state_specific(self, state)
        return self

    def make_rdm1s(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                   ncore=None, **kwargs):
        '''One-particle density matrices for alpha and beta spin on AO basis
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1a, casdm1b = self.fcisolver.make_rdm1s(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1b = numpy.dot(mocore, mocore.conj().T)
        dm1a = dm1b + reduce(numpy.dot, (mocas, casdm1a, mocas.conj().T))
        dm1b += reduce(numpy.dot, (mocas, casdm1b, mocas.conj().T))
        return dm1a, dm1b

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        '''One-particle density matrix in AO representation
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1 = self.fcisolver.make_rdm1(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1 = numpy.dot(mocore, mocore.conj().T) * 2
        dm1 = dm1 + reduce(numpy.dot, (mocas, casdm1, mocas.conj().T))
        return dm1

    def fix_spin_(self, shift=PENALTY, ss=None):
        r'''Use level shift to control FCI solver spin.

        .. math::

            (H + shift*S^2) |\Psi\rangle = E |\Psi\rangle

        Kwargs:
            shift : float
                Energy penalty for states which have wrong spin
            ss : number
                S^2 expection value == s*(s+1)
        '''
        fci.addons.fix_spin_(self.fcisolver, shift, ss)
        return self
    fix_spin = fix_spin_

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.density_fit(self, auxbasis, with_df)

    def sfx2c1e(self):
        from pyscf.x2c import sfx2c1e
        self._scf = sfx2c1e.sfx2c1e(self._scf).run()
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        return self
    x2c = x2c1e = sfx2c1e

    def nuc_grad_method(self):
        raise NotImplementedError

class CASCI(CASBase):
    def get_h2eff(self, mo_coeff=None):
        '''Compute the active space two-particle Hamiltonian.
        '''
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            ncore = self.ncore
            mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        if hasattr(self._scf, '_eri') and self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        return self.kernel(mo_coeff, ci0, verbose)
    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        self.check_sanity()
        self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=log)

        if self.canonicalization:
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)
        elif self.natorb:
            # FIXME (pyscf-2.0): Whether to transform natural orbitals in
            # active space when this flag is enabled?
            log.warn('The attribute .natorb of mcscf object affects only the '
                     'orbital canonicalization.\n'
                     'If you would like to get natural orbitals in active space '
                     'without touching core and external orbitals, an explicit '
                     'call to mc.cas_natorb_() is required')

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('CASCI converged')
            else:
                log.info('CASCI not converged')
        else:
            self.converged = True
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    as_scanner = as_scanner

    def nuc_grad_method(self):
        from pyscf.grad import casci
        return casci.Gradients(self)

scf.hf.RHF.CASCI = scf.rohf.ROHF.CASCI = lib.class_as_method(CASCI)
scf.uhf.UHF.CASCI = None

del (WITH_META_LOWDIN, LARGE_CI_TOL, PENALTY)
