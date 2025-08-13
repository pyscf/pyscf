#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

import numpy as np
from pyscf.lib import logger, tag_array, current_memory
from pyscf.dft.gen_grid import BLKSIZE
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf.mcpdft.pdft_eff import _ERIS
from pyscf.mcpdft.tfnal_derivs import contract_fot
import gc


def kernel(ot, dm1s, cascm2, c_dm1s, c_cascm2, mo_coeff, ncore, ncas, max_memory=2000, hermi=1, paaa_only=False,
           aaaa_only=False, jk_pc=False, delta=False):
    r'''Get the 1- and 2-body effective gradient responses from MC-PDFT. The
    $\rho \cdot \mathbf{F}$ terms, or Hessian vector products.

    Args:
        ot : an instance of otfnal class
        dm1s : ndarray of shape (2, nao, nao)
            Contains the spin-separated one-body density matrices to evaluate
            the kernel at
        cascm2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed two-body cumulant density matrix in the active space to
            evaluate the kernel at
        c_dm1s : ndarray of shape (2, nao, nao)
            Contains the spin-separated one-body density matrices to contract
            the kernel with.
        c_cascm2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed two-body cumulant density matrix in the active space to
            contract the kernel with.
        mo_coeff : ndarray of shape (nao, nmo)
            containing molecular orbital coefficients
        ncore : integer
            number of inactive orbitals
        ncas : integer
            number of active orbitals

    Kwargs:
        max_memory : int or float
            maximum cache size in MB
            default is 2000
        hermi : int
            1 if 1rdms are assumed hermitian, 0 otherwise
        paaa_only : logical
            If true, only compute the paaa range of papa and ppaa (all other
            elements set to zero)
        aaaa_only : logical
            If true, only compute the aaaa range of papa and ppaa (all other
            elements set to zero; overrides paaa_only)
        jk_pc : logical
            If true, compute the ppii=pipi elements of veff2 (otherwise, these
            are set to zero)
        delta : logical
            If true, then contract with the delta density. The delta density is the c_dm1s-dm1s and similarly for the
            2rdm element (though care is taken since 2rdm elements are expressed in cumulant form).

    Returns:
        feff1 : ndarray of shape (nao, nao)
            1-body effective gradient response
        feff2 : object of class pdft_eff._ERIS
            2-body effective gradient response
    '''
    nocc = ncore + ncas
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    nao = mo_coeff.shape[0]
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]
    shls_slice = (0, ot.mol.nbas)
    ao_loc = ot.mol.ao_loc_nr()

    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc)
    hyb_x, hyb_c = hyb
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")

    if abs(hyb_x) > 1e-11 or abs(hyb_c) > 1e-11:
        raise NotImplementedError("effective potential for hybrid functionals")

    feff1 = np.zeros((nao, nao), dtype=dm1s.dtype)
    feff2 = _ERIS(ot.mol, mo_coeff, ncore, ncas, paaa_only=paaa_only,
                  aaaa_only=aaaa_only, jk_pc=jk_pc, verbose=ot.verbose,
                  stdout=ot.stdout)

    t0 = (logger.process_clock(), logger.perf_counter())

    # Density matrices
    dm_core = mo_core @ mo_core.T
    dm_cas = dm1s - dm_core[None, :, :]
    dm_core *= 2

    # Propagate speedup tags
    if hasattr(dm1s, 'mo_coeff') and hasattr(dm1s, 'mo_occ'):
        dm_core = tag_array(dm_core, mo_coeff=dm1s.mo_coeff[0, :, :ncore],
                            mo_occ=dm1s.mo_occ[:, :ncore].sum(0))
        dm_cas = tag_array(dm_cas, mo_coeff=dm1s.mo_coeff[:, :, ncore:nocc],
                           mo_occ=dm1s.mo_occ[:, ncore:nocc])

    # rho generators
    make_rho_c = ni._gen_rho_evaluator(ot.mol, dm_core, hermi)[0]
    make_rho_a = ni._gen_rho_evaluator(ot.mol, dm_cas, hermi)[0]
    make_rho = ni._gen_rho_evaluator(ot.mol, dm1s, hermi)[0]
    make_crho = ni._gen_rho_evaluator(ot.mol, c_dm1s, hermi)[0]

    # memory block size
    gc.collect()
    remaining_floats = (max_memory - current_memory()[0]) * 1e6 / 8
    nderiv_rho = (1, 4, 10)[dens_deriv]  # ?? for meta-GGA
    nderiv_Pi = (1, 4)[ot.Pi_deriv]
    ncols = 4 + nderiv_rho * nao  # ao, weight, coords
    ncols += nderiv_rho * 4 + nderiv_Pi  # rho, rho_a, rho_c, Pi
    ncols += 1 + nderiv_rho + nderiv_Pi  # eot, vot
    ncols += (nderiv_rho*nderiv_Pi*(nderiv_rho*nderiv_Pi + 1))/2 # fot

    # Asynchronous part
    nfeff1 = nderiv_rho * (nao + 1)  # footprint of get_eff_1body
    nfeff2 = feff2._accumulate_ftpt() * nderiv_Pi
    ncols += np.amax([nfeff1, nfeff2])  # asynchronous fns
    pdft_blksize = int(remaining_floats / (ncols * BLKSIZE)) * BLKSIZE
    if ot.grids.coords is None:
        ot.grids.build(with_non0tab=True)
    ngrids = ot.grids.coords.shape[0]
    ngrids_blk = int(ngrids / BLKSIZE) * BLKSIZE
    pdft_blksize = max(BLKSIZE, min(pdft_blksize, ngrids_blk, BLKSIZE * 1200))
    logger.debug(ot, ('{} MB used of {} available; block size of {} chosen'
        'for grid with {} points').format(current_memory()[0], max_memory,
                                          pdft_blksize, ngrids))

    for ao, mask, weight, coords in ni.block_loop(ot.mol, ot.grids, nao,
                                                  dens_deriv, max_memory,
                                                  blksize=pdft_blksize):
        rho = np.asarray([make_rho(i, ao, mask, xctype) for i in range(2)])
        crho = np.asarray([make_crho(i, ao, mask, xctype) for i in range(2)])
        rho_a = sum([make_rho_a(i, ao, mask, xctype) for i in range(2)])
        rho_c = make_rho_c(0, ao, mask, xctype)
        t0 = logger.timer(ot, 'untransformed densities (core and total)', *t0)

        Pi = get_ontop_pair_density(ot, rho, ao, cascm2, mo_cas,
                                    dens_deriv, mask)
        cPi = get_ontop_pair_density(ot, crho, ao, c_cascm2, mo_cas,
                                     dens_deriv, mask)
        t0 = logger.timer(ot, 'on-top pair density calculation', *t0)

        if delta:
            crho -= rho
            cPi -= Pi

        vot, fot = ot.eval_ot(rho, Pi, weights=weight, dderiv=2,
                              _unpack_vot=False)[1:]
        frho, fPi = contract_fot(ot, fot, rho, Pi, crho, cPi, unpack=True,
                                 vot_packed=vot)

        t0 = logger.timer(ot, 'effective gradient response kernel calculation',
                          *t0)

        if ao.ndim == 2:
            ao = ao[None, :, :]

        feff1 += ot.get_eff_1body(ao, weight, frho, non0tab=mask,
                                  shls_slice=shls_slice, ao_loc=ao_loc,
                                  hermi=1)
        t0 = logger.timer(ot, '1-body effective gradient response calculation',
                          *t0)

        feff2._accumulate(ot, ao, weight, rho_c, rho_a, fPi, mask, shls_slice,
                          ao_loc)
        t0 = logger.timer(ot, '2-body effective gradient response calculation',
                          *t0)

    feff2._finalize()
    t0 = logger.timer(ot, 'Finalizing 2-body gradient response calculation',
                      *t0)

    return feff1, feff2


def lazy_kernel(ot, dm1s, cascm2, c_dm1s, c_cascm2, mo_cas, hermi=1, max_memory=2000, delta=False):
    '''1- and 2-body gradient response (hessian-vector products) from MC-PDFT.
    This is the lazy way and doesn't care about memory.'''
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    nao = mo_cas.shape[0]

    feff1 = np.zeros_like(dm1s[0])
    feff2 = np.zeros((nao, nao, nao, nao), dtype=feff1.dtype)

    t0 = (logger.process_clock(), logger.perf_counter())

    make_rho = tuple(ni._gen_rho_evaluator(ot.mol, dm1s[i, :, :], hermi) for i
        in range(2))
    make_crho = tuple(ni._gen_rho_evaluator(ot.mol, c_dm1s[i, :, :], hermi) for
        i in range(2))

    for ao, mask, weight, coords in ni.block_loop(ot.mol, ot.grids, nao,
                                                  dens_deriv, max_memory):
        rho = np.asarray([m[0](0, ao, mask, xctype) for m in make_rho])
        crho = np.asarray([m[0](0, ao, mask, xctype) for m in make_crho])
        t0 = logger.timer(ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density(ot, rho, ao, cascm2, mo_cas, dens_deriv,
                                    mask)
        cPi = get_ontop_pair_density(ot, crho, ao, c_cascm2, mo_cas,
                                     dens_deriv, mask)
        t0 = logger.timer(ot, 'on-top pair density calculation', *t0)

        if delta:
            crho -= rho
            cPi -= Pi

        vot, fot = ot.eval_ot(rho, Pi, weights=weight, dderiv=2,
                              _unpack_vot=False)[1:]
        frho, fPi = contract_fot(ot, fot, rho, Pi, crho, cPi, unpack=True,
                                 vot_packed=vot)

        t0 = logger.timer(ot, 'effective gradient response kernel calculation',
                          *t0)
        if ao.ndim == 2:
            ao = ao[None, :, :]

        feff1 += ot.get_eff_1body(ao, weight, frho)
        t0 = logger.timer(ot, '1-body effective gradient response calculation',
                          *t0)

        feff2 += ot.get_eff_2body(ao, weight, fPi, aosym=1)
        t0 = logger.timer(ot, '2-body effective gradient response calculation',
                          *t0)

    return feff1, feff2


def get_feff_1body(otfnal, rho, Pi, crho, cPi, ao, weight, kern=None, non0tab=None,
                   shls_slice=None, ao_loc=None, hermi=0, **kwargs):
    r"""Get the terms [\Delta F]_{pq}

    Args:
        rho : ndarray of shape (2,*,ngrids)
            Spin-density [and derivatives]
        Pi : ndarray with shape (*,ngrids)
            On-top pair density [and derivatives]
        crho : ndarray of shape (2,*,ngrids)
            Spin-density [and derivatives] to contract the hessian with
        cPi : ndarray with shape (*,ngrids)
            On-top pair density [and derivatives] to contract Hessian with
        ao : ndarray or 2 ndarrays of shape (*,ngrids,nao)
            contains values and derivatives of nao. 2 different ndarrays can
            have different nao but not different ngrids
        weight : ndarray of shape (ngrids)
            containing numerical integration weights

    Kwargs:
        kern : ndarray of shape (*,ngrids)
            the hessian-vector product. If not provided, it is calculated.
        non0tab : ndarray of shape (nblk, nbas)
            Identifies blocks of grid points which are nonzero on each AO shell
            so as to exploit sparsity. If you want the "ao" array to be in the
            MO basis, just leave this as None. If hermi == 0, it only applies
            to the bra index ao array, even if the ket index ao array is the
            same (so probably always pass hermi = 1 in that case)
        shls_slice : sequence of integers of len 2
            Identifies starting and stopping indices of AO shells
        ao_loc : ndarray of length nbas
            Offset to first AO of each shell
        hermi : integer or logical
            Toggle whether feff is supposed to be a Hermitian matrix You can
            still pass two different ao arrays for the bra and the ket indices,
            for instance if one of them is supposed to be a higher derivative.
            They just have to have the same nao in that case.

    Returns : ndarray of shape (nao[0],nao[1])
        The 1-body effective gradient response corresponding to this on-top
        pair density exchange-correlation functional, in the atomic-orbital
        basis. In PDFT this functional is always spin-symmetric.
    """
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[0]
        rho = np.squeeze(rho)
        Pi = np.squeeze(Pi)
        crho = np.squeeze(crho)
        cPi = np.squeeze(cPi)

    return otfnal.get_eff_1body(ao, weight, kern=kern, non0tab=non0tab,
                                shls_slice=shls_slice, ao_loc=ao_loc,
                                hermi=hermi)


def get_feff_2body(otfnal, rho, Pi, crho, cPi, ao, weight, aosym='s4',
                   kern=None, fao=None, **kwargs):
    r'''Get the terms [\Delta F]_{pqrs}

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]
            crho : ndarray of shape (2,*,ngrids)
                Spin-density [and derivatives] to contract the hessian with
            cPi : ndarray with shape (*,ngrids)
                On-top pair density [and derivatives] to contract Hessian with
            ao : ndarray of shape (*,ngrids,nao)
                OR list of ndarrays of shape (*,ngrids,*) values and
                derivatives of atomic or molecular orbitals in which space to
                calculate the 2-body veff If a list of length 4, the
                corresponding set of eri-like elements are returned
            weight : ndarray of shape (ngrids)
                containing numerical integration weights

        Kwargs:
            aosym : int or str
                Index permutation symmetry of the desired integrals. Valid
                options are 1 (or '1' or 's1'), 4 (or '4' or 's4'), '2ij' (or
                's2ij'), and '2kl' (or 's2kl'). These have the same meaning as
                in PySCF's ao2mo module. Currently all symmetry exploitation is
                extremely slow and unparallelizable for some reason so trying
                to use this is not recommended until I come up with a C
                routine.
            kern : ndarray of shape (*,ngrids)
                the hessian-vector product. If not provided, it is calculated.
            fao : ndarray of shape (*,ngrids,nao,nao) or
                (*,ngrids,nao*(nao+1)//2). An intermediate in which the kernel
                and the k,l orbital indices have been contracted. Overrides
                kl_symm

        Returns : eri-like ndarray
            The two-body effective gradient response corresponding to this on-top
            pair density exchange-correlation functional or elements
            thereof, in the provided basis.
        '''
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[1]
        rho = np.squeeze(rho)
        Pi = np.squeeze(Pi)
        crho = np.squeeze(crho)
        cPi = np.squeeze(cPi)

    return otfnal.get_eff_2body(ao, weight, kern, aosym=aosym, eff_ao=fao)


def get_feff_2body_kl(otfnal, rho, Pi, crho, cPi, ao_k, ao_l, weight,
                      symm=False, kern=None, **kwargs):
    r''' get the two-index intermediate Mkl of [\Delta \cdot F]_{pqrs}

        Args:
            rho : ndarray of shape (2,*,ngrids)
                containing spin-density [and derivatives]
            Pi : ndarray with shape (*,ngrids)
                containing on-top pair density [and derivatives]
            crho : ndarray of shape (2,*,ngrids)
                Spin-density [and derivatives] to contract the hessian with
            cPi : ndarray with shape (*,ngrids)
                On-top pair density [and derivatives] to contract Hessian with
            ao_k : ndarray of shape (*,ngrids,nao)
                OR list of ndarrays of shape (*,ngrids,*) values and
                derivatives of atomic or molecular orbitals corresponding to
                index k
            ao_l : ndarray of shape (*,ngrids,nao)
                OR list of ndarrays of shape (*,ngrids,*) values and
                derivatives of atomic or molecular orbitals corresponding to
                index l
            weight : ndarray of shape (ngrids)
                containing numerical integration weights

        Kwargs:
            symm : logical
                Index permutation symmetry of the desired integral wrt k,l
            kern : ndarray of shape (*,ngrids)
                the hessian-vector product. If not provided, it is calculated.

        Returns : ndarray of shape (*,ngrids,nao,nao)
            or (*,ngrids,nao*(nao+1)//2). An intermediate for calculating the
            two-body effective gradient response corresponding to this on-top
            pair density exchange-correlation functional in the provided basis.
        '''
    if kern is None:
        if rho.ndim == 2:
            rho = np.expand_dims(rho, 1)
            Pi = np.expand_dims(Pi, 0)

        if crho.ndim == 2:
            crho = np.expand_dims(crho, 1)
            cPi = np.expand_dims(cPi, 0)

        fot = otfnal.eval_ot(rho, Pi, dderiv=2, **kwargs)[2]
        kern = contract_fot(otfnal, fot, rho, Pi, crho, cPi)[1]

    return otfnal.get_eff_2body_kl(ao_k, ao_l, weight, kern=kern, symm=symm)
