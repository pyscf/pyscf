#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
import time
import gc
from functools import reduce
from scipy import linalg
from pyscf import gto
from pyscf import mcscf, lib, ao2mo
from pyscf.grad import lagrange
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.grad import casscf as casscf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
from pyscf.ao2mo.outcore import balance_partition
from pyscf.mcscf import mc1step, mc1step_symm, newton_casscf
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.df.grad import casscf as dfcasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf.df.grad.rhf import _int3c_wrapper
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import spin_square0
from pyscf.fci import cistring
from pyscf.df.grad.casdm2_util import (solve_df_rdm2, solve_df_eri,
                                      grad_elec_dferi,
                                      grad_elec_auxresponse_dferi)

def Lorb_dot_dgorb_dx (Lorb, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, eris=None, verbose=None,
                       auxbasis_response=True):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the orbital
    Lagrange term nuclear gradient (sum_pq Lorb_pq d2_Ecas/d_lambda d_kpq)
    This involves removing nuclear-nuclear terms and making the substitution
    (D_[p]q + D_p[q]) -> D_pq
    (d_[p]qrs + d_pq[r]s + d_p[q]rs + d_pqr[s]) -> d_pqrs
    Where [] around an index implies contraction with Lorb from the left, so that the external index
    (regardless of whether the index on the rdm is bra or ket) is always the first index of Lorb. '''

    # dmo = smoT.dao.smo
    # dao = mo.dmo.moT
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = dfrhf_grad.Gradients (mc._scf)
    if mc.frozen is not None:
        raise NotImplementedError

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    # MRH: new 'effective' MO coefficients including contraction from the Lagrange multipliers
    moL_coeff = np.dot (mo_coeff, Lorb)
    s0_inv = np.dot (mo_coeff,  mo_coeff.T)
    moL_core = moL_coeff[:,:ncore]
    moL_cas = moL_coeff[:,ncore:nocc]

    # MRH: these SHOULD be state-averaged! Use the actual sacasscf object!
    casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

    # gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    # MRH: each index exactly once!
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    # MRH: new density matrix terms
    dmL_core = np.dot(moL_core, mo_core.T) * 2
    dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
    dmL_core += dmL_core.T
    dmL_cas += dmL_cas.T
    dm1 = dm_core + dm_cas
    dm1L = dmL_core + dmL_cas
    # MRH: end new density matrix terms
    # MRH: wrap the integral instead of the density matrix. I THINK the sign is the same!
    # mo sets 0 and 2 should be transposed, 1 and 3 should be not transposed; this will lead to correct sign
    # Except I can't do this for the external index, because the external index is contracted to ovlp matrix,
    # not the 2RDM
    aapa = np.zeros ((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    aapaL = np.zeros ((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    for i in range (nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        aapa[:,:,i,:] = jbuf[ncore:nocc,:,:].transpose (1,2,0)
        aapaL[:,:,i,:] += np.tensordot (jbuf, Lorb[:,ncore:nocc], axes=((0),(0)))
        kbuf = np.tensordot (kbuf, Lorb[:,ncore:nocc], axes=((1),(0))).transpose (1,2,0)
        aapaL[:,:,i,:] += kbuf + kbuf.transpose (1,0,2)
    # MRH: new vhf terms
    vj, vk   = mc._scf.get_jk(mol, (dm_core,  dm_cas))
    vjL, vkL = mc._scf.get_jk(mol, (dmL_core, dmL_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    vhfL_c = vjL[0] - vkL[0] * .5
    vhfL_a = vjL[1] - vkL[1] * .5
    # MRH: I rewrote this Feff calculation completely, double-check it
    gfock  = np.dot (h1, dm1L) # h1e
    gfock += np.dot ((vhf_c + vhf_a), dmL_core) # core-core and active-core, 2nd 1RDM linked
    gfock += np.dot ((vhfL_c + vhfL_a), dm_core) # core-core and active-core, 1st 1RDM linked
    gfock += np.dot (vhfL_c, dm_cas) # core-active, 1st 1RDM linked
    gfock += np.dot (vhf_c, dmL_cas) # core-active, 2nd 1RDM linked
    gfock  = np.dot (s0_inv, gfock) # Definition of quantity is in MO's; going (AO->MO->AO) incurs an inverse ovlp
    gfock += reduce (np.dot, (mo_coeff, np.einsum('uviw,uvtw->it', aapaL, casdm2), mo_cas.T)) # active-active
    # MRH: I have to contract this external 2RDM index explicitly on the 2RDM but fortunately I can do so here
    gfock += reduce (np.dot, (mo_coeff, np.einsum('uviw,vuwt->it', aapa, casdm2), moL_cas.T))
    # MRH: As of 04/18/2019, the two-body part of this is including aapaL is definitely, unambiguously correct
    dme0 = (gfock+gfock.T)/2 # This transpose is for the overlap matrix later on
    aapa = vj = vk = vhf_c = vhf_a = None

    if atmlst is None:
        atmlst = list (range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de_aux = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    #vhf1c, vhf1a, vhf1cL, vhf1aL = mf_grad.get_veff(mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    vj, vk = mf_grad.get_jk (mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    vhf1c, vhf1a, vhf1cL, vhf1aL = list (vj - vk * 0.5)
    if auxbasis_response:
        de_aux = vj.aux - 0.5 * vk.aux
        #              D.T     +    T.D
        de_aux = ((de_aux[0,2] + de_aux[2,0]) # core-core
                + (de_aux[0,3] + de_aux[2,1]) # core-active
                + (de_aux[1,2] + de_aux[3,0])) # active-core
    vj = vk = None
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)


    t0 = lib.logger.timer (mc, 'SA-CASSCF Lorb_dot_dgorb 1-electron part', *t0)

    # I am trying to contract the eris with a notional casdm2 which has four separate terms.
    casdm2 += casdm2.transpose (1,0,3,2) # Now I should only need 2 separate terms...
    # The bare 3-center eris and the auxbasis derivatives are always symmetric wrt AOs
    # grad_elec_dferi is explicitly symmetrized wrt AOs.
    # If this fails I can always debug it by kludging ncore, ncas -> 0, nmo
    dfcasdm2  = solve_df_rdm2 (mc, mo_cas=(mo_cas, moL_cas), casdm2=casdm2)
    de_eri += grad_elec_dferi (mc, mo_cas=mo_cas, dfcasdm2=dfcasdm2, atmlst=atmlst, max_memory=mc.max_memory)[0]
    if auxbasis_response:
        de_aux += grad_elec_auxresponse_dferi (mc, mo_cas=mo_cas, dfcasdm2=dfcasdm2, atmlst=atmlst,
                                               max_memory=mc.max_memory)[0]
    dfcasdm2  = solve_df_rdm2 (mc, mo_cas=mo_cas, casdm2=casdm2)
    de_eri += grad_elec_dferi (mc, mo_cas=(mo_cas, moL_cas), dfcasdm2=dfcasdm2, atmlst=atmlst,
                               max_memory=mc.max_memory)[0]
    if auxbasis_response:
        de_aux += grad_elec_auxresponse_dferi (mc, mo_cas=(mo_cas, moL_cas), dfcasdm2=dfcasdm2, atmlst=atmlst,
                                               max_memory=mc.max_memory)[0]
    dfcasdm2 = casdm2 = None

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: h1e and Feff terms
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1L)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        # MRH: core-core and core-active 2RDM terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm1L[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1cL[:,p0:p1], dm1[p0:p1]) * 2
        # MRH: active-core 2RDM terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dmL_core[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1aL[:,p0:p1], dm_core[p0:p1]) * 2

    # MRH: deleted the nuclear-nuclear part to avoid double-counting
    # lesson learned from debugging - mol.intor computes -1 * the derivative and only
    # for one index
    # on the other hand, mf_grad.hcore_generator computes the actual derivative of
    # h1 for both indices and with the correct sign

    lib.logger.debug (mc, f"Orb lagrange hcore component:\n{de_hcore}")
    lib.logger.debug (mc, f"Orb lagrange renorm component:\n{de_renorm}")
    lib.logger.debug (mc, f"Orb lagrange eri component:\n{de_eri}")
    lib.logger.debug (mc, f"Orb lagrange aux component:\n{de_aux}")
    de = de_hcore + de_renorm + de_eri + de_aux

    return de

def Lci_dot_dgci_dx (Lci, weights, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, eris=None, verbose=None,
                     auxbasis_response=True):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the CI
    Lagrange term nuclear gradient (sum_IJ Lci_IJ d2_Ecas/d_lambda d_PIJ)
    This involves removing all core-core and nuclear-nuclear terms and making the substitution
    sum_I w_I<L_I|p'q|I> + c.c. -> <0|p'q|0>
    sum_I w_I<L_I|p'r'sq|I> + c.c. -> <0|p'r'sq|0>
    The active-core terms (sum_I w_I<L_I|x'iyi|I>, sum_I w_I <L_I|x'iiy|I>, c.c.) must be retained.'''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = dfrhf_grad.Gradients (mc._scf)
    if mc.frozen is not None:
        raise NotImplementedError

    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    # MRH: TDMs + c.c. instead of RDMs; 06/30/2020: new interface in mcscf.addons makes this much more transparent
    casdm1, casdm2 = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
    casdm1 += casdm1.transpose (1,0)
    casdm2 += casdm2.transpose (1,0,3,2)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = np.zeros ((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    for i in range (nmo):
        aapa[:,:,i,:] = eris.ppaa[i][ncore:nocc,:,:].transpose (1,2,0)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    # MRH: delete h1 + vhf_c from the first line below (core and core-core stuff)
    # Also extend gfock to span the whole space
    gfock = np.zeros_like (dm_cas)
    gfock[:,:nocc]   = reduce(np.dot, (mo_coeff.T, vhf_a, mo_occ)) * 2
    gfock[:,ncore:nocc]  = reduce(np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1))
    gfock[:,ncore:nocc] += np.einsum('uvpw,vuwt->pt', aapa, casdm2)
    dme0 = reduce(np.dot, (mo_coeff, (gfock+gfock.T)*.5, mo_coeff.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de_aux = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    #vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))
    vj, vk = mf_grad.get_jk (mol, (dm_core, dm_cas))
    if auxbasis_response:
        de_aux = vj.aux - 0.5 * vk.aux
        de_aux = de_aux[0,1] + de_aux[1,0]
        # ^ de_aux[0,0] not included b/c this is CAS lagrange multipliers
    vhf1c, vhf1a = list (vj - vk * 0.5)
    vj = vk = None
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    dfcasdm2 = casdm2 = solve_df_rdm2 (mc, mo_cas=mo_cas, casdm2=casdm2)
    de_eri = grad_elec_dferi (mc, mo_cas=mo_cas, dfcasdm2=dfcasdm2, atmlst=atmlst,
        max_memory=mc.max_memory)[0]
    if auxbasis_response:
        de_aux += grad_elec_auxresponse_dferi (mc, mo_cas=mo_cas, dfcasdm2=dfcasdm2,
            atmlst=atmlst, max_memory=mc.max_memory)[0]
    dfcasdm2 = casdm2 = None

    t0 = lib.logger.timer (mc, 'SA-CASSCF Lci_dot_dgci 1-electron part', *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: dm1 -> dm_cas in the line below
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm_cas)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        # MRH: dm1 -> dm_cas in the line below. Also eliminate core-core terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm_cas[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    lib.logger.debug (mc, f"CI lagrange hcore component:\n{de_hcore}")
    lib.logger.debug (mc, f"CI lagrange renorm component:\n{de_renorm}")
    lib.logger.debug (mc, f"CI lagrange eri component:\n{de_eri}")
    lib.logger.debug (mc, f"CI lagrange aux component:\n{de_aux}")
    de = de_hcore + de_renorm + de_eri + de_aux
    return de


def _grad_elec_df_response_direct(mc, mf_grad, dms, pair_weights,
                                  mo_df_pairs, atmlst, max_memory,
                                  auxbasis_response=True):
    '''Directly contract all DF response terms into atomic gradients.

    This specializes ``pyscf.df.grad.rhf.get_jk`` by contracting only the
    selected density pairs directly into atomic gradients, while sharing the
    three-center integral passes with the active-space DF-RDM2 response.  The
    direct-contraction strategy was also used by
    ``gpu4pyscf/df/grad/jk.py:get_grad_vjk`` in GPU4PySCF commit
    ``69036d5181a16c534565092342b341e6a409fb33``.

    ``pair_weights[i,j]`` selects the one-particle density pairs required by
    the SA-CASSCF response, avoiding the dense nset-by-nset auxiliary tensor.
    The active-space DF-RDM2 terms share the same ip1 and ip2 integral loops.
    '''
    mol = mc.mol
    auxmol = mc.with_df.auxmol
    nao, nbas, naux = mol.nao, mol.nbas, auxmol.nao
    dms = np.asarray(dms).reshape(-1, nao, nao)
    nset = len(dms)
    pair_weights = np.asarray(pair_weights)
    if pair_weights.shape != (nset, nset):
        raise ValueError('pair_weights must have shape (nset,nset)')
    active_pairs = np.argwhere(abs(pair_weights) > 1e-14)
    exchange_pairs = {(int(i), int(j)) for i, j in active_pairs}
    exchange_pairs.update((j, i) for i, j in tuple(exchange_pairs))

    # The AO derivative matrices are never formed.  For each source density,
    # combine all right-hand densities before entering the integral loop.
    right_dms = np.einsum('ij,jpq->ipq', pair_weights, dms)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    dm_tril = dms + dms.transpose(0,2,1)
    dm_tril = lib.pack_tril(dm_tril)
    dm_tril[:,diag_idx] *= .5

    orbol, orbor = dfrhf_grad._decompose_rdm1_svd(mf_grad, mol, dms)
    rhoj, get_rhok = dfrhf_grad._cho_solve_rhojk(
        mf_grad, mol, auxmol, orbol, orbor)
    nocc = [orb.shape[-1] for orb in orbor]

    # Prepare the two active-space DF-RDM2 contractions.  Their zero-order
    # metric solves are still reusable targets for a subsequent optimization.
    prepared = []
    if auxbasis_response:
        for mo0, mo1, dfcasdm2 in mo_df_pairs:
            mosym, nmo_pair, mo_conc, mo_slice = _conc_mos(
                mo0, mo1, compact=True)
            dm2 = np.array(dfcasdm2, copy=True)
            if 's2' in mosym:
                nmo = mo0.shape[1]
                dm2 = dm2.reshape(naux, nmo, nmo)
                dm2 += dm2.transpose(0,2,1)
                idx = np.arange(nmo)
                idx = idx * (idx+1) // 2 + idx
                dm2 = lib.pack_tril(np.ascontiguousarray(dm2))
                dm2[:,idx] *= .5
            dm2 = dm2.reshape(naux, nmo_pair)
            prepared.append((mo0, mo1, mosym, nmo_pair, mo_conc,
                             mo_slice, dm2))

    de_aux_active = np.zeros((naux, 3))
    de_aux_j = np.zeros((naux, 3))
    de_aux_k = np.zeros((naux, 3))
    rhok_oo = {}
    if auxbasis_response:
        int2c_e1 = auxmol.intor('int2c2e_ip1')

        # Active-space two-center metric response.
        for mo0, mo1, mosym, nmo_pair, mo_conc, mo_slice, dm2 in prepared:
            dferi = solve_df_eri(mc, mo_cas=(mo0, mo1)).reshape(
                naux, nmo_pair)
            metric_response = np.dot(int2c_e1, dferi)
            de_aux_active += lib.einsum(
                'pi,xpi->px', dm2, metric_response)

        # Only form exchange intermediates for selected density pairs.  The
        # generic DF J/K driver constructs all nset**2 combinations.
        avail_memory = max_memory - lib.current_memory()[0]
        max_occ = max(nocc)
        blksize = int(min(max(avail_memory * .5e6 / 8 /
                              (naux * max_occ), 20), naux))
        for i, j in sorted(exchange_pairs):
            buf = np.empty((naux, nocc[i], nocc[j]))
            for p0, p1 in lib.prange(0, naux, blksize):
                rhok = get_rhok(i, p0, p1).reshape(
                    (p1-p0)*nocc[i], nao)
                buf[p0:p1] = lib.dot(rhok, orbol[j]).reshape(
                    p1-p0, nocc[i], nocc[j])
            rhok_oo[i,j] = buf

        # Two-center J/K metric response, accumulated directly rather than
        # stored as (nset,nset,3,naux).
        metric_j = {
            int(j): lib.einsum('xpq,q->px', int2c_e1, rhoj[int(j)])
            for j in np.unique(active_pairs[:,1])
        }
        for i_, j_ in active_pairs:
            i, j = int(i_), int(j_)
            weight = pair_weights[i,j]
            de_aux_j -= weight * rhoj[i,:,None] * metric_j[j]
            metric_k = lib.einsum(
                'pij,qji->pq', rhok_oo[i,j], rhok_oo[j,i])
            de_aux_k -= weight * lib.einsum(
                'xpq,pq->px', int2c_e1, metric_k)

    de_ao = np.zeros((nao, 3))
    get_int3c_ip1 = _int3c_wrapper(
        mol, auxmol, 'int3c2e_ip1', 's1')
    avail_memory = max_memory - lib.current_memory()[0]
    blksize = int(min(max(avail_memory * .5e6 / 8 /
                          (nao**2 * 3), 20), naux, 240))
    aux_loc = auxmol.ao_loc
    aux_ranges = balance_partition(aux_loc, blksize)

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s1
    null = lib.c_null_ptr()

    # One ip1 pass for both one-particle J/K and active-space DF-RDM2.
    for shl0, shl1, nL in aux_ranges:
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c_ao = get_int3c_ip1(
            (0, nbas, 0, nbas, shl0, shl1))

        # Active-space contribution.
        for mo0, mo1, dfcasdm2 in mo_df_pairs:
            intbuf = lib.einsum('xuvp,vj->xupj', int3c_ao, mo1)
            dm2buf = lib.einsum('ui,pij->upj', mo0,
                                dfcasdm2[p0:p1])
            de_ao -= np.einsum('upj,xupj->ux', dm2buf, intbuf)
            intbuf = lib.einsum('xuvp,vj->xupj', int3c_ao, mo0)
            dm2buf = lib.einsum('uj,pij->upi', mo1,
                                dfcasdm2[p0:p1])
            de_ao -= np.einsum('upj,xupj->ux', dm2buf, intbuf)

        # One-particle J/K contribution.  Contract each temporary derivative
        # matrix immediately with its pre-combined right-hand density.
        int3c = np.ascontiguousarray(int3c_ao.transpose(0,3,2,1))
        for i in range(nset):
            if not np.any(pair_weights[i]):
                continue
            vj = np.empty((3, nao, nao))
            for x in range(3):
                vj[x] = np.dot(
                    rhoj[i,p0:p1],
                    int3c[x].reshape(p1-p0, -1)).reshape(nao, nao).T

            tmp = np.empty((3, p1-p0, nocc[i], nao),
                           dtype=orbol[i].dtype)
            fdrv(ftrans, fmmm,
                 tmp.ctypes.data_as(ctypes.c_void_p),
                 int3c.ctypes.data_as(ctypes.c_void_p),
                 orbol[i].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(3*(p1-p0)), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nocc[i], 0, nao),
                 null, ctypes.c_int(0))
            rhok = get_rhok(i, p0, p1)
            vk = lib.einsum('xpoi,pok->xik', tmp, rhok)
            veff = -(vj - .5 * vk)
            de_ao += lib.einsum(
                'xuv,uv->ux', veff, right_dms[i]) * 2

    de_aux = np.zeros((naux, 3))
    if auxbasis_response:
        get_int3c_ip2 = _int3c_wrapper(
            mol, auxmol, 'int3c2e_ip2', 's2ij')
        npair = nao * (nao + 1) // 2
        max_pair = max(item[3] for item in prepared)
        avail_memory = max_memory - lib.current_memory()[0]
        blklen = 3 * (npair + max_pair)
        blksize = int(min(max(avail_memory * 1e6 / 8 / blklen, 20),
                          240))
        aux_ranges = balance_partition(aux_loc, blksize)
        fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2

        # Group selected pairs by their left density so the ip2 AO-to-MO
        # transformation is shared by all requested right densities.
        right_by_left = {}
        for i_, j_ in active_pairs:
            i, j = int(i_), int(j_)
            right_by_left.setdefault(i, []).append(j)

        for shl0, shl1, nL in aux_ranges:
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            int3c_ao = get_int3c_ip2(
                (0, nbas, 0, nbas, shl0, shl1))
            int3c = np.ascontiguousarray(
                int3c_ao.transpose(0,2,1).reshape(3*(p1-p0), npair))

            drhoj = lib.dot(int3c, dm_tril.T).reshape(
                3, p1-p0, nset)
            for i_, j_ in active_pairs:
                i, j = int(i_), int(j_)
                de_aux_j[p0:p1] += pair_weights[i,j] * (
                    drhoj[:,:,i] * rhoj[j,p0:p1][None,:]).T

            for i, js in right_by_left.items():
                buf = np.empty((3, p1-p0, nocc[i], nao),
                               dtype=orbol[i].dtype)
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     int3c.ctypes.data_as(ctypes.c_void_p),
                     orbol[i].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(3*(p1-p0)), ctypes.c_int(nao),
                     (ctypes.c_int*4)(0, nocc[i], 0, nao),
                     null, ctypes.c_int(0))
                for j in js:
                    int3c_ij = lib.dot(buf.reshape(-1, nao), orbor[j])
                    int3c_ij = int3c_ij.reshape(
                        3, p1-p0, nocc[i], nocc[j])
                    de_aux_k[p0:p1] += pair_weights[i,j] * lib.einsum(
                        'xpij,pij->px', int3c_ij,
                        rhok_oo[i,j][p0:p1])

            for mo0, mo1, mosym, nmo_pair, mo_conc, mo_slice, dm2 in prepared:
                intbuf = _ao2mo.nr_e2(
                    int3c, mo_conc, mo_slice, aosym='s2', mosym=mosym)
                intbuf = np.ascontiguousarray(
                    intbuf.reshape(3, p1-p0, nmo_pair))
                de_aux_active[p0:p1] -= lib.einsum(
                    'pi,xpi->px', dm2[p0:p1], intbuf)

        de_aux = de_aux_active - (de_aux_j - .5 * de_aux_k)

    aoslices = mol.aoslice_by_atom()
    de_ao = np.asarray([de_ao[p0:p1].sum(axis=0)
                        for p0, p1 in aoslices[:,2:]])
    auxslices = auxmol.aoslice_by_atom()
    de_aux = np.asarray([de_aux[p0:p1].sum(axis=0)
                         for p0, p1 in auxslices[:,2:]])
    atmlst = np.asarray(list(atmlst), dtype=int)
    return np.ascontiguousarray(de_ao[atmlst] + de_aux[atmlst])


def Lorb_Lci_dot_dgorb_dgci_dx(Lorb, Lci, weights, mc, mo_coeff=None,
                               ci=None, atmlst=None, mf_grad=None, eris=None,
                               verbose=None, fcasscf=None, ci_state=None,
                               auxbasis_response=True,
                               lagrange_intermediates=None):
    '''Combined DF Hamiltonian, orbital, and CI SA-CASSCF response.

    Selected J/K density pairs and the effective active-space DF densities
    are contracted directly in one ``int3c2e_ip1`` pass and, when requested,
    one ``int3c2e_ip2`` auxiliary-response pass.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = dfrhf_grad.Gradients(mc._scf)
    if mc.frozen is not None:
        raise NotImplementedError

    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    if lagrange_intermediates is None:
        lagrange_intermediates = sacasscf_grad.make_sa_lagrange_response_intermediates(
            Lorb, Lci, mc, mo_coeff=mo_coeff, ci=ci, eris=eris)
    common, orbital_response, ci_response = lagrange_intermediates
    mol = common['mol']
    if atmlst is None:
        atmlst = list(range(mol.natm))
    else:
        atmlst = list(atmlst)
    mo_coeff, ci, eris = common['mo_coeff'], common['ci'], common['eris']
    ncore, ncas, nocc = common['ncore'], common['ncas'], common['nocc']
    nmo = common['nmo']
    mo_core, mo_cas = common['mo_core'], common['mo_cas']
    dm_core, s0_inv, aapa = common['dm_core'], common['s0_inv'], common['aapa']
    moL_cas = orbital_response['moL_cas']
    casdm2 = orbital_response['casdm2']
    dm_cas = orbital_response['dm_cas']
    dmL_core = orbital_response['dmL_core']
    dmL_cas = orbital_response['dmL_cas']
    dm1L = orbital_response['dm1L']
    aapaL = orbital_response['aapaL']
    casdm1_ci, casdm2_ci = ci_response['casdm1'], ci_response['casdm2']
    dm_cas_ci = ci_response['dm_cas']

    with_ham_response = fcasscf is not None or ci_state is not None
    if with_ham_response:
        if fcasscf is None or ci_state is None:
            raise ValueError('fcasscf and ci_state must be supplied together')
        casdm1_ham, casdm2_ham = fcasscf.fcisolver.make_rdm12(
            ci_state, ncas, fcasscf.nelecas)
        dm_cas_ham = reduce(np.dot, (mo_cas, casdm1_ham, mo_cas.T))
        dm1_ham = dm_core + dm_cas_ham

    jk_dms = (dm_core, dm_cas, dmL_core, dmL_cas, dm_cas_ci)
    if with_ham_response:
        jk_dms += (dm_cas_ham,)

    # Note that this can be problematic if the mc and mf have different auxbasis.
    # vj, vk = mc._scf.get_jk(mol, jk_dms)
    # I have replaced with the mc.get_jk call.
    vj, vk = mc.get_jk(mol, jk_dms)
    vhf = vj - vk * .5
    vhf_c, vhf_a, vhfL_c, vhfL_a, vhf_a_ci = vhf[:5]
    h1 = mc.get_hcore()

    gfock = np.dot(h1, dm1L)
    gfock += np.dot(vhf_c + vhf_a, dmL_core)
    gfock += np.dot(vhfL_c + vhfL_a, dm_core)
    gfock += np.dot(vhfL_c, dm_cas)
    gfock += np.dot(vhf_c, dmL_cas)
    gfock = np.dot(s0_inv, gfock)
    gfock += reduce(np.dot, (mo_coeff,
                             np.einsum('uviw,uvtw->it', aapaL, casdm2),
                             mo_cas.T))
    gfock += reduce(np.dot, (mo_coeff,
                             np.einsum('uviw,vuwt->it', aapa, casdm2),
                             moL_cas.T))
    dme0 = (gfock + gfock.T) / 2

    gfock_ci = np.zeros((nmo,nmo), dtype=dm_cas_ci.dtype)
    gfock_ci[:,:nocc] = reduce(
        np.dot, (mo_coeff.T, vhf_a_ci, mo_coeff[:,:nocc])) * 2
    gfock_ci[:,ncore:nocc] = reduce(
        np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1_ci))
    gfock_ci[:,ncore:nocc] += np.einsum(
        'uvpw,vuwt->pt', aapa, casdm2_ci)
    dme0_ci = reduce(
        np.dot, (mo_coeff, (gfock_ci + gfock_ci.T) * .5, mo_coeff.T))

    if with_ham_response:
        vhf_a_ham = vhf[5]
        gfock_ham = np.zeros((nmo,nmo), dtype=dm_cas_ham.dtype)
        gfock_ham[:,:ncore] = reduce(
            np.dot, (mo_coeff.T, h1 + vhf_c + vhf_a_ham,
                     mo_core)) * 2
        gfock_ham[:,ncore:nocc] = reduce(
            np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1_ham))
        gfock_ham[:,ncore:nocc] += np.einsum(
            'uviw,vuwt->it', aapa, casdm2_ham)
        dme0_ham = reduce(
            np.dot, (mo_coeff, (gfock_ham + gfock_ham.T) * .5,
                     mo_coeff.T))
    aapa = aapaL = vj = vk = None

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm1_hcore = dm1L + dm_cas_ci
    dme0_total = dme0 + dme0_ci
    if with_ham_response:
        dm1_hcore += dm1_ham
        dme0_total += dme0_ham

    casdm2_orb = casdm2 + casdm2.transpose(1,0,3,2)
    regular_dm2 = [casdm2_ci, casdm2_orb]
    if with_ham_response:
        regular_dm2.append(casdm2_ham)
    df_regular = solve_df_rdm2(mc, mo_cas=mo_cas,
                               casdm2=regular_dm2)
    df_ci, df_orb = df_regular[:2]
    df_orb_internal_L = solve_df_rdm2(
        mc, mo_cas=(mo_cas, moL_cas), casdm2=casdm2_orb)[0]
    df_external_regular = df_ci + df_orb_internal_L
    if with_ham_response:
        df_external_regular += df_regular[2]
    mo_df_pairs = ((mo_cas, mo_cas, df_external_regular),
                   (mo_cas, moL_cas, df_orb))

    pair_weights = np.zeros((len(jk_dms), len(jk_dms)))
    for i, j in ((0,2), (2,0), (0,3), (2,1), (1,2), (3,0),
                 (0,4), (4,0)):
        pair_weights[i,j] += 1
    if with_ham_response:
        for i, j in ((0,0), (0,5), (5,0)):
            pair_weights[i,j] += 1
    de_df = _grad_elec_df_response_direct(
        mc, mf_grad, jk_dms, pair_weights, mo_df_pairs, atmlst,
        mc.max_memory, auxbasis_response=auxbasis_response)

    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1_hcore)
        de_renorm[k] -= np.einsum(
            'xij,ij->x', s1[:,p0:p1], dme0_total[p0:p1]) * 2

    lib.logger.debug(mc, f'Combined DF hcore component:\n{de_hcore}')
    lib.logger.debug(mc, f'Combined DF renorm component:\n{de_renorm}')
    lib.logger.debug(mc, f'Combined direct DF component:\n{de_df}')
    lib.logger.timer(mc, 'Combined DF SA-CASSCF response', *t0)
    return de_hcore + de_renorm + de_df

def as_scanner(mcscf_grad, state=None):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.1', verbose=0)
    >>> mc_grad_scanner = mcscf.CASSCF(scf.RHF(mol), 4, 4).nuc_grad_method().as_scanner()
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    if state is None and (not hasattr (mcscf_grad, 'state') or (mcscf_grad.state is None)):
        return dfcasscf_grad.as_scanner (mcscf_grad)

    lib.logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)
    name = mcscf_grad.__class__.__name__ + CASSCF_GradScanner.__name_mixin__
    return lib.set_class(CASSCF_GradScanner(mcscf_grad, state),
                         (CASSCF_GradScanner, mcscf_grad.__class__), name)

class CASSCF_GradScanner(lib.GradScanner):
    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        mc_scanner = self.base
        e_tot = mc_scanner(mol)
        if hasattr (mc_scanner, 'e_mcscf'): self.e_mcscf = mc_scanner.e_mcscf
        #if isinstance (e_tot, (list, tuple, np.ndarray)): e_tot = e_tot[self.state]
        if hasattr (mc_scanner, 'e_states'): e_tot = mc_scanner.e_states[self.state]
        self.mol = mol
        if not ('state' in kwargs):
            kwargs['state'] = self.state
        de = self.kernel(**kwargs)
        return e_tot, de


class Gradients (sacasscf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    def __init__(self, mc, state=None):
        self.auxbasis_response = True
        sacasscf_grad.Gradients.__init__(self, mc, state=state)

    def kernel (self, **kwargs):
        mf_grad = kwargs['mf_grad'] if 'mf_grad' in kwargs else None
        if mf_grad is None: kwargs['mf_grad'] = dfrhf_grad.Gradients (self.base._scf)
        # The below only works because dfcasscf_grad is NOT a child of casscf_grad
        # For instance, I can't monkeypatch rhf_grad this way b/c dfrhf_grad refers to rhf_grad
        # Maybe it should be, in which case I will have to change this
        # But on the other hand maybe it can be even simpler?
        with lib.temporary_env (casscf_grad, Gradients=dfcasscf_grad.Gradients):
            return sacasscf_grad.Gradients.kernel (self, **kwargs)

    def get_LdotJnuc (self, Lvec, **kwargs):
        with lib.temporary_env (sacasscf_grad, Lci_dot_dgci_dx=Lci_dot_dgci_dx, Lorb_dot_dgorb_dx=Lorb_dot_dgorb_dx):
            return sacasscf_grad.Gradients.get_LdotJnuc (self, Lvec, **kwargs)

    def get_nuc_response(self, Lvec, state=None, atmlst=None, verbose=None,
                         mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        '''Return the combined DF SA-CASSCF nuclear response.'''
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo(mo)
        elif eris is None:
            eris = self.eris
        if mf_grad is None:
            mf_grad = dfrhf_grad.Gradients(self.base._scf)

        Lorb, Lci = self.unpack_uniq_var(Lvec)
        fcasscf = self.make_fcasscf(state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        lagrange_intermediates = sacasscf_grad.make_sa_lagrange_response_intermediates(
            Lorb, Lci, self.base, mo_coeff=mo, ci=ci, eris=eris)

        de = Lorb_Lci_dot_dgorb_dgci_dx(
            Lorb, Lci, self.weights, self.base, mo_coeff=mo, ci=ci,
            atmlst=atmlst, mf_grad=mf_grad, eris=eris, verbose=verbose,
            fcasscf=fcasscf, ci_state=ci[state],
            auxbasis_response=self.auxbasis_response,
            lagrange_intermediates=lagrange_intermediates)
        de += self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            de = self.symmetrize(de, atmlst)
        return de

    to_gpu = lib.to_gpu
