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

from pyscf import mcscf, lib, ao2mo
from pyscf.grad import lagrange
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.grad import casscf as casscf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.mcscf import mc1step, mc1step_symm, newton_casscf
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.df.grad import casscf as dfcasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import spin_square0
from pyscf.fci import cistring
import numpy as np
import copy, time, gc
from functools import reduce
from scipy import linalg
from pyscf.df.grad.casdm2_util import solve_df_rdm2, grad_elec_dferi, grad_elec_auxresponse_dferi

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

    lib.logger.debug (mc, "Orb lagrange hcore component:\n{}".format (de_hcore))
    lib.logger.debug (mc, "Orb lagrange renorm component:\n{}".format (de_renorm))
    lib.logger.debug (mc, "Orb lagrange eri component:\n{}".format (de_eri))
    lib.logger.debug (mc, "Orb lagrange aux component:\n{}".format (de_aux))
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

    lib.logger.debug (mc, "CI lagrange hcore component:\n{}".format (de_hcore))
    lib.logger.debug (mc, "CI lagrange renorm component:\n{}".format (de_renorm))
    lib.logger.debug (mc, "CI lagrange eri component:\n{}".format (de_eri))
    lib.logger.debug (mc, "CI lagrange aux component:\n{}".format (de_aux))
    de = de_hcore + de_renorm + de_eri + de_aux
    return de

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

    class CASSCF_GradScanner(mcscf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
            if state is None:
                self.state = g.state
            else:
                self.state = state
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
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

    return CASSCF_GradScanner(mcscf_grad)


class Gradients (sacasscf_grad.Gradients):

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
