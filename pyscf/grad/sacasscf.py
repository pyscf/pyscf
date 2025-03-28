#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import gc
import numpy as np
from functools import reduce
from itertools import product
from scipy import linalg
from pyscf import gto
from pyscf.grad import lagrange
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageFCISolver
from pyscf.mcscf.addons import StateAverageMixFCISolver, state_average_mix_
from pyscf.grad.mp2 import _shell_prange
from pyscf.mcscf import mc1step, mc1step_symm, newton_casscf
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.addons import fix_spin_, SpinPenaltyFCISolver
from pyscf.fci.spin_op import spin_square
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf import lib, ao2mo, mcscf

# ref. Mol. Phys., 99, 103 (2001); DOI: 10.1080/002689700110005642

def Lorb_dot_dgorb_dx (Lorb, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, eris=None,
                       verbose=None):
    ''' Modification of single-state CASSCF electronic energy nuclear gradient to compute instead
    the orbital Lagrange term nuclear gradient:

    sum_pq Lorb_pq d2_Ecas/d_lambda d_kpq

    This involves the effective density matrices
    ~D_pq   = L_pr*D_rq   + L_qr*D_pr
    ~d_pqrs = L_pt*d_tqrs + L_rt*d_pqts + L_qt*d_ptrs + L_st*d_pqrt
    (NB: L_pq = -L_qp)
    '''

    # dmo = smoT.dao.smo
    # dao = mo.dmo.moT
    t0 = (logger.process_clock(), logger.perf_counter())

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

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
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    # MRH: new density matrix terms
    dmL_core = np.dot(moL_core, mo_core.T) * 2
    dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
    dmL_core += dmL_core.T
    dmL_cas += dmL_cas.T
    dm1 = dm_core + dm_cas
    dm1L = dmL_core + dmL_cas
    # MRH: wrap the integral instead of the density matrix.
    # g_prst*~d_qrst = (g_pust*L_ur + g_prut*L_us + g_prsu*L_ut)*d_qrst + g_prst*L_uq*d_urst
    #                = 'aapaL'_prst*d_qrst        [ERI TERM 1]
    #                = 'aapa'_prst*L_uq*d_urst    [ERI TERM 2]
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
    gfock  = np.dot (h1, dm1L) # h1e
    gfock += np.dot ((vhf_c + vhf_a), dmL_core) # core-core and active-core, 2nd 1RDM linked
    gfock += np.dot ((vhfL_c + vhfL_a), dm_core) # core-core and active-core, 1st 1RDM linked
    gfock += np.dot (vhfL_c, dm_cas) # core-active, 1st 1RDM linked
    gfock += np.dot (vhf_c, dmL_cas) # core-active, 2nd 1RDM linked
    gfock  = np.dot (s0_inv, gfock) # Definition in MO's; going (AO->MO->AO) incurs inverse ovlp
    # [ERI TERM 1]
    gfock += reduce (np.dot, (mo_coeff, np.einsum('uviw,uvtw->it', aapaL, casdm2), mo_cas.T))
    # [ERI TERM 2]
    gfock += reduce (np.dot, (mo_coeff, np.einsum('uviw,vuwt->it', aapa, casdm2), moL_cas.T))
    dme0 = (gfock+gfock.T)/2 # This transpose is for the overlap matrix later on
    aapa = vj = vk = vhf_c = vhf_a = None

    vj, vk = mf_grad.get_jk (mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    vhf1c, vhf1a, vhf1cL, vhf1aL = vj - vk * 0.5
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    # MRH: contract the final two indices of the active-active 2RDM with L as you change to AOs
    # note tensordot always puts indices in the order of the arguments.
    dm2Lbuf = np.zeros ((ncas**2,nmo,nmo))
    # MRH: The second line below transposes the L; the third line transposes the derivative
    # Both the L and the derivative have to explore all indices
    Lcasdm2 = np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,2)).transpose (1,2,0,3)
    dm2Lbuf[:,:,ncore:nocc] = Lcasdm2.reshape (ncas**2,nmo,ncas)
    Lcasdm2 = np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,3)).transpose (1,2,3,0)
    dm2Lbuf[:,ncore:nocc,:] += Lcasdm2.reshape (ncas**2,ncas,nmo)
    Lcasdm2 = None
    dm2Lbuf += dm2Lbuf.transpose (0,2,1)
    dm2Lbuf = np.ascontiguousarray (dm2Lbuf)
    dm2Lbuf = ao2mo._ao2mo.nr_e2(dm2Lbuf.reshape (ncas**2,nmo**2), mo_coeff.T,
                                 (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    dm2Lbuf = lib.pack_tril(dm2Lbuf)
    dm2Lbuf[:,diag_idx] *= .5
    dm2Lbuf = dm2Lbuf.reshape(ncas,ncas,nao_pair)

    if atmlst is None:
        atmlst = list (range(mol.natm))
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / (4*(aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    # MRH: 3 components of eri array and 1 density matrix array:
    # FOUR arrays of this size are required!
    blksize = min(nao, max(2, blksize))
    logger.info (mc, 'SA-CASSCF Lorb_dot_dgorb memory remaining for eri manipulation: %f MB; using'
                 ' blocksize = %d', max_memory, blksize)
    t0 = logger.timer (mc, 'SA-CASSCF Lorb_dot_dgorb 1-electron part', *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: h1e and Feff terms
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm1L)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao  = lib.einsum('ijw,pi,qj->pqw', dm2Lbuf, mo_cas[p0:p1], mo_cas[q0:q1])
            # MRH: contract first two indices of active-active 2RDM with L as you go MOs -> AOs
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, moL_cas[p0:p1], mo_cas[q0:q1])
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], moL_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            # MRH: I still don't understand why there is a minus here!
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = dm2_ao = None
            t0 = logger.timer (mc, 'SA-CASSCF Lorb_dot_dgorb atom {} ({},{}|{})'.format (ia, p1-p0,
                               nf, nao_pair), *t0)
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

    logger.debug (mc, "Orb lagrange hcore component:\n{}".format (de_hcore))
    logger.debug (mc, "Orb lagrange renorm component:\n{}".format (de_renorm))
    logger.debug (mc, "Orb lagrange eri component:\n{}".format (de_eri))
    de = de_hcore + de_renorm + de_eri

    return de

def Lci_dot_dgci_dx (Lci, weights, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None,
                     eris=None, verbose=None):
    ''' Modification of single-state CASSCF electronic energy nuclear gradient to compute instead
    the CI Lagrange term nuclear gradient:

    sum_IJ Lci_IJ d2_Ecas/d_lambda d_PIJ

    This involves the effective density matrices
    ~D_pq = sum_I w_I<L_I|p'q|I> + c.c.
    ~d_pqrs = sum_I w_I<L_I|p'r'sq|I> + c.c.
    (NB: All-core terms ~D_ii, ~d_iijj = 0
     However, active-core terms ~d_xyii, ~d_xiiy != 0)
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError

    t0 = (logger.process_clock(), logger.perf_counter())
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    # MRH: TDMs + c.c. instead of RDMs
    # MRH, 06/30/2020: new interface in mcscf.addons makes this much more transparent
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

    vj, vk = mf_grad.get_jk (mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj - vk * 0.5
    #vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(casdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst),3))
    de_renorm = np.zeros((len(atmlst),3))
    de_eri = np.zeros((len(atmlst),3))
    de = np.zeros((len(atmlst),3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*.9e6/8 / (4*(aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    # MRH: 3 components of eri array and 1 density matrix array:
    # FOUR arrays of this size are required!
    blksize = min(nao, max(2, blksize))
    logger.info (mc, 'SA-CASSCF Lci_dot_dgci memory remaining for eri manipulation: %f MB; using '
                 'blocksize = %d', max_memory, blksize)
    t0 = logger.timer (mc, 'SA-CASSCF Lci_dot_dgci 1-electron part', *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        # MRH: dm1 -> dm_cas in the line below
        de_hcore[k] += np.einsum('xij,ij->x', h1ao, dm_cas)
        de_renorm[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = dm2_ao = None
            t0 = logger.timer (mc, 'SA-CASSCF Lci_dot_dgci atom {} ({},{}|{})'.format (ia, p1-p0,
                               nf, nao_pair), *t0)
        # MRH: dm1 -> dm_cas in the line below. Also eliminate core-core terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm_cas[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    logger.debug (mc, "CI lagrange hcore component:\n{}".format (de_hcore))
    logger.debug (mc, "CI lagrange renorm component:\n{}".format (de_renorm))
    logger.debug (mc, "CI lagrange eri component:\n{}".format (de_eri))
    de = de_hcore + de_renorm + de_eri
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
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    #if state is None and (not hasattr (mcscf_grad, 'state') or (mcscf_grad.state is None)):
    #    return casscf_grad.as_scanner (mcscf_grad)

    logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)
    name = mcscf_grad.__class__.__name__ + CASSCF_GradScanner.__name_mixin__
    return lib.set_class(CASSCF_GradScanner(mcscf_grad, state),
                         (CASSCF_GradScanner, mcscf_grad.__class__), name)

class CASSCF_GradScanner(lib.GradScanner):
    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is None:
            self.state = g.state
        else:
            self.state = state
        self._converged = False

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)
        if 'state' in kwargs: self.state = kwargs['state']
        mc_scanner = self.base
        e_tot = mc_scanner(mol)
        if hasattr (mc_scanner, 'e_mcscf'): self.e_mcscf = mc_scanner.e_mcscf
        if hasattr (mc_scanner, 'e_states') and self.state is not None:
            e_tot = mc_scanner.e_states[self.state]
        if not ('state' in kwargs):
            kwargs['state'] = self.state
        de = self.kernel(**kwargs)
        return e_tot, de

    @property
    def converged(self):
        return self._converged
    @converged.setter
    def converged(self, x):
        self._converged = x


class Gradients (lagrange.Gradients):

    _keys = {
        'ngorb', 'nroots', 'spin_states', 'na_states', 'nb_states', 'nroots',
        'nci', 'state', 'eris', 'weights', 'e_states', 'max_cycle', 'ncas',
        'e_cas', 'nelecas', 'mo_occ', 'mo_energy', 'mo_coeff', 'callback',
        'chkfile', 'nlag', 'frozen', 'level_shift', 'extrasym', 'fcisolver',
    }

    def __init__(self, mc, state=None):
        self.__dict__.update (mc.__dict__)
        nmo = mc.mo_coeff.shape[-1]
        self.ngorb = np.count_nonzero (mc.uniq_var_indices (nmo, mc.ncore, mc.ncas, mc.frozen))
        self.nroots = mc.fcisolver.nroots
        neleca, nelecb = _unpack_nelec (mc.nelecas)
        self.spin_states = [neleca - nelecb,] * self.nroots
        self.na_states = [cistring.num_strings (mc.ncas, neleca),] * self.nroots
        self.nb_states = [cistring.num_strings (mc.ncas, nelecb),] * self.nroots
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            self.nroots = p0 = 0
            for solver in mc.fcisolver.fcisolvers:
                self.nroots += solver.nroots
                nea, neb = mc.fcisolver._get_nelec (solver, (neleca, nelecb))
                nstr_a = cistring.num_strings (mc.ncas, nea)
                nstr_b = cistring.num_strings (mc.ncas, neb)
                for p1 in range (p0, self.nroots):
                    self.spin_states[p1] = nea - neb
                    self.na_states[p1] = nstr_a
                    self.nb_states[p1] = nstr_b
                p0 = self.nroots
        self.nci = sum ([na * nb for na, nb in zip (self.na_states, self.nb_states)])
        if state is not None:
            self.state = state
        elif hasattr (mc, 'nuc_grad_state'):
            self.state = mc.nuc_grad_state
        else:
            self.state = None
        self.eris = None
        self.weights = np.array ([1])
        try:
            self.e_states = np.asarray (mc.e_states)
        except AttributeError:
            self.e_states = np.asarray (mc.e_tot)
        if isinstance (mc, StateAverageMCSCFSolver):
            self.weights = np.asarray (mc.weights)
        if np.amax (self.weights) - np.amin (self.weights) > 1e-8:
            raise NotImplementedError ("Unequal weights in SA-CASSCF gradients")
        assert (len (self.weights) == self.nroots), '{} {} {}'.format (
            mc.fcisolver.__class__, self.weights, self.nroots)
        lagrange.Gradients.__init__(self, mc, self.ngorb+self.nci)
        self.max_cycle = mc.max_cycle_macro

    def pack_uniq_var (self, xorb, xci):
        # TODO: point-group symmetry of the xci components? CSFs?
        xorb = self.base.pack_uniq_var (xorb)
        xci = np.concatenate ([x.ravel () for x in xci])
        return np.append (xorb, xci)

    def unpack_uniq_var (self, x):
        # TODO: point-group symmetry of the xci components? CSFs?
        xorb, x = self.base.unpack_uniq_var (x[:self.ngorb]), x[self.ngorb:]
        xci = []
        for na, nb in zip (self.na_states, self.nb_states):
            xci.append (x[:na*nb].reshape (na, nb))
            x = x[na*nb:]
        return xorb, xci

    def make_fcasscf (self, state=None, casscf_attr={}, fcisolver_attr={}):
        ''' SA-CASSCF nuclear gradients require 1) first derivatives wrt wave function variables
        and nuclear shifts of the target state's energy, AND 2) first and second derivatives of the
        objective function used to determine the MO coefficients and CI vectors. This function
        addresses 1).

        Kwargs:
            state : integer
                The specific state whose energy is being differentiated. This kwarg is necessary
                in the context of state_average_mix, where the number of electrons and the
                make_rdm* functions differ from state to state.
            casscf_attr : dictionary
                Extra attributes to apply to fcasscf. Relevant to child methods (i.e., MC-PDFT;
                NACs)
            fcisolver_attr : dictionary
                Extra attributes to apply to fcasscf.fcisolver. Relevant to child methods (i.e.,
                MC-PDFT; NACs)

        Returns:
            fcasscf : object of :class:`mc1step.CASSCF`
                Set up to evaluate first derivatives of state "state". Only functions, classes,
                and the nelecas variable are set up; the caller should assign MO coefficients
                and CI vectors explicitly post facto.
        '''
        fcasscf = mcscf.CASSCF (self.base._scf, self.base.ncas, self.base.nelecas)
        fcasscf.__dict__.update (self.base.__dict__)

        nelecas = self.base.nelecas
        if isinstance (fcasscf.fcisolver, StateAverageFCISolver):
            if isinstance (fcasscf.fcisolver, StateAverageMixFCISolver):
                p0 = 0
                for solver in fcasscf.fcisolver.fcisolvers:
                    p1 = p0 + solver.nroots
                    if p0 <= state < p1:
                        solver_class = solver.__class__
                        solver_obj = solver
                        nelecas = fcasscf.fcisolver._get_nelec (solver_obj, nelecas)
                        break
                    p0 = p1
            else:
                solver_class = self.base.fcisolver._base_class
                solver_obj = self.base.fcisolver
            fcasscf.fcisolver = solver_obj.view(solver_class)
            fcasscf.fcisolver.nroots = 1
        # Spin penalty method is inapplicable to response calc'ns
        # It must be deactivated for Lagrange multipliers to converge
        if isinstance (fcasscf.fcisolver, SpinPenaltyFCISolver):
            fcasscf.fcisolver = fcasscf.fcisolver.undo_fix_spin()
        fcasscf.__dict__.update (casscf_attr)
        fcasscf.nelecas = nelecas
        fcasscf.fcisolver.__dict__.update (fcisolver_attr)
        fcasscf.verbose, fcasscf.stdout = self.verbose, self.stdout
        fcasscf._tag_gfock_ov_nonzero = True
        return fcasscf

    def make_fcasscf_sa (self, casscf_attr={}, fcisolver_attr={}):
        ''' SA-CASSCF nuclear gradients require 1) first derivatives wrt wave function variables
        and nuclear shifts of the target state's energy, AND 2) first and second derivatives of the
        objective function used to determine the MO coefficients and CI vectors. This function
        addresses 2). Note that penalty methods etc. must be removed, and that child methods such
        as MC-PDFT which do not reoptimize the orbitals also do not alter this function.

        Kwargs:
            casscf_attr : dictionary
                Extra attributes to apply to fcasscf. Just in case.
            fcisolver_attr : dictionary
                Extra attributes to apply to fcasscf.fcisolver. Just in case.

        Returns:
            fcasscf : object of :class:`StateAverageMCSCFSolver`
                Set up to evaluate second derivatives of SA-CASSCF average energy in the
                absence of (i.e., spin) penalties.
        '''
        fcasscf = self.make_fcasscf (state=0, casscf_attr={}, fcisolver_attr={})
        fcasscf.__dict__.update (self.base.__dict__)
        if isinstance (self.base, StateAverageMCSCFSolver):
            if isinstance (self.base.fcisolver, StateAverageMixFCISolver):
                fcisolvers = [f.copy() for f in self.base.fcisolver.fcisolvers]
                # Spin penalty method is inapplicable to response calc'ns
                # It must be deactivated for Lagrange multipliers to converge
                for i in range (len (fcisolvers)):
                    if isinstance (fcisolvers[i], SpinPenaltyFCISolver):
                        fcisolvers[i].ss_penalty = 0
                fcasscf = state_average_mix_(fcasscf, fcisolvers,
                                             self.base.weights)
            else:
                fcasscf.state_average_(self.base.weights)
        # Spin penalty method is inapplicable to response calc'ns
        # It must be deactivated for Lagrange multipliers to converge
        if isinstance (fcasscf.fcisolver, SpinPenaltyFCISolver):
            fcasscf.fcisolver = fcasscf.fcisolver.copy()
            fcasscf.fcisolver.ss_penalty = 0
        fcasscf.__dict__.update (casscf_attr)
        fcasscf.fcisolver.__dict__.update (fcisolver_attr)
        return fcasscf

    def kernel (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None,
                mf_grad=None, e_states=None, level_shift=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        if state is None:
            return casscf_grad.Gradients (self.base).kernel (
                mo_coeff=mo, ci=ci, atmlst=atmlst, verbose=verbose)
        if e_states is None:
            try:
                e_states = self.e_states = np.asarray (self.base.e_states)
            except AttributeError:
                e_states = self.e_states = np.asarray (self.base.e_tot)
        if level_shift is None: level_shift=self.level_shift
        return lagrange.Gradients.kernel (
            self, state=state, atmlst=atmlst, verbose=verbose, mo=mo, ci=ci, eris=eris,
            mf_grad=mf_grad, e_states=e_states, level_shift=level_shift, **kwargs)

    def get_wfn_response (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        ndet = self.na_states[state] * self.nb_states[state]
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        eris = fcasscf.ao2mo (mo)
        g_all_state = newton_casscf.gen_g_hop (fcasscf, mo, ci[state], eris, verbose)[0]
        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_state[:self.ngorb]
        # No need to reshape or anything, just use the magic of repeated slicing
        offs = sum ([na * nb for na, nb in zip(self.na_states[:state],
                                               self.nb_states[:state])]) if state > 0 else 0
        g_all[self.ngorb:][offs:][:ndet] = g_all_state[self.ngorb:]
        return g_all

    def get_Aop_Adiag (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, eris=None,
                       level_shift=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if not isinstance (self.base, StateAverageMCSCFSolver) and isinstance (ci, list):
            ci = ci[0]
        fcasscf = self.make_fcasscf_sa ()
        Aop, Adiag = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris, verbose)[2:]
        # Eliminate the component of Aop (x) which is parallel to the state-average space
        # The Lagrange multiplier equations are not defined there
        return self.project_Aop (Aop, ci, state), Adiag

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None,
                          mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        fcasscf_grad = casscf_grad.Gradients (self.make_fcasscf (state))
        # Mute some misleading messages
        fcasscf_grad._finalize = lambda: None
        return fcasscf_grad.kernel (mo_coeff=mo, ci=ci[state], atmlst=atmlst, verbose=verbose)

    def get_LdotJnuc (self, Lvec, state=None, atmlst=None, verbose=None, mo=None, ci=None,
                      eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[state]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris

        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb, Lci = self.unpack_uniq_var (Lvec)
        #Lorb = self.base.unpack_uniq_var (Lvec[:self.ngorb])
        #Lci = Lvec[self.ngorb:].reshape (self.nroots, -1)
        #ci = np.ravel (ci).reshape (self.nroots, -1)

        # CI part
        t0 = (logger.process_clock(), logger.perf_counter())
        de_Lci = Lci_dot_dgci_dx(Lci, self.weights, self.base, mo_coeff=mo, ci=ci,
                                 atmlst=atmlst, mf_grad=mf_grad, eris=eris, verbose=verbose)
        logger.info (self, '--------------- %s gradient Lagrange CI response ---------------',
                     self.base.__class__.__name__)
        if verbose >= logger.INFO: rhf_grad._write(self, self.mol, de_Lci, atmlst)
        logger.info (self, '----------------------------------------------------------------')
        t0 = logger.timer (self, '{} gradient Lagrange CI response'.format (
            self.base.__class__.__name__), *t0)

        # Orb part
        de_Lorb = Lorb_dot_dgorb_dx(Lorb, self.base, mo_coeff=mo, ci=ci,
                                    atmlst=atmlst, mf_grad=mf_grad, eris=eris, verbose=verbose)
        logger.info (self, '--------------- %s gradient Lagrange orbital response ---------------',
                     self.base.__class__.__name__)
        if verbose >= logger.INFO: rhf_grad._write(self, self.mol, de_Lorb, atmlst)
        logger.info (self, '---------------------------------------------------------------------')
        t0 = logger.timer (self, '{} gradient Lagrange orbital response'.format (
            self.base.__class__.__name__), *t0)

        return de_Lci + de_Lorb

    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, state=None, mo=None, ci=None, **kwargs):
        # This needs to be rewritten substantially to work properly with state_average_mix
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        def _debug_cispace (xci, label):
            xci_norm = [np.dot (c.ravel (), c.ravel ()) for c in xci]
            try:
                xci_ss = self.base.fcisolver.states_spin_square (
                    xci, self.base.ncas, self.base.nelecas)[0]
            except AttributeError:
                nelec = sum (_unpack_nelec (self.base.nelecas))
                xci_ss = [spin_square (x, self.base.ncas, ((nelec+m)//2,(nelec-m)//2))[0]
                          for x, m in zip (xci, self.spin_states)]
            xci_ss = [x / max (y, 1e-8) for x, y in zip (xci_ss, xci_norm)]
            xci_multip = [np.sqrt (x+.25) - .5 for x in xci_ss]
            for ix, (norm, ss, multip) in enumerate (zip (xci_norm, xci_ss, xci_multip)):
                logger.debug (self,
                              ' State {} {} norm = {:.7e} ; <S^2> = {:.7f} ; 2S+1 = {:.7f}'.format(
                                  ix, label, norm, ss, multip))
        borb, bci = self.unpack_uniq_var (bvec)
        logger.debug (self, 'Orbital rotation gradient norm = {:.7e}'.format (linalg.norm (borb)))
        _debug_cispace (bci, 'CI gradient')
        Aorb, Aci = self.unpack_uniq_var (Adiag)
        logger.debug (self, 'Orbital rotation Hamiltonian diagonal norm = {:.7e}'.format (
            linalg.norm (Aorb)))
        _debug_cispace (Aci, 'Hamiltonian diagonal')
        Lorb, Lci = self.unpack_uniq_var (Lvec)
        logger.debug (self, 'Orbital rotation Lagrange vector norm = {:.7e}'.format (
            linalg.norm (Lorb)))
        _debug_cispace (Lci, 'Lagrange vector')

    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        return SACASLagPrec (Adiag=Adiag, level_shift=level_shift, ci=ci, grad_method=self)

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        def my_call (x):
            itvec[0] += 1
            geff = geff_op (x)
            deltax = x - Lvec_last
            gorb, gci = self.unpack_uniq_var (geff)
            deltaorb, deltaci = self.unpack_uniq_var (deltax)
            gci = np.concatenate ([g.ravel () for g in gci])
            deltaci = np.concatenate ([d.ravel () for d in deltaci])
            logger.info(self, ('Lagrange optimization iteration {}, |gorb| = {}, |gci| = {}, '
                               '|dLorb| = {}, |dLci| = {}').format (
                                   itvec[0], linalg.norm (gorb), linalg.norm (gci),
                                   linalg.norm (deltaorb), linalg.norm (deltaci)))
            Lvec_last[:] = x[:]
        return my_call

    def project_Aop (self, Aop, ci, state):
        ''' Wrap the Aop function to project out redundant degrees of freedom for the CI part.
            What's redundant changes between SA-CASSCF and MC-PDFT so modify this part in child
            classes. '''
        def my_Aop (x):
            Ax = Aop (x)
            Ax_orb, Ax_ci = self.unpack_uniq_var (Ax)
            for i, j in product (range (self.nroots), repeat=2):
                # I'm assuming the only symmetry here that's actually built into the data structure
                # is solver.spin. This will be the case as long as the various solvers are
                # determinants with a common total charge occupying a common set of orbitals
                if self.spin_states[i] != self.spin_states[j]: continue
                Ax_ci[i] -= np.dot (Ax_ci[i].ravel (), ci[j].ravel ()) * ci[j]
            #Ax_ci = Ax[self.ngorb:].reshape (self.nroots, -1)
            #ci_arr = np.asarray (ci).reshape (self.nroots, -1)
            #ovlp = np.dot (ci_arr.conjugate (), Ax_ci.T)
            #Ax_ci -= np.dot (ovlp.T, ci_arr)
            #Ax[self.ngorb:] = Ax_ci.ravel ()
            return self.pack_uniq_var (Ax_orb, Ax_ci)
        return my_Aop

    as_scanner = as_scanner

class SACASLagPrec (lagrange.LagPrec):
    ''' A callable preconditioner for solving the Lagrange equations.
    Based on Mol. Phys. 99, 103 (2001).
    Attributes:

    nroots : integer
        Number of roots in the SA space
    nlag : integer
        Number of Lagrange degrees of freedom
    ngorb : integer
        Number of Lagrange degrees of freedom which are orbital rotations
    level_shift : float
        numerical shift applied to CI rotation Hessian
    ci : ndarray of shape (nroots, ndet or ncscf)
        Ci vectors of the SA space
    Rorb : ndarray of shape (ngorb)
        Diagonal inverse Hessian matrix for orbital rotations
    Rci : ndarray of shape (nroots, ndet or ncsf)
        Diagonal inverse Hessian matrix for CI rotations including a level shift
    Rci_sa : ndarray of shape (nroots (I), ndet or ncsf, nroots (K))
        First two factors of the inverse diagonal CI Hessian projected into SA space:
        Rci(I)|J> <J|Rci(I)|K>^{-1} <K|Rci(I)
        note: right-hand bra and R_I factor not included due to storage considerations
        Make the operand's matrix element with <K|Rci(I) before taking the dot product!
'''

    _keys = {
        'level_shift', 'nroots', 'nlag', 'ngorb', 'spin_states', 'na_states',
        'nb_states', 'grad_method', 'Rorb', 'ci', 'Rci', 'Rci_sa',
    }

    # TODO: fix me (subclass me? wrap me?) for state_average_mix
    def __init__(self, Adiag=None, level_shift=None, ci=None, grad_method=None):
        self.level_shift = level_shift
        self.nroots = grad_method.nroots
        self.nlag = grad_method.nlag
        self.ngorb = grad_method.ngorb
        self.spin_states = grad_method.spin_states
        self.na_states = grad_method.na_states
        self.nb_states = grad_method.nb_states
        self.grad_method = grad_method
        Aorb, Aci = self.unpack_uniq_var (Adiag)
        self._init_orb (Aorb)
        self._init_ci (Aci, ci)

    def unpack_uniq_var (self, x):
        return self.grad_method.unpack_uniq_var (x)

    def pack_uniq_var (self, xorb, xci):
        return self.grad_method.pack_uniq_var (xorb, xci)

    def _init_orb (self, Aorb):
        self.Rorb = Aorb
        self.Rorb[abs(self.Rorb)<1e-8] = 1e-8
        self.Rorb = 1./self.Rorb

    def _init_ci (self, Aci_spins, ci_spins):
        self.ci = []
        self.Rci = []
        self.Rci_sa = []
        for [Aci, ci] in self._iterate_ci (Aci_spins, ci_spins):
            nroots = Aci.shape[0]
            Rci = Aci + self.level_shift
            Rci[abs(Rci)<1e-8] = 1e-8
            Rci = 1./Rci
            # R_I|J>
            # Indices: I, det, J
            Rci_cross = Rci[:,:,None] * ci.T[None,:,:]
            # S(I)_JK = <J|R_I|K> (first index of CI contract with middle index of R_I|J>)
            # and reshape to put I first
            Sci = np.tensordot (ci.conjugate (), Rci_cross, axes=(1,1)).transpose (1,0,2)
            # R_I|J> S(I)_JK^-1 (can only loop explicitly because of necessary call to linalg.inv)
            # Indices: I, det, K
            Rci_sa = np.zeros_like (Rci_cross)
            for iroot in range (nroots):
                Rci_sa[iroot] = np.dot (Rci_cross[iroot], linalg.inv (Sci[iroot]))
            self.ci.append (ci)
            self.Rci.append (Rci)
            self.Rci_sa.append (Rci_sa)

    def _iterate_ci (self, *args):
        # All args must be iterables over CI vectors in input order
        # Eventually, get rid of copying (np.asarray, etc.)
        # Don't assume args are ndarrays on input
        for my_spin in np.unique (self.spin_states):
            idx = np.where (self.spin_states == my_spin)[0]
            yield [np.asarray ([arg[i] for i in idx]).reshape (len (idx), -1) for arg in args]

    def __call__(self, x):
        xorb, xci = self.unpack_uniq_var (x)
        Mxorb = self.orb_prec (xorb)
        Mxci = self.ci_prec (xci)
        return self.pack_uniq_var (Mxorb, Mxci)

    def orb_prec (self, xorb):
        return self.Rorb * xorb

    def ci_prec (self, xci_spins):
        Mxci = [None,] * self.nroots
        for ix_spin, [xci, desort_spin] in enumerate (
                self._iterate_ci (xci_spins, list(range(self.nroots)))):
            desort_spin = np.atleast_1d (np.squeeze (desort_spin))
            nroots = xci.shape[0]
            ci = self.ci[ix_spin]
            Rci = self.Rci[ix_spin]
            Rci_sa = self.Rci_sa[ix_spin]
            # R_I|H I> (indices: I, det)
            Rx = Rci * xci
            # <J|R_I|H I> (indices: J, I)
            sa_ovlp = np.dot (ci.conjugate (), Rx.T)
            # R_I|J> S(I)_JK^-1 <K|R_I|H I> (indices: I, det)
            Rx_sub = np.zeros_like (Rx)
            for iroot in range (nroots):
                Rx_sub[iroot] = np.dot (Rci_sa[iroot], sa_ovlp[:,iroot])
            for i, j in enumerate (desort_spin):
                try:
                    Mxci[j] = Rx[i] - Rx_sub[i]
                except Exception as e:
                    print (i, j, desort_spin)
                    raise (e)
        assert (all (i is not None for i in Mxci))
        return Mxci

mcscf.addons.StateAverageMCSCFSolver.Gradients = lib.class_as_method(Gradients)
