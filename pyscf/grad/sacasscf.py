from pyscf.grad import lagrange
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from pyscf.grad.mp2 import _shell_prange
from pyscf.mcscf import mc1step, mc1step_symm, newton_casscf
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import spin_square0
from pyscf.fci import cistring
from pyscf import lib, ao2mo
import numpy as np
import copy, time, gc
from functools import reduce
from scipy import linalg

def Lorb_dot_dgorb_dx (Lorb, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, eris=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the orbital
    Lagrange term nuclear gradient (sum_pq Lorb_pq d2_Ecas/d_lambda d_kpq)
    This involves removing nuclear-nuclear terms and making the substitution
    (D_[p]q + D_p[q]) -> D_pq
    (d_[p]qrs + d_pq[r]s + d_p[q]rs + d_pqr[s]) -> d_pqrs
    Where [] around an index implies contraction with Lorb from the left, so that the external index
    (regardless of whether the index on the rdm is bra or ket) is always the first index of Lorb. '''

    # dmo = smoT.dao.smo
    # dao = mo.dmo.moT
    t0 = (time.clock (), time.time ())

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

    mo_occ = mo_coeff[:,:nocc]
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

    vhf1c, vhf1a, vhf1cL, vhf1aL = mf_grad.get_veff(mol, (dm_core, dm_cas, dmL_core, dmL_cas))
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
    # MRH: The second line below transposes the L; the third line transposes the derivative later on
    # Both the L and the derivative have to explore all indices
    dm2Lbuf[:,:,ncore:nocc]  = np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,2)).transpose (1,2,0,3).reshape (ncas**2,nmo,ncas)
    dm2Lbuf[:,ncore:nocc,:] += np.tensordot (Lorb[:,ncore:nocc], casdm2, axes=(1,3)).transpose (1,2,3,0).reshape (ncas**2,ncas,nmo)
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
    # MRH: 3 components of eri array and 1 density matrix array: FOUR arrays of this size are required!
    blksize = min(nao, max(2, blksize))
    lib.logger.info (mc, 'SA-CASSCF Lorb_dot_dgorb memory remaining for eri manipulation: {} MB; using blocksize = {}'.format (max_memory, blksize)) 
    t0 = lib.logger.timer (mc, 'SA-CASSCF Lorb_dot_dgorb 1-electron part', *t0)

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
            # MRH: now contract the first two indices of the active-active 2RDM with L as you go from MOs to AOs
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, moL_cas[p0:p1], mo_cas[q0:q1])
            dm2_ao += lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], moL_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            gc.collect ()
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            # MRH: I still don't understand why there is a minus here!
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = dm2_ao = None
            gc.collect ()
            t0 = lib.logger.timer (mc, 'SA-CASSCF Lorb_dot_dgorb atom {} ({},{}|{})'.format (ia, p1-p0, nf, nao_pair), *t0)
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
    de = de_hcore + de_renorm + de_eri

    return de

def Lci_dot_dgci_dx (Lci, weights, mc, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, eris=None, verbose=None):
    ''' Modification of pyscf.grad.casscf.kernel to compute instead the CI
    Lagrange term nuclear gradient (sum_IJ Lci_IJ d2_Ecas/d_lambda d_PIJ)
    This involves removing all core-core and nuclear-nuclear terms and making the substitution
    sum_I w_I<L_I|p'q|I> + c.c. -> <0|p'q|0>
    sum_I w_I<L_I|p'r'sq|I> + c.c. -> <0|p'r'sq|0>
    The active-core terms (sum_I w_I<L_I|x'iyi|I>, sum_I w_I <L_I|x'iiy|I>, c.c.) must be retained.'''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError

    t0 = (time.clock (), time.time ())
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    nroots = ci.shape[0]

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    # MRH: TDMs + c.c. instead of RDMs
    casdm1 = np.zeros ((nroots, ncas, ncas))
    casdm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas))
    for iroot in range (nroots):
        #print ("norm of Lci, ci for root {}: {} {}".format (iroot, linalg.norm (Lci[iroot]), linalg.norm (ci[iroot])))
        casdm1[iroot], casdm2[iroot] = mc.fcisolver.trans_rdm12 (Lci[iroot], ci[iroot], ncas, nelecas)
    casdm1 = (casdm1 * weights[:,None,None]).sum (0)
    casdm2 = (casdm2 * weights[:,None,None,None,None]).sum (0)
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

    vhf1c, vhf1a = mf_grad.get_veff(mol, (dm_core, dm_cas))
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
    # MRH: 3 components of eri array and 1 density matrix array: FOUR arrays of this size are required!
    blksize = min(nao, max(2, blksize))
    lib.logger.info (mc, 'SA-CASSCF Lci_dot_dgci memory remaining for eri manipulation: {} MB; using blocksize = {}'.format (max_memory, blksize)) 
    t0 = lib.logger.timer (mc, 'SA-CASSCF Lci_dot_dgci 1-electron part', *t0)

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
            gc.collect ()
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de_eri[k] -= np.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = dm2_ao = None
            gc.collect ()
            t0 = lib.logger.timer (mc, 'SA-CASSCF Lci_dot_dgci atom {} ({},{}|{})'.format (ia, p1-p0, nf, nao_pair), *t0)
        # MRH: dm1 -> dm_cas in the line below. Also eliminate core-core terms
        de_eri[k] += np.einsum('xij,ij->x', vhf1c[:,p0:p1], dm_cas[p0:p1]) * 2
        de_eri[k] += np.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    lib.logger.debug (mc, "CI lagrange hcore component:\n{}".format (de_hcore))
    lib.logger.debug (mc, "CI lagrange renorm component:\n{}".format (de_renorm))
    lib.logger.debug (mc, "CI lagrange eri component:\n{}".format (de_eri))
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
    from pyscf import gto
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    if state is None and (not hasattr (mcscf_grad, 'state') or (mcscf_grad.state is None)):
        return casscf_grad.as_scanner (mcscf_grad)

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


class Gradients (lagrange.Gradients):

    def __init__(self, mc, state=None):
        self.__dict__.update (mc.__dict__)
        nmo = mc.mo_coeff.shape[-1]
        self.ngorb = np.count_nonzero (mc.uniq_var_indices (nmo, mc.ncore, mc.ncas, mc.frozen))
        self.nroots = mc.fcisolver.nroots
        if hasattr (mc.fcisolver, 'fcisolvers'):
            self.nroots = sum ([s.nroots for s in mc.fcisolver.fcisolvers])
        neleca, nelecb = _unpack_nelec (mc.nelecas)
        self.nci = cistring.num_strings (mc.ncas, neleca) * cistring.num_strings (mc.ncas, nelecb) * self.nroots
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
        except AttributeError as e:
            self.e_states = np.asarray (mc.e_tot)
        if hasattr (mc, 'weights'):
            self.weights = np.asarray (mc.weights)
        assert (len (self.weights) == self.nroots), '{} {} {}'.format (mc.fcisolver.__class__, self.weights, self.nroots)
        lagrange.Gradients.__init__(self, mc, self.ngorb+self.nci)
        self.max_cycle = mc.max_cycle_macro

    def make_fcasscf (self, casscf_attr={}, fcisolver_attr={}):
        ''' Make a fake CASSCF object for ostensible single-state calculations '''
        if isinstance (self.base, mc1step_symm.CASSCF):
            fcasscf = mc1step_symm.CASSCF (self.base._scf, self.base.ncas, self.base.nelecas)
        else:
            fcasscf = mc1step.CASSCF (self.base._scf, self.base.ncas, self.base.nelecas)
        fcasscf.__dict__.update (self.base.__dict__)
        # Fix me for state_average_mix!
        if hasattr (self.base, 'weights'):
            fcasscf.fcisolver = self.base.fcisolver._base_class (self.base.mol)
            fcasscf.nroots = 1
            fcasscf.fcisolver.__dict__.update (self.base.fcisolver.__dict__)
        fcasscf.__dict__.update (casscf_attr)
        fcasscf.fcisolver.__dict__.update (fcisolver_attr)
        fcasscf.verbose, fcasscf.stdout = self.verbose, self.stdout
        fcasscf._tag_gfock_ov_nonzero = True
        return fcasscf

    def make_fcasscf_sa (self, casscf_attr={}, fcisolver_attr={}):
        ''' Make a fake SA-CASSCF object to get around weird inheritance conflicts '''
        # Fix me for state_average_mix!
        fcasscf = self.make_fcasscf (casscf_attr={}, fcisolver_attr={})
        if hasattr (self.base, 'weights'):
            fcasscf.state_average_(self.base.weights)
        return fcasscf

    def kernel (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, e_states=None, level_shift=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        if state is None:
            return casscf_grad.Gradients (self.base).kernel (mo_coeff=mo, ci=ci, atmlst=atmlst, verbose=verbose)
        if e_states is None:
            try:
                e_states = self.e_states = np.asarray (self.base.e_states)
            except AttributeError as e:
                e_states = self.e_states = np.asarray (self.base.e_tot)
        if level_shift is None: level_shift=self.level_shift
        return lagrange.Gradients.kernel (self, state=state, atmlst=atmlst, verbose=verbose, mo=mo, ci=ci, eris=eris, mf_grad=mf_grad, e_states=e_states, level_shift=level_shift, **kwargs)

    def get_wfn_response (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        ndet = ci[state].size
        fcasscf = self.make_fcasscf ()
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        eris = fcasscf.ao2mo (mo)
        g_all_state = newton_casscf.gen_g_hop (fcasscf, mo, ci[state], eris, verbose)[0]
        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_state[:self.ngorb]
        # No need to reshape or anything, just use the magic of repeated slicing
        g_all[self.ngorb:][ndet*state:][:ndet] = g_all_state[self.ngorb:]
        return g_all

    def get_Aop_Adiag (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, eris=None, level_shift=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if not isinstance (self.base, StateAverageMCSCFSolver) and isinstance (ci, list): ci = ci[0]
        fcasscf = self.make_fcasscf_sa ()
        Aop, Adiag = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris, verbose)[2:]
        # Eliminate the component of Aop (x) which is parallel to the state-average space
        # The Lagrange multiplier equations are not defined there
        return self.project_Aop (Aop, ci, state), Adiag


    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        fcasscf_grad = casscf_grad.Gradients (self.make_fcasscf ())
        fcasscf_grad.mo_coeff = mo
        fcasscf_grad.ci = ci[state]
        return fcasscf_grad.kernel (mo_coeff=mo, ci=ci[state], atmlst=atmlst, verbose=verbose)

    def get_LdotJnuc (self, Lvec, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[state]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        if getattr(self.base.fcisolver, 'gen_linkstr', None):
            linkstr  = self.base.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstr  = None

        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb = self.base.unpack_uniq_var (Lvec[:self.ngorb])
        Lci = Lvec[self.ngorb:].reshape (self.nroots, -1)
        ci = np.ravel (ci).reshape (self.nroots, -1)

        # CI part
        t0 = (time.clock (), time.time ())
        de_Lci = Lci_dot_dgci_dx (Lci, self.weights, self.base, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, eris=eris, verbose=verbose)
        lib.logger.info (self, '--------------- %s gradient Lagrange CI response ---------------',
                    self.base.__class__.__name__)
        if verbose >= lib.logger.INFO: rhf_grad._write(self, self.mol, de_Lci, atmlst)
        lib.logger.info (self, '----------------------------------------------------------------')
        t0 = lib.logger.timer (self, '{} gradient Lagrange CI response'.format (self.base.__class__.__name__), *t0)

        # Orb part
        de_Lorb = Lorb_dot_dgorb_dx (Lorb, self.base, mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad, eris=eris, verbose=verbose)
        lib.logger.info (self, '--------------- %s gradient Lagrange orbital response ---------------',
                    self.base.__class__.__name__)
        if verbose >= lib.logger.INFO: rhf_grad._write(self, self.mol, de_Lorb, atmlst)
        lib.logger.info (self, '----------------------------------------------------------------------')
        t0 = lib.logger.timer (self, '{} gradient Lagrange orbital response'.format (self.base.__class__.__name__), *t0)

        return de_Lci + de_Lorb
    
    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, state=None, mo=None, ci=None, **kwargs):
        return
        # This needs to be rewritten substantially to work properly with state_average_mix
        #if state is None: state = self.state
        #if mo is None: mo = self.base.mo_coeff
        #if ci is None: ci = self.base.ci
        #lib.logger.info (self, '{} gradient: state = {}'.format (self.base.__class__.__name__, state))
        #ngorb = self.ngorb
        #nci = self.nci
        #nroots = self.nroots
        #ndet = nci // nroots
        #ncore = self.base.ncore
        #ncas = self.base.ncas
        #nelecas = self.base.nelecas
        #nocc = ncore + ncas
        #nlag = self.nlag
        #ci = np.asarray (self.base.ci).reshape (nroots, -1)
        #err = Aop (Lvec) + bvec
        #eorb = self.base.unpack_uniq_var (err[:ngorb])
        #eci = err[ngorb:].reshape (nroots, -1)
        #borb = self.base.unpack_uniq_var (bvec[:ngorb])
        #bci = bvec[ngorb:].reshape (nroots, -1)
        #Lorb = self.base.unpack_uniq_var (Lvec[:ngorb])
        #Lci = Lvec[ngorb:].reshape (nroots, ndet)
        #Aci = Adiag[ngorb:].reshape (nroots, ndet)
        #Lci_ci_ovlp = (np.asarray (ci).reshape (nroots,-1).conjugate () @ Lci.T).T
        #Lci_Lci_ovlp = (Lci.conjugate () @ Lci.T).T
        #eci_ci_ovlp = (np.asarray (ci).reshape (nroots,-1).conjugate () @ eci.T).T
        #bci_ci_ovlp = (np.asarray (ci).reshape (nroots,-1).conjugate () @ bci.T).T
        #ci_ci_ovlp = ci.conjugate () @ ci.T
        #lib.logger.debug (self, "{} gradient RHS, inactive-active orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, borb[:ncore,ncore:nocc]))
        #lib.logger.debug (self, "{} gradient RHS, inactive-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, borb[:ncore,nocc:]))
        #lib.logger.debug (self, "{} gradient RHS, active-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, borb[ncore:nocc,nocc:]))
        #lib.logger.debug (self, "{} gradient residual, inactive-active orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, eorb[:ncore,ncore:nocc]))
        #lib.logger.debug (self, "{} gradient residual, inactive-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, eorb[:ncore,nocc:]))
        #lib.logger.debug (self, "{} gradient residual, active-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, eorb[ncore:nocc,nocc:]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, inactive-active orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, Lorb[:ncore,ncore:nocc]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, inactive-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, Lorb[:ncore,nocc:]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, active-external orbital rotations:\n{}".format (
        #    self.base.__class__.__name__, Lorb[ncore:nocc,nocc:]))
        #'''
        #lib.logger.debug (self, "{} gradient RHS, inactive-inactive orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, borb[:ncore,:ncore]))
        #lib.logger.debug (self, "{} gradient RHS, active-active orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, borb[ncore:nocc,ncore:nocc]))
        #lib.logger.debug (self, "{} gradient RHS, external-external orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, borb[nocc:,nocc:]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, inactive-inactive orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, Lorb[:ncore,:ncore]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, active-active orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, Lorb[ncore:nocc,ncore:nocc]))
        #lib.logger.debug (self, "{} gradient Lagrange factor, external-external orbital rotations (redundant!):\n{}".format (
        #    self.base.__class__.__name__, Lorb[nocc:,nocc:]))
        #'''
        #lib.logger.debug (self, "{} gradient Lagrange factor, CI part overlap with true CI SA space:\n{}".format ( 
        #    self.base.__class__.__name__, Lci_ci_ovlp))
        #lib.logger.debug (self, "{} gradient Lagrange factor, CI part self overlap matrix:\n{}".format ( 
        #    self.base.__class__.__name__, Lci_Lci_ovlp))
        #lib.logger.debug (self, "{} gradient Lagrange factor, CI vector self overlap matrix:\n{}".format ( 
        #    self.base.__class__.__name__, ci_ci_ovlp))
        #lib.logger.debug (self, "{} gradient Lagrange factor, CI part response overlap with SA space:\n{}".format ( 
        #    self.base.__class__.__name__, bci_ci_ovlp))
        #lib.logger.debug (self, "{} gradient Lagrange factor, CI part residual overlap with SA space:\n{}".format ( 
        #    self.base.__class__.__name__, eci_ci_ovlp))
        #neleca, nelecb = _unpack_nelec (nelecas)
        #spin = neleca - nelecb + 1
        #csf = CSFTransformer (ncas, neleca, nelecb, spin)
        #ecsf = csf.vec_det2csf (eci, normalize=False, order='C')
        #err_norm_det = linalg.norm (err)
        #err_norm_csf = linalg.norm (np.append (eorb, ecsf.ravel ()))
        #lib.logger.debug (self, "{} gradient: determinant residual = {}, CSF residual = {}".format (
        #    self.base.__class__.__name__, err_norm_det, err_norm_csf))
        #ci_lbls, ci_csf   = csf.printable_largest_csf (ci,  10, isdet=True, normalize=True,  order='C')
        #bci_lbls, bci_csf = csf.printable_largest_csf (bci, 10, isdet=True, normalize=False, order='C')
        #eci_lbls, eci_csf = csf.printable_largest_csf (eci, 10, isdet=True, normalize=False, order='C')
        #Lci_lbls, Lci_csf = csf.printable_largest_csf (Lci, 10, isdet=True, normalize=False, order='C')
        #Aci_lbls, Aci_csf = csf.printable_largest_csf (Aci, 10, isdet=True, normalize=False, order='C')
        #ncsf = bci_csf.shape[1]
        #for iroot in range (self.nroots):
        #    lib.logger.debug (self, "{} gradient Lagrange factor, CI part root {} spin square: {}".format (
        #        self.base.__class__.__name__, iroot, spin_square0 (Lci[iroot], ncas, nelecas)))
        #    lib.logger.debug (self, "Base CI vector")
        #    for icsf in range (ncsf):
        #        lib.logger.debug (self, '{} {}'.format (ci_lbls[iroot,icsf], ci_csf[iroot,icsf]))
        #    lib.logger.debug (self, "CI gradient:")
        #    for icsf in range (ncsf):
        #        lib.logger.debug (self, '{} {}'.format (bci_lbls[iroot,icsf], bci_csf[iroot,icsf]))
        #    lib.logger.debug (self, "CI residual:")
        #    for icsf in range (ncsf):
        #        lib.logger.debug (self, '{} {}'.format (eci_lbls[iroot,icsf], eci_csf[iroot,icsf]))
        #    lib.logger.debug (self, "CI Lagrange vector:")
        #    for icsf in range (ncsf):
        #        lib.logger.debug (self, '{} {}'.format (Lci_lbls[iroot,icsf], Lci_csf[iroot,icsf]))
        #    lib.logger.debug (self, "Diagonal of Hessian matrix CI part:")
        #    for icsf in range (ncsf):
        #        lib.logger.debug (self, '{} {}'.format (Aci_lbls[iroot,icsf], Aci_csf[iroot,icsf]))
        #'''
        #Afull = np.zeros ((nlag, nlag))
        #dum = np.zeros ((nlag))
        #for ix in range (nlag):
        #    dum[ix] = 1
        #    Afull[ix,:] = Aop (dum)
        #    dum[ix] = 0
        #Afull_orborb = Afull[:ngorb,:ngorb]
        #Afull_orbci = Afull[:ngorb,ngorb:].reshape (ngorb, nroots, ndet)
        #Afull_ciorb = Afull[ngorb:,:ngorb].reshape (nroots, ndet, ngorb)
        #Afull_cici = Afull[ngorb:,ngorb:].reshape (nroots, ndet, nroots, ndet).transpose (0, 2, 1, 3)
        #lib.logger.debug (self, "Orb-orb Hessian:\n{}".format (Afull_orborb))
        #for iroot in range (nroots):
        #    lib.logger.debug (self, "Orb-ci Hessian root {}:\n{}".format (iroot, Afull_orbci[:,iroot,:]))
        #    lib.logger.debug (self, "Ci-orb Hessian root {}:\n{}".format (iroot, Afull_ciorb[iroot,:,:]))
        #    for jroot in range (nroots):
        #        lib.logger.debug (self, "Ci-ci Hessian roots {},{}:\n{}".format (iroot, jroot, Afull_cici[iroot,jroot,:,:]))
        #'''


    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        return SACASLagPrec (nroots=self.nroots, nlag=self.nlag, ngorb=self.ngorb, Adiag=Adiag, 
            level_shift=level_shift, ci=ci, **kwargs)

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        def my_call (x):
            itvec[0] += 1
            geff = geff_op (x)
            deltax = x - Lvec_last
            gorb = geff[:self.ngorb]
            xci = x[self.ngorb:]
            gci = geff[self.ngorb:]
            deltaorb = deltax[:self.ngorb]
            deltaci = deltax[self.ngorb:]
            lib.logger.info (self, ('Lagrange optimization iteration {}, |gorb| = {}, |gci| = {}, '
                '|dLorb| = {}, |dLci| = {}').format (itvec[0], linalg.norm (gorb), linalg.norm (gci),
                linalg.norm (deltaorb), linalg.norm (deltaci))) 
            Lvec_last[:] = x[:]
            #ci_arr = np.array (self.ci).reshape (self.nroots, -1)
            #deltaci_ovlp = ci_arr @ deltaci.reshape (self.nroots, -1).T
            #gci_ovlp = ci_arr @ gci.reshape (self.nroots, -1).T
            #xci_ovlp = ci_arr @ xci.reshape (self.nroots, -1).T
            #print (xci_ovlp)
            #print (linalg.norm (xci - (ci_arr.T @ xci_ovlp).ravel ()))
        return my_call

    def project_Aop (self, Aop, ci, state):
        ''' Wrap the Aop function to project out redundant degrees of freedom for the CI part.  What's redundant
            changes between SA-CASSCF and MC-PDFT so modify this part in child classes. '''
        def my_Aop (x):
            Ax = Aop (x)
            Ax_ci = Ax[self.ngorb:].reshape (self.nroots, -1)
            ci_arr = np.asarray (ci).reshape (self.nroots, -1)
            ovlp = np.dot (ci_arr.conjugate (), Ax_ci.T)
            Ax_ci -= np.dot (ovlp.T, ci_arr)
            Ax[self.ngorb:] = Ax_ci.ravel ()
            return Ax
        return my_Aop

    as_scanner = as_scanner

class SACASLagPrec (lagrange.LagPrec):
    ''' A callable preconditioner for solving the Lagrange equations. Based on Mol. Phys. 99, 103 (2001).
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

    def __init__(self, nroots=None, nlag=None, ngorb=None, Adiag=None, ci=None, level_shift=None, **kwargs):
        self.level_shift = level_shift
        self.nroots = nroots
        self.nlag = nlag
        self.ngorb = ngorb
        self.ci = np.asarray (ci).reshape (self.nroots, -1)
        self._init_orb (Adiag)
        self._init_ci (Adiag)

    def _init_orb (self, Adiag):
        self.Rorb = Adiag[:self.ngorb]
        self.Rorb[abs(self.Rorb)<1e-8] = 1e-8
        self.Rorb = 1./self.Rorb

    def _init_ci (self, Adiag):
        self.Rci = Adiag[self.ngorb:].reshape (self.nroots, -1) + self.level_shift
        self.Rci[abs(self.Rci)<1e-8] = 1e-8
        self.Rci = 1./self.Rci
        # R_I|J> 
        # Indices: I, det, J
        Rci_cross = self.Rci[:,:,None] * self.ci.T[None,:,:]
        # S(I)_JK = <J|R_I|K> (first index of CI contract with middle index of R_I|J> and reshape to put I first)
        Sci = np.tensordot (self.ci.conjugate (), Rci_cross, axes=(1,1)).transpose (1,0,2)
        # R_I|J> S(I)_JK^-1 (can only loop explicitly because of necessary call to linalg.inv)
        # Indices: I, det, K
        self.Rci_sa = np.zeros_like (Rci_cross)
        for iroot in range (self.nroots):
            self.Rci_sa[iroot] = np.dot (Rci_cross[iroot], linalg.inv (Sci[iroot]))

    def __call__(self, x):
        xorb = self.orb_prec (x)
        xci = self.ci_prec (x)
        return np.append (xorb, xci.ravel ())

    def orb_prec (self, x):
        return self.Rorb * x[:self.ngorb]

    def ci_prec (self, x):
        xci = x[self.ngorb:].reshape (self.nroots, -1)
        # R_I|H I> (indices: I, det)
        Rx = self.Rci * xci
        # <J|R_I|H I> (indices: J, I)
        sa_ovlp = np.dot (self.ci.conjugate (), Rx.T) 
        # R_I|J> S(I)_JK^-1 <K|R_I|H I> (indices: I, det)
        Rx_sub = np.zeros_like (Rx)
        for iroot in range (self.nroots): 
            Rx_sub[iroot] = np.dot (self.Rci_sa[iroot], sa_ovlp[:,iroot])
        return Rx - Rx_sub

from pyscf import mcscf
mcscf.addons.StateAverageMCSCFSolver.Gradients = lib.class_as_method(Gradients)


