import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf.mcscf import newton_casscf
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import sacasscf as sacasscf_grad
from functools import reduce

# The extension from gradients -> NACs has three basic steps:
# 0. ("state" index integer -> tuple)
# 1. fcisolver.make_rdm12 -> fcisolver.trans_rdm12
# 2. remove core-orbital and nuclear contributions to everything
# 3. option to include the "csf contribution"
# Additional good ideas:
# a. Option to multiply NACs by the energy difference to control
#    singularities

def _unpack_state(state):
    assert len(state) == 2, "derivative couplings are defined between 2 states"
    return state[0], state[1]


def grad_elec_core(mc_grad, mo_coeff=None, atmlst=None, eris=None, mf_grad=None):
    """Compute the core-electron part of the CASSCF (Hellmann-Feynman)
    gradient using a modified RHF grad_elec call."""
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if eris is None: eris = mc.ao2mo (mo_coeff)
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    ncore = mc.ncore
    moH = mo_coeff.conj ().T
    f0 = (moH @ mc.get_hcore () @ mo_coeff) + eris.vhf_c
    mo_energy = f0.diagonal ().copy ()
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:ncore] = 2.0
    f0 *= mo_occ[None,:]
    dme0 = lambda * args: mo_coeff @ ((f0+f0.T)*.5) @ moH
    with lib.temporary_env (mf_grad, make_rdm1e=dme0, verbose=0):
        with lib.temporary_env (mf_grad.base, mo_coeff=mo_coeff, mo_occ=mo_occ):
            # Second level there should become unnecessary in future, if anyone
            # ever gets around to cleaning up pyscf.df.grad.rhf & pyscf.grad.rhf
            de = mf_grad.grad_elec (mo_coeff=mo_coeff, mo_energy=mo_energy,
                                    mo_occ=mo_occ, atmlst=atmlst)
    return de

def grad_elec_active (mc_grad, mo_coeff=None, ci=None, atmlst=None,
                      eris=None, mf_grad=None, verbose=None):
    '''Compute the active-electron part of the CASSCF (Hellmann-Feynman)
    gradient by subtracting the core-electron part.'''
    t0 = (logger.process_clock (), logger.perf_counter ())
    mc = mc_grad.base
    log = logger.new_logger (mc_grad, verbose)
    if mf_grad is None: mf_grad=mc._scf.nuc_grad_method ()
    de = mc_grad.grad_elec (mo_coeff=mo_coeff, ci=ci, atmlst=atmlst,
                            verbose=0)
    de -= grad_elec_core (mc_grad, mo_coeff=mo_coeff, atmlst=atmlst,
                          eris=eris, mf_grad=mf_grad)
    log.debug ('CASSCF active-orbital gradient:\n{}'.format (de))
    log.timer ('CASSCF active-orbital gradient', *t0)
    return de

def gen_g_hop_active (mc, mo, ci0, eris, verbose=None):
    '''Compute the active-electron part of the orbital rotation gradient
    by patching out the appropriate block of eris.vhf_c'''
    moH = mo.conj ().T
    ncore = mc.ncore
    vnocore = eris.vhf_c.copy ()
    vnocore[:,:ncore] = -moH @ mc.get_hcore () @ mo[:,:ncore]
    with lib.temporary_env (eris, vhf_c=vnocore):
        return newton_casscf.gen_g_hop (mc, mo, ci0, eris, verbose=verbose)

def _nac_csf (mol, mf_grad, tm1, atmlst):
    if atmlst is None: atmlst = list (range (mol.natm))
    aoslices = mol.aoslice_by_atom ()
    s1 = mf_grad.get_ovlp (mol)
    # if libcint documentation is to be trusted, mf_grad.get_ovlp
    # corresponds to differentiating on the SECOND index: <p|dq/dR>
    nac = np.zeros ((len(atmlst), 3))
    for k, ia in enumerate (atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        nac[k] += 0.5*np.einsum ('xij,ij->x', s1[:,p0:p1], tm1[p0:p1])
    return nac

def nac_csf (mc_grad, mo_coeff=None, ci=None, state=None, mf_grad=None,
             atmlst=None):
    '''Compute the "CSF contribution" to the SA-CASSCF NAC'''
    mc = mc_grad.base
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if state is None: state = mc_grad.state
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    if atmlst is None: atmlst = mc_grad.atmlst
    mol = mc.mol
    ket, bra = _unpack_state (state)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    castm1 = direct_spin1.trans_rdm1 (ci[bra], ci[ket], ncas, nelecas)
    # if PySCF commentary is to be trusted, trans_rdm1[p,q] is
    # <bra|q'p|ket>. I want <bra|p'q - q'p|ket>.
    castm1 = castm1.conj ().T - castm1
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    tm1 = reduce (np.dot, (mo_cas, castm1, mo_cas.conj ().T))
    return _nac_csf (mol, mf_grad, tm1, atmlst)

class NonAdiabaticCouplings (sacasscf_grad.Gradients):
    '''SA-CASSCF non-adiabatic couplings (NACs) between states

    kwargs/attributes:

    state : tuple of length 2
        The NACs returned are <state[1]|d(state[0])/dR>.
        In other words, state = (ket, bra).
    mult_ediff : logical
        If True, returns NACs multiplied by the energy difference.
        Useful near conical intersections to avoid numerical problems.
    use_etfs : logical
        If True, use the ``electron translation factors'' of Fatehi and
        Subotnik [JPCL 3, 2039 (2012)], which guarantee conservation of
        total electron + nuclear momentum when the nuclei are moving
        (i.e., in non-adiabatic molecular dynamics). This corresponds
        to omitting the so-called ``CSF contribution'' [cf. JCTC 12,
        3636 (2016)].
    '''

    def __init__(self, mc, state=None, mult_ediff=False, use_etfs=False):
        self.mult_ediff = mult_ediff
        self.use_etfs = use_etfs
        if state is not None:
            assert len(state) == 2, "derivative couplings are defined between 2 states"
        sacasscf_grad.Gradients.__init__(self, mc, state=state)

    def make_fcasscf_nacs (self, state=None, casscf_attr=None,
                           fcisolver_attr=None):
        if state is None: state = self.state
        if casscf_attr is None: casscf_attr = {}
        if fcisolver_attr is None: fcisolver_attr = {}
        ket, bra = _unpack_state (state)
        ci, ncas, nelecas = self.base.ci, self.base.ncas, self.base.nelecas
        # TODO: use fcisolver.fcisolvers in state-average mix case for this
        castm1, castm2 = direct_spin1.trans_rdm12 (ci[bra], ci[ket], ncas,
                                                   nelecas)
        castm1 = 0.5 * (castm1 + castm1.T)
        castm2 = 0.5 * (castm2 + castm2.transpose (1,0,3,2))
        fcisolver_attr['make_rdm12'] = lambda *args, **kwargs : (castm1, castm2)
        fcisolver_attr['make_rdm1'] = lambda *args, **kwargs : castm1
        fcisolver_attr['make_rdm2'] = lambda *args, **kwargs : castm2
        return sacasscf_grad.Gradients.make_fcasscf (self,
            state=ket, casscf_attr=casscf_attr, fcisolver_attr=fcisolver_attr)


    def get_wfn_response (self, atmlst=None, state=None, verbose=None, mo=None, ci=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        log = logger.new_logger (self, verbose)
        ket, bra = _unpack_state (state)
        fcasscf = self.make_fcasscf_nacs (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[ket]
        eris = fcasscf.ao2mo (mo)
        g_all_ket = gen_g_hop_active (fcasscf, mo, ci[ket], eris, verbose)[0]
        g_all = np.zeros (self.nlag)
        g_all[:self.ngorb] = g_all_ket[:self.ngorb]
        # The fun thing about the ci sector is that you swap them (&/2):
        # <I|[H,|A><I|-|I><A|]|J> = <A|H|J> = <J|[H,|A><J|-|J><A|]|J>/2
        # (It should be zero for converged SA-CASSCF anyway, though)
        g_ci_bra = 0.5 * g_all_ket[self.ngorb:]
        g_all_bra = gen_g_hop_active (fcasscf, mo, ci[bra], eris, verbose)[0]
        g_ci_ket = 0.5 * g_all_bra[self.ngorb:]
        # I have to make sure they don't talk to each other because the
        # preconditioner doesn't explore that space at all. Should I
        # instead solve at the init_guess step, like in MC-PDFT?
        # In practice it should all be zeros but how tightly does
        # everything have to be converged?
        ndet_ket = (self.na_states[ket], self.nb_states[ket])
        ndet_bra = (self.na_states[bra], self.nb_states[bra])
        if ndet_ket==ndet_bra:
            ket2bra = np.dot (ci[bra].conj ().ravel (), g_ci_ket)
            bra2ket = np.dot (ci[ket].conj ().ravel (), g_ci_bra)
            log.debug ('SA-CASSCF <bra|H|ket>,<ket|H|bra> check: %5.3g , %5.3g',
                       ket2bra, bra2ket)
            g_ci_ket -= ket2bra * ci[bra].ravel ()
            g_ci_bra -= bra2ket * ci[ket].ravel ()
        ndet_ket = ndet_ket[0]*ndet_ket[1]
        ndet_bra = ndet_bra[0]*ndet_bra[1]
        # No need to reshape or anything, just use the magic of repeated slicing
        offs_ket = (sum ([na * nb for na, nb in zip(
                         self.na_states[:ket], self.nb_states[:ket])])
                    if ket > 0 else 0)
        offs_bra = (sum ([na * nb for na, nb in zip(
                         self.na_states[:bra], self.nb_states[:bra])])
                    if ket > 0 else 0)
        g_all[self.ngorb:][offs_ket:][:ndet_ket] = g_ci_ket
        g_all[self.ngorb:][offs_bra:][:ndet_bra] = g_ci_bra
        return g_all


    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None,
                          ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        use_etfs = kwargs.get ('use_etfs', self.use_etfs)
        ket, bra = _unpack_state (state)
        fcasscf_grad = casscf_grad.Gradients (self.make_fcasscf_nacs (state))
        nac = grad_elec_active (fcasscf_grad, mo_coeff=mo, ci=ci[ket],
                                eris=eris, atmlst=atmlst, verbose=verbose)
        if not use_etfs: nac += self.nac_csf (
            mo_coeff=mo, ci=ci, state=state, mf_grad=mf_grad, atmlst=atmlst)
        return nac

    def nac_csf (self, mo_coeff=None, ci=None, state=None, mf_grad=None, atmlst=None):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if mf_grad is None: mf_grad = self.base._scf.nuc_grad_method ()
        nac = nac_csf (self, mo_coeff=mo_coeff, ci=ci, state=state,
                       mf_grad=mf_grad, atmlst=atmlst)
        ket, bra = _unpack_state (state)
        e_bra = self.base.e_states[bra]
        e_ket = self.base.e_states[ket]
        nac *= e_bra - e_ket
        return nac

    def kernel (self, *args, **kwargs):
        mult_ediff = kwargs.get ('mult_ediff', self.mult_ediff)
        state = kwargs.get ('state', self.state)
        assert len(state) == 2, "derivative couplings are defined between 2 states"
        if state[0] == state[1]:
            mol = kwargs.get('mol', self.mol)
            atmlst = kwargs.get('atmlst', range(mol.natm))
            return np.zeros((len(atmlst), 3))

        nac = sacasscf_grad.Gradients.kernel (self, *args, **kwargs)
        if not mult_ediff:
            ket, bra = _unpack_state (state)
            e_bra = self.base.e_states[bra]
            e_ket = self.base.e_states[ket]
            nac /= e_bra - e_ket
        return nac

if __name__=='__main__':
    from pyscf import gto, scf, mcscf
    from scipy import linalg
    mol = gto.M (atom = 'Li 0 0 0; H 0 0 1.5', basis='sto-3g',
                 output='sacasscf_nacs.log', verbose=lib.logger.INFO)
    mf = scf.RHF (mol).run ()
    mc = mcscf.CASSCF (mf, 2, 2).fix_spin_(ss=0).state_average ([0.5,0.5]).run (conv_tol=1e-10)
    openmolcas_energies = np.array ([-7.85629118, -7.72175252])
    print ("energies:",mc.e_states)
    print ("disagreement w openmolcas:", np.around (mc.e_states-openmolcas_energies, 8))
    mc_nacs = NonAdiabaticCouplings (mc)
    print ("no csf contr")
    nac_01 = mc_nacs.kernel (state=(0,1), use_etfs=True)
    nac_10 = mc_nacs.kernel (state=(1,0), use_etfs=True)
    nac_01_mult = mc_nacs.kernel (state=(0,1), use_etfs=True, mult_ediff=True)
    nac_10_mult = mc_nacs.kernel (state=(1,0), use_etfs=True, mult_ediff=True)
    print ("antisym")
    print (nac_01)
    print ("checking antisym:",linalg.norm(nac_01+nac_10))
    print ("sym")
    print (nac_01_mult)
    print ("checking sym:",linalg.norm(nac_01_mult-nac_10_mult))


    print ("incl csf contr")
    nac_01 = mc_nacs.kernel (state=(0,1), use_etfs=False)
    nac_10 = mc_nacs.kernel (state=(1,0), use_etfs=False)
    nac_01_mult = mc_nacs.kernel (state=(0,1), use_etfs=False, mult_ediff=True)
    nac_10_mult = mc_nacs.kernel (state=(1,0), use_etfs=False, mult_ediff=True)
    print ("antisym")
    print (nac_01)
    print ("checking antisym:",linalg.norm(nac_01+nac_10))
    print ("sym")
    print (nac_01_mult)
    print ("checking sym:",linalg.norm(nac_01_mult-nac_10_mult))

    print ("Check gradients")
    mc_grad = mc.nuc_grad_method ()
    de_0 = mc_grad.kernel (state=0)
    print (de_0)
    de_1 = mc_grad.kernel (state=1)
    print (de_1)
