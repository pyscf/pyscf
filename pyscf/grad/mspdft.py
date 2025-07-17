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
import math
import numpy as np
from scipy import linalg
from pyscf import mcpdft
from pyscf.grad import mcpdft as mcpdft_grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf.mcscf import mc1step, newton_casscf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import casscf as casscf_grad
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf import __config__
from itertools import product

# PySCF-Forge installation check
try:
    from pyscf.csf_fci.csf import CSFFCISolver
except ModuleNotFoundError:
    class CSFFCISolver:
        pass

CONV_TOL_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_conv_tol_diabatize', 1e-8)
SING_TOL_DIABATIZE = getattr(__config__, 'mcpdft_mspdft_sing_tol_diabatize', 1e-8)
SING_STEP_TOL = getattr(__config__, 'grad_mspdft_sing_step_tol', 2*math.pi)

def _unpack_state (state):
    if hasattr (state, '__len__'): return state[0], state[1]
    return state, state

# TODO: state-average-mix generalization ?
def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket):
    '''Compute <bra|O|ket> - sum_i <i|O|i>, where O is the 1- and 2-RDM
    operator product, and |bra> and |ket> are both states spanning the
    vector space of |i>, which are multi-determinantal many-electron
    states in an active space.

    Args:
        mc : object of class CASCI or CASSCF
            Only "ncas" and "nelecas" are used, to determine Hilbert
            of ci
        ci : ndarray or list of length (nroots)
            Contains CI vectors spanning a model space
        si_bra : ndarray of shape (nroots)
            Coefficients of ci elements for state |bra>
        si_ket : ndarray of shape (nroots)
            Coefficients of ci elements for state |ket>

    Returns:
        casdm1 : ndarray of shape [ncas,]*2
            Contains O = p'q case
        casdm2 : ndarray of shape [ncas,]*4
            Contains O = p'q'sr case
    '''
    ncas, nelecas = mc.ncas, mc.nelecas
    nroots = len (ci)
    ci_arr = np.asarray (ci)
    ci_bra = np.tensordot (si_bra, ci_arr, axes=1)
    ci_ket = np.tensordot (si_ket, ci_arr, axes=1)
    casdm1, casdm2 = direct_spin1.trans_rdm12 (ci_bra, ci_ket, ncas, nelecas)
    ddm1 = np.zeros ((nroots, ncas, ncas), dtype=casdm1.dtype)
    ddm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas), dtype=casdm1.dtype)
    for i in range (nroots):
        ddm1[i,...], ddm2[i,...] = direct_spin1.make_rdm12 (ci[i], ncas,
            nelecas)
    si_diag = si_bra * si_ket
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    casdm2 -= np.tensordot (si_diag, ddm2, axes=1)
    return casdm1, casdm2

# TODO: docstring?
def mspdft_heff_response (mc_grad, mo=None, ci=None,
        si_bra=None, si_ket=None, state=None,
        heff_mcscf=None, eris=None):
    '''Compute the orbital and intermediate-state rotation response
    vector in the context of an MS-PDFT gradient calculation '''
    mc = mc_grad.base
    if mo is None: mo = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    ket, bra = _unpack_state (state)
    if si_bra is None: si_bra = mc.si[:,bra]
    if si_ket is None: si_ket = mc.si[:,ket]
    if heff_mcscf is None: heff_mcscf = mc.heff_mcscf
    if eris is None: eris = mc.ao2mo (mo)
    nroots, ncore = mc_grad.nroots, mc.ncore
    moH = mo.conj ().T

    # Orbital rotation (no all-core DM terms allowed!)
    # (Factor of 2 is convention difference between mc1step and newton_casscf)
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    vnocore = eris.vhf_c.copy ()
    vnocore[:,:ncore] = -moH @ mc.get_hcore () @ mo[:,:ncore]
    with lib.temporary_env (eris, vhf_c=vnocore):
        g_orb = 2 * mc1step.gen_g_hop (mc, mo, 1, casdm1, casdm2, eris)[0]
    g_orb = mc.unpack_uniq_var (g_orb)

    # Intermediate state rotation (TODO: state-average-mix generalization)
    braH = np.dot (si_bra, heff_mcscf)
    Hket = np.dot (heff_mcscf, si_ket)
    si2 = si_bra * si_ket
    g_is  = np.multiply.outer (si_ket, braH)
    g_is += np.multiply.outer (si_bra, Hket)
    g_is -= 2 * si2[:,None] * heff_mcscf
    g_is -= g_is.T
    g_is = g_is[np.tril_indices (nroots, k=-1)]

    return g_orb, g_is

# TODO: docstring?
def mspdft_heff_HellmanFeynman (mc_grad, atmlst=None, mo=None, ci=None,
        si=None, si_bra=None, si_ket=None, state=None, eris=None, mf_grad=None,
        verbose=None, **kwargs):
    mc = mc_grad.base
    if atmlst is None: atmlst = mc_grad.atmlst
    if mo is None: mo = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = getattr (mc, 'si', None)
    if state is None: state = mc_grad.state
    ket, bra = _unpack_state (state)
    if si_bra is None: si_bra = si[:,bra]
    if si_ket is None: si_ket = si[:,ket]
    if eris is None: eris = mc.ao2mo (mo)
    if mf_grad is None: mf_grad = mc.get_rhf_base ().nuc_grad_method ()
    if verbose is None: verbose = mc_grad.verbose
    ncore = mc.ncore
    log = logger.new_logger (mc_grad, verbose)
    ci0 = np.zeros_like (ci[0])

    # CASSCF grad with effective RDMs
    t0 = (logger.process_clock (), logger.perf_counter ())
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    dm12 = lambda * args: (casdm1, casdm2)
    fcasscf = mc_grad.make_fcasscf (state=ket,
        fcisolver_attr={'make_rdm12' : dm12})
    # TODO: DFeri functionality
    # Perhaps by patching fcasscf.nuc_grad_method?
    fcasscf_grad = fcasscf.nuc_grad_method ()
    #fcasscf_grad = casscf_grad.Gradients (fcasscf)
    de = fcasscf_grad.kernel (mo_coeff=mo, ci=ci0, atmlst=atmlst, verbose=0)

    # subtract nuc-nuc and core-core (patching out simplified gfock terms)
    moH = mo.conj ().T
    f0 = (moH @ mc.get_hcore () @ mo) + eris.vhf_c
    mo_energy = f0.diagonal ().copy ()
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:ncore] = 2.0
    f0 *= mo_occ[None,:]
    dme0 = lambda * args: mo @ ((f0+f0.T)*.5) @ moH
    with lib.temporary_env (mf_grad, make_rdm1e=dme0, verbose=0):
        with lib.temporary_env (mf_grad.base, mo_coeff=mo, mo_occ=mo_occ):
            # Second level there should become unnecessary in future, if anyone
            # ever gets around to cleaning up pyscf.df.grad.rhf & pyscf.grad.rhf
            dde = mf_grad.kernel (mo_coeff=mo, mo_energy=mo_energy, mo_occ=mo_occ,
                atmlst=atmlst)
    de -= dde
    log.debug ('MS-PDFT gradient off-diagonal H-F terms:\n{}'.format (de))
    log.timer ('MS-PDFT gradient off-diagonal H-F terms', *t0)
    return de

def get_diabfns (obj):
    '''Interpret the name of the MS-PDFT method as a pair of functions
    which compute the derivatives of a particular objective function
    with respect to wave function parameters and geometry perturbations,
    excluding first and second derivatives wrt intermediate state
    rotations, which is handled by the energy-class version of this
    function.

    Args:
        obj : string
            Specify particular MS-PDFT method. Currently, only "CMS" is
            supported. Not case-sensitive.

    Returns:
        diab_response : callable
            Computes the orbital-rotation and CI-transfer sectors of the
            Hessian-vector product of the MS objective function for a
            vector of intermediate-state rotations
        diab_grad : callable
            Computes the gradient of the MS objective function wrt
            geometry perturbation
    '''
    if obj.upper () == 'CMS':
        from pyscf.grad.cmspdft import diab_response, diab_grad
    else:
        raise RuntimeError ('MS-PDFT type not supported')
    return diab_response, diab_grad

# TODO: docstring? especially considering the "si_bra," "si_ket"
# functionality??
# TODO: figure out how to log the gradients with the right method name!
class Gradients (mcpdft_grad.Gradients):

    # Preconditioner solves the IS problem; hence, get_init_guess rewrite is
    # unnecessary
    get_init_guess = sacasscf_grad.Gradients.get_init_guess
    project_Aop = sacasscf_grad.Gradients.project_Aop

    def __init__(self, mc):
        self.conv_rtol = 0
        def_tol0 = getattr (mc, 'conv_tol_grad', None)
        if def_tol0 is None:
            def_tol0 = np.sqrt (getattr (mc, 'conv_tol', 1e-7))
        def_tol1 = getattr (mc, 'conv_tol_diabatize',
                            CONV_TOL_DIABATIZE)
        self.conv_atol = min (def_tol0, def_tol1)
        self.sing_step_tol = SING_STEP_TOL
        mcpdft_grad.Gradients.__init__(self, mc)
        r, g = get_diabfns (self.base.diabatization)
        self._diab_response = r
        self._diab_grad = g
        self.nlag += self.nis

    @property
    def nis (self):
        return self.nroots * (self.nroots - 1) // 2

    def diab_response (self, Lis, **kwargs):
        return self._diab_response (self, Lis, **kwargs)
    def diab_grad (self, Lis, **kwargs):
        return self._diab_grad (self, Lis, **kwargs)

    def kernel (self, state=None, mo=None, ci=None, si=None, _freeze_is=False,
            **kwargs):
        '''Cache the Hamiltonian and effective Hamiltonian terms, and
        pass around the IS hessian

        eris, veff1, veff2, and d2f should be available to all top-level
        functions: get_wfn_response, get_Aop_Adiag, get_ham_response,
        and get_LdotJnuc

        freeze_is == True sets the is component of the response to zero
        for debugging purposes
        '''
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        nroots = self.nroots
        veff1 = []
        veff2 = []
        d2f = self.base.diabatizer (ci=ci)[2]
        for ix in range (nroots):
            v1, v2 = self.base.get_pdft_veff (mo, ci, incl_coul=True,
                paaa_only=True, state=ix)
            veff1.append (v1)
            veff2.append (v2)
        return mcpdft_grad.Gradients.kernel (self, state=state, mo=mo, ci=ci,
            si=si, d2f=d2f, veff1=veff1, veff2=veff2, _freeze_is=_freeze_is,
            **kwargs)

    def pack_uniq_var (self, xorb, xci, xis=None):
        x = sacasscf_grad.Gradients.pack_uniq_var (self, xorb, xci)
        if xis is not None: x = np.append (x, xis)
        return x

    def unpack_uniq_var (self, x):
        ngorb, nci, nis = self.ngorb, self.nci, self.nis
        x, xis = x[:ngorb+nci], x[ngorb+nci:]
        xorb, xci = sacasscf_grad.Gradients.unpack_uniq_var (self, x)
        if len (xis)==nis: return xorb, xci, xis
        return xorb, xci

    def _get_is_component (self, xci, ci=None, symm=-1):
        # TODO: state-average-mix
        if ci is None: ci = self.base.ci
        nroots = self.nroots
        xci = np.asarray (xci).reshape (nroots, -1)
        ci = np.asarray (ci).reshape (nroots, -1)
        xis = np.dot (xci.conj (), ci.T)
        if symm > -1: xis -= xis.T
        else:
            assert (np.amax (np.abs (xis + xis.T)) < 1e-8), '{}'.format (xis)
        return xis[np.tril_indices (nroots, k=-1)]

    def _separate_is_component (self, xci, ci=None, symm=-1):
        # TODO: state-average-mix
        is_list = isinstance (xci, list)
        is_tuple = isinstance (xci, tuple)
        if ci is None: ci = self.base.ci
        nroots = self.nroots
        ishape = np.asarray (xci).shape
        xci = np.asarray (xci).reshape (nroots, -1)
        xci = np.asarray (xci).reshape (nroots, -1)
        ci = np.asarray (ci).reshape (nroots, -1)
        xis = np.dot (xci.conj (), ci.T)
        xci -= np.dot (xis.conj (), ci)
        xci = xci.reshape (ishape)
        if is_list: xci = list (xci)
        elif is_tuple: xci = tuple (xci)
        if symm > -1: xis -= xis.T
        #else:
        #    assert (np.amax (np.abs (xis + xis.T)) < 1e-8), '{}'.format (xis)
        xis = xis[np.tril_indices (nroots, k=-1)]
        return xci, xis


    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, si=None, eris=None, veff1=None, veff2=None,
            _freeze_is=False, d2f=None, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if state is None: state = self.state
        ket, bra = _unpack_state (state)
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        if d2f is None: d2f = self.base.diabatizer (ci=ci)[2]
        log = lib.logger.new_logger (self, self.verbose)
        si_diag = si_bra * si_ket

        # Diagonal: PDFT component
        nlag = self.nlag-self.nis
        g_all_pdft = np.zeros (nlag)
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            g_i = mcpdft_grad.Gradients.get_wfn_response (self,
                state=i, mo=mo, ci=ci, veff1=v1, veff2=v2, nlag=nlag, **kwargs)
            g_all_pdft += amp * g_i
            if self.verbose >= lib.logger.DEBUG:
                g_orb, g_ci = self.unpack_uniq_var (g_i)
                g_ci, g_is = self._separate_is_component (g_ci, ci=ci, symm=0)
                log.debug ('g_is pdft state {} component:\n{} * {}'.format (i,
                    amp, g_is))

        # DEBUG
        g_orb_pdft, g_ci = self.unpack_uniq_var (g_all_pdft)
        g_ci, g_is_pdft = self._separate_is_component (g_ci, ci=ci, symm=0)

        # Off-diagonal: heff component
        g_orb_heff, g_is_heff = mspdft_heff_response (self, mo=mo, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris)

        log.debug ('g_is pdft total component:\n{}'.format (g_is_pdft))
        log.debug ('g_is heff component:\n{}'.format (g_is_heff))

        # Combine
        g_orb = g_orb_pdft + g_orb_heff
        g_is = g_is_pdft + g_is_heff
        if _freeze_is: g_is[:] = 0.0
        g_all = self.pack_uniq_var (g_orb, g_ci, g_is)

        # DEBUG
        d2f_evals, d2f_evecs = linalg.eigh (d2f)
        g_is_modes = np.dot (g_is, d2f_evecs)
        g_is_pdft = np.dot (g_is_pdft, d2f_evecs)
        g_is_heff = np.dot (g_is_heff, d2f_evecs)
        log.debug ("IS sector Lagrange multiplier solution:")
        for i, (denom, num) in enumerate (zip (d2f_evals, g_is_modes)):
            log.debug ('%d %e / %e = %e (%e %e)', i, -num, denom, -num/denom,
                       g_is_pdft[i], g_is_heff[i])

        return g_all

    def get_Aop_Adiag (self, verbose=None, mo=None, ci=None, eris=None,
            level_shift=None, d2f=None, **kwargs):
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        if d2f is None: d2f = self.base.diabatizer (ci=ci)[2]
        ham_od = self.base.get_heff_offdiag ()
        ham_od += ham_od.T # This corresponds to the arbitrary newton_casscf*2
        fcasscf = self.make_fcasscf_sa ()
        hop, Adiag = newton_casscf.gen_g_hop (fcasscf, mo, ci, eris,
            verbose)[2:]
        ngorb, nci = self.ngorb, self.nci
        # TODO: cacheing diab_response? or an x=0 branch?
        def Aop (x):
            x_v, x_is = x[:ngorb+nci], x[ngorb+nci:]
            Ax_v = hop (x_v) + self.diab_response (x_is, mo=mo, ci=ci,
                eris=eris)
            x_c = self.unpack_uniq_var (x_v)[1]
            Ax_is = np.dot (d2f, x_is)
            Ax_o, Ax_c = self.unpack_uniq_var (Ax_v)
            Ax_c, Ax_is2 = self._separate_is_component (Ax_c)
            Ax_c_od = list (np.tensordot (-ham_od, np.stack (x_c, axis=0),
                axes=1))
            Ax_c = [a1 + (w*a2) for a1, a2, w in zip (Ax_c, Ax_c_od,
                self.base.weights)]
            return self.pack_uniq_var (Ax_o, Ax_c, Ax_is)
        return Aop, Adiag

    def get_lagrange_precond (self, Adiag, level_shift=None, ci=None, d2f=None,
            **kwargs):
        if level_shift is None: level_shift = self.level_shift
        if ci is None: ci = self.base.ci
        if d2f is None: d2f = self.base.diabatizer (ci=ci)[2]
        return MSPDFTLagPrec (Adiag=Adiag, level_shift=level_shift, ci=ci,
            d2f=d2f, grad_method=self)

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, si=None, eris=None, veff1=None, veff2=None, mf_grad=None,
            atmlst=None, verbose=None, **kwargs):
        '''write mspdft heff Hellmann-Feynman calculator; sum over
        diagonal PDFT Hellmann-Feynman terms
        '''
        if atmlst is None: atmlst = self.atmlst
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if state is None: state = self.state
        ket, bra = _unpack_state (state)
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        if mf_grad is None: mf_grad = self.base.get_rhf_base ().nuc_grad_method ()
        if verbose is None: verbose = self.verbose
        si_diag = si_bra * si_ket
        log = logger.new_logger (self, verbose)

        # Fix messed-up counting of the nuclear part
        de_nuc = mf_grad.grad_nuc (self.mol, atmlst)
        log.debug ('MS-PDFT gradient n-n terms:\n{}'.format (de_nuc))
        de = si_diag.sum () * de_nuc.copy ()

        # Diagonal: PDFT component
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            de_i = mcpdft_grad.Gradients.get_ham_response (self, state=i,
                mo=mo, ci=ci, veff1=v1, veff2=v2, eris=eris, mf_grad=mf_grad,
                verbose=0, **kwargs) - de_nuc
            log.debug ('MS-PDFT gradient int-state {} EPDFT terms:\n{}'.format
                (i, de_i))
            log.debug ('Factor for these terms: {}'.format (amp))
            de += amp * de_i
        log.debug ('MS-PDFT gradient diag H-F terms:\n{}'.format (de))

        # Off-diagonal: heff component
        de_o = mspdft_heff_HellmanFeynman (self, mo_coeff=mo, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris, state=state,
            mf_grad=mf_grad, **kwargs)
        log.debug ('MS-PDFT gradient offdiag H-F terms:\n{}'.format (de_o))
        de += de_o

        return de

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None, mo=None,
            ci=None, eris=None, mf_grad=None, d2f=None, **kwargs):
        ''' Add the IS component '''
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris

        ngorb, nci = self.ngorb, self.nci
        Lvec_v, Lvec_is = Lvec[:ngorb+nci], Lvec[ngorb+nci:]

        # Double-check Lvec_v sanity
        Lvec_orb, Lvec_ci = self.unpack_uniq_var (Lvec_v)
        Lvec_is2 = self._get_is_component (Lvec_ci, symm=0)
        assert (np.amax (np.abs (Lvec_is2)) < 1e-8), '{} {}'.format (Lvec_is,
            Lvec_is2)

        # Orbital and CI components
        de_Lv = sacasscf_grad.Gradients.get_LdotJnuc (self, Lvec_v,
            atmlst=atmlst, verbose=verbose, ci=ci, eris=eris, mf_grad=mf_grad,
            **kwargs)

        # SI component
        t0 = (logger.process_clock(), logger.perf_counter())
        de_Lis = self.diab_grad (Lvec_is, atmlst=atmlst, mf_grad=mf_grad,
            eris=eris, mo=mo, ci=ci, **kwargs)
        logger.info (self,
            '--------------- %s gradient Lagrange IS response ---------------',
            self.base.__class__.__name__)
        if verbose >= logger.INFO:
            rhf_grad._write(self, self.mol, de_Lis, atmlst)
        logger.info (self,
            '----------------------------------------------------------------')
        t0 = logger.timer (self, '{} gradient Lagrange IS response'.format (
            self.base.__class__.__name__), *t0)
        return de_Lv + de_Lis

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        # TODO: state-average-mix
        log = logger.new_logger (self, self.verbose)
        if isinstance (self.base.fcisolver, CSFFCISolver):
            transf = self.base.fcisolver.transformer
            def _debug_csfs (xci, tag):
                xci_csf = transf.vec_det2csf (xci, normalize=False)
                xci_p = transf.vec_csf2det (xci_csf, normalize=False)
                xci_bs_norm = linalg.norm (np.concatenate (
                    [x.ravel () - y.ravel () for x, y in zip (xci_p, xci)]))
                log.debug ('Broken-spin |{}| = {}'.format (tag, xci_bs_norm))
        else:
            def _debug_csfs (xci, tag):
                pass
        def my_call (x):
            itvec[0] += 1
            geff = geff_op (x)
            deltax = x - Lvec_last
            gorb, gci, gis = self.unpack_uniq_var (geff)
            deltaorb, deltaci, deltais = self.unpack_uniq_var (deltax)
            gci_norm = linalg.norm (np.asarray (gci).ravel ())
            deltaci_norm = linalg.norm (np.asarray (deltaci).ravel ())
            logger.info(self, ('Lagrange optimization iteration {}, |gorb| = '
                '{}, |gci| = {}, |gis| = {} |dLorb| = {}, |dLci| = {}, |dLis|'
                ' = {}').format (itvec[0], linalg.norm (gorb),
                gci_norm, linalg.norm (gis), linalg.norm (deltaorb),
                deltaci_norm, linalg.norm (deltais)))
            _debug_csfs (gci, 'gci')
            _debug_csfs (deltaci, 'dLci')
            Lvec_last[:] = x[:]
        return my_call

    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, state=None, mo=None,
            ci=None, d2f=None, verbose=None, eris=None, **kwargs):
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None: eris = self.base.ao2mo (mo)
        if verbose is None: verbose = self.verbose
        nroots = self.nroots
        log = logger.new_logger (self, verbose)
        if isinstance (self.base.fcisolver, CSFFCISolver):
            transf = self.base.fcisolver.transformer
            def _debug_csfs (xci, label, normalize=False):
                strs, vecs = transf.printable_largest_csf (np.asarray (xci),
                    10, isdet=True, normalize=normalize, order='C')
                log.debug ('Leading CSFs for %s', label)
                for iroot in range (nroots):
                    log.debug (' Root %d', iroot)
                    for s, v in zip (strs[iroot], vecs[iroot]):
                        log.debug ('  |%s>: %s', s, v)
        else:
            def _debug_csfs (*args, **kwargs):
                pass
        def _debug_cispace (xci, label):
            log.debug ('%s', label)
            xci_norm = [np.dot (c.ravel (), c.ravel ()) for c in xci]
            try:
                xci_ss = self.base.fcisolver.states_spin_square (xci,
                    self.base.ncas, self.base.nelecas)[0]
            except AttributeError:
                from pyscf.fci.direct_spin1 import _unpack_nelec, spin_square
                nelec = sum (_unpack_nelec(self.base.nelecas))
                xci_ss = [spin_square (x, self.base.ncas,
                          ((nelec+m)//2,(nelec-m)//2))[0]
                          for x, m in zip (xci, self.spin_states)]

            xci_ss = [x / max (y, 1e-8) for x, y in zip (xci_ss, xci_norm)]
            xci_multip = [np.sqrt (x+.25) - .5 for x in xci_ss]
            xci_norm = np.sqrt (xci_norm)
            for ix, (norm, ss, multip) in enumerate (zip (xci_norm, xci_ss,
                    xci_multip)):
                log.debug ((' State {} norm = {:.7e} ; <S^2> = {:.7f} ; 2S+1'
                            ' = {:.7f}').format (ix, norm, ss, multip))
            ovlp = np.zeros ((nroots, nroots), dtype=xci[0].dtype)
            for i, j in product (range (nroots), repeat=2):
                if self.spin_states[i] != self.spin_states[j]: continue
                ovlp[i,j] = np.dot (xci[i].ravel (), ci[j].ravel ())
            log.debug (' Overlap matrix with CI array:')
            fmt_str = '  ' + ' '.join (['{:8.1e}',]*nroots)
            for row in ovlp: log.debug (fmt_str.format (*row))
        _debug_csfs (ci, 'CI vector', normalize=True)
        borb, bci, bis = self.unpack_uniq_var (bvec)
        log.debug ('Orbital rotation gradient (b) norm = {:.6e}'.format (
            linalg.norm (borb)))
        _debug_cispace (bci, 'CI gradient (b)')
        _debug_csfs (bci, 'CI gradient (b)')
        Aorb, Aci = self.unpack_uniq_var (Adiag)
        log.debug ('Orbital rotation Hessian (A) diagonal norm = {:.7e}'.format
            (linalg.norm (Aorb)))
        _debug_cispace (Aci, 'CI Hessian (A) diagonal')
        _debug_csfs (Aci, 'CI Hessian (A) diagonal')
        Lorb, Lci, Lis = self.unpack_uniq_var (Lvec)
        log.debug ('Orbital rotation Lagrange vector (x) norm = {:.7e}'.format
            (linalg.norm (Lorb)))
        _debug_cispace (Lci, 'CI Lagrange (x) vector')
        _debug_csfs (Lci, 'CI Lagrange (x) vector')
        log.debug ('{} Constraint Jacobian (A):'.format (self.base.__class__.__name__))
        fmt = ' ' + ' '.join (['{:12.5e}' for i in range (self.nis)])
        for row in d2f: log.debug (fmt.format (*row))
        log.debug (' {:>12s} {:>12s}'.format ('Gradient (b)', 'Vector (x)'))
        for g, v in zip (bis, Lis):
            log.debug (' {:12.5e} {:12.5e}'.format (g, v))

class MSPDFTLagPrec (sacasscf_grad.SACASLagPrec):
    ''' Solve IS part exactly, then do everything else the same '''

    def __init__(self, Adiag=None, level_shift=None, ci=None, grad_method=None,
            d2f=None, **kwargs):
        sacasscf_grad.SACASLagPrec.__init__(self, Adiag=Adiag,
            level_shift=level_shift, ci=ci, grad_method=grad_method)
        self.grad_method = grad_method
        self.sing_tol = getattr (grad_method.base, 'sing_tol_diabatize',
                                 SING_TOL_DIABATIZE)
        self.sing_step_tol = getattr (grad_method.base, 'sing_step_tol',
                                      SING_STEP_TOL)
        self.sing_warned = False
        self.log = logger.new_logger (self.grad_method,
            self.grad_method.verbose)
        self._init_d2f (d2f=d2f, **kwargs)
        self.verbose = self.grad_method.verbose

    def _init_d2f (self, d2f=None, **kwargs):
        self.d2f=d2f
        self.d2f_evals, self.d2f_evecs = linalg.eigh (d2f)
        idx_sing = np.abs (self.d2f_evals) < self.sing_tol
        self.log.debug ('IS component Hessian eigenvalues: {}'.format (
            self.d2f_evals))
        if np.any (idx_sing): self.do_sing_warn ()
        self.d2f_evals = self.d2f_evals[~idx_sing]
        self.d2f_evecs = self.d2f_evecs[:,~idx_sing]

    def unpack_uniq_var (self, x):
        return self.grad_method.unpack_uniq_var (x)

    def pack_uniq_var (self, x0, x1, x2=None):
        return self.grad_method.pack_uniq_var (x0, x1, x2)

    def __call__(self, x):
        xorb, xci, xis = self.unpack_uniq_var (x)
        Mxorb = self.orb_prec (xorb)
        Mxci = self.ci_prec (xci)
        Mxis = self.is_prec (xis)
        return self.pack_uniq_var (Mxorb, Mxci, Mxis)

    def is_prec (self, xis):
        xis = np.dot (xis, self.d2f_evecs)
        Mxis = xis/self.d2f_evals
        idx_sing = (np.abs (Mxis) >= self.sing_step_tol)
        if np.any (idx_sing): self.do_sing_warn ()
        Mxis[idx_sing] = 0
        Mxis = np.dot (self.d2f_evecs, Mxis)
        return Mxis

    def do_sing_warn (self):
        if self.sing_warned: return
        self.log.warn ('Model-space frame-rotation Hessian is singular! '
                        'Response equations may not be solvable to arbitrary '
                        'precision!')
        self.sing_warned = True



if __name__ == '__main__':
    # Test mspdft_heff_response and mspdft_heff_HellmannFeynman by trying to
    # reproduce SA-CASSCF derivatives in an arbitrary basis
    import math
    from pyscf import scf, gto, mcscf
    from pyscf.fci import csf_solver
    xyz = '''O  0.00000000   0.08111156   0.00000000
             H  0.78620605   0.66349738   0.00000000
             H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='mspdft.log',
        verbose=lib.logger.DEBUG)
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).set (fcisolver = csf_solver (mol, 1))
    mc = mc.multi_state ([1.0/3,]*3, 'cms').run ()
    mc_grad = Gradients (mc)
    de = np.stack ([mc_grad.kernel (state=i) for i in range (3)], axis=0)

