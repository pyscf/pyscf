#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

import numpy as np
from pyscf import mcpdft
from pyscf.grad import mspdft as mspdft_grad
from pyscf import lib
from pyscf.fci import direct_spin1
from pyscf.nac import sacasscf as sacasscf_nacs
from functools import reduce

_unpack_state = mspdft_grad._unpack_state
_nac_csf = sacasscf_nacs._nac_csf

def nac_model (mc_grad, mo_coeff=None, ci=None, si_bra=None, si_ket=None,
               mf_grad=None, atmlst=None):
    '''Compute the "model-state contribution" to the MS-PDFT NAC'''
    mc = mc_grad.base
    mol = mc.mol
    ci_bra = np.tensordot (si_bra, ci, axes=1)
    ci_ket = np.tensordot (si_ket, ci, axes=1)
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    castm1 = direct_spin1.trans_rdm1 (ci_bra, ci_ket, ncas, nelecas)
    # if PySCF commentary is to be trusted, trans_rdm1[p,q] is
    # <bra|q'p|ket>. I want <bra|p'q - q'p|ket>.
    castm1 = castm1.conj ().T - castm1
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    tm1 = reduce (np.dot, (mo_cas, castm1, mo_cas.conj ().T))
    return _nac_csf (mol, mf_grad, tm1, atmlst)

class NonAdiabaticCouplings (mspdft_grad.Gradients):
    '''MS-PDFT non-adiabatic couplings (NACs) between states

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
        to omitting the ``model state contribution''.
    '''

    def __init__(self, mc, state=None, mult_ediff=False, use_etfs=False):
        self.mult_ediff = mult_ediff
        self.use_etfs = use_etfs
        self.state = state
        mspdft_grad.Gradients.__init__(self, mc)

    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, si=None,
                          verbose=None, **kwargs):
        g_all = mspdft_grad.Gradients.get_wfn_response (
            self, si_bra=si_bra, si_ket=si_ket, state=state, si=si,
            verbose=verbose, **kwargs
        )
        g_orb, g_ci, g_is = self.unpack_uniq_var (g_all)
        if state is None: state = self.state
        ket, bra = _unpack_state (state)
        if si is None: si = self.base.si
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        nroots = self.nroots
        log = lib.logger.new_logger (self, verbose)
        g_model = np.multiply.outer (si_bra.conj (), si_ket)
        g_model -= g_model.T
        g_model *= self.base.e_states[bra]-self.base.e_states[ket]
        g_model = g_model[np.tril_indices (nroots, k=-1)]
        log.debug (f"NACs g_is additional component:\n{g_model}")
        return self.pack_uniq_var (g_orb, g_ci, g_is+g_model)

    def get_ham_response (self, **kwargs):
        nac = mspdft_grad.Gradients.get_ham_response (self, **kwargs)
        use_etfs = kwargs.get ('use_etfs', self.use_etfs)
        if not use_etfs:
            verbose = kwargs.get ('verbose', self.verbose)
            log = lib.logger.new_logger (self, verbose)
            nac_model = self.nac_model (**kwargs)
            log.info (f'NACs model-state contribution:\n{nac_model}')
            nac += nac_model
        return nac

    def nac_model (self, mo_coeff=None, ci=None, si=None, si_bra=None,
                   si_ket=None, state=None, mf_grad=None, atmlst=None,
                   **kwargs):
        if state is None: state = self.state
        ket, bra = _unpack_state (state)
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        if mf_grad is None: mf_grad = self.base.get_rhf_base ().nuc_grad_method ()
        if atmlst is None: atmlst = self.atmlst
        nac = nac_model (self, mo_coeff=mo_coeff, ci=ci, si_bra=si_bra,
                         si_ket=si_ket, mf_grad=mf_grad, atmlst=atmlst)
        e_bra = self.base.e_states[bra]
        e_ket = self.base.e_states[ket]
        nac *= e_bra - e_ket
        return nac

    def kernel (self, *args, **kwargs):
        mult_ediff = kwargs.get ('mult_ediff', self.mult_ediff)
        state = kwargs.get ('state', self.state)
        nac = mspdft_grad.Gradients.kernel (self, *args, **kwargs)
        if not mult_ediff:
            ket, bra = _unpack_state (state)
            e_bra = self.base.e_states[bra]
            e_ket = self.base.e_states[ket]
            nac /= e_bra - e_ket
        return nac
