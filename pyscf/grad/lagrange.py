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

import numpy as np
from scipy import linalg, optimize
from scipy.sparse import linalg as sparse_linalg
from pyscf import lib, __config__
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.soscf import ciah

default_level_shift = getattr(__config__, 'grad_lagrange_Gradients_level_shift', 1e-8)
default_conv_atol = getattr (__config__, 'grad_lagrange_Gradients_conv_atol', 1e-12)
default_conv_rtol = getattr (__config__, 'grad_lagrange_Gradients_conv_rtol', 1e-7)
default_max_cycle = getattr (__config__, 'grad_lagrange_Gradients_max_cycle', 50)

class Gradients (rhf_grad.GradientsMixin):
    r''' Dummy parent class for calculating analytical nuclear gradients using the technique of
    Lagrange multipliers:
    L = E + \sum_i z_i L_i
    dE/dx = \partial L/\partial x iff all L_i = 0 for the given wave function
    I.E., the Lagrange multipliers L_i cancel the direct dependence of the wave function on the
    nuclear coordinates and allow the Hellmann-Feynman theorem to be used for some non-variational
    methods. '''

    ####################### Child classes MUST overwrite the methods below ########################

    def get_wfn_response (self, **kwargs):
        ''' Return first derivative of the energy wrt wave function parameters conjugate to the
            Lagrange multipliers. Used to calculate the value of the Lagrange multipliers. '''
        return np.zeros(self.nlag)

    def get_Aop_Adiag (self, **kwargs):
        ''' Return a function calculating Lvec . J_wfn, where J_wfn is the Jacobian of the Lagrange
            cofactors (e.g., in state-averaged CASSCF, the Hessian of the state-averaged energy wrt
            wfn parameters) along with the diagonal of the Jacobian. '''
        def Aop (Lvec):
            return np.zeros(self.nlag)
        Adiag = np.zeros(self.nlag)
        return Aop, Adiag

    def get_ham_response (self, **kwargs):
        ''' Return expectation values <dH/dx> where x is nuclear displacement.
        I.E., the gradient if the method were variational.
        '''
        return np.zeros((self.mol.natm, 3))

    def get_LdotJnuc (self, Lvec, **kwargs):
        ''' Return Lvec . J_nuc, where J_nuc is the Jacobian of the Lagrange cofactors wrt nuclear
        displacement. This is the second term of the final gradient expectation value.
        '''
        return np.zeros((self.mol.natm, 3))

    ####################### Child classes SHOULD overwrite the methods below ######################

    def __init__(self, method, nlag):
        self._conv = False
        self.Lvec = None
        self.nlag = nlag
        self.level_shift = default_level_shift
        self.conv_atol = default_conv_atol
        self.conv_rtol = default_conv_rtol
        self.max_cycle = default_max_cycle
        rhf_grad.GradientsMixin.__init__(self, method)

    def debug_lagrange (self, Lvec, bvec, Aop, Adiag, **kwargs):
        logger.debug (self, "{} gradient Lagrange factor debugging not enabled".format (
            self.base.__class__.__name__))

    def get_lagrange_callback (self, Lvec_last, itvec, geff_op):
        def my_call (x):
            itvec[0] += 1
            logger.info (self, 'Lagrange optimization iteration %d, |geff| = %.8g, |dLvec| = %.8g',
                         itvec[0], linalg.norm (geff_op (x)), linalg.norm (x - Lvec_last))
            Lvec_last[:] = x[:]
        return my_call

    def get_lagrange_precond (self, Adiag, level_shift=None, **kwargs):
        if level_shift is None: level_shift = self.level_shift
        return LagPrec (Adiag=Adiag, level_shift=level_shift, **kwargs)

    def get_init_guess (self, bvec, Adiag, Aop, precond):
        return precond (-bvec)

    @property
    def converged (self):
        return self._conv and getattr (self.base, 'converged', True)
    @converged.setter
    def converged (self, x):
        self._conv = x
        return self._conv and getattr (self.base, 'converged', True)

    ####################### Child classes SHOULD NOT overwrite the methods below ##################

    def solve_lagrange (self, Lvec_guess=None, level_shift=None, **kwargs):
        bvec = self.get_wfn_response (**kwargs)
        Aop, Adiag = self.get_Aop_Adiag (**kwargs)
        def my_geff (x):
            return bvec + Aop (x)
        Lvec_last = np.zeros_like (bvec)
        def my_Lvec_last ():
            return Lvec_last
        precond = self.get_lagrange_precond (Adiag, level_shift=level_shift, **kwargs)
        it = np.asarray ([0])
        logger.debug(self, 'Lagrange multiplier determination intial gradient norm: %.8g',
                     linalg.norm(bvec))
        my_call = self.get_lagrange_callback (Lvec_last, it, my_geff)
        Aop_obj = sparse_linalg.LinearOperator ((self.nlag,self.nlag), matvec=Aop,
                                                dtype=bvec.dtype)
        prec_obj = sparse_linalg.LinearOperator ((self.nlag,self.nlag), matvec=precond,
                                                 dtype=bvec.dtype)
        x0_guess = self.get_init_guess (bvec, Adiag, Aop, precond)
        Lvec, info_int = sparse_linalg.cg(Aop_obj, -bvec, x0=x0_guess,
                                          tol=self.conv_rtol, atol=self.conv_atol,
                                          maxiter=self.max_cycle, callback=my_call, M=prec_obj)
        logger.info (self, ('Lagrange multiplier determination {} after {} iterations\n'
                            '   |geff| = {}, |Lvec| = {}').format (
                                'converged' if info_int == 0 else 'not converged',
                                it[0], linalg.norm (my_geff (Lvec)), linalg.norm (Lvec)))
        if info_int < 0:
            logger.info (self, 'Lagrange multiplier determination error code {}'.format (info_int))
        return (info_int==0), Lvec, bvec, Aop, Adiag

    def kernel (self, level_shift=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.new_logger(self, self.verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']
        #self.natm = len (self.atmlst)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.converged, self.Lvec, bvec, Aop, Adiag = self.solve_lagrange (
            level_shift=level_shift, **kwargs)
        if self.verbose >= logger.INFO:
            self.debug_lagrange (self.Lvec, bvec, Aop, Adiag, **kwargs)
            cput1 = logger.timer (self, 'Lagrange gradient multiplier solution', *cput0)

        ham_response = self.get_ham_response (**kwargs)
        if self.verbose >= logger.INFO:
            logger.info(self, '--------------- %s gradient Hamiltonian response ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, ham_response, self.atmlst)
            logger.info(self, '----------------------------------------------')
            cput1 = logger.timer (self, 'Lagrange gradient Hellmann-Feynman determination', *cput1)

        LdotJnuc = self.get_LdotJnuc (self.Lvec, **kwargs)
        if self.verbose >= logger.INFO:
            logger.info(self, '--------------- %s gradient Lagrange response ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, LdotJnuc, self.atmlst)
            logger.info(self, '----------------------------------------------')
            cput1 = logger.timer (self, 'Lagrange gradient Jacobian', *cput1)

        self.de = ham_response + LdotJnuc
        log.timer('Lagrange gradients', *cput0)
        self._finalize()
        return self.de

class LagPrec (object):
    ''' A callable preconditioner for solving the Lagrange equations.
        Default is 1/(Adiagd+level_shift)
    '''

    def __init__(self, Adiag=None, level_shift=None, **kwargs):
        self.Adiag = Adiag
        self.level_shift = level_shift

    def __call__(self, x):
        Adiagd = self.Adiag + self.level_shift
        Adiagd[abs(Adiagd)<1e-8] = 1e-8
        x /= Adiagd
        return x

