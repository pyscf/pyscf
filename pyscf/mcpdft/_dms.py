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
# Common density-matrix manipulations

import numpy as np
from pyscf import lib
from pyscf.mcscf.addons import StateAverageFCISolver
from pyscf.mcscf.addons import StateAverageMixFCISolver
from scipy import linalg

# DMRG solvers require special handling but dmrgscf is not always installed
try:
    from pyscf import dmrgscf
    DMRGCI = dmrgscf.DMRGCI
except ImportError:
    class DMRGCI :
        pass

def _get_fcisolver (mc, ci, state=0):
    '''Find the appropriate FCI solver, CI vector, and nelecas tuple to
    build single-state reduced density matrices. If state_average or
    state_average_mix is involved this takes a bit of work.

    The better solution, of course, is to edit StateAverage*FCI classes
    to have quick density-matrices-of-one-state API...
    '''
    nelecas = mc.nelecas
    nroots = getattr (mc.fcisolver, 'nroots', 1)
    fcisolver = mc.fcisolver
    solver_state_index = state
    if nroots>1: ci = ci[state]
    if isinstance (mc.fcisolver, StateAverageMixFCISolver):
        p0 = 0
        fcisolver = None
        for s in mc.fcisolver.fcisolvers:
            p1 = p0 + s.nroots
            if p0 <= state and state < p1:
                fcisolver = s
                nelecas = mc.fcisolver._get_nelec (s, nelecas)
                solver_state_index = state - p0
                break
            p0 = p1
        if fcisolver is None:
            raise RuntimeError ("Can't find FCI solver for state", state)
    elif isinstance (mc.fcisolver, StateAverageFCISolver):
        fcisolver = fcisolver.undo_state_average ()
    if isinstance (fcisolver, DMRGCI):
        ci = solver_state_index # DMRGCI takes state index in place of ci vector
    return fcisolver, ci, nelecas

def make_one_casdm1s (mc, ci, state=0):
    '''
    Construct the spin-separated active-space one-body reduced density
    matrix for a single state. This API is not consistently available
    in the StateAverageFCISolver functions without wasted effort (i.e.,
    without constructing the dms for all states and then discarding most
    of them
    '''
    ncas = mc.ncas
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    return fcisolver.make_rdm1s (ci, ncas, nelecas)

def make_one_casdm2 (mc, ci, state=0):
    '''
    Construct the spin-summed active-space two-body reduced density
    matrix for a single state. This API is not consistently available
    in the StateAverageFCISolver functions without wasted effort (i.e.,
    without constructing the dms for all states and then discarding most
    of them
    '''
    ncas = mc.ncas
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    try:
        casdm2 = fcisolver.make_rdm2 (ci, ncas, nelecas)
    except AttributeError:
        # Hail Mary: maybe the fcisolver class only has make_rdm12
        # but not make_rdm2 implemented?
        _, casdm2 = fcisolver.make_rdm12 (ci, ncas, nelecas)
    return casdm2


def dm2_cumulant (dm2, dm1s):
    '''
    Evaluate the spin-summed two-body cumulant reduced density
    matrix:

    cm2[p,q,r,s] = (dm2[p,q,r,s] - dm1[p,q]*dm1[r,s]
                       + dm1s[0][p,s]*dm1s[0][r,q]
                       + dm1s[1][p,s]*dm1s[1][r,q])

    Args:
        dm2 : ndarray of shape [norb,]*4
            Contains spin-summed 2-RDMs
        dm1s : ndarray (or compatible) of overall shape [2,norb,norb]
            Contains spin-separated 1-RDMs

    Returns:
        cm2 : ndarray of shape [norb,]*4
    '''

    dm1s = np.asarray (dm1s)
    if len (dm1s.shape) < 3:
        dm1 = dm1s.copy ()
        dm1s = dm1 / 2
        dm1s = np.stack ((dm1s, dm1s), axis=0)
    else:
        dm1 = dm1s[0] + dm1s[1]
    cm2  = dm2.copy ()
    cm2 -= np.multiply.outer (dm1, dm1)
    cm2 += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0, 3, 2, 1)
    cm2 += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0, 3, 2, 1)
    return cm2

def dm2s_cumulant (dm2s, dm1s):
    '''Evaluate the spin-summed two-body cumulant reduced density
    matrix:

    cm2s[0][p,q,r,s] = (dm2s[0][p,q,r,s] - dm1s[0][p,q]*dm1s[0][r,s]
                       + dm1s[0][p,s]*dm1s[0][r,q])
    cm2s[1][p,q,r,s] = (dm2s[1][p,q,r,s] - dm1s[0][p,q]*dm1s[1][r,s])
    cm2s[2][p,q,r,s] = (dm2s[2][p,q,r,s] - dm1s[1][p,q]*dm1s[1][r,s]
                       + dm1s[1][p,s]*dm1s[1][r,q])

    Args:
        dm2s : ndarray of shape [norb,]*4
            Contains spin-separated 2-RDMs
        dm1s : ndarray (or compatible) of overall shape [2,norb,norb]
            Contains spin-separated 1-RDMs

    Returns:
        cm2s : (cm2s[0], cms2[1], cm2s[2])
            ndarrays of shape [norb,]*4; contain spin components
            aa, ab, bb respectively
    '''
    dm1s = np.asarray (dm1s)
    if len (dm1s.shape) < 3:
        dm1 = dm1s.copy ()
        dm1s = dm1 / 2
        dm1s = np.stack ((dm1s, dm1s), axis=0)
    #cm2  = dm2 - np.einsum ('pq,rs->pqrs', dm1, dm1)
    #cm2 +=    0.5 * np.einsum ('ps,rq->pqrs', dm1, dm1)
    cm2s = [i.copy () for i in dm2s]
    cm2s[0] -= np.multiply.outer (dm1s[0], dm1s[0])
    cm2s[1] -= np.multiply.outer (dm1s[0], dm1s[1])
    cm2s[2] -= np.multiply.outer (dm1s[1], dm1s[1])
    cm2s[0] += np.multiply.outer (dm1s[0], dm1s[0]).transpose (0, 3, 2, 1)
    cm2s[2] += np.multiply.outer (dm1s[1], dm1s[1]).transpose (0, 3, 2, 1)
    return tuple (cm2s)

def casdm1s_to_dm1s (mc, casdm1s, mo_coeff=None, ncore=None, ncas=None):
    '''Generate AO-basis spin-separated 1-RDM from active space part.
    This is necessary because the StateAverageMCSCFSolver class doesn't
    have API for getting the AO-basis density matrix of a single state.

    Args:
        mc : object of CASCI or CASSCF class
        casdm1s : ndarray or compatible of shape (2,ncas,ncas)
            Active-space spin-separated 1-RDM

    Kwargs:
        ncore : integer
            Number of occupied inactive orbitals
        ncas : integer
            Number of active orbitals

    Returns:
        dm1s : ndarray of shape (2,nao,nao)
    '''
    if mo_coeff is None: mo_coeff=mc.mo_coeff
    if ncore is None: ncore=mc.ncore
    if ncas is None: ncas=mc.ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    moH_core = mo_core.conj ().T
    moH_cas = mo_cas.conj ().T

    casdm1s = np.asarray (casdm1s)
    dm1s_cas = np.dot (casdm1s, moH_cas)
    dm1s_cas = np.dot (mo_cas, dm1s_cas).transpose (1,0,2)
    dm1s_core = np.dot (mo_core, moH_core)
    dm1s = dm1s_cas + dm1s_core[None,:,:]

    # Tags for speeding up rho generators and DF fns
    no_coeff = mo_coeff[:,:ncore+ncas]
    no_coeff = np.stack ([no_coeff, no_coeff], axis=0)
    no_occ = np.zeros ((2,ncore+ncas), dtype=no_coeff.dtype)
    no_occ[:,:ncore] = 1.0
    no_cas = no_coeff[:,:,ncore:]
    for i in range (2):
        no_occ[i,ncore:], umat = linalg.eigh (-casdm1s[i])
        no_cas[i,:,:] = np.dot (no_cas[i,:,:], umat)
    no_occ[:,ncore:] *= -1
    dm1s = lib.tag_array (dm1s, mo_coeff=no_coeff, mo_occ=no_occ)

    return dm1s


def make_weighted_casdm1s(mc, ci=None, weights=None):
    '''Compute the weighted average 1-electron spin-separated CAS density.

    Args:
        mc : instance of class _PDFT

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        weights : ndarray of length nroots
            Weight for each state. If none, uses weights from SA-CASSCF
            calculation

    Returns:
        Weighted average of casdm1s
    '''
    if ci is None: ci = mc.ci
    if weights is None: weights = mc.weights

    # There might be a better way to construct all of them, but this should be
    # more cost-effective than what is currently in the _dms file.
    casdm1s_all = [make_one_casdm1s(mc, ci, state) for state in range(len(ci))]
    casdm1s_0 = np.tensordot(weights, casdm1s_all, axes=1)
    return tuple(casdm1s_0)

def make_weighted_casdm2(mc, ci=None, weights=None):
    '''Compute the weighted average 2-electron spin-summed CAS density.

    Args:
        mc : instance of class _PDFT

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        weights : ndarray of length nroots
            Weight for each state. If none, uses weights from SA-CASSCF
            calculation

    Returns:
        Weighted average of casdm2
    '''
    if ci is None: ci = mc.ci
    if weights is None: weights = mc.weights

    # There might be a better way to construct all of them, but this should be
    # more cost-effective than what is currently in the _dms file.
    casdm2_all = [make_one_casdm2(mc, ci, state) for state in range(len(ci))]
    return np.tensordot(weights, casdm2_all, axes=1)
