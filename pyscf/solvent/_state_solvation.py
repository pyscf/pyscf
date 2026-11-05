#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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
# Author: Arpan Choudhury (University of Chicago) <arpanchoudhury29@gmail.com>
#

'''
State-specific and state-averaged solvation kernels for continuum models.

The solvent polarization is partitioned into a slow (orientational) part,
and a fast (electronic) part.

These kernels are opt-in: a solvent model dispatches here only when .refdm or
.rfroot is set in mcscf/mcpdft.
'''

import numpy
from pyscf.lib import logger

PI = numpy.pi


def _get_K_opt_R_opt(solvent_obj):
    '''K and R matrices evaluated at the optical dielectric constant.

    Computed on demand rather than in build(), since the optical dielectric
    constant is only needed for nonequilibrium solvation. The result is cached
    in _intermediates and invalidated when the optical dielectric changes.
    '''
    if not solvent_obj._intermediates:
        solvent_obj.build()

    epsilon_opt = solvent_obj.get_eps_optical()

    intermediates = solvent_obj._intermediates
    if (intermediates.get('eps_optical_cached') == epsilon_opt
            and 'K_opt' in intermediates and 'R_opt' in intermediates):
        return intermediates['K_opt'], intermediates['R_opt']

    D = intermediates['D']
    A = intermediates['A']
    S = intermediates['S']

    method = solvent_obj.method.upper()
    if method in ('C-PCM', 'CPCM'):
        f_epsilon_opt = (epsilon_opt - 1.0) / epsilon_opt
        K_opt = S
        R_opt = -f_epsilon_opt * numpy.eye(S.shape[0])
    elif method == 'COSMO':
        f_epsilon_opt = (epsilon_opt - 1.0) / (epsilon_opt + 1.0/2.0)
        K_opt = S
        R_opt = -f_epsilon_opt * numpy.eye(S.shape[0])
    elif method in ('IEF-PCM', 'IEFPCM'):
        f_epsilon_opt = (epsilon_opt - 1.0) / (epsilon_opt + 1.0)
        DA = D*A
        DAS = numpy.dot(DA, S)
        K_opt = S - f_epsilon_opt/(2.0*PI) * DAS
        R_opt = -f_epsilon_opt * (numpy.eye(K_opt.shape[0]) - 1.0/(2.0*PI)*DA)
    else:
        raise NotImplementedError(
            f'Nonequilibrium solvation is not implemented for {solvent_obj.method}.')

    intermediates['K_opt'] = K_opt
    intermediates['R_opt'] = R_opt
    intermediates['eps_optical_cached'] = epsilon_opt
    return K_opt, R_opt


def _get_vind_pekar(solvent_obj, dms):
    '''Fast (electronic) component of the surface charges, evaluated with the
    optical dielectric constant.
    '''
    if not solvent_obj._intermediates:
        solvent_obj.build()

    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    if dms.shape[0] == 2:
        dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

    K_opt, R_opt = _get_K_opt_R_opt(solvent_obj)
    v_grids_e = solvent_obj._get_v(dms)
    v_grids = solvent_obj.v_grids_n - v_grids_e

    b = numpy.dot(R_opt, v_grids.T)
    q = numpy.linalg.solve(K_opt, b).T

    vK_1 = numpy.linalg.solve(K_opt.T, v_grids.T)
    qt = numpy.dot(R_opt.T, vK_1).T
    q_sym = (q + qt)/2.0

    return q_sym[0], v_grids[0]


# Kernels

def _check_state_dms(solvent_obj, dm):
    '''Validate the per-state density matrices of a state-averaged kernel.'''
    dms = numpy.asarray(dm)
    if dms.ndim != 3:
        raise ValueError(
            f'rfroot={solvent_obj.rfroot} requires one density matrix per state, '
            f'but received an array of shape {dms.shape}. If this solvent '
            'object is shared with another method, give the state-averaged '
            'method its own copy.')
    n_states = dms.shape[0]
    phi = solvent_obj.rfroot
    if not 0 <= phi < n_states:
        raise ValueError(
            f'rfroot={phi} is out of range for {n_states} states.')
    return dms, phi


def kernel_ss_neq(solvent_obj, dm):
    '''State-specific nonequilibrium solvation.

    Slow charges are taken from the reference density .refdm, fast charges
    from dm.
    '''
    refdm = solvent_obj.refdm
    q_ref_fast, v_grids_ref = _get_vind_pekar(solvent_obj, refdm)
    solvent_obj._get_vind(refdm)
    q_ref = solvent_obj._intermediates['q_sym']
    q_ref_slow = q_ref - q_ref_fast

    q_fast, v_grids = _get_vind_pekar(solvent_obj, dm)
    q_neq = q_ref_slow + q_fast

    vmat = solvent_obj._get_vmat(q_neq)[0]

    e_fast = 0.5 * numpy.dot(q_fast, v_grids)
    e_slow = numpy.dot((v_grids - 0.5 * v_grids_ref), q_ref_slow)
    e_pol = e_fast + e_slow

    logger.info(solvent_obj, 'Nonequilibrium state-specific solvation: slow charges from '
                 'refdm, fast charges from the current state')
    solvent_obj.e, solvent_obj.v = e_pol, vmat
    logger.info(solvent_obj, '%s E_diel = %.15g (nonequilibrium)',
                solvent_obj.__class__.__name__, e_pol)
    return e_pol, vmat


def kernel_sa_neq(solvent_obj, dm):
    '''State-averaged nonequilibrium solvation.

    Slow charges are taken from state 0 and fast charges from state .rfroot.
    Returns a list of per-state polarization energies.
    '''
    dms, phi = _check_state_dms(solvent_obj, dm)

    # slow charges from state 0
    solvent_obj._get_vind(dms[0])
    q0_total = solvent_obj._intermediates['q_sym']
    v_grids0 = solvent_obj._intermediates['v_grids']
    q0_fast, _ = _get_vind_pekar(solvent_obj, dms[0])
    q0_slow = q0_total - q0_fast

    # fast charges from state rfroot
    qphi_fast, _ = _get_vind_pekar(solvent_obj, dms[phi])

    slow_self = -0.5 * numpy.dot(v_grids0, q0_slow)

    logger.info(solvent_obj, 'Nonequilibrium state-averaged solvation: slow charges from '
                 'state 0, fast charges from state %d', phi)

    # reaction field from the total NEQ charge Q^slow(0) + Q^fast(phi)
    q_neq = q0_slow + qphi_fast
    vmat = solvent_obj._get_vmat(q_neq)[0]

    name = solvent_obj.__class__.__name__
    e_pols = []
    for i, d in enumerate(dms):
        if i == 0:
            e_pol_i = 0.5 * numpy.dot(v_grids0, q0_total)
            logger.info(solvent_obj, '%s E_diel(state %d) = %.15g (equilibrium)',
                        name, i, e_pol_i)
        else:
            qk_fast, v_grids_k = _get_vind_pekar(solvent_obj, d)
            fast_self_k = 0.5 * numpy.dot(v_grids_k, qk_fast)
            slow_inter_k = numpy.dot(v_grids_k, q0_slow)
            e_pol_i = fast_self_k + slow_inter_k + slow_self
            logger.info(solvent_obj, '%s E_diel(state %d) = %.15g (nonequilibrium)',
                        name, i, e_pol_i)
        e_pols.append(e_pol_i)

    solvent_obj.e, solvent_obj.v = e_pols, vmat
    return e_pols, vmat


def kernel_sa_eq(solvent_obj, dm):
    '''State-averaged equilibrium solvation.

    The reaction field is polarized by state .rfroot. Returns a list of
    per-state polarization energies.
    '''
    dms, phi = _check_state_dms(solvent_obj, dm)

    logger.info(solvent_obj, 'Equilibrium state-averaged solvation: total charges from '
                 'state %d', phi)
    _, vmat = solvent_obj._get_vind(dms[phi])

    name = solvent_obj.__class__.__name__
    e_pols = []
    for i, d in enumerate(dms):
        e_pol_k, _ = solvent_obj._get_vind(d)
        e_pols.append(e_pol_k)
        logger.info(solvent_obj, '%s E_diel(state %d) = %.15g', name, i, e_pol_k)

    solvent_obj.e, solvent_obj.v = e_pols, vmat
    return e_pols, vmat


_kernels = {
    'ss_neq': kernel_ss_neq,
    'sa_neq': kernel_sa_neq,
    'sa_eq': kernel_sa_eq,
}
