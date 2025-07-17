#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Matthew Hennefarth <mhennefarth@uchicago.com>

from functools import reduce
import numpy as np
from scipy import linalg

from pyscf.mcpdft import _dms
from pyscf.fci import direct_spin1


def fock_h1e_for_cas(mc, sa_casdm1, mo_coeff=None, ncas=None, ncore=None):
    '''Compute the CAS SA Fock Matrix 1-electron integrals.

    Args:
        mc : instance of class _PDFT

        sa_casdm1 : ndarray of shape (ncas,ncas)
            1-RDM in the active space generated from the state-average
            density.

        mo_coeff : ndarray of shape (nao,nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ncas : int
            Number of active space molecular orbitals

        ncore : int
            Number of core molecular orbitals

    Returns:
        A tuple, the first is the one-electron integrals defined in the CAS
        space and the second is the constant, core part.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    nocc = ncore + ncas
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    hcore_eff = mc.get_fock(casdm1=sa_casdm1)
    energy_core = mc._scf.energy_nuc()

    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        energy_core += np.tensordot(core_dm, hcore_eff).real

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore_eff, mo_cas))

    return h1eff, energy_core


def make_fock_mcscf(mc, mo_coeff=None, ci=None, weights=None):
    '''Compute the SA-Fock Hamiltonian/Matrix

    Args:
        mc : instance of class _PDFT

        mo_coeff : ndarray of shape (nao,nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ci : ndarray of shape (nroots)
            CI vectors should be from a converged CASSCF/CASCI calculation

        weights : ndarray of length nroots
            Weight for each state. If none, uses weights from SA-CASSCF
            calculation

    Returns:
        safock_ham : ndarray of shape (nroots,nroots)
            State-average Fock matrix expressed in the basis provided by the CI
            vectors.

    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    ncas = mc.ncas

    sa_casdm1 = sum(_dms.make_weighted_casdm1s(mc, ci=ci, weights=weights))
    h1, h0 = fock_h1e_for_cas(mc, sa_casdm1, mo_coeff=mo_coeff)
    hc_all = [direct_spin1.contract_1e(h1, c, ncas, mc.nelecas) for c in ci]
    safock_ham = np.tensordot(ci, hc_all, axes=((1, 2), (1, 2)))
    idx = np.diag_indices_from(safock_ham)
    safock_ham[idx] += h0

    return safock_ham


def diagonalize_safock(mc, mo_coeff=None, ci=None):
    '''Diagonalizes the SA-Fock matrix. Returns the eigenvalues and
    eigenvectors. '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    fock = make_fock_mcscf(mc, mo_coeff=mo_coeff, ci=ci)
    return linalg.eigh(fock)


def safock_energy(mc, **kwargs):
    '''Diabatizer Function

    The "objective" function we are optimizing when solving for the
    SA-Fock eigenstates is that the SA-Fock energy (average) is
    minimized with the constraint to the final states being orthonormal.
    Its the whole saddle point thing similar to in SA-MC-PDFT gradients. For now
    I just omit this whole issue since the first derivative is by default 0 and
    don't worry about the Hessian (or second derivatives) since I don't care.
    It should fail if we try and do gradients but I can put a hard
    RaiseImplementation error

    Returns:
        SA-Fock Energy : float
            weighted sum of SA-Fock energies
        dSA-Fock Energy : ndarray of shape npair = nroots*(nroots - 1)/2
            first derivative of the SA-Fock energy wrt interstate rotation
            This is zero by default since we diagonalize a matrix
        d2SA-Fock Energy : ndarray of shape (npair,npair)
            Should be the Lagrange multiplier terms. Currently returning None
            since we cannot do gradients yet.
    '''
    dsa_fock = np.zeros(
        int(mc.fcisolver.nroots * (mc.fcisolver.nroots - 1) / 2))
    # TODO fix, this redundancy...no need to compute fock matrix twice..
    e_states, _ = diagonalize_safock(mc)

    return np.dot(e_states, mc.weights), dsa_fock, None


def solve_safock(mc, mo_coeff=None, ci=None, ):
    '''Diabatize Function. Finds the SA-Fock Eigenstates within the model space
    spanned by the CI vectors.

    Args:
        mc : instance of class _PDFT

        mo_coeff : ndarray of shape (nao,nmo)
            A full set of molecular orbital coefficients. Taken from
            self if not provided.

        ci : ndarray of shape (nroots)
            CI vectors should be from a converged CASSCF/CASCI calculation

    Returns:
        conv : bool
            Always true. If there is a convergence issue, scipy will raise an
            exception.

        ci : list of ndarrays of length = nroots
            CI vectors of the optimized diabatic states
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    e_states, si_pdft = diagonalize_safock(mc, mo_coeff=mo_coeff, ci=ci)

    ci = np.tensordot(si_pdft.T, ci, 1)
    conv = True

    return conv, ci


if __name__ == "__main__":
    from pyscf import scf, gto
    from pyscf import mcpdft

    xyz = '''O  0.00000000   0.08111156   0.00000000
                 H  0.78620605   0.66349738   0.00000000
                 H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M(atom=xyz, basis='sto-3g', symmetry=False,
                verbose=5)
    mf = scf.RHF(mol).run()
    mc = mcpdft.CASSCF(mf, 'tPBE', 4, 4)
    mc.fix_spin_(ss=0)
    mc = mc.multi_state([1.0 / 3, ] * 3, 'xms').run()
