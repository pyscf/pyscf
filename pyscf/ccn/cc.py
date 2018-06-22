#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Auto-generated coupled-cluster equations of arbitrary order
#
# Author: Artem Pulkin
#
# Initial implementation includes ground-state, lambda, IP-EOM and EA-EOM kernels for CCS, CCSD, CCD and CCSDT
# unrestricted (general) theory.

from .equations import *
from .pyscf_helpers import kernel_solve, kernel_eig, eris_hamiltonian, koopmans_guess_ip, koopmans_guess_ea

from collections import OrderedDict


def kernel_ground_state_s(cc, tolerance=1e-10, maxiter=50):
    """
    A ground-state kernel (singles).
    Args:
        cc (object): one of pyscf ccsd objects;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state amplitudes and energy.
    """
    t, e = kernel_solve(cc.ao2mo(), eq_gs_s, ("t1",), tolerance=tolerance, equation_energy=energy_gs_s,
                        dim_spec=("ov",), maxiter=maxiter)
    return e, t["t1"]


def kernel_ground_state_sd(cc, tolerance=1e-10, maxiter=50):
    """
    A ground-state kernel (singles and doubles).
    Args:
        cc (object): one of pyscf ccsd objects;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state amplitudes and energy.
    """
    t, e = kernel_solve(cc.ao2mo(), eq_gs_sd, ("t1", "t2",), tolerance=tolerance, equation_energy=energy_gs_sd,
                        dim_spec=("ov", "oovv"), maxiter=maxiter)
    return e, t["t1"], t["t2"]


def kernel_ground_state_sdt(cc, tolerance=1e-10, maxiter=50):
    """
    A ground-state kernel (singles, doubles and triples).
    Args:
        cc (object): one of pyscf ccsd objects;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state amplitudes and energy.
    """
    t, e = kernel_solve(cc.ao2mo(), eq_gs_sdt, ("t1", "t2", "t3"), tolerance=tolerance, equation_energy=energy_gs_sdt,
                        dim_spec=("ov", "oovv", "ooovvv"), maxiter=maxiter)
    return e, t["t1"], t["t2"], t["t3"]


def kernel_ground_state_d(cc, tolerance=1e-10, maxiter=50):
    """
    A ground-state kernel (doubles only).
    Args:
        cc (object): one of pyscf ccsd objects;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state amplitudes and energy.
    """
    t, e = kernel_solve(cc.ao2mo(), eq_gs_d, ("t2",), tolerance=tolerance, equation_energy=energy_gs_d,
                        dim_spec=("oovv",), maxiter=maxiter)
    return e, t["t2"]


def kernel_lambda_s(cc, t1, tolerance=1e-10, maxiter=50):
    """
    A ground-state lambda kernel (singles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state lambda amplitudes.
    """
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1,
    ))
    l = kernel_solve(hamiltonian, eq_lambda_s, ("a1",), tolerance=tolerance, dim_spec=("ov",), maxiter=maxiter)
    return l["a1"]


def kernel_lambda_sd(cc, t1, t2, tolerance=1e-10, maxiter=50):
    """
    A ground-state lambda kernel (singles and doubles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state lambda amplitudes.
    """
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1,
        t2=t2,
    ))
    l = kernel_solve(hamiltonian, eq_lambda_sd, ("a1", "a2"), tolerance=tolerance, dim_spec=("ov", "oovv"),
                     maxiter=maxiter)
    return l["a1"], l["a2"]


def kernel_lambda_sdt(cc, t1, t2, t3, tolerance=1e-10, maxiter=50):
    """
    A ground-state lambda kernel (singles, doubles and triples).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        t3 (numpy.ndarray): the t3 amplitudes;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state lambda amplitudes.
    """
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1, t2=t2, t3=t3,
    ))
    l = kernel_solve(hamiltonian, eq_lambda_sdt, ("a1", "a2", "a3"), tolerance=tolerance,
                     dim_spec=("ov", "oovv", "ooovvv"), maxiter=maxiter)
    return l["a1"], l["a2"], l["a3"]


def kernel_lambda_d(cc, t2, tolerance=1e-10, maxiter=50):
    """
    A ground-state lambda kernel (doubles only).
    Args:
        cc (object): one of pyscf ccsd objects;
        t2 (numpy.ndarray): the t2 amplitudes;
        tolerance (float): tolerance to converge to;
        maxiter (int): the maximal number of iterations;

    Returns:
        Ground state lambda amplitudes.
    """
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t2=t2,
    ))
    l = kernel_solve(hamiltonian, eq_lambda_d, ("a2",), tolerance=tolerance, dim_spec=("oovv",), maxiter=maxiter)
    return l["a2"]


def kernel_ip_s(cc, t1, nroots=1, tolerance=1e-10):
    """
    EOM-IP kernel (singles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-IP energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1,
    ))
    initial_guess_ip = list(
        koopmans_guess_ip(nocc, nvirt, OrderedDict((("r_ip1", 1),)), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ip_s, initial_guess_ip, tolerance=tolerance)


def kernel_ip_sd(cc, t1, t2, nroots=1, tolerance=1e-10):
    """
    EOM-IP kernel (singles and doubles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-IP energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1, t2=t2,
    ))
    initial_guess_ip = list(
        koopmans_guess_ip(nocc, nvirt, OrderedDict((("r_ip1", 1), ("r_ip2", 2))), i, dtype=float)
        for i in range(nroots)
    )
    #print initial_guess_ip
    return kernel_eig(hamiltonian, eq_ip_sd, initial_guess_ip, tolerance=tolerance)


def kernel_ip_sdt(cc, t1, t2, t3, nroots=1, tolerance=1e-10):
    """
    EOM-IP kernel (singles, doubles and triples).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        t3 (numpy.ndarray): the t3 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-IP energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1, t2=t2, t3=t3,
    ))
    initial_guess_ip = list(
        koopmans_guess_ip(nocc, nvirt, OrderedDict((("r_ip1", 1), ("r_ip2", 2), ("r_ip3", 3))), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ip_sdt, initial_guess_ip, tolerance=tolerance)


def kernel_ip_d(cc, t2, nroots=1, tolerance=1e-10):
    """
    EOM-IP kernel (doubles only).
    Args:
        cc (object): one of pyscf ccsd objects;
        t2 (numpy.ndarray): the t2 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-IP energies and amplitudes.
    """
    nocc, _, nvirt, _ = t2.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t2=t2,
    ))
    initial_guess_ip = list(
        koopmans_guess_ip(nocc, nvirt, OrderedDict((("r_ip2", 2))), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ip_d, initial_guess_ip, tolerance=tolerance)


def kernel_ea_s(cc, t1, nroots=1, tolerance=1e-10):
    """
    EOM-EA kernel (singles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-EA energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1,
    ))
    initial_guess_ea = list(
        koopmans_guess_ea(nocc, nvirt, OrderedDict((("r_ea1", 1),)), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ea_s, initial_guess_ea, tolerance=tolerance)


def kernel_ea_sd(cc, t1, t2, nroots=1, tolerance=1e-10):
    """
    EOM-EA kernel (singles and doubles).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-EA energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1, t2=t2,
    ))
    initial_guess_ea = list(
        koopmans_guess_ea(nocc, nvirt, OrderedDict((("r_ea1", 1), ("r_ea2", 2))), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ea_sd, initial_guess_ea, tolerance=tolerance)


def kernel_ea_sdt(cc, t1, t2, t3, nroots=1, tolerance=1e-10):
    """
    EOM-EA kernel (singles, doubles and triples).
    Args:
        cc (object): one of pyscf ccsd objects;
        t1 (numpy.ndarray): the t1 amplitudes;
        t2 (numpy.ndarray): the t2 amplitudes;
        t3 (numpy.ndarray): the t3 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-EA energies and amplitudes.
    """
    nocc, nvirt = t1.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t1=t1, t2=t2, t3=t3,
    ))
    initial_guess_ea = list(
        koopmans_guess_ea(nocc, nvirt, OrderedDict((("r_ea1", 1), ("r_ea2", 2), ("r_ea3", 3))), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ea_sdt, initial_guess_ea, tolerance=tolerance)


def kernel_ea_d(cc, t2, nroots=1, tolerance=1e-10):
    """
    EOM-EA kernel (doubles only).
    Args:
        cc (object): one of pyscf ccsd objects;
        t2 (numpy.ndarray): the t2 amplitudes;
        nroots (int): the number of roots to find;
        tolerance (float): tolerance to converge to;

    Returns:
        EOM-EA energies and amplitudes.
    """
    nocc, _, nvirt, _ = t2.shape
    hamiltonian = eris_hamiltonian(cc.ao2mo())
    hamiltonian.update(dict(
        t2=t2,
    ))
    initial_guess_ea = list(
        koopmans_guess_ea(nocc, nvirt, OrderedDict((("r_ea2", 2))), i, dtype=float)
        for i in range(nroots)
    )
    return kernel_eig(hamiltonian, eq_ea_d, initial_guess_ea, tolerance=tolerance)


if __name__ == "__main__":
    from pyscf import scf, gto, cc
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H  0.758602  0.000000  0.504284; H  0.758602  0.000000  -0.504284"
    mol.unit = "angstrom"
    mol.basis = 'cc-pvdz'
    mol.build()

    mf = scf.GHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()

    ccsd = cc.GCCSD(mf, frozen=2)
    ccsd.kernel()

    e, t1, t2 = kernel_ground_state_sd(ccsd)
    print(e)
