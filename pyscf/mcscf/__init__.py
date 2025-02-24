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

'''CASCI and CASSCF

When using results of this code for publications, please cite the following paper:
"A general second order complete active space self-consistent-field solver for large-scale systems", Q. Sun, J. Yang, and G. K.-L. Chan, Chem. Phys. Lett. 683, 291 (2017).

Simple usage::

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 6, 6)
    >>> mc.kernel()[0]
    -108.980200816243354
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    >>> mc = mcscf.CASSCF(mf, 4, 4)
    >>> cas_list = [5,6,8,9] # pick orbitals for CAS space, 1-based indices
    >>> mo = mcscf.sort_mo(mc, mf.mo_coeff, cas_list)
    >>> mc.kernel(mo)[0]
    -109.007378939813691

:func:`mcscf.CASSCF` or :func:`mcscf.CASCI` returns a proper instance of CASSCF/CASCI class.
There are some parameters to control the CASSCF/CASCI method.

    verbose : int
        Print level.  Default value equals to :class:`Mole.verbose`.
    max_memory : float or int
        Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
    ncas : int
        Active space size.
    nelecas : tuple of int
        Active (nelec_alpha, nelec_beta)
    ncore : int or tuple of int
        Core electron number.  In UHF-CASSCF, it's a tuple to indicate the different core electron numbers.
    natorb : bool
        Whether to restore the natural orbital during CASSCF optimization. Default is not.
    canonicalization : bool
        Whether to canonicalize orbitals.  Default is True.
    fcisolver : an instance of :class:`FCISolver`
        The pyscf.fci module provides several FCISolver for different scenario.  Generally,
        fci.direct_spin1.FCISolver can be used for all RHF-CASSCF.  However, a proper FCISolver
        can provide better performance and better numerical stability.  One can either use
        :func:`fci.solver` function to pick the FCISolver by the program or manually assigen
        the FCISolver to this attribute, e.g.

        >>> from pyscf import fci
        >>> mc = mcscf.CASSCF(mf, 4, 4)
        >>> mc.fcisolver = fci.solver(mol, singlet=True)
        >>> mc.fcisolver = fci.direct_spin1.FCISolver(mol)

        You can control FCISolver by setting e.g.::

            >>> mc.fcisolver.max_cycle = 30
            >>> mc.fcisolver.conv_tol = 1e-7

        For more details of the parameter for FCISolver, See :mod:`fci`.

        By replacing this fcisolver, you can easily use the CASCI/CASSCF solver
        with other FCI replacements,  such as DMRG, QMC.  See :mod:`dmrgscf` and
        :mod:`fciqmcscf`.

The Following attributes are used for CASSCF

    conv_tol : float
        Converge threshold.  Default is 1e-7
    conv_tol_grad : float
        Converge threshold for CI gradients and orbital rotation gradients.
        If not specified, it is set to sqrt(conv_tol).
    max_stepsize : float
        The step size for orbital rotation.  Small step size is prefered.
        Default is 0.02.  
        (NOTE although the default step size is small enough for many systems,
        it happens that the orbital optimizor crosses the barrier of local
        minimum and converge to the neighbour solution, e.g. the CAS(4,4) for
        C2H4 in the test files.  In these systems, adjusting max_stepsize,
        max_ci_stepsize and max_cycle_micro and ah_start_tol may be helpful)

        >>> mc = mcscf.CASSCF(mf, 6, 6)
        >>> mc.max_stepsize = .01
        >>> mc.max_cycle_micro = 1
        >>> mc.max_cycle_macro = 100
        >>> mc.ah_start_tol = 1e-6

    max_cycle_macro : int
        Max number of macro iterations.  Default is 50.
    max_cycle_micro : int
        Max number of micro iterations in each macro iteration.  Depending on
        systems, increasing this value might reduce the total macro
        iterations.  Generally, 2 - 5 steps should be enough.  Default is 4.
    frozen : int or list
        If integer is given, the inner-most orbitals are excluded from optimization.
        Given the orbital indices (0-based) in a list, any doubly occupied core
        orbitals, active orbitals and external orbitals can be frozen.
    ah_level_shift : float, for AH solver.
        Level shift for the Davidson diagonalization in AH solver.  Default is 0.
    ah_conv_tol : float, for AH solver.
        converge threshold for Davidson diagonalization in AH solver.  Default is 1e-8.
    ah_max_cycle : float, for AH solver.
        Max number of iterations allowd in AH solver.  Default is 20.
    ah_lindep : float, for AH solver.
        Linear dependence threshold for AH solver.  Default is 1e-16.
    ah_start_tol : float, for AH solver.
        In AH solver, the orbital rotation is started without completely solving the AH problem.
        This value is to control the start point. Default is 2.5.
    ah_start_cycle : int, for AH solver.
        In AH solver, the orbital rotation is started without completely solving the AH problem.
        This value is to control the start point. Default is 3.

        ``ah_conv_tol``, ``ah_max_cycle``, ``ah_lindep``, ``ah_start_tol`` and ``ah_start_cycle``
        can affect the accuracy and performance of CASSCF solver.  Lower
        ``ah_conv_tol`` and ``ah_lindep`` can improve the accuracy of CASSCF
        optimization, but slow down the performance.

        >>> from pyscf import gto, scf, mcscf
        >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
        >>> mf = scf.UHF(mol)
        >>> mf.scf()
        >>> mc = mcscf.CASSCF(mf, 6, 6)
        >>> mc.conv_tol = 1e-10
        >>> mc.ah_conv_tol = 1e-5
        >>> mc.kernel()
        -109.044401898486001
        >>> mc.ah_conv_tol = 1e-10
        >>> mc.kernel()
        -109.044401887945668

    chkfile : str
        Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
        Default is the checkpoint file of mean field object.


Saved results

    e_tot : float
        Total MCSCF energy (electronic energy plus nuclear repulsion)
    ci : ndarray
        CAS space FCI coefficients
    converged : bool, for CASSCF only
        It indicates CASSCF optimization converged or not.
    mo_energy: ndarray,
        Diagonal elements of general Fock matrix
    mo_coeff : ndarray, for CASSCF only
        Optimized CASSCF orbitals coefficients
        Note the orbitals are NOT natural orbitals by default.  There are two
        inbuilt methods to convert the mo_coeff to natural orbitals.
        1. Set .natorb attribute.  It can be used before calculation.
        2. call .cas_natorb_ method after the calculation to in-place convert the orbitals
'''


from pyscf.mcscf import mc1step
from pyscf.mcscf import mc1step_symm
from pyscf.mcscf import casci
from pyscf.mcscf import casci_symm
from pyscf.mcscf import addons
from pyscf.mcscf import ucasci
casci_uhf = ucasci  # for backward compatibility
from pyscf.mcscf import umc1step
mc1step_uhf = umc1step  # for backward compatibility
from pyscf.mcscf.addons import *
from pyscf.mcscf import chkfile

def CASSCF(mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
    from pyscf import gto
    from pyscf import scf
    from pyscf.df.df_jk import _DFHF
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.RHF()
    else:
        mf = mf_or_mol

    if isinstance(mf, scf.uhf.UHF):
        mf = mf.to_rhf()
    if isinstance(mf, _DFHF) and mf.with_df:
        return DFCASSCF(mf, ncas, nelecas, ncore, frozen)

    if mf.mol.symmetry and mf.mol.groupname != 'C1':
        mc = mc1step_symm.CASSCF(mf, ncas, nelecas, ncore, frozen)
    else:
        mc = mc1step.CASSCF(mf, ncas, nelecas, ncore, frozen)
    return mc

RCASSCF = CASSCF


def CASCI(mf_or_mol, ncas, nelecas, ncore=None):
    from pyscf import gto
    from pyscf import scf
    from pyscf.df.df_jk import _DFHF
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.RHF()
    else:
        mf = mf_or_mol

    if isinstance(mf, scf.uhf.UHF):
        mf = mf.to_rhf()

    if isinstance(mf, _DFHF) and mf.with_df:
        return DFCASCI(mf, ncas, nelecas, ncore)

    if mf.mol.symmetry and mf.mol.groupname != 'C1':
        mc = casci_symm.CASCI(mf, ncas, nelecas, ncore)
    else:
        mc = casci.CASCI(mf, ncas, nelecas, ncore)
    return mc

RCASCI = CASCI


def UCASCI(mf_or_mol, ncas, nelecas, ncore=None):
    from pyscf import gto
    from pyscf import scf
    from pyscf.df.df_jk import _DFHF
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.UHF()
    else:
        mf = mf_or_mol

    if not isinstance(mf, scf.uhf.UHF):
        mf = mf.to_uhf()
    if isinstance(mf, _DFHF) and mf.with_df:
        from pyscf.lib import logger
        logger.warn(mf, f'DF-UCASCI for DFHF method {mf} is not available. '
                    'Normal UCASCI method is called.')
        mf = mf.undo_df()
    mc = ucasci.UCASCI(mf, ncas, nelecas, ncore)
    return mc


def UCASSCF(mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
    from pyscf import gto
    from pyscf import scf
    from pyscf.df.df_jk import _DFHF
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.UHF()
    else:
        mf = mf_or_mol

    if not isinstance(mf, scf.uhf.UHF):
        mf = mf.to_uhf()
    if isinstance(mf, _DFHF) and mf.with_df:
        from pyscf.lib import logger
        logger.warn(mf, f'DF-UCASSCF for DFHF method {mf} is not available. '
                    'Normal UCASSCF method is called.')
        mf = mf.undo_df()
    mc = umc1step.UCASSCF(mf, ncas, nelecas, ncore, frozen)
    return mc

def newton(mc):
    return mc.newton()


from pyscf.mcscf import df
def DFCASSCF(mf_or_mol, ncas, nelecas, auxbasis=None, ncore=None,
             frozen=None):
    from pyscf import gto
    from pyscf import scf
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.RHF().density_fit()
    else:
        mf = mf_or_mol

    if isinstance(mf, scf.uhf.UHF):
        mf = mf.to_rhf()

    if mf.mol.symmetry and mf.mol.groupname != 'C1':
        mc = mc1step_symm.CASSCF(mf, ncas, nelecas, ncore, frozen)
    else:
        mc = mc1step.CASSCF(mf, ncas, nelecas, ncore, frozen)
    return df.density_fit(mc, auxbasis)

def DFCASCI(mf_or_mol, ncas, nelecas, auxbasis=None, ncore=None):
    from pyscf import gto
    from pyscf import scf
    if isinstance(mf_or_mol, gto.MoleBase):
        mf = mf_or_mol.RHF().density_fit()
    else:
        mf = mf_or_mol

    if isinstance(mf, scf.uhf.UHF):
        mf = mf.to_rhf()

    if mf.mol.symmetry and mf.mol.groupname != 'C1':
        mc = casci_symm.CASCI(mf, ncas, nelecas, ncore)
    else:
        mc = casci.CASCI(mf, ncas, nelecas, ncore)
    return df.density_fit(mc, auxbasis)

approx_hessian = df.approx_hessian

def density_fit(mc, auxbasis=None, with_df=None):
    return mc.density_fit(auxbasis, with_df)
