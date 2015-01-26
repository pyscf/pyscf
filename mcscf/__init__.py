#!/usr/bin/env python

'''CASCI and CASSCF

Simple usage::

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.Mole()
    >>> mol.build(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASCI(mol, mf, 6, 6)
    >>> mc.kernel()[0]
    -108.980200816243354
    >>> mc = mcscf.CASSCF(mol, mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    >>> mc = mcscf.CASSCF(mol, mf, 4, 4)
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
        Core electron number.  In UHF-CASSCF, it's a tuple to indicate the different core eletron numbers.
    fcisolver : an instance of :class:`FCISolver`
        The pyscf.fci module provides several FCISolver for different scenario.  Generally,
        fci.direct_spin1.FCISolver can be used for all RHF-CASSCF.  However, a proper FCISolver
        can provide better performance and better numerical stability.  One can either use
        :func:`fci.solver` function to pick the FCISolver by the program or manually assigen
        the FCISolver to this attribute, e.g.

        >>> from pyscf import fci
        >>> mc = mcscf.CASSCF(mol, mf, 4, 4)
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
        Default is 1e-4
    max_orb_stepsize : float
        The step size for orbital rotation.  Small step size is prefered.
        Default is 0.03.
    max_ci_stepsize : float
        The max size for approximate CI updates.  The approximate updates are
        used in 1-step algorithm, to estimate the change of CI wavefunction wrt
        the orbital rotation.  Small step size is prefered.  Default is 0.01.
    max_cycle_macro : int
        Max number of macro iterations.  Default is 50.
    max_cycle_micro : int
        Max number of micro iterations in each macro iteration.  Depending on
        systems, increasing this value might reduce the total macro
        iterations.  Generally, 2 - 3 steps should be enough.  Default is 2.
    max_cycle_micro_inner : int
        Max number of steps for the orbital rotations allowed for the augmented
        hessian solver.  It can affect the actual size of orbital rotation.
        Even with a small max_orb_stepsize, a few max_cycle_micro_inner can
        accumulate the rotation and leads to a significant change of the CAS
        space.  Depending on systems, increasing this value migh reduce the
        total number of macro iterations.  The value between 2 - 8 is preferred.
        Default is 4.
    ah_level_shift : float, for AH solver.
        Level shift for the Davidson diagonalization in AH solver.  Default is 0.
    ah_conv_tol : float, for AH solver.
        converge threshold for Davidson diagonalization in AH solver.  Default is 1e-8.
    ah_max_cycle : float, for AH solver.
        Max number of iterations allowd in AH solver.  Default is 20.
    ah_lindep : float, for AH solver.
        Linear dependence threshold for AH solver.  Default is 1e-16.
    ah_start_tol : flat, for AH solver.
        In AH solver, the orbital rotation is started without completely solving the AH problem.
        This value is to control the start point. Default is 1e-4.
    ah_start_cycle : int, for AH solver.
        In AH solver, the orbital rotation is started without completely solving the AH problem.
        This value is to control the start point. Default is 3.

        ``ah_conv_tol``, ``ah_max_cycle``, ``ah_lindep``, ``ah_start_tol`` and ``ah_start_cycle``
        can affect the accuracy and performance of CASSCF solver.  Lower
        ``ah_conv_tol`` and ``ah_lindep`` can improve the accuracy of CASSCF
        optimization, but decrease the performance.
        
        >>> from pyscf import gto, scf, mcscf
        >>> mol = gto.Mole()
        >>> mol.build(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
        >>> mf = scf.UHF(mol)
        >>> mf.scf()
        >>> mc = mcscf.CASSCF(mol, mf, 6, 6)
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
    natorb : bool
        Whether to restore the natural orbital during CASSCF optimization. Default is not.


Saved results

    e_tot : float
        Total MCSCF energy (electronic energy plus nuclear repulsion)
    ci : ndarray
        CAS space FCI coefficients
    converged : bool, for CASSCF only
        It indicates CASSCF optimization converged or not.
    mo_coeff : ndarray, for CASSCF only
        Optimized CASSCF orbitals coefficients
'''


from pyscf.mcscf import mc1step
from pyscf.mcscf import mc1step_symm
from pyscf.mcscf import casci
from pyscf.mcscf import casci_symm
from pyscf.mcscf import addons
from pyscf.mcscf import casci_uhf
from pyscf.mcscf import mc1step_uhf
from pyscf.mcscf.addons import *

def CASSCF(mol, mf, *args, **kwargs):
    if mol.symmetry:
        if mf.__class__.__name__ in ('RHF', 'ROHF'):
            mc = mc1step_symm.CASSCF(mol, mf, *args, **kwargs)
        else:
            mc = mc1step_uhf.CASSCF(mol, mf, *args, **kwargs)
    else:
        if mf.__class__.__name__ in ('RHF', 'ROHF'):
            mc = mc1step.CASSCF(mol, mf, *args, **kwargs)
        else:
            mc = mc1step_uhf.CASSCF(mol, mf, *args, **kwargs)
    return mc

def CASCI(mol, mf, *args, **kwargs):
    if mol.symmetry:
        if mf.__class__.__name__ in ('RHF', 'ROHF'):
            mc = casci_symm.CASCI(mol, mf, *args, **kwargs)
        else:
            mc = casci_uhf.CASCI(mol, mf, *args, **kwargs)
    else:
        if mf.__class__.__name__ in ('RHF', 'ROHF'):
            mc = casci.CASCI(mol, mf, *args, **kwargs)
        else:
            mc = casci_uhf.CASCI(mol, mf, *args, **kwargs)
    return mc
