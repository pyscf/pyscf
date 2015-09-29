'''
**************************************
A Python package for quantum chemistry
**************************************

Features
--------

1. Common quantum chemistry methods
    * Hartree-Fock
    * DFT
    * CASSCF and FCI
    * Full CI
    * MP2
    * SC-NEVPT2
    * CCSD
    * CCSD lambda
    * EOM-CCSD
    * Density fitting
    * relativistic correction
    * General integral transformation
    * Gradient
    * NMR
2. Interface to integral package `Libcint <https://github.com/sunqm/libcint>`_
3. Interface to DMRG `CheMPS2 <https://github.com/SebWouters/CheMPS2>`_
4. Interface to DMRG `Block <https://github.com/sanshar/Block>`_
5. Interface to FCIQMC `NECI <https://github.com/ghb24/NECI_STABLE>`_

How to use
----------
There are two ways to access the documentation: the docstrings come with
the code, and an online program reference, available from
http://www.sunqm.net/pyscf/index.html

We recommend the enhanced Python interpreter `IPython <http://ipython.org>`_
and the web-based Python IDE `Ipython notebook <http://ipython.org/notebook.html>`_
to try out the package::

    >>> from pyscf import gto, scf
    >>> mol = gto.Mole()
    >>> mol.build(atom='H 0 0 0; H 0 0 1.2', basis='cc-pvdz')
    >>> m = scf.RHF(mol)
    >>> m.scf()
    converged SCF energy = -1.06111199785749
    -1.06111199786

Submodules
----------
In pyscf, submodules require explict import::

    >>> from pyscf import gto, scf

:mod:`gto`
    Molecular structure and basis sets.
scf
    Non-relativistic and relativistic Hartree-Fock.
mcscf
    CASCI, 1-step and 2-step CASSCF
ao2mo
    General 2-electron AO to MO integral transformation
dft
    Non-relativistic DFT
df
    Density fitting
fci
    Full CI
cc
    Coupled Cluster
dmrgscf
    2-step DMRG-CASSCF
fciqmcscf
    2-step FCIQMC-CASSCF
grad
    Gradients
lo
    Localization and orbital orthogonalization
lib
    Basic functions and C extensions
nmr
    NMR
mp
    Moller-Plesset perturbation theory
symm
    Symmetry
tools
    fcidump, molden etc


Pure function and Class
-----------------------
The class are designed to hold only the final results and the control
parameters such as maximum number of iterations, convergence threshold, etc.
The intermediate status are not saved in the class.  If the .kernel() function
is finished without any errors,  the solution will be saved in the class (see
documentation).

Most useful functions are implemented at module level, and can be accessed in
both class or module,  e.g.  ``scf.hf.get_jk(mol, dm)`` and
``SCF(mol).get_jk(mol, dm)`` have the same functionality.  As a result, most
functions and class are **pure**, i.e. no status are saved, and the argument
are not changed inplace.  Exceptions (destructive functions and methods) are
suffixed with underscore in the function name,  eg  ``scf.hf.get_fock_``
function may change the status of the argument ``adiis``

'''

__version__ = '0.11'

import os
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import ao2mo

# modules in ./future are in test
__path__.append(os.path.join(os.path.dirname(__file__), 'future'))
__path__.append(os.path.join(os.path.dirname(__file__), 'tools'))
