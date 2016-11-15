.. _pbc:

******************************************
:mod:`pbc` --- Periodic boundary condition
******************************************
The :mod:`pbc` module provides the electronic structure implementation
with the periodic boundary condition based on the periodic Gaussian
basis functions.  The PBC program supports both all-electron and pseudo
potential (including quantum chemistry ECP) descriptions.

The PBC implementation has tight relation to the molecular implementation.
The module and function names and layouts of PBC code are the same to
those of molecular code.  The PBC program supports the use of basis sets
and the pseudo potential (PP) developed by quantum chemistry community.
The program allows one mixing the PBC-specific basis sets and PP with
the quantum chemistry basis sets and ECPs.  This feature offers high
flexibility for the choice of basis sets, methods in the PBC calculations.
Moreover, many post-mean-field methods defined in molecular code can be
seamlessly mixed with the PBC gamma point calculations.  Eg, one can
start from PBC gamma point Hartree-Fock calculation, followed by CCSD,
TDHF methods etc which are implemented in the molecular code.

In the k-point sampling calculation, we make small changes on data structure
based on the gamma point program and export K-HF, K-DFT methods.
On top of the K-HF methods,  we developed the k-point CCSD, and k-point EOM-CCSD
methods in which the computing work and data distribution are carefully
optimized.

The list of modules described in this chapter is:

.. toctree::

   pbc/gto.rst
   pbc/scf.rst
   pbc/dft.rst
   pbc/df.rst
   pbc/cc.rst
   pbc/tools.rst
   pbc/mix_mol.rst

