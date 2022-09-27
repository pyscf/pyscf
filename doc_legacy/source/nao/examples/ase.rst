.. _nao_examples_qmd_C60:

Interfacing with ASE and Siesta
**********************************************
The Atomic Simulation Environment (`ASE <https://wiki.fysik.dtu.dk/ase/index.html>`_)
Greatly simplify the scripting of your problem and combine calculations
from the ground state with `Siesta <https://departments.icmab.es/leem/siesta/>`_ 
and excited states with PYSCF-NAO. 

Simple Polarizability Caculation
================================

To perform a simple calculation returning the polariazbility of the system
one just need the following script,

.. literalinclude:: script_pyscf_ase.py 

You will need first to install Siesta, ASE and to setup the interface between
Siesta and ASE. Everything is explain on the `Siesta webpage of ASE <https://wiki.fysik.dtu.dk/ase/ase/calculators/siesta.html#module-ase.calculators.siesta>`_

Non-Resonant Raman Spectra
==========================

In ASE there is also the possibility to calculate Non-Resonnant Raman spectra
using the Siesta calculator together with PYSCF,

.. literalinclude:: script_raman_pyscf.py 
