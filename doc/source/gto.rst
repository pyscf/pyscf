.. _gto:

gto --- Molecular structure and GTO basis
*****************************************
 
This module provides the functions to parse the command line options,
the molecular geometry and format the basic functions for `libcint`
integral library.  In :file:`mole`, a basic class :class:`Mole` is
defined to hold the global parameters, which will be used throughout the
package.


.. automodule:: pyscf.gto
 

mole
====

:class:`Mole` class handles three layers: input, internal format, libcint arguments.
The relationship of the three layers are::

  .atom (input)  <=>  ._atom (for python) <=> ._atm (for libcint)
  .basis (input) <=> ._basis (for python) <=> ._bas (for libcint)

input layer does not talk to libcint directly.  Data are held in python
internal fomrat layer.  Most of methods defined in this class only operates
on the internal format.  Exceptions are make_env, make_atm_env, make_bas_env,
:func:`set_common_orig_`, :func:`set_rinv_orig_` which are used to
manipulate the libcint arguments.


.. automodule:: pyscf.gto.mole
   :members:

.. autoclass:: Mole
   :members:


.. _gto_moleintor:

moleintor
=========

.. automodule:: pyscf.gto.moleintor
   :members:


.. _gto_basis:

basis
=====

Basis sets

.. automodule:: pyscf.gto.basis
   :members:

Optimized contraction
---------------------

