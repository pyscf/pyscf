.. _pbc_df:

pbc.df --- PBC denisty fitting
******************************

.. module:: pbc.df
   :synopsis: Density fitting and RI approximation with periodic boundary conditions
.. sectionauthor:: Qiming Sun <osirpt.sun@gmail.com>.

Introduction
============

The :mod:`pbc.df` module provides the fundamental functions to handle the
DF integral tensors required by the gamma-point k-point PBC calculations.


FFTDF --- density fitting with FFT integrals
--------------------------------------------

* When gs is not enough, this module will produce warning message.

* Low-dimension system is not supported.


AFTDF --- density fitting with AFT integrals
--------------------------------------------


.. _pbc_gdf:

Gaussian density fitting
------------------------

* Saving/loading DF integrals the same way as in the


.. _pbc_mdf:

Mixed density fitting
---------------------

In principle, the accuracy of MDF method can be improved by adding more and more
plane waves in the auxiliary basis set.  In practice, large number of plane
waves may lead to strong linear dependence which has strong effects to the
numerical stability.  The optimal accuracy requires a reasonable plan wave basis
size with a reasonable linear dependence threshold.


Low-dimension system
--------------------


All-electron system
-------------------



Program reference
=================

FFTDF class
-----------

.. autoclass:: pyscf.pbc.df.fft.FFTDF

FFTDF helper functions
----------------------

.. automodule:: pyscf.pbc.df.fft_jk

.. automodule:: pyscf.pbc.df.fft_ao2mo

AFTDF class
-----------

.. autoclass:: pyscf.pbc.df.aft.AFTDF

AFTDF helper functions
----------------------

.. automodule:: pyscf.pbc.df.aft_jk

.. automodule:: pyscf.pbc.df.aft_ao2mo


GDF class
---------

.. autoclass:: pyscf.pbc.df.df.GDF

GDF helper functions
--------------------

.. automodule:: pyscf.pbc.df.df_jk

.. automodule:: pyscf.pbc.df.df_ao2mo


MDF class
---------

.. autoclass:: pyscf.pbc.df.mdf.MDF

MDF helper functions
--------------------

.. automodule:: pyscf.pbc.df.mdf_jk

.. automodule:: pyscf.pbc.df.mdf_ao2mo


