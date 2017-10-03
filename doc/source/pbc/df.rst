.. _pbc_df:

pbc.df --- PBC denisty fitting
******************************

.. module:: pbc.df
   :synopsis: Density fitting and RI approximation with periodic boundary conditions
.. sectionauthor:: Qiming Sun <osirpt.sun@gmail.com>.

Introduction
============

The :mod:`pbc.df` module provides the fundamental functions to handle the
density fitting (DF) integral tensors required by the gamma-point and k-point
PBC calculations.  There are four types of DF methods available for PBC
systems.  They are FFTDF (plane-wave density fitting with fast Fourier
transformation), AFTDF (plane-wave density fitting with analytical Fourier
transformation), GDF (Gaussian density fitting) and MDF (mixed density fitting).
The Coulomb integrals and nuclear attraction integrals in the PBC calculations
are all computed with DF technique.  The default scheme is FFTDF.


FFTDF --- FFT-based density fitting
-----------------------------------

FFTDF represents the method to compute electron repulsion integrals in
reciprocal space with the Fourier transformed Coulomb kernel

.. math::
    (ij|kl) = \rho_{ij}(\mathbf{G}) \frac{4\pi}{G^2} \rho_{kl}(-\mathbf{G})

:math:`\mathbf{G}` is the plane wave vector.
:math:`\rho_{ij}(\mathbf{G})` is the Fourier transformed orbital pair

.. math::
    \rho_{ij}(\mathbf{G}) = \sum_{r} e^{-\mathbf{G}\cdot\mathbf{r}} \phi_i(\mathbf{r})\phi_j(\mathbf{r}))

As the default integral scheme of PBC calculations, FFTDF can be accessed
through :attr:`with_df` of mean-field object::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> mf = scf.RHF(cell)
    >>> print(mf.with_df)
    <pyscf.pbc.df.fft.FFTDF>

* Nuclear-electron interaction integrals and PP integrals are also computed by this FFT technique

* get_eri and get_jk and ao2mo

* When gs is not enough, this module will produce warning message.

* Specify :attr:`kpts`

* Low-dimension system is not supported.


AFTDF --- AFT-based density fitting
-----------------------------------


.. _pbc_gdf:

GDF --- Gaussian density fitting
--------------------------------

GDF represents the method to compute electron repulsion integrals in
* Saving/loading DF integrals the same way as in the

GDF can be::
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='def2-tzvp')
    mf = scf.RHF(mol).density_fit().run()
    mc = mcscf.CASSCF(mf, 8, 10).density_fit().run()


.. _pbc_mdf:

MDF --- mixed density fitting
-----------------------------

In principle, the accuracy of MDF method can be increased by adding more and more
plane waves in the auxiliary basis set.  In practice, large number of plane
waves may lead to strong linear dependency which may lead to numerical stability
issue.  The optimal accuracy requires a reasonable number of plan wave basis
with a reasonable linear dependence threshold.  Big threshold would remove
more auxiliary functions while small threshold would cause numerical error.


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


