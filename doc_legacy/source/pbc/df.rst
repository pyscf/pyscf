.. _pbc_df:

pbc.df --- PBC density fitting
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

The characters of these PBC DF methods are summarized in the following table

========================= =========== =========== ========== ==============
Subject                   FFTDF       AFTDF       GDF        MDF
------------------------- ----------- ----------- ---------- --------------
Initialization            No          No          Slow       Slow
HF Coulomb matrix (J)     Fast        Slow        Fast       Moderate
HF exchange matrix (K)    Slow        Slow        Fast       Moderate
Building ERIs             Slow        Slow        Fast       Moderate
All-electron calculation  Huge error  Large error Accurate   Most accurate
Low-dimension system      N/A         0D,1D,2D    0D,1D,2D   0D,1D,2D
========================= =========== =========== ========== ==============


.. _fftdf:

FFTDF --- FFT-based density fitting
-----------------------------------

FFTDF represents the method to compute electron repulsion integrals in
reciprocal space with the Fourier transformed Coulomb kernel

.. math::
    (ij|kl) = \sum_G \rho_{ij}(\mathbf{G}) \frac{4\pi}{G^2} \rho_{kl}(-\mathbf{G})

:math:`\mathbf{G}` is the plane wave vector.
:math:`\rho_{ij}(\mathbf{G})` is the Fourier transformed orbital pair

.. math::
    \rho_{ij}(\mathbf{G}) = \sum_{r} e^{-\mathbf{G}\cdot\mathbf{r}} \phi_i(\mathbf{r})\phi_j(\mathbf{r})

Here are some examples to initialize FFTDF object::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> fftdf = df.FFTDF(cell)
    >>> print(fftdf)
    <pyscf.pbc.df.fft.FFTDF object at 0x7f599dbd6450>
    >>> mf = scf.RHF(cell)
    >>> print(mf.with_df)
    <pyscf.pbc.df.fft.FFTDF object at 0x7f59a1a10c50>

As the default integral scheme of PBC calculations, FFTDF is created when
initializing the PBC mean-field object and held in the attribute :attr:`with_df`.


Nuclear type integrals
^^^^^^^^^^^^^^^^^^^^^^

PBC nuclear-electron interaction and pseudo-potential (PP) integrals can be
computed with the FFTDF methods :func:`FFTDF.get_nuc` and :func:`FFTDF.get_pp`.
:func:`FFTDF.get_nuc` function only evaluates the integral of the point charge.
If PP was specified in the cell object, :func:`FFTDF.get_nuc` produces the
integrals of the point nuclei with the effective charges.  If PP was not
defined in the cell object, :func:`FFTDF.get_pp` and :func:`FFTDF.get_nuc`
produce the same integrals.  Depending on the input k-point(s),
the two functions can produce the nuclear-type integrals for a single k-point or
a list of nuclear-type integrals for the k-points.  By default, they compute the
nuclear-type integrals of Gamma point::

    >>> vnuc = fftdf.get_pp()
    >>> print(vnuc.shape)
    (2, 2)
    >>> kpts = cell.make_kpts([2,2,2])
    >>> vnuc = fftdf.get_pp(kpts)
    >>> print(vnuc.shape)
    (8, 2, 2)
    >>> vnuc = fftdf.get_pp(kpts)
    >>> print(vnuc.shape)
    (2, 2)


Hartree-Fock Coulomb and exchange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`FFTDF` class provides a method :func:`FFTDF.get_jk` to compute
Hartree-Fock Coulomb matrix (J) and exchange matrix (K).  This method can take
one density matrix or a list of density matrices as input and return the J and K
matrices for each density matrix::

    >>> dm = numpy.random.random((2,2))
    >>> j, k = fftdf.get_jk(dm)
    >>> print(j.shape)
    (2, 2)
    >>> dm = numpy.random.random((3,2,2))
    >>> j, k = fftdf.get_jk(dm)
    >>> print(j.shape)
    (3, 2, 2)

When k-points are specified, the input density matrices should have the correct
shape that matches the number of k-points::

    >>> kpts = cell.make_kpts([1,1,3])
    >>> dm = numpy.random.random((3,2,2))
    >>> j, k = fftdf.get_jk(dm, kpts=kpts)
    >>> print(j.shape)
    (3, 2, 2)
    >>> dm = numpy.random.random((5,3,2,2))
    >>> j, k = fftdf.get_jk(dm, kpts=kpts)
    >>> print(j.shape)
    (5, 3, 2, 2)


4-index ERI tensor and integral transformation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4-index electron repulsion integrals can be computed with :func:`FFTDF.get_eri`
and :func:`FFTDF.ao2mo` methods.  Given 4 k-points(s) (corresponding to the 4
AO indices), :func:`FFTDF.get_eri` method produce the regular 4-index ERIs
:math:`(ij|kl)` in AO basis.  The 4 k-points should follow the law of momentum
conservation

.. math::
    (\mathbf{k}_j - \mathbf{k}_i + \mathbf{k}_l - \mathbf{k}_k) \cdot a = 2n\pi.

By default, four :math:`\Gamma`-points are assigned to the four AO indices.
As the format of molecular ERI tensor, the PBC ERI tensor is reshaped to a 2D
array::

    >>> eri = fftdf.get_eri()
    >>> print(eri.shape)
    (4, 4)
    >>> eri = fftdf.get_eri([kpts[0],kpts[0],kpts[1],kpts[1]])
    >>> print(eri.shape)
    (4, 4)

:func:`FFTDF.ao2mo` function applies integral transformation for the given four
sets of orbital coefficients, four input k-points.  The four k-points need to
follow the momentum conservation law.  Similar to :func:`FFTDF.get_eri`, the
returned integral tensor is shaped to a 2D array::

    >>> orbs = numpy.random.random((4,2,2))
    >>> eri_mo = fftdf.get_eri(orbs, [kpts[0],kpts[0],kpts[1],kpts[1]])
    >>> print(eri_mo.shape)
    (4, 4)


Kinetic energy cutoff
^^^^^^^^^^^^^^^^^^^^^

The accuracy of FFTDF integrals are affected by the kinetic energy cutoff.  The
default kinetic energy cutoff is a conservative estimation based on the basis
set and the lattice parameter.  You can adjust the attribute :attr:`FFTDF.gs`
(the numbers of grid points in each positive direction) to change the kinetic
energy cutoff.  If any values in :attr:`FFTDF.gs` is too small to reach the
required accuracy :attr:`cell.precision`, :class:`FFTDF` may output a warning
message, eg::

  WARN: ke_cutoff/gs (12.437 / [3, 4, 4]) is not enough for FFTDF to get integral accuracy 1e-08.
  Coulomb integral error is ~ 2.6 Eh.
  Recomended ke_cutoff/gs are 538.542 / [20 20 20].

In this warning message, ``Coulomb integral error`` is a rough estimation for
the largest error of the matrix elements of the two-electron Coulomb integrals.
The overall computational error may be varied by 1 - 2 orders of magnitude.


AFTDF --- AFT-based density fitting
-----------------------------------

AFTDF mans that the Fourier transform of the orbital pair is computed
analytically

.. math::
    \rho_{ij}(\mathbf{G}) = \int e^{-\mathbf{G}\cdot\mathbf{r}} \phi_i(\mathbf{r})\phi_j(\mathbf{r}) d^3\mathbf{r}

To enable AFTDF in the calculation, :class:`AFTDF` object can be initialized
and assigned to :attr:`with_df` object of mean-field object::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> aft = df.AFTDF(cell)
    >>> print(aft)
    <pyscf.pbc.df.aft.AFTDF object at 0x7ff8b1893d90>
    >>> mf = scf.RHF(cell)
    >>> mf.with_df = aft

Generally, AFTDF is slower than FFTDF method.

:class:`AFTDF` class offers the same methods as the :class:`FFTDF` class.
Nuclear and PP integrals, Hartree-Fock J and K matrices, electron repulsion
integrals and integral transformation can be computed with functions
:func:`AFTDF.get_nuc`, :func:`AFTDF.get_pp`, :func:`AFTDF.get_jk`,
:func:`AFTDF.get_eri` and :func:`AFTDF.ao2mo` using the same calling APIs as the
analogy functions in :ref:`fftdf`.


Kinetic energy cutoff
^^^^^^^^^^^^^^^^^^^^^

:class:`AFTDF` also makes estimation on the kinetic energy cutoff.  When the
any values of :attr:`AFTDF.gs` are too small for required accuracy
:attr:`cell.precision`, this class also outputs the
``Coulomb integral error`` warning message as the :class:`FFTDF` class.


.. _pbc_gdf:

GDF --- Gaussian density fitting
--------------------------------

GDF is an analogy of the conventional density fitting method with periodic
boundary condition.  The auxiliary fitting basis in PBC GDF is periodic Gaussian
function (To ensure the long range Coulomb integrals converging in the real
space lattice summation, the multipoles are removed from the auxiliary basis).
:class:`GDF` object can be initialized and enabled in the SCF calculation in two
ways::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> gdf = df.GDF(cell)
    >>> mf = scf.RHF(cell)
    >>> mf.with_df = gdf
    >>> mf.run()
    >>> # Using SCF.density_fit method
    >>> mf = scf.RHF(cell).density_fit().run()
    >>> print(mf.with_df)
    <pyscf.pbc.df.df.GDF object at 0x7fec7722aa10>

Similar to the molecular code, :func:`SCF.density_fit` method returns a
mean-field object with :class:`GDF` as the integral engine.  

In the :class:`GDF` method, the DF-integral tensor is precomputed and stored
on disk.  :class:`GDF` method supports both the :math:`\Gamma`-point ERIs and
the ERIs of different k-points.  :attr:`GDF.kpts` should be specified before
initializing :class:`GDF` object.  :class:`GDF` class provides the same APIs as
the :class:`FFTDF` class to compute nuclear integrals and electron Coulomb
repulsion integrals::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> gdf = df.GDF(cell)
    >>> gdf.kpts = cell.make_kpts([2,2,2])
    >>> gdf.get_eri([kpts[0],kpts[0],kpts[1],kpts[1]])

In the mean-field calculation, assigning :attr:`kpts` attribute to mean-field
object updates the :attr:`kpts` attribute of the underlying DF method::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> mf = scf.KRHF(cell).density_fit()
    >>> kpts = cell.make_kpts([2,2,2])
    >>> mf.kpts = kpts
    >>> mf.with_df.get_eri([kpts[0],kpts[0],kpts[1],kpts[1]])

Once the GDF integral tensor was initialized, the :class:`GDF` can be only used
with certain k-points calculations.  An incorrect :attr:`kpts` argument can lead
to a runtime error::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    >>> gdf = df.GDF(cell, kpts=cell.make_kpts([2,2,2]))
    >>> kpt = np.random.random(3)
    >>> gdf.get_eri([kpt,kpt,kpt,kpt])
    RuntimeError: j3c for kpts [[ 0.53135523  0.06389596  0.19441766]
     [ 0.53135523  0.06389596  0.19441766]] is not initialized.
    You need to update the attribute .kpts then call .build() to initialize j3c.

The GDF initialization is very expensive.  To reduce the initialization cost in
a series of calculations, it would be useful to cache the GDF integral tensor in
a file then load them into the calculation when needed.  The GDF integral tensor
can be saved and loaded the same way as we did for the molecular DF method (see
:ref:`sl_cderi`)::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    gdf = df.GDF(cell, kpts=cell.make_kpts([2,2,2]))
    gdf._cderi_to_save = 'df_ints.h5'  # To save the GDF integrals
    gdf.build()

    mf = scf.KRHF(cell, kpts=cell.make_kpts([2,2,2])).density_fit() 
    mf.with_df._cderi = 'df_ints.h5'   # To load the GDF integrals
    mf.run()


Auxiliary Gaussian basis
^^^^^^^^^^^^^^^^^^^^^^^^

GDF method requires a set of Gaussian functions as the density fitting auxiliary basis.
See also :ref:`df_auxbasis` and :ref:`df_etb_auxbasis` for the choices of DF auxiliary
basis in PySCF GDF code.  There are not many optimized auxiliary basis sets available
for PBC AO basis.  You can use the even-tempered Gaussian functions as the
auxiliary basis in the PBC GDF method::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    gdf = df.GDF(cell, kpts=cell.make_kpts([2,2,2]))
    gdf.auxbasis = df.aug_etb(cell, beta=2.0)
    gdf.build()


Kinetic energy cutoff
^^^^^^^^^^^^^^^^^^^^^

GDF method does not require the specification of kinetic energy cutoff.
:attr:`cell.ke_cutoff` and :attr:`cell.gs` are ignored in the :class:`GDF`
class.  Internally, a small set of planewaves is used in the GDF method to
accelerate the convergence of GDF integrals in the real space lattice summation.
The estimated energy cutoff is generated in the :class:`GDF` class and stored in
the attribute :class:`GDF.gs`.  It is not recommended to change this parameter.


.. _pbc_mdf:

MDF --- mixed density fitting
-----------------------------

MDF method combines the AFTDF and GDF in the same framework.  The MDF auxiliary
basis is Gaussian and plane-wave mixed basis.  :class:`MDF` object can be
created in two ways::

    >>> import numpy as np
    >>> from pyscf.pbc import gto, df, scf
    >>> cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g', ke_cutoff=10)
    >>> mdf = df.MDF(cell)
    >>> print(mdf)
    <pyscf.pbc.df.mdf.MDF object at 0x7f4025120a10>
    >>> mf = scf.RHF(cell).mix_density_fit().run()
    >>> print(mf.with_df)
    <pyscf.pbc.df.mdf.MDF object at 0x7f7963390a10>

The kinetic energy cutoff is specified in this example to constrain the number of
planewaves.  The number of planewaves can also be controlled by through
attribute :attr:`MDF.gs`.

In principle, the accuracy of MDF method can be increased by adding
more plane waves in the auxiliary basis.  In practice, the linear dependency
between plane waves and Gaussians may lead to numerical stability issue.
The optimal accuracy (with reasonable computational cost) requires a reasonable
size of plan wave basis with a reasonable linear dependency threshold.  A
threshold too large would remove many auxiliary functions while a threshold too
small would cause numerical instability.
.. In our preliminary test, ``ke_cutoff=10`` is able to produce 0.1 mEh accuracy in
.. total energy.
The default linear dependency threshold is 1e-10.  The threshold can be adjusted
through the attribute :attr:`MDF.linear_dep_threshold`.

Like the GDF method, it is also very demanding to initialize the 3-center
Gaussian integrals in the MDF method.  The 3-center Gaussian integral tensor can
be cached in a file and loaded to :class:`MDF` object at the runtime::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g')
    mdf = df.MDF(cell, kpts=cell.make_kpts([2,2,2]))
    mdf._cderi_to_save = 'df_ints.h5'  # To save the GDF integrals
    mdf.build()

    mf = scf.KRHF(cell, kpts=cell.make_kpts([2,2,2])).mix_density_fit() 
    mf.with_df._cderi = 'df_ints.h5'   # To load the GDF integrals
    mf.run()


All-electron calculation
------------------------

All-electron calculations with FFTDF or AFTDF methods requires high energy cutoff
for most elements.  It is recommended to use GDF or MDF methods in the
all-electron calculations.  In fact, GDF and MDF can also be used in PP
calculations to reduce the number of planewave basis if steep functions are
existed in the AO basis.


Low-dimension system
--------------------

.. In 1.4 release, FFTDF module does not support low-dimension pbc system.

:class:`AFTDF` supports the systems with 0D (molecule), 1D and 2D periodic
boundary conditions.  When computing the integrals of low-dimension systems, an
infinite vacuum is placed on the free boundary.  You can set the
:attr:`cell.dimension`, to enable the integral algorithms for
low-dimension systems in :class:`AFTDF` class::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g', dimension=1)
    aft = df.AFTDF(cell)
    aft.get_eri()

:class:`GDF` and :class:`MDF` all support the integrals of low-dimension system.
Similar to the usage of AFTDF method, you need to set :attr:`cell.dimension` for
the low-dimension systems::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g', dimension=1)
    gdf = df.GDF(cell)
    gdf.get_eri()

See more examples in ``examples/pbc/31-low_dimensional_pbc.py``


Interface to molecular DF-post-HF methods
=========================================

PBC DF object is compatible to the molecular DF object.  The
:math:`\Gamma`-point PBC SCF object can be directly passed to molecular DF
post-HF methods for an electron correlation calculations in PBC::

    import numpy as np
    from pyscf.pbc import gto, df, scf
    from pyscf import cc as mol_cc
    cell = gto.M(atom='He 1 1 1', a=np.eye(3)*2, basis='3-21g', dimension=1)
    mf = scf.RHF(cell).density_fit()
    mol_cc.RCCSD(mf).run()


Examples
========

DF relevant examples can be found in the PySCF examples directory::

    examples/pbc/10-gamma_point_scf.py
    examples/pbc/11-gamma_point_all_electron_scf.py
    examples/pbc/12-gamma_point_post_hf.py
    examples/pbc/20-k_points_scf.py
    examples/pbc/21-k_points_all_electron_scf.py
    examples/pbc/30-ao_integrals.py
    examples/pbc/30-ao_value_on_grid.py
    examples/pbc/30-mo_integrals.py
    examples/pbc/31-low_dimensional_pbc.py


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

