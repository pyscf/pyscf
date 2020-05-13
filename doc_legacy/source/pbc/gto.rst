.. _pbc_gto:

pbc.gto --- Crystal cell structure
**********************************
This module provides functions to setup the basic information of a PBC calculation.  The
:mod:`pyscf.pbc.gto` module is analogous to the basic molecular :mod:`pyscf.gto` module.
The :class:`Cell` class for crystal structure unit cells is defined in this module and is
analogous to the basic molecular :class:`Mole` class.  Among other details, the basis set
and pseudopotentials are parsed in this module.

:class:`Cell` class
===================
The :class:`.Cell` class is defined as an extension of the molecular
:class:`pyscf.gto.mole.Mole` class.  The :class:`Cell` object offers much of the same
functionality as the :class:`Mole` object.  For example, one can use the :class:`Cell`
object to access the atomic structure, basis functions, pseudopotentials, and certain
analytical periodic integrals.

Similar to the input in a molecular calculation, one first creates a :class:`Cell` object.
After assigning the crystal parameters, one calls :func:`build` to fully initialize the
:class:`Cell` object.  A shortcut function :func:`M` is available at the module level to
simplify the input.

.. literalinclude:: ../../../examples/pbc/00-input_cell.py

Beyond the basic parameters :attr:`atom` and :attr:`basis`, one needs to set the unit cell
lattice vectors :attr:`a` (a 3x3 array, where each row is a real-space primitive vector)
and the numbers of grid points in the FFT-mesh in each positive direction :attr:`gs` (a
length-3 list or 1x3 array); the total number of grid points is 2 :attr:`gs` +1.

In certain cases, it is convenient to choose the FFT-mesh density based on the kinetic
energy cutoff.  The :class:`Cell` class offers an alternative attribute
:attr:`ke_cutoff` that can be used to set the FFT-mesh.  If :attr:`ke_cutoff` is set and
:attr:`gs` is ``None``, the :class:`Cell` initialization function will convert the
:attr:`ke_cutoff` to the equivalent FFT-mesh 
according to the relation :math:`\mathbf{g} = \frac{\sqrt{2E_{\mathrm{cut}}}}{2\pi}\mathbf{a}^T`
and will overwrite the :attr:`gs` attribute.

Many PBC calculations are best performed using pseudopotentials, which are set via
the :attr:`pseudo` attribute.  Pseudopotentials alleviate the need for impractically
dense FFT-meshes, although they represent a potentially uncontrolled source of error.
See :ref:`pseudo` for further details and a list of available pseudopotentials.

The input parameters ``.a`` and ``.pseudo`` are immutable in the :class:`Cell` object.  We
emphasize that the input format might be different from the internal format used by PySCF.
Similar to the convention in :class:`Mole`, an internal Python data layer is created to
hold the formatted ``.a`` and ``.pseudo`` parameters used as input.

_pseudo
  The internal format to hold PBC pseudo potential parameters.  It is
  represented with nested Python lists only.

Nuclear-nuclear interaction energies are evaluated by means of Ewald summation, which
depends on three parameters: the truncation radius for real-space lattice sums
:attr:`rcut`, the Gaussian model charge :attr:`ew_eta`, and the energy cutoff
:attr:`ew_cut`.  Although they can be set manually, these parameters are by default chosen
automatically according to the attribute :attr:`precision`, which likewise can be set
manually or left to its default value.

Besides the methods and parameters provided by :class:`Mole` class (see Chapter
:ref:`gto`), there are some parameters frequently used in the code to access the
information of the crystal.

kpts
  The scaled or absolute k-points (nkpts x 3 array). This variable is not held as an
  attribute in :class:`Cell` object; instead, the :class:`Cell` object provides functions
  to generate the k-points and convert the k-points between the scaled (fractional) value
  and absolute value::

    # Generate k-points
    n_kpts_each_direction = [2,2,2]
    abs_kpts = cell.make_kpts(n_kpts_each_direction)

    # Convert k-points between two convention, the scaled and the absoulte values
    scaled_kpts = cell.get_scaled_kpts(abs_kpts)
    abs_kpts = cell.get_abs_kpts(scaled_kpts)

Gv
  The (N x 3) array of plane waves associated to :attr:`gs`.  :attr:`gs` defines
  the number of FFT grids in each direction.  :meth:`Cell.Gv` or :meth:`get_Gv`
  convert the FFT-mesh to the plane waves.  ``Gv`` are the the plane wave bases
  of 3D-FFT transformation.  Given ``gs = [nx,ny,nz]``, the number of vectors in
  ``Gv`` is ``(2*nx+1)*(2*ny+1)*(2*nz+1)``.

vol
  :attr:`Cell.vol` gives the volume of the unit cell (in atomic unit).

reciprocal_vectors
  A 3x3 array.  Each row is a reciprocal space primitive vector.

energy_nuc
  Similar to the :func:`energy_nuc` provided by :class:`Mole` class, this
  function also return the energy associated to the nuclear repulsion.  The
  nuclear repulsion energy is computed with Ewald summation technique.  The
  background contribution is removed from the nuclear repulsion energy otherwise
  this term is divergent.

pbc_intor
  PBC analytic integral driver.  It allows user to compute the PBC integral
  array in bulk, for given integral descriptor ``intor`` (see also
  :meth:`Mole.intor` function :ref:`gto_moleintor`).  In the :class:`Cell` object,
  we didn't overload the :meth:`intor` method.  So one can access both the
  periodic integrals and free-boundary integrals within the :class:`Cell`
  object.  It allows you to input the cell object into the molecule program to
  run the free-boundary calculation (see :ref:`cell_to_mol`).

.. note::
  :meth:`pbc_intor` does not support Coulomb type integrals.  Calling pbc_intor
  with Coulomb type integral descriptor such as ``cint1e_nuc_sph`` leads to
  divergent integrals.  The Coulomb type PBC integrals should be evaluated with
  density fitting technique (see Chapter :ref:`pbc_df`).


Attributes and methods
----------------------

.. autoclass:: pyscf.pbc.gto.Cell
   :members:


.. _cell_to_mol:

Connection to :class:`Mole` class
---------------------------------
:class:`.Cell` class is compatible with the molecule
:class:`pyscf.gto.mole.Mole` class.  They shared most data structure and
methods.  It gives the freedom to mix the finite size calculation and the PBC
calculation.  If you feed the cell object to molecule module/functions, the
molecule program will not check whether the given :class:`Mole` object is the
true :class:`Mole` or not.  It simply treats the :class:`Cell` object as the
:class:`Mole` object and run the finite size calculations.  Because the same
module names were used in PBC program and molecule program, you should be
careful with the imported modules since no error message will be raised if you
by mistake input the :class:`Cell` object into the molecule program.

.. In a solid surface program, it allows you easily switching between the
.. periodicity and non-periodicity by calling different module with the same
.. ``Cell`` object.  However, for code readability, it is recommended to cast the
.. cell object into the regular molecule object using :meth:`Cell.to_mol`.

Although we reserve the flexibility to mix the :class:`Cell` and :class:`Mole`
objects in the same code, it should be noted that the serialization methods of
the two objects are not completely compatible.  When you dumps/loads the cell
object in the molecule program, informations of the :class:`Cell` object or the
faked :class:`Mole` object may be lost.


Serialization
-------------
:class:`Cell` class has two set of functions to serialize Cell object in
different formats.

* JSON format is the default serialization format used by :mod:`pyscf.lib.chkfile`
  module.  It can be serialized by :func:`Cell.dumps` function and deserialized
  by :func:`Cell.loads` function.

* In the old version, :func:`Mole.pack` and :func:`Mole.unpack` functions are
  used to convert the :class:`Mole` object to and from Python dict.  The Python
  dict is then serialized by pickle module.  This serialization method is not
  used anymore in the new PySCF code.  To keep the backward compatibility, the
  two methods are defined in :class:`Cell` class.


Basis set
=========
The pbc module supports all-electron calculation.  The all-electron basis sets
developed by quantum chemistry community can be directly used in the pbc
calculation.  The :class:`Cell` class supports to mix the QC all-electron basis
and PBC basis in the same calculation.

.. literalinclude:: ../../../examples/pbc/04-input_basis.py

.. note::

  The default PBC Coulomb type integrals are computed using FFT transformation.
  If the all-electron basis are used, you might need very high energy cutoff to
  converge the integrals.  It is recommended to use mixed density fitting
  technique (:ref:`pbc_df`) to handle the all-electron calculations.


.. _pseudo:

Pseudo potential
================
Quantum chemistry community developed a wide range of pseudo potentials (which
are called ECP, effective core potential) for heavy elements.  ECP works quite
successful in finite system.  It has high flexibility to choose different core
size and relevant basis sets to satisfy different requirements on accuracy,
efficiency in different simulation scenario.  Extending ECP to PBC code enriches
the pseudo potential database.  PySCF PBC program supports both the PBC
conventional pseudo potential and ECP and the mix of the two kinds of potentials
in the same calculation.

.. literalinclude:: ../../../examples/pbc/05-input_pp.py

