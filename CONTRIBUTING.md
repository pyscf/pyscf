# Contributing to PySCF

The development guideline of [PySCF](https://github.com/sunqm/pyscf) program is
to provide an environment which is convenient for method developing, quick
testing and calculations for systems of moderate size.  We emphasize first the
simplicity, next the generality, finally the efficiency.  We favor the
implementations which have clear structure with optimization at Python level.
When Python performance becomes a big bottleneck, C code can be implemented to
improve efficiency.  Following we specify the standard for C/Python interface
and other coding rules in the PySCF program.  The following is a set of
guidelines for contributing to package.  These are just guidelines, not rules.
Feel free to propose changes.


Code standard
=============

* Code at least should work under python-2.7, gcc-4.8.
 
* 90/10 functional/OOP, unless performance critical, functions are pure.
 
* 90/10 Python/C, only computational hot spots were written in C.
 
* To extend python function with C/Fortran:

* Following C99 standard for C code.

* Following Fortran 95 standard (http://j3-fortran.org/doc/standing/archive/007/97-007r2/pdf/97-007r2.pdf)
  for Fortran code.
   
* Using ctypes to interface C/python functions

* Do **not** use other program languages (to keep the package light-weight).

* External C/Fortran libraries.
  These are libraries to be compiled and linked in the PySCF C libraries.  Their
  compiling/linking flags are resolved in the cmake config system.
  - BLAS, FFTW: Yes.
  - LAPACK, ARPACK: Yes but not recommended.  These libraries can be used in the
    PySCF C level library. But we recommend to restructure your code and move
    the relevant linear algebra and sparse matrix operations to Python code.
  - MPI and other parallel libraries: No.  MPI is the only library that can be
    used in the code. The MPI communications should be implemented at python
    level through MPI4py library.

* Minimal requirements on 3rd party programs or libraries.
  - For 3rd party Python library, implementing either back up plan or
    error/exception handler to avoid breaking the import chain
  - 3rd party C library should not be compiled by default. A cmake option in
    CMakeLists.txt can be provided to enable these C libraries

* Not enforced but recommended
  - Compatible with Python 2.6, 2.7, 3.2-3.*.
    To write code that works for both Python 2 and Python 3, you have to choose
    the common set of Python 2 and Python 3.  Please watch out the following
    Python language features which may be regularly used in the program but
    behave differently in Python 2 and 3:
    + Avoiding relative import.
    + Use the % format for print function or `from future import print_function`.
    + Distinguish / and // in python 2.
    + map, zip and filter functions return generator in python 3. Changing the
      return value of map, zip and filter to list, eg `list(zip(a,b))`, to make
      them work in the same way in python 2 and 3.
    + Always import `from functools import reduce` before using reduce function.
    + Avoid dict.items() method.
    + Avoid xrange function.
    + String should contain only ASCII character

  - Python coding style. Whenever possible, follow PEP-8
    https://www.python.org/dev/peps/pep-0008/

  - New features are first placed in dev branch.


Name convention
===============

* The prefix or suffix underscore in the function names have special meanings.
  - functions with prefix-underscore like `_fn` are private functions. They
    are typically not documented, and not recommended to be used outside the
    file where it is defined.
  - functions with suffix-underscore like `fn_` means that they have side
    effects.  The side effects include the change of the input argument, the
    runtime modification of the class definitions (attributes or members), or
    module definitions (global variables or functions) etc.
  - regular (pure) functions do not have underscore in the prefix or suffix.

API convention
==============

* :class:`gto.Mole` and :class:`pbc.gto.Cell` holds all global parameters, like
  the log level, the largest memory allowed etc.  They are used as the default
  value for all other classes.

* Method class.

  - Most QC method classes (like HF, CASSCF, FCI, ...) directly take three
    attributes `verbose`, `stdout` and `max_memory` from
    :class:`gto.Mole`.  Overwriting them only affects the behavior of the local
    instance for that method class.  In the following example, `mf.verbose`
    mutes the output produced by :class:`RHF` method, and the output of
    :class:`MP2` is written in the log file `example.log`::

        >>> from pyscf import gto, scf, mp
        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', verbose=5)
        >>> mf = scf.RHF(mol)
        >>> mf.verbose = 0
        >>> mf.kernel()
        >>> mp2 = mp.MP2(mf)
        >>> mp2.stdout = open('example.log', 'w')
        >>> mp2.kernel()

  - Method class are only to hold the options or environments (like convergence
    threshold, max iterations, ...) to control the behavior/convergence of the
    method.  The intermediate status are **not** supposed to be saved in the
    method class (during the computation).  However, the final results or
    solutions are kept in the method object for convenience.  Once the results
    are generated in the method instance, they are assumed to be read only. To
    simplify the function input arguments, class member functions can take the
    results as the default arguments if the caller did not provide enough
    arguments.

  - In `__init__` function, initialize/define the problem size.  The problem size
    (like num orbitals etc) can be considered as environment parameter.  The
    environment parameters are not supposed to be changed by other functions.
    It is recommended to inherit the class from the :class:`pyscf.lib.StreamObj`,
    and initialize attribute ._keys in the `__init__` function.  Attribute ._keys
    is used for sanity check.

  - Kernel functions
    It is recommended to provide an entrance method called `kernel` for the
    method class.  The kernel function should be able to guide the program flow
    to the right driver function.

  - Return value.
    Create return value for all functions whenever possible.  For methods
    defined in class, return self instead of None if the method does not have
    particular return values.


Unit Tests and Example Scripts
==============================

* Examples to run modules should be placed in the appropriate directory within
  the /examples directory.  While the examples should be easy enough to run on a
  modest personal computer; however, should not be trivial and instead showcase
  the functionality of the module.  The format for naming examples is::

    /examples/name_of_module/XX-function_name.py

  where XX is a two-digit numeric string.

* Test cases are placed in the /test/name_of_module directory and performed with
  nosetest (https://nose.readthedocs.io/en/latest/).  These tests are to ensure
  the robustness of both simple functions and more complex drivers between
  version changes.
