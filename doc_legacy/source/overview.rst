An overview of PySCF
********************

Python-based simulations of chemistry framework (PySCF) is a general-purpose
electronic structure platform designed from the ground up to emphasize code
simplicity, so as to facilitate new method development and enable flexible
computational workflows. The package provides a wide range of tools to support
simulations of finite-size systems, extended systems with periodic boundary
conditions, low-dimensional periodic systems, and custom Hamiltonians, using
mean-field and post-mean-field methods with standard Gaussian basis functions.
To ensure ease of extensibility, PySCF uses the Python language to implement
almost all of its features, while computationally critical paths are
implemented with heavily optimized C routines. Using this combined Python/C
implementation, the package is as efficient as the best existing C or Fortran-
based quantum chemistry programs.


How to cite
===========
Bibtex entry::

  @Misc{PYSCF,
    title = {PySCF: the Python‐based simulations of chemistry framework},
    author = {Qiming Sun and Timothy C. Berkelbach and Nick S. Blunt and George H. Booth and Sheng Guo and Zhendong Li and Junzi Liu and James D. McClain and Elvira R. Sayfutyarova and Sandeep Sharma and Sebastian Wouters and Garnet Kin‐Lic Chan},
    year = {2017},
    journal = {Wiley Interdisciplinary Reviews: Computational Molecular Science},
    volume = {8},
    number = {1},
    pages = {e1340},
    doi = {10.1002/wcms.1340},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/wcms.1340},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1340},
  }

In addition, if you use Libcint to compute integrals, please cite the following paper:

"Libcint: An efficient general integral library for Gaussian basis functions", Q. Sun, J. Comp. Chem. 36, 1664 (2015).

Features
========

.. include:: features.txt

* Interface to integral package `Libcint <https://github.com/sunqm/libcint>`_

* Interface to DMRG `CheMPS2 <https://github.com/SebWouters/CheMPS2>`_

* Interface to DMRG `Block <https://github.com/sanshar/Block>`_

* Interface to FCIQMC `NECI <https://github.com/ghb24/NECI_STABLE>`_

* Interface to XC functional library `XCFun <https://github.com/dftlibs/xcfun>`_

* Interface to XC functional library `Libxc <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc>`_

* Interface to tensor contraction library `TBLIS <https://github.com/devinamatthews/tblis>`_

* Interface to Heat-bath Selected CI program `Dice <https://sanshar.github.io/Dice/>`_

* Interface to geometry optimizer `Pyberny <https://github.com/jhrmnn/pyberny>`_

.. * Interface to `pyWannier90 <https://github.com/hungpham2017/pyWannier90>`_


Designs
=======
Kernel and Stream functions
---------------------------

Every class has the ``kernel`` method which serves as the entry or the driver of
the method. Once an object of one method was created, you can always call
``.kernel()`` to start or restart a calculation.

The return value of kernel method is different for different class. To unify the
return value, the package introduces the stream methods to pipe the computing
stream.  A stream method of an object only return the object itself.  There are
three general stream methods available for most method classes. They are:

1 ``.set`` method to update object attributes, for example::

  mf = scf.RHF(mol).set(conv_tol=1e-5)

is identical to two lines of statements::

  mf = scf.RHF(mol)
  mf.conv_tol = 1e-5

2 ``.run`` method to pass the call to the ``.kernel`` method.  If arguments are
presented in ``.run`` method, the arguments will be passed to the kernel
function.  If keyword arguments are given, ``.run`` method will first
call ``.set`` method to update the attributes then execute the ``.kernel``
method.  For example::

  mf = scf.RHF(mol).run(dm_init, conv_tol=1e-5)

is identical to three lines of statements::

  mf = scf.RHF(mol)
  mf.conv_tol = 1e-5
  mf.kernel(dm_init)

3 ``.apply`` method to pass the current object (as the first argument) to the
given function/class and return a new object.  If arguments and keyword
arguments are presented, they will all be passed to the function/class. For
example::

  mc = mol.apply(scf.RHF).run().apply(mcscf.CASSCF, 6, 4, frozen=4)
  
is identical to::

  mf = scf.RHF(mol)
  mf.kernel()
  mc = mcscf.CASSCF(mf, 6, 4, frozen=4)

Aside from the three general stream methods, the regular class methods may
return the objects as well when the methods do not have particular value to
return.  Using the stream methods, you can evaluate certain quantities with one
line of code::

  dm = gto.M(atom='H 0 0 0; H 0 0 1') \
  .apply(scf.RHF) \
  .dump_flags() \
  .run() \
  .make_rdm1()



Pure function and Class
-----------------------

Class are designed to hold only the final results and the control parameters
such as maximum number of iterations, convergence threshold, etc.
Intermediates are NOT saved in the class.  After calling the ``.kernel()`` or
``.run()`` method, results will be generated and saved in the object. For
example::

  from pyscf import gto, scf, ccsd
  mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='ccpvtz')
  mf = scf.RHF(mol).run()
  mycc = ccsd.CCSD(mf).run()
  print(mycc.e_tot)
  print(mycc.e_corr)
  print(mycc.t1.shape)
  print(mycc.t2.shape)

Many useful functions are defined at both the module level and class level. They
can be accessed from either the module functions or the class methods and the
return values should be the same::

  vj, vk = scf.hf.get_jk(mol, dm)
  vj, vk = SCF(mol).get_jk(mol, dm)

Note some module functions may require the class as the first argument.

Most functions and classes are **pure**, i.e. no intermediate status are held
within the classes, and the argument of the methods and functions are immutable
during calculations.  These functions can be called arbitrary times in
arbitrary order and their returns should be always the same.

Exceptions are often suffixed with underscore in the function name, e.g.
``mcscf.state_average_(mc)`` where the attributes of ``mc`` object may be
changed or overwritten by the ``state_average_`` method.  Cautious should be
taken when you see the functions or methods with ugly suffices.


.. _global_config:

Global configurations
---------------------

Global configuration file is a Python script that contains PySCF configurations.
When importing ``pyscf`` module in a Python program (or Python interpreter), the
package will preload the global configuration file and take the configurations
as the default values of the parameters of functions or attributes of classes
during initialization.  For example, the configuration file below detects the
available memory in the operate system at the runtime and set the maximum memory
for PySCF::

  $ cat ~/.pyscf_conf.py
  import psutil
  total, available, percent, used, free, active, inactive, buffers, cached, shared = psutil.virtual_memory()
  MAX_MEMORY = available

By setting ``MAX_MEMORY`` in the global configuration file, you don't need the
statement to set the ``max_memory`` attribute in every script. The dynamically
determined ``max_memory`` will be loaded during the program initialization step
automatically.

There are two methods to let the PySCF package load the global configurations.
One is to create a configuration file ``.pyscf_conf.py`` in home directory or
in work directory.  Another is to set the environment variable
``PYSCF_CONFIG_FILE`` which points to the configuration file (with the absolute
path).  The environment variable ``PYSCF_CONFIG_FILE`` has high priority than
the configuration file in default locations (home directory or work directory).
If environment variable ``PYSCF_CONFIG_FILE`` is available, the program will
read the configurations from the ``$PYSCF_CONFIG_FILE``. If
``PYSCF_CONFIG_FILE`` is not set or the file it points to does not exist, the
program will turn to the default location for the file ``.pyscf_conf.py``.  If
none of the configuration file exists, the program will use the built-in
configurations which are generally conservative settings.

In the source code, global configurations are loaded by importing
:mod:`pyscf.__config__` module::

  from pyscf import __config__
  MAX_MEMORY = getattr(__config__, 'MAX_MEMORY')

Please refer to the source code for the available configurations.


.. _scanner:

Scanner
-------

Scanner is a function that takes an ``Mole`` (or ``Cell``) object as input and
return the energy or nuclear gradients of the given ``Mole`` (or ``Cell``)
object.  Scanner can be considered as a shortcut function for a sequence of
statements which includes the initialization of a required calculation model
with necessary precomputing, next updating the attributes based on the settings
of the referred object, then calling kernel function and finally returning
results.  For example::

  cc_scanner = gto.M().apply(scf.RHF).apply(cc.CCSD).as_scanner()
  for r in (1.0, 1.1, 1.2):
    print(cc_scanner(gto.M(atom='H 0 0 0; H 0 0 %g'%r)))

An equivalent but slightly complicated code is::

  for r in (1.0, 1.1, 1.2):
    mol = gto.M(atom='H 0 0 0; H 0 0 %g'%r)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    print(mycc.e_tot)

There are two types of scanner available in the package.  They are *energy
scanner* and *nuclear gradients scanner*.  The example above is the energy
scanner.  Energy scanner only returns the energy of the given molecular
structure while the nuclear gradients scanner returns the nuclear gradients in
addition.

Scanner is a special derived object of the caller.  Most methods which are
defined in the caller class can be used with the scanner object. For example::

  mf_scanner = gto.M().apply(scf.RHF).as_scanner()
  mf_scanner(gto.M(atom='H 0 0 0; H 0 0 1.2'))
  mf_scanner.analyze()
  dm1 = mf_scanner.make_rdm1()

  mf_grad_scanner = mf_scanner.nuc_grad_method().as_scanner()
  mf_grad_scanner(gto.M(atom='H 0 0 0; H 0 0 1.2'))

As shown in the example above, the scanner works pretty close to the relevant
class object except that the scanner does not need the ``kernel`` or ``run``
methods to run a calculation.  Given molecule structure, the scanner
automatically checks and updates the necessary object dependence and passes the
work flow to the ``kernel`` method.  The computational results are held in the
scanner object as the regular class object does.

To make structure of scanner object uniform for all methods, two attributes
(``.e_tot`` and ``.converged``) are defined for all energy scanner
and three attributes (``.e_tot``, ``.de`` and ``.converged``) are defined for
all nuclear gradients scanner.

