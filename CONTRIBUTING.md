# Contributing to PySCF

The following is a set of guidelines for contributing to
[PySCF](https://github.com/sunqm/pyscf) package.  These are just guidelines,
not rules.  Feel free to propose changes.

* New features are first placed in dev branch.

* Code at least should work with python-2.7, gcc-4.8.
  - Writing python code compatibile with Python 2.6, 2.7, 3.2 - 3.6.
  - Except complex value and variable length array, following C89 standard for C code.

* Using ctypes to bridge C/python functions.
