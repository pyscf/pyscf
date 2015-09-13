.. PySCF documentation master file, created by
   sphinx-quickstart on Thu Jan 15 01:55:04 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySCF's documentation!
=================================

Pyscf is a quantum chemistry package written in python.  The package aims to
provide a simple, light-weight and efficient platform for quantum chemistry
code developing and calculation.  The program is developed in the principle of

* Easy to install, to use, to extend and to be embedded;

* Minimal requirements on libraries (No Boost, MPI) and computing
  resources (perhaps losing efficiency to reduce I/O);

* 90/10 Python/C, only computational hot spots were written in C;

* 90/10 functional/OOP, unless performance critical, functions are pure.

Contents
--------

.. toctree::
   :maxdepth: 2

   overview.rst
   tutorial.rst
   advanced.rst
   install.rst
   gto.rst
   lib.rst
   scf.rst
   ao2mo.rst
   mcscf.rst
   fci.rst
   symm.rst
   df.rst
   dft.rst
   tools.rst

   benchmark.rst
   code-rule.rst
   version.rst

..   dmrgscf.rst
..   fciqmcscf.rst
..   cc.rst
..   lo.rst


You can also download the `PDF version
<http://www.pyscf.org/pdf/PySCF-1.0.pdf>`_ of this manual.


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

