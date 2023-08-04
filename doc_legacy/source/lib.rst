:mod:`lib` --- Helper functions, parameters, and C extensions
*************************************************************

.. module:: lib
   :synopsis: Helper functions, parameters, and C extensions.
 

:mod:`lib.parameters`
=====================

.. automodule:: lib.parameters


:mod:`lib.logger`
=================

.. automodule:: lib.logger

Logger object
-------------
.. autoclass:: lib.logger.Logger

.. autofunction:: lib.logger.new_logger


numpy extensions
================

.. automodule:: lib.numpy_helper
   :members:


scipy extensions
================

.. automodule:: lib.linalg_helper
   :members:


:mod:`lib.chkfile`
==================

Chkfile is a HDF5 file.

Functions to access key/value in chkfile
----------------------------------------

.. automodule:: lib.chkfile
   :members: save, load, save_mol, load_mol


Quickly loading object from chkfile
-----------------------------------

The results of SCF and MCSCF methods are saved as a Python dictionary in
the chkfile.  One can fast load the results and update the SCF and MCSCF
objects using the python built in methods ``.__dict__.update``, e.g.::

    from pyscf import gto, scf, mcscf, lib
    mol = gto.M(atom='N 0 0 0; N 1 1 1', basis='ccpvdz')
    mf = mol.apply(scf.RHF).set(chkfile='n2.chk).run()
    mc = mcscf.CASSCF(mf, 6, 6).set(chkfile='n2.chk').run()

    # load SCF results
    mf = scf.RHF(mol)
    mf.__dict__.update(lib.chkfile.load('n2.chk', 'scf'))

    # load MCSCF results
    mc = mcscf.CASCI(mf, 6, 6)
    mc.__dict__.update(lib.chkfile.load('n2.chk', 'mcscf'))
    mc.kernel()


:mod:`lib.diis`
===============

.. automodule:: lib.diis
   :members: DIIS, restore


Other helper functions
======================

Background mode
---------------
.. autofunction:: lib.call_in_background


Temporary HDF5 file
-------------------
.. autoclass:: lib.H5TmpFile


OpenMP threads
--------------
.. autofunction:: lib.num_threads

.. autoclass:: lib.with_omp_threads


Capture stdout
--------------
.. autoclass:: lib.capture_stdout


Other helper functions in :mod:`lib.misc`
-----------------------------------------
.. automodule:: lib.misc
  :members: flatten, light_speed
