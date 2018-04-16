lib --- Helper functions, parameters, and C extensions
******************************************************

.. automodule:: pyscf.lib
 

parameters
==========

Some PySCF environment parameters are defined in this module.

Scratch directory
-----------------

The PySCF scratch directory is specified by :data:`TMPDIR`.  Its default value
is the same to the system environment variable ``TMPDIR``.  It can be
overwritten by the system environment variable ``PYSCF_TMPDIR``.


.. _max_mem:

Maximum memory
--------------

The variable :data:`MAX_MEMORY` defines the maximum memory that PySCF can be
used in the calculation.  Its unit is MB.  The default value is 4000 MB.  It can
be overwritten by the system environment variable ``PYSCF_MAX_MEMORY``.

.. note:: Some calculations may exceed the max_memory limit, especially
  when the attribute :attr:`Mole.incore_anyway` was set.

.. automodule:: pyscf.lib.parameters
   :members:


logger
======

.. automodule:: pyscf.lib.logger
   :members:


numpy helper
============

.. automodule:: pyscf.lib.numpy_helper
   :members:

.. automodule:: pyscf.lib.linalg_helper
   :members:


chkfile
=======

.. automodule:: pyscf.lib.chkfile
   :members:


Fast load
---------

The results of SCF and MCSCF methods are saved as a Python dictionary in
the chkfile.  One can fast load the results and update the SCF and MCSCF
objects using the python built in methods ``.__dict__.update``, eg::

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


Other helper functions
======================

Background mode
---------------

Macro :func:`call_in_background`


Temporary HDF5 file
-------------------
