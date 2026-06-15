#!/usr/bin/env python
'''
Save and load PySCF objects via the TREXIO file format.

TREXIO (https://github.com/TREX-CoE/trexio) is a portable, standardized
file format used in particular for handoff between quantum chemistry
codes and QMC packages.

Requires the ``trexio`` Python package: ``pip install trexio``.
'''

from pyscf import gto, scf
from pyscf.tools import trexio

# 1) Run an SCF and dump everything (geometry, basis, MOs, AO integrals,
#    ERIs) into a single TREXIO file.
mol = gto.M(
    atom='''
    O      0.000  0.000  0.117
    H      0.000  0.756 -0.467
    H      0.000 -0.756 -0.467
    ''',
    basis='ccpvdz',
)
mf = scf.RHF(mol).run()
trexio.to_trexio(mf, 'h2o.trexio',
                 backend='TEXT',     # use 'HDF5' if trexio was built with HDF5
                 with_ao_ints=True,  # write S, T, V_ne, h_core, dipole
                 with_eri=True,      # write 8-fold-symmetric AO ERIs
                 with_mo_eri=True)   # write 8-fold-symmetric MO ERIs

# 2) Reconstruct a mean-field object from the file.
mf2 = trexio.scf_from_trexio('h2o.trexio', backend='TEXT')
print('SCF energy from file: %.10f' % mf2.e_tot)

# 3) Read just the geometry/basis if you only need the Mole.
mol2 = trexio.mol_from_trexio('h2o.trexio', backend='TEXT')
print('nao =', mol2.nao_nr())

# 4) Pull AO integrals back (reordered to PySCF AO order).
ints = trexio.read_ao_1e_integrals('h2o.trexio', mol=mol2, backend='TEXT')
print('overlap shape =', ints['overlap'].shape)
eri = trexio.read_ao_2e_integrals('h2o.trexio', mol=mol2, backend='TEXT')
print('AO eri shape =', eri.shape)
mo_eri = trexio.read_mo_2e_integrals('h2o.trexio', backend='TEXT')
print('MO eri shape =', mo_eri.shape)
