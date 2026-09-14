#!/usr/bin/env python

'''
Dump PySCF objects to TREXIO files.

trexio.to_trexio dispatches on the type of the object passed in (Mole/Cell,
SCF, or MCSCF) and writes the appropriate quantities.  The examples below
show a few common patterns, taken from the trexio.to_trexio docstring.
'''

# 1. Export only molecular geometry and basis (no SCF)

from pyscf import gto
from pyscf.tools import trexio
mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
trexio.to_trexio(mol, 'hf_mol.h5')

# 2. Export SCF results without integrals or density matrices

from pyscf import gto, scf
from pyscf.tools import trexio
mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
trexio.to_trexio(mf, 'hf_scf.h5',
    write_ao_eri=False, write_mo_eri=False, eri_sym='s1', write_mo_rdm=False)

# 3. Export SCF results with MO integrals and density matrices

from pyscf import gto, scf
from pyscf.tools import trexio
mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
trexio.to_trexio(
    mf, 'hf_full.h5',
    write_ao_eri=False, write_mo_eri=True, eri_sym='s4',
    write_mo_rdm=True,
)

# 4. Export CASSCF results with active-space integrals and density matrices

from pyscf import gto, scf, mcscf
from pyscf.tools import trexio
mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).run()
trexio.to_trexio(
    mc, 'cas.h5',
    write_mcscf_eri=True,
    write_mcscf_rdm=True,
    ci_threshold=1e-3,
)

