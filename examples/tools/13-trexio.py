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


# 5. Check that the written file is valid
#
# trexio-validate (https://github.com/TREX-CoE/trexio-validate) recomputes the
# contents of a TREXIO file from the basis set stored in that same file, using
# libcint, and compares the result with the stored data.  It checks the file
# against the TREXIO specification rather than against PySCF's own
# conventions, so it catches mistakes in the AO ordering, the normalization
# and the index order of the integrals that reading the file back into PySCF
# would not reveal.
#
# It is not on PyPI; build it from source (see the test module
# pyscf/tools/test/test_trexio_validate.py for the cmake invocation) and
# either put its Python module on PYTHONPATH or its executable on PATH.

from pyscf import gto, scf
from pyscf.tools import trexio
mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
trexio.to_trexio(mf, 'hf_ao.h5', write_ao_eri=True, eri_sym='s8')

try:
    import trexio_validate
except ImportError:
    print('trexio-validate is not installed; skipping the validation')
else:
    report = trexio_validate.validate_file('hf_ao.h5',
                                           require=['mo_orthonormality',
                                                    'ao_1e_int_overlap',
                                                    'ao_2e_int_eri'])
    print(report)
    report.raise_for_failure()

# From the command line, the same check is
#
#     trexio-validate --require mo_orthonormality,ao_1e_int_overlap hf_ao.h5
