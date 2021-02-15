#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Assign FCI wavefunction symmetry
'''

import numpy
from pyscf import gto, scf, fci
from pyscf import ao2mo, symm, mcscf

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='631g', symmetry=True)
m = scf.RHF(mol).run()
norb = m.mo_coeff.shape[1]
nelec = mol.nelec

fs = fci.FCI(mol, m.mo_coeff)
fs.wfnsym = 'E1x'
e, c = fs.kernel(verbose=5)
print('Energy of %s state %.12f' % (fs.wfnsym, e))

#
# Symmetry adapted FCI solver can be called with arbitrary Hamiltonian.  In the
# following example, you need to prepare 1-electron and 2-electron integrals,
# core energy shift, and the symmetry of orbitals.
#
cas_idx = numpy.arange(2,10)
core_idx = numpy.arange(2)
mo_cas = m.mo_coeff[:,cas_idx]
mo_core = m.mo_coeff[:,core_idx]
dm_core = mo_core.dot(mo_core.T) * 2
vhf_core = m.get_veff(mol, dm_core)
h1 = mo_cas.T.dot( m.get_hcore() + vhf_core ).dot(mo_cas)
h2 = ao2mo.kernel(mol, mo_cas)
ecore = (numpy.einsum('ij,ji', dm_core, m.get_hcore())
         + .5 * numpy.einsum('ij,ji', dm_core, vhf_core) + m.energy_nuc())
orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_cas)

norb = mo_cas.shape[1]
ncore = mo_core.shape[1]
nelec = mol.nelectron - ncore * 2
fs = fci.direct_spin0_symm.FCI(mol)
fs.wfnsym = 'A2'
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore, orbsym=orbsym, verbose=5)
print('Energy of %s state %.12f' % (fs.wfnsym, e))

#
# mcscf module has a function h1e_for_cas to generate 1-electron Hamiltonian and
# core energy.
#
cas_idx = numpy.arange(2,10)
core_idx = numpy.arange(2)
mo_cas = m.mo_coeff[:,cas_idx]
mo_core = m.mo_coeff[:,core_idx]
norb = mo_cas.shape[1]
ncore = mo_core.shape[1]
nelec = mol.nelectron - ncore * 2
h1, ecore = mcscf.casci.h1e_for_cas(m, m.mo_coeff, norb, ncore)
h2 = ao2mo.kernel(mol, mo_cas)
orbsym = scf.hf_symm.get_orbsym(mol, m.mo_coeff)[cas_idx]
fs = fci.direct_spin0_symm.FCI(mol)
fs.wfnsym = 'A2'
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore, orbsym=orbsym)
print('Energy of %s state %.12f' % (fs.wfnsym, e))

#
# Using the CASCI object, initialization of 1-electron and 2-electron integrals
# can be further simplified.
#
cas_idx = numpy.arange(2,10)
norb = len(cas_idx)
ncore = 2
nelec = mol.nelectron - ncore * 2
mc = mcscf.CASCI(m, norb, nelec)
mo = mc.sort_mo(cas_idx, base=0)
h1, ecore = mc.get_h1eff(mo)
h2 = mc.get_h2eff(mo)
orbsym = scf.hf_symm.get_orbsym(mol, m.mo_coeff)[cas_idx]
fs = fci.direct_spin0_symm.FCI(mol)
fs.wfnsym = 'A2'
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore, orbsym=orbsym)
print('Energy of %s state %.12f' % (fs.wfnsym, e))


#
# In the following example, the default initial guess of the regular FCI
# solver is different to the symmetry of HF wfn.  In the symmetry adapted FCI
# solver, the initial guess of required wfnsym is generated.  The FCI solution
# has the same symmetry as the HF wfn. 
#
mol = gto.M(atom='C 0. 0. 0.; C 0. 0. 1.24253', basis='6-31g', symmetry=True)
mf = scf.RHF(mol).run()
cas_idx = [2, 3, 4, 5, 6]
norb = len(cas_idx)
ncore = 2
nelec = mol.nelectron - ncore * 2
mc = mcscf.CASCI(mf, norb, nelec)
h1, ecore = mc.get_h1eff(mf.mo_coeff)
h2 = mc.get_h2eff(mf.mo_coeff)
orbsym = scf.hf_symm.get_orbsym(mol, mf.mo_coeff)[cas_idx]

fs = fci.direct_spin0.FCI(mol)  # Regular FCI solver
fs.davidson_only = True
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore)
print('''Using regular FCI solver, FCI converges to
  wfnsym = %s  E(FCI) = %.12f''' % (fci.addons.guess_wfnsym(c, len(cas_idx), nelec, orbsym), e))
hf_as_civec = numpy.zeros_like(c)
hf_as_civec[0,0] = 1
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore, ci0=hf_as_civec)
print('''Using HF determinant as initial guess, FCI converges to
  wfnsym = %s  E(FCI) = %.12f''' % (fci.addons.guess_wfnsym(c, len(cas_idx), nelec, orbsym), e))

fs = fci.direct_spin0_symm.FCI(mol)  # Symmetry adapted FCI solver
fs.wfnsym = 0
e, c = fs.kernel(h1, h2, norb, nelec, ecore=ecore, orbsym=orbsym)
print('''Using symmetry adapted FCI solver, FCI converges to
  wfnsym = %s  E(FCI) = %.12f''' % (fci.addons.guess_wfnsym(c, len(cas_idx), nelec, orbsym), e))

