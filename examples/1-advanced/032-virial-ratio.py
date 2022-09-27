#!/usr/bin/env python

'''
Virial ratio and correlated virial ratio

Ref. JCP, 118, 2491
'''

import numpy
from pyscf import gto, scf, ci, cc

mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
mf = scf.RHF(mol).run()
dm = mf.make_rdm1()
t_hf = numpy.einsum('ij,ji', dm, mol.intor('int1e_kin'))
v_hf = mf.e_tot - t_hf
print('HF virial ration', -v_hf/t_hf)

myci = ci.CISD(mf).run()
dm1_mo = myci.make_rdm1()
dm1_ao = mf.mo_coeff.dot(dm1_mo).dot(mf.mo_coeff.T)
t_ci = numpy.einsum('ij,ji', dm1_ao, mol.intor('int1e_kin'))
v_ci = myci.e_tot - t_ci
print('CISD virial ration', -v_ci/t_ci)
print('CISD correlated virial ration', -(v_ci-v_hf)/(t_ci-t_hf))

mycc = cc.CCSD(mf).run()
dm1_mo = mycc.make_rdm1()
dm1_ao = mf.mo_coeff.dot(dm1_mo).dot(mf.mo_coeff.T)
t_cc = numpy.einsum('ij,ji', dm1_ao, mol.intor('int1e_kin'))
v_cc = mycc.e_tot - t_cc
print('CCSD virial ration', -v_cc/t_cc)
print('CCSD correlated virial ration', -(v_cc-v_hf)/(t_cc-t_hf))
