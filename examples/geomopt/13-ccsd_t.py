#!/usr/bin/env python

'''
CCSD(T) does not have the interface to the geometry optimizer berny_solver.
You need to define a function to compute CCSD(T) total energy and gradients
then use "as_pyscf_method" to pass them to berny_solver.

See also  examples/geomopt/02-as_pyscf_method.py
'''

from pyscf import gto
from pyscf import scf
from pyscf import cc
from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
from pyscf.grad import ccsd_t as ccsd_t_grad
from pyscf.geomopt import berny_solver

mol = gto.M(
    verbose = 3,
    atom = [
        ['O' , (0. , 0.     , 0.    )],
        ['H' , (0. ,-0.757  ,-0.587)],
        ['H' , (0. , 0.757  ,-0.587)]],
    basis = 'ccpvdz'
)
mf = scf.RHF(mol)
cc_scan = cc.CCSD(mf).as_scanner()

def f(mol):
    # Compute CCSD(T) energy
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    et_correction = mycc.ccsd_t()
    e_tot = mycc.e_tot + et_correction

    # Compute CCSD(T) gradients
    g = ccsd_t_grad.Gradients(mycc).kernel()
    print('CCSD(T) nuclear gradients:')
    print(g)
    return e_tot, g

fake_method = berny_solver.as_pyscf_method(mol, f)

new_mol = berny_solver.optimize(fake_method)

print('Old geometry:')
print(mol.tostring())

print('New geometry:')
print(new_mol.tostring())

