#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Largest CI coefficients
'''

from pyscf import gto, scf, mcscf, fci

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = 'N, 0., 0., 0. ; N,  0., 0., 1.4',
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

ncas = 6
nelec = 6
mc = mcscf.CASSCF(m, 6, 6)
mc.kernel()

# Output all determinants coefficients
print('   det-alpha,    det-beta,    CI coefficients')
occslst = fci.cistring.gen_occslst(range(ncas), nelec//2)
for i,occsa in enumerate(occslst):
    for j,occsb in enumerate(occslst):
        print('   %s       %s      %.12f' % (occsa, occsb, mc.ci[i,j]))

# Only output determinants which have coefficients > 0.05
nelec = (3,3)  # 3 spin-up electrons and 3 spin-down electrons
print('   det-alpha,    det-beta,    CI coefficients')
for c,ia,ib in mc.fcisolver.large_ci(mc.ci, ncas, nelec, tol=.05, return_strs=False):
    print('   %s       %s      %.12f' % (ia, ib, c))
