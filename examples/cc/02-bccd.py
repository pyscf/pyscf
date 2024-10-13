#!/usr/bin/env python

'''
Brueckner coupled-cluster doubles (BCCD) and BCCD(T) calculations for RHF, UHF and GHF.
See also the relevant discussions in https://github.com/pyscf/pyscf/issues/1591 .
'''

from pyscf import gto, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz',
    verbose = 0,
    spin = 0,
)
myhf = mol.HF().run()

mycc = cc.BCCD(myhf).run()
e_r = mycc.e_tot
e_ccsd_t = mycc.ccsd_t()
PSI4_reference = -0.002625521337000
print(f'RHF-BCCD total energy {mycc.e_tot}.')
print(f'BCCD(T) correlation energy {e_ccsd_t}. Difference to Psi4 {e_ccsd_t - PSI4_reference}')
print(f'Max. value in BCCD T1 amplitudes {abs(mycc.t1).max()}')

mygcc = cc.BCCD(myhf.to_ghf()).run()
e_g = mygcc.e_tot
print(f'GHF-BCCD total energy {mygcc.e_tot}.')

# Run BCCD with frozen orbitals
myhf = mol.UHF().run()
myucc = cc.BCCD(myhf)
myucc.frozen = [0]
myucc.kernel()
print(f'UHF-BCCD total energy {myucc.e_tot}.')
