from pyscf import gto, scf, cc
import numpy as np
mol = gto.Mole()
dista = 1.5
mol.atom = [['Be',(0.0, 0.0, 0.0)], ['H',(dista, 1.344-0.46*dista, 0.0)],['H',(dista, -1.344+0.46*dista, 0.0)]]
mol.spin = 0
mol.basis = 'cc-pvdz'
mol.build()

# df-cRHF
dfrhf = scf.RHF(mol).density_fit()
dfrhf.verbose = 4
dfrhf.kernel()

dm = dfrhf.make_rdm1() + 0j
dm[0,:] += .1j
dm[:,0] -= .1j
dfrhf.kernel(dm)

stab = dfrhf.stability()
dm = 2 * np.dot(stab[0][:,:mol.nelectron // 2],stab[0][:,:mol.nelectron // 2].T.conj())
dfrhf.kernel(dm)

# df-cCCSD
dfccc = cc.RCCSD(dfrhf)
dfccc.kernel()

# cRCCSD
rrhf = scf.RHF(mol)
rrhf.mo_coeff = dfccc.mo_coeff
rrhf.mo_occ = dfccc.mo_occ
eri = mol.intor('int2e')
rrhf._eri = eri
rcc = cc.CCSD(rrhf)
rcc.kernel()

dE = abs(rcc.e_corr - dfccc.e_corr)
assert dE < 1e-4
