from pyscf import gto, scf, dft, mcscf

mol = gto.M(atom='H 0 0 0; O 0 0 1.2', basis='ccpvdz', spin=1, verbose=4)
mf = scf.UHF(mol)
mf.kernel()
mc = mcscf.CASSCF(mf, 4, 3)
mc.kernel()

mf = scf.UKS(mol)
mf.kernel()
mc = mcscf.CASSCF(mf, 4, 3)
mc.kernel()
