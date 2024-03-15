import numpy
from pyscf import gto, scf, dft

mol = gto.M(atom='Li 0 0 0; Li 2 0 0', basis='6-31g', verbose=9)
mf = scf.HF(mol)
mf.kernel()

#mf2 = dft.RKS(mol, xc='b3lyp')
mf2 = scf.HF(mol)
m3 = scf.M3SOSCF(mf2, 4, init_guess=mf.mo_coeff)
m3.kernel()


