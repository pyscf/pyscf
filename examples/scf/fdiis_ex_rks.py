from pyscf import gto, dft
import numpy
import pyscf.scf

atom_str = 'C 0.0 0.0 0.0; O -1.0 0.0 0.0; O 1.0 1.0 1.0'
mol = gto.M(atom=atom_str, basis='6-31g')

rks = dft.RKS(mol, xc='cam-b3lyp')
rks.kernel()


m3 = pyscf.scf.M3SOSCF(rks, 10)
m3.converge()


