'''
Direct RPA correlation energy
'''

from pyscf import gto, dft, gw
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'sto3g',
    verbose = 5,
    )

mf = dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()

rpa = gw.rpa.dRPA(mf)
rpa.kernel()

