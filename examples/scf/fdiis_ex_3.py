from pyscf import gto, scf, dft
import scipy
import scipy.linalg
import time
import numpy.linalg
import numpy
import sys

atom_str = 'Li 0.0 0.0 0.0; F 4.5 0.0 0.0'
mol = gto.M(atom=atom_str, basis='6-31+g', spin=0, verbose=4)

mf = scf.RHF(mol)
m3 = scf.M3SOSCF(mf, 4, stepsize=0.3, initGuess='1e')
m3.converge()





