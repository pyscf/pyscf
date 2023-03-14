from pyscf import gto, scf, dft, symm
import scipy
import scipy.linalg
import time
import numpy.linalg
import numpy
import sys

atom_str = 'H 0.0 0.0 0.0; He 1.0 0.0 0.0; He -1.0 0.0 0.0; He 0.0 1.0 0.0; He 0.0 -1.0 0.0; He 0.0 0.0 1.0; He 0.0 0.0 -1.0'
mol = gto.M(atom=atom_str, basis='6-31+g*', spin=1, verbose=4)

mf = scf.UHF(mol)
m3 = scf.M3SOSCF(mf, 5)
m3.converge()
