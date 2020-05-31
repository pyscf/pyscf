import numpy as np
import pyscf
import scipy
import scipy.linalg

mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 0.74',
    basis = '6-31g',
)

mf = mol.HF()
S = mf.get_ovlp()
n = S.shape[-1]

e, C = scipy.linalg.eigh(S, b=S)
assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(n))

# Should be the same??
A = np.dot(np.linalg.inv(S), S)
assert n.allclose(A, A.T)
assert np.allclose(A, np.eye(n))
e, C = scipy.linalg.eigh(A)

assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(n))
