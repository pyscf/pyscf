import numpy as np
import pyscf
import scipy
import scipy.linalg

mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 0.74'
    basis = '6-31g',
)

mf = mol.HF()
S = mf.get_ovlp()

#local = np.s_[:2]
local = np.s_[:]

def make_S121():
    """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
    S1 = mf.get_ovlp()
    nao = mol.nao_nr()
    S2 = S1[local,local]
    S21 = S1[local]
    #s2_inv = np.linalg.inv(s2)
    #p_21 = np.dot(s2_inv, s21)
    # Better: solve with Cholesky decomposition
    # Solve: S2 * p_21 = S21 for p_21
    p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
    #p_12 = np.eye(nao)[:,self.indices]
    p = np.dot(S21.T, p_21)
    return p

S121 = make_S121()
assert np.allclose(S, S121)

e, C = scipy.linalg.eigh(S, b=S)

n = S.shape[-1]
assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(n))




e, v = scipy.linalg.eigh(np.eye(n))
print(e)
print(v)
n = S.shape[-1]
assert np.allclose(np.linalg.multi_dot((v.T, S, v)), np.eye(n))





