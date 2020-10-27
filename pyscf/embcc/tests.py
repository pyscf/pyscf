import numpy as np

__all__ = [
        "is_hermitian",
        "is_definite",
        ]

def is_hermitian(a, **kwargs):
    return np.allclose(a, a.T.conj(), **kwargs)

def is_definite(a, positive=True, tol=1e-12):
    if not is_hermitian(a):
        return False
    e, v = np.linalg.eigh(a)
    if positive is True:
        return np.all(e > -tol)
    elif positive is False:
        return np.all(e < tol)
    elif positive == "any":
        return (np.all(e > -tol) or np.all(e < -tol))
