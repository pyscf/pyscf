import numpy as np

default_minao = {
        "gth-dzv" : "gth-szv",
        "gth-dzvp" : "gth-szv",
        "gth-tzvp" : "gth-szv",
        "gth-tzv2p" : "gth-szv",
        }

def einsum(*args, **kwargs):
    kwargs["optimize"] = kwargs.pop("optimize", True)
    return np.einsum(*args, **kwargs)

def get_minimal_basis(basis):
    minao = default_minao.get(basis, "minao")
    return minao

def indices_to_bools(indices, n):
    bools = np.zeros(n, dtype=bool)
    bools[np.asarray(indices)] = True
    return bools

def transform_amplitudes(t1, t2, u_occ, u_vir):
    if t1 is not None:
        t1 = einsum("ia,ix,ay->xy", t1, u_occ, u_vir)
    else:
        t1 = None
    if t2 is not None:
        t2 = einsum("ijab,ix,jy,az,bw->xyzw", t2, u_occ, u_occ, u_vir, u_vir)
    else:
        t2 = None
    return t1, t2
