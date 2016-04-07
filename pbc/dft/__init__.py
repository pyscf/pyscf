from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft.gen_grid import UniformGrids, BeckeGrids


def RKS(mol, *args, **kwargs):
    from pyscf.pbc.dft import rks
    return rks.RKS(mol, *args, **kwargs)


def KRKS(mol, *args, **kwargs):
    from pyscf.pbc.dft import krks
    return krks.KRKS(mol, *args, **kwargs)
