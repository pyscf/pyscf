'''
Density functional theory
=========================

Simple usage::

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='def2-tzvp')
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'pbe,pbe'
    >>> mf.run()
'''

try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    pass
try:
    from pyscf.dft import xcfun
except (ImportError, OSError):
    pass
from pyscf.dft import rks
from pyscf.dft import roks
from pyscf.dft import uks
from pyscf.dft import rks_symm
from pyscf.dft import uks_symm
from pyscf.dft import gen_grid as grid
from pyscf.dft import radi
from pyscf.df import density_fit
from pyscf.dft.gen_grid import sg1_prune, nwchem_prune, treutler_prune, \
        stratmann, original_becke, Grids
from pyscf.dft.radi import BRAGG_RADII, COVALENT_RADII, \
        delley, mura_knowles, gauss_chebyshev, treutler, treutler_ahlrichs, \
        treutler_atomic_radii_adjust, becke_atomic_radii_adjust


def RKS(mol, *args):
    if mol.nelectron == 1:
        return uks.UKS(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        if mol.spin > 0:
            return roks.ROKS(mol, *args)
        else:
            return rks.RKS(mol, *args)
    else:
        if mol.spin > 0:
            return rks_symm.ROKS(mol, *args)
        else:
            return rks_symm.RKS(mol, *args)

def ROKS(mol, *args):
    if mol.nelectron == 1:
        return uks.UKS(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        return roks.ROKS(mol, *args)
    else:
        return rks_symm.ROKS(mol, *args)

def UKS(mol, *args):
    if not mol.symmetry or mol.groupname is 'C1':
        return uks.UKS(mol, *args)
    else:
        return uks_symm.UKS(mol, *args)

