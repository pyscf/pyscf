import sys
from pyscf.dft import vxc
from pyscf.dft import rks
from pyscf.dft import uks
from pyscf.dft import rks_symm
from pyscf.dft import uks_symm
from pyscf.dft import gen_grid as grid
from pyscf.dft import radi
from pyscf.df import density_fit

# register the XC keywords in module
curmod = sys.modules[__name__]
for k,v in vxc.XC_CODES.iteritems():
    setattr(curmod, k, v)
del(k)
del(v)


def RKS(mol, *args):
    if mol.nelectron == 1:
        return uks.UKS(mol)
    elif not mol.symmetry or mol.groupname is 'C1':
        if mol.spin > 0:
            return rks.ROKS(mol, *args)
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
        return rks.ROKS(mol, *args)
    else:
        return rks_symm.ROKS(mol, *args)

def UKS(mol, *args):
    if not mol.symmetry or mol.groupname is 'C1':
        return uks.UKS(mol, *args)
    else:
        return uks_symm.UKS(mol, *args)

