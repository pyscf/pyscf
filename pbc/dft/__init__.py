import sys
from pyscf.dft import vxc
from pyscf.pbc.dft import rks
#from pyscf.df import density_fit

# register the XC keywords in module
curmod = sys.modules[__name__]
for k,v in vxc.XC_CODES.items():
    setattr(curmod, k, v)


def RKS(mol, *args):
    return rks.RKS(mol, *args)
