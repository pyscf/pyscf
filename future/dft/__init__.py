import sys
import vxc

import rks

curmod = sys.modules[__name__]
for k,v in vxc.XC_CODES.items():
    setattr(curmod, k, v)


def RKS(mol, *args):
    if not mol.symmetry or mol.pgname is 'C1':
        return rks.RKS(mol, *args)
    else:
        raise ValueError('symmetry is not implemented')
        return rks_symm.RKS(mol, *args)
