'''
Moller-Plesset perturbation theory
'''

from pyscf import scf
from pyscf.mp import mp2
from pyscf.mp import dfmp2
from pyscf.mp import ump2
from pyscf.mp import gmp2

def MP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = mp2.MP2.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)

def RMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = mp2.RMP2.__doc__
    scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)

def UMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = ump2.UMP2.__doc__
    mf = scf.addons.convert_to_uhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-UMP2')
    else:
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)

def GMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = gmp2.GMP2.__doc__
    mf = scf.addons.convert_to_ghf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-GMP2')
    else:
        return gmp2.GMP2(mf, frozen, mo_coeff, mo_occ)

