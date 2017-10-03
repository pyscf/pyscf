from pyscf.mp import mp2
from pyscf.mp import dfmp2
from pyscf.mp.ump2 import UMP2

def MP2(mf, frozen=[], mo_coeff=None, mo_occ=None):
    __doc__ = mp2.MP2.__doc__
    from pyscf import scf

    mf = scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        return dfmp2.MP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.MP2(mf, frozen, mo_coeff, mo_occ)

