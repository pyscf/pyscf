from pyscf.pbc import scf
from pyscf.pbc.mp import mp2

def RMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)

MP2 = RMP2

def UMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_uhf(mf)
    return mp2.UMP2(mf, frozen, mo_coeff, mo_occ)

def GMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_ghf(mf)
    return mp2.GMP2(mf, frozen, mo_coeff, mo_occ)
