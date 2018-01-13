from pyscf.pbc import scf
from pyscf.pbc.ci import cisd

def RCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)

CISD = RCISD

def UCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_uhf(mf)
    return cisd.UCISD(mf, frozen, mo_coeff, mo_occ)

def GCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_ghf(mf)
    return cisd.GCISD(mf, frozen, mo_coeff, mo_occ)
