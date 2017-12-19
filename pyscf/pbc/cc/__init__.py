from pyscf.pbc import scf
from pyscf.pbc.cc import ccsd

def RCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

CCSD = RCCSD

def UCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_uhf(mf)
    return ccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)

def GCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_ghf(mf)
    return ccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KGCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd
    mf = scf.addons.convert_to_ghf(mf)
    return kccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd_rhf
    mf = scf.addons.convert_to_rhf(mf)
    return kccsd_rhf.RCCSD(mf, frozen, mo_coeff, mo_occ)

KCCSD = KRCCSD

def KUCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    raise NotImplementedError
    from pyscf.pbc.cc import kccsd_uhf
    mf = scf.addons.convert_to_uhf(mf)
    return kccsd_uhf.UCCSD(mf, frozen, mo_coeff, mo_occ)
