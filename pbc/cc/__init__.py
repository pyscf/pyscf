from pyscf.pbc.cc import ccsd

def CCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def RCCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

def KCCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd
    return kccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd_rhf
    return kccsd_rhf.RCCSD(mf, frozen, mo_coeff, mo_occ)
