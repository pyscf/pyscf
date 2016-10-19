from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm

CCSD = ccsd.CCSD

def RCCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    from pyscf.cc import rccsd
    return rccsd.RCCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)

def UCCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    from pyscf.cc import uccsd
    return uccsd.UCCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)
