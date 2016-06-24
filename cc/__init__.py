from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm

def CCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    return ccsd.CCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)

def EOMCCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    from pyscf.cc import ccsd_eom
    return ccsd_eom.CCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)

def RCCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    from pyscf.cc import rccsd_eom
    return rccsd_eom.RCCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)

def UCCSD(mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
    from pyscf.cc import uccsd_eom
    return uccsd_eom.UCCSD(mf, frozen, mo_energy, mo_coeff, mo_occ)
