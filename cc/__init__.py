from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm

def CCSD(mf, frozen=[]):
    return ccsd.CCSD(mf, frozen)

def EOMCCSD(mf, frozen=[]):
    from pyscf.cc import ccsd_eom
    return ccsd_eom.CCSD(mf, frozen)
