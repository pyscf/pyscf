from pyscf.cc import ccsd
from pyscf.cc import ccsd_lambda
from pyscf.cc import ccsd_rdm

def CCSD(mf, frozen=[]):
    return ccsd.CCSD(mf, frozen)
