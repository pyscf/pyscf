from pyscf.pbc.cc import ccsd

def CCSD(mf, frozen=[]):
    return ccsd.CCSD(mf, frozen)

def RCCSD(mf, frozen=[]):
    return ccsd.RCCSD(mf, frozen)
