from pyscf.cc import ccsd

def CCSD(mf, frozen=[]):
    return ccsd.CCSD(mf, frozen)
