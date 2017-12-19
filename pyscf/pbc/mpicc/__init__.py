from pyscf.pbc.cc import ccsd

def KRCCSD(mf, frozen=[]):
    from pyscf.pbc.mpicc import kccsd_rhf
    return kccsd_rhf.RCCSD(mf, frozen)
