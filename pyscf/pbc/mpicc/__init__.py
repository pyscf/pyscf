from pyscf.pbc.mpicc import ccsd

#def CCSD(mf, frozen=[]):
#    return ccsd.CCSD(mf, frozen)

#def RCCSD(mf, frozen=[]):
#    return ccsd.RCCSD(mf, frozen)

#def KCCSD(mf, frozen=[]):
#    from pyscf.pbc.mpicc import kccsd
#    return kccsd.CCSD(mf, frozen)

def KRCCSD(mf, frozen=[]):
    from pyscf.pbc.mpicc import kccsd_rhf
    return kccsd_rhf.RCCSD(mf, frozen)
