from pyscf.pbc.mp import kmp2

def KMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    return kmp2.KMP2(mf, frozen, mo_coeff, mo_occ)

