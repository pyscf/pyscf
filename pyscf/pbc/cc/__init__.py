from pyscf.pbc.cc import ccsd

def CCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = _convert_to_uhf(mf)
    return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def RCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = _convert_to_rhf(mf)
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

def KCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd
    return kccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def KRCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd_rhf
    return kccsd_rhf.RCCSD(mf, frozen, mo_coeff, mo_occ)

def _convert_to_rhf(mf):
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_rhf(mf)
    return mf

def _convert_to_uhf(mf):
    from pyscf.pbc import scf
    if not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)
    return mf
