from pyscf.pbc.cc import ccsd

def RCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = _convert_to_rhf(mf)
    return ccsd.RCCSD(mf, frozen, mo_coeff, mo_occ)

CCSD = RCCSD

def UCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    mf = _convert_to_uhf(mf)
    return ccsd.UCCSD(mf, frozen, mo_coeff, mo_occ)

def GCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.scf import ghf
    assert(isinstance(mf, ghf.GHF))
    return ccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

def KGCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd
    return kccsd.GCCSD(mf, frozen, mo_coeff, mo_occ)

KCCSD = KGCCSD

def KRCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf.pbc.cc import kccsd_rhf
    mf = _convert_to_rhf(mf)
    return kccsd_rhf.RCCSD(mf, frozen, mo_coeff, mo_occ)

def KUCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    raise NotImplementedError
    from pyscf.pbc.cc import kccsd_uhf
    mf = _convert_to_uhf(mf)
    return kccsd_uhf.UCCSD(mf, frozen, mo_coeff, mo_occ)

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
