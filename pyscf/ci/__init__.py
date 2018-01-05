from pyscf import lib
from pyscf import scf
from pyscf.ci import cisd
from pyscf.ci import ucisd
from pyscf.ci import gcisd

def CISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCISD method is called.')
        mf = scf.addons.convert_to_uhf(mf)
        return UCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return RCISD(mf, frozen, mo_coeff, mo_occ)

def RCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = cisd.RCISD.__doc__
    scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-RCISD')
    else:
        return cisd.RCISD(mf, frozen, mo_coeff, mo_occ)

def UCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = ucisd.UCISD.__doc__
    mf = scf.addons.convert_to_uhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-UCISD')
    else:
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)


def GCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = gcisd.GCISD.__doc__
    mf = scf.addons.convert_to_ghf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-GCISD')
    else:
        return gcisd.GCISD(mf, frozen, mo_coeff, mo_occ)
