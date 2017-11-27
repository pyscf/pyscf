from pyscf.ci import cisd
from pyscf.ci import ucisd

def CISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    from pyscf import lib
    from pyscf import scf

    if 'dft' in str(mf.__module__):
        sys.stderr.write('GCISD Warning: The first argument mf is a DFT object. '
                         'GCISD calculation should be used with HF object')

    if isinstance(mf, scf.uhf.UHF):
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RCISD method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UCISD method is called.')
        mf = scf.addons.convert_to_uhf(mf)
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)
    else:
        return cisd.CISD(mf, frozen, mo_coeff, mo_occ)


def UCCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = ucisd.UCISD.__doc__
    import sys
    from pyscf import scf

    if 'dft' in str(mf.__module__):
        sys.stderr.write('UCISD Warning: The first argument mf is a DFT object. '
                         'UCISD calculation should be used with HF object')

    mf = scf.addons.convert_to_uhf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-UCISD')
    else:
        return ucisd.UCISD(mf, frozen, mo_coeff, mo_occ)


def GCISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    import sys
    from pyscf import scf
    from pyscf.ci import gcisd

    if 'dft' in str(mf.__module__):
        sys.stderr.write('GCISD Warning: The first argument mf is a DFT object. '
                         'GCISD calculation should be used with HF object')

    mf = scf.addons.convert_to_ghf(mf)
    if hasattr(mf, 'with_df') and mf.with_df:
        raise NotImplementedError('DF-GCISD')
    else:
        return gcisd.GCISD(mf, frozen, mo_coeff, mo_occ)
