from pyscf.ci import cisd

def CISD(mf, frozen=[], mo_coeff=None, mo_occ=None):
    from pyscf import scf
    if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
        raise NotImplementedError('RO-CISD, UCISD are not available in this pyscf version')
    return cisd.CISD(mf, frozen, mo_coeff, mo_occ)
