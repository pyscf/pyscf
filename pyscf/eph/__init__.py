from pyscf.eph import rhf
from pyscf.eph import uhf
from pyscf.eph import rks
from pyscf.eph import uks
from pyscf import scf, dft

def EPH(mf, **kwargs):
    if isinstance(mf, dft.uks.UKS):
        return uks.EPH(mf, **kwargs)
    elif isinstance(mf, dft.rks.RKS):
        return rks.EPH(mf, **kwargs)
    elif isinstance(mf, scf.uhf.UHF):
        return uhf.EPH(mf, **kwargs)
    elif isinstance(mf, scf.rhf.RHF):
        return rhf.EPH(mf, **kwargs)
    else:
        raise TypeError("EPH only supports RHF, UHF, RKS and UKS")
