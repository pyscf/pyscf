from pyscf import lib
from pyscf.ci import cisd
from pyscf.ci import ucisd
from pyscf.ci import gcisd

class RCISD(cisd.RCISD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        cisd.RCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)
    def ao2mo(self, mo_coeff=None):
        from pyscf.cc import rccsd
        ao2mofn = _gen_ao2mofn(self._scf)
        return rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class UCISD(ucisd.UCISD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        ucisd.UCISD.__init__(self, mf, frozen, mo_coeff, mo_occ)
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return ucisd.uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class GCISD(gcisd.GCISD):
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return gcisd.gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

def _gen_ao2mofn(mf):
    def ao2mofn(mo_coeff):
        return mf.with_df.ao2mo(mo_coeff, mf.kpt, compact=False)
    return ao2mofn
