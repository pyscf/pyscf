from pyscf import lib
from pyscf.lib import logger

from pyscf.cc import rccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd

class RCCSD(rccsd.RCCSD):
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class UCCSD(uccsd.UCCSD):
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class GCCSD(gccsd.GCCSD):
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

def _gen_ao2mofn(mf):
    def ao2mofn(mo_coeff):
        return mf.with_df.ao2mo(mo_coeff, mf.kpt, compact=False)
    return ao2mofn
