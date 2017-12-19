from pyscf import lib
from pyscf.mp import mp2
from pyscf.mp import ump2
from pyscf.mp import gmp2

class RMP2(mp2.RMP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        mp2.RMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return mp2._make_eris(self, mo_coeff, ao2mofn, self.verbose)

class UMP2(ump2.UMP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if abs(mf.kpt).max() > 1e-9:
            raise NotImplementedError
        ump2.UMP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return ump2._make_eris(self, mo_coeff, ao2mofn, self.verbose)

class GMP2(gmp2.GMP2):
    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return gmp2._make_eris_incore(self, mo_coeff, ao2mofn, self.verbose)

def _gen_ao2mofn(mf):
    def ao2mofn(mo_coeff):
        return mf.with_df.ao2mo(mo_coeff, mf.kpt, compact=False)
    return ao2mofn
