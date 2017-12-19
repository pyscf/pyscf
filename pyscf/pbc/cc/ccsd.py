from pyscf import lib
from pyscf.lib import logger

from pyscf.cc import rccsd
from pyscf.cc import uccsd
from pyscf.cc import gccsd
from pyscf.pbc import mp

class RCCSD(rccsd.RCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            pt = mp.RMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = numpy.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2
        return rccsd.RCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return rccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class UCCSD(uccsd.UCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            pt = mp.UMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocca, nvira = self.nocc
            nmoa, nmoa = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            self.t1 = (numpy.zeros((nocca,nvira)), numpy.zeros((noccb,nvirb)))
            return self.e_corr, self.t1, self.t2
        return uccsd.UCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return uccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

class GCCSD(gccsd.GCCSD):
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            from pyscf.pbc.mp import mp2
            pt = mp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = numpy.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2
        return gccsd.GCCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        ao2mofn = _gen_ao2mofn(self._scf)
        return gccsd._make_eris_incore(self, mo_coeff, ao2mofn=ao2mofn)

def _gen_ao2mofn(mf):
    def ao2mofn(mo_coeff):
        return mf.with_df.ao2mo(mo_coeff, mf.kpt, compact=False)
    return ao2mofn
