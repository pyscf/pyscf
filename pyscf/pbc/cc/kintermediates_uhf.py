import numpy as np
from functools import reduce

from pyscf.pbc.cc import uccsd_khf
from pyscf.pbc.cc import kccsd
from pyscf.pbc.lib import kpts_helper


def uccsd_cc_Fvv(cc, t1, t2, uccsd_eris):
    orbspin = uccsd_eris._kccsd_eris.orbspin
    kconserv = kpts_helper.get_kconserv(cc._scf.cell, cc.kpts)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    t1 = kccsd.spin2spatial((t1a, t1b), orbspin, kconserv)
    t2 = kccsd.spin2spatial((t2aa, t2ab, t2bb), orbspin, kconserv)
    uccsd_eris.vovv # Chemist's notation, the goal is to use uccsd_eris.vovv

    # mimic the UCCSD contraction with the kccsd.cc_Fvv function as below.
    # It should be removed when finishing the project
    uccsd_eris._kccsd_eris.vovv # the KCCSD spin-orbital tensor, anti-symmetrized, Physist's notation
    gkccsd_Fvv = cc_Fvv(cc, t1, t2, uccsd_eris._kccsd_eris)

    Fvv, FVV = kccsd_uhf._eri_spin2spatial(gkccsd_Fvv, 'vv', uccsd_eris)
    return Fvv, FVV
