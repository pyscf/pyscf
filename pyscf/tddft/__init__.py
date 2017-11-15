#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.tddft import rks
from pyscf.tddft import rhf
from pyscf.tddft.rhf import TDHF, CIS, RPA

def TD(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    return TDDFT(mf)

def TDA(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'xc'):
        return rks.TDA(mf)
    else:
        return rhf.TDA(mf)

def TDDFT(mf):
    from pyscf import scf
    mf = scf.addons.convert_to_rhf(mf)
    if hasattr(mf, 'xc'):
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            return rks.TDDFT(mf)
        else:
            return rks.TDDFTNoHybrid(mf)
    else:
        return rhf.TDHF(mf)
