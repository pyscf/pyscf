#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.tddft import rks
from pyscf.tddft import rhf
from pyscf.tddft.rhf import TDHF, CIS, RPA

def TD(mf):
    return TDDFT(mf)

def TDA(mf):
    if hasattr(mf, 'xc'):
        return rks.TDA(mf)
    else:
        return rhf.TDA(mf)

def TDDFT(mf):
    if hasattr(mf, 'xc'):
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            return rks.TDDFT(mf)
        else:
            return rks.TDDFTNoHybrid(mf)
    else:
        return rhf.TDHF(mf)
