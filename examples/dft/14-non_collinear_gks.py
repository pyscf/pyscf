#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
GKS with non-collinear functional
'''

from pyscf import gto

mol = gto.M(atom="H 0 0 0; F 0 0 1", basis='unc-sto3g', verbose=4)
mf = mol.GKS()
mf.xc = 'pbe'
# Enable non-collinear functional. GKS calls collinear functional by default
# mcol is short for multi-collinear functional. This is one treatment of
# non-collinear method which avoids the singularity issue in functional.
# For more details of multi-collinear method, please see
#   Noncollinear density functional theory, Zhichen Pu, et. al., Rev. Research 5, 013036
mf.collinear = 'mcol'
mf.kernel()
