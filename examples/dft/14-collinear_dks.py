#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
DKS with collinear functional
'''

from pyscf import gto

mol = gto.M(atom="H 0 0 0; F 0 0 1", basis='unc-sto3g', verbose=4)
mf = mol.DKS()
mf.xc = 'pbe'
# Enable collinear functional. DKS calls non-collinear functional by default
mf.collinear = 'mcol'
mf.kernel()
