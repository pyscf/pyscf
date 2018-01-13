#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

'''
Analytical nuclear gradients
============================

Simple usage::

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz')
    >>> mf = scf.RHF(mol).run()
    >>> grad.RHF(mf).kernel()
'''

from pyscf.scf import rhf_grad as rhf
from pyscf.scf import dhf_grad as dhf
from pyscf.grad import uhf
from pyscf.grad import rohf
from pyscf.cc import ccsd_grad as ccsd
from pyscf.ci import cisd_grad as cisd
#from pyscf.grad import rks
RHF = rhf.Gradients
DHF = dhf.Gradients
from pyscf.grad.uhf import Gradients as UHF
from pyscf.grad.rohf import Gradients as ROHF
#from pyscf.grad.rks import Gradients as RKS
#CCSD = ccsd.Gradients

grad_nuc = rhf.grad_nuc
