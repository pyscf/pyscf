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

from pyscf.grad import rhf
from pyscf.grad import dhf
from pyscf.grad import rks
from pyscf.grad import ccsd
from pyscf.grad.rhf  import Gradients as RHF
from pyscf.grad.dhf  import Gradients as DHF
from pyscf.grad.rks  import Gradients as RKS
#from pyscf.grad.ccsd import Gradients as CCSD

from pyscf.grad.rhf import grad_nuc
