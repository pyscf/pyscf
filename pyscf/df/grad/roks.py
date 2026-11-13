#!/usr/bin/env python
#

from pyscf.df.grad import uks
from pyscf.grad import rohf

__all__ = ['Gradients', 'Grad']

class Gradients (uks.Gradients):
    make_rdm1e = rohf.make_rdm1e
    _tag_rdm1 = rohf._tag_rdm1

Grad = Gradients
