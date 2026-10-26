#!/usr/bin/env python
#

from pyscf.df.grad import uhf
from pyscf.grad import rohf

__all__ = ['Gradients', 'Grad']

class Gradients (uhf.Gradients):
    make_rdm1e = rohf.make_rdm1e
    _tag_rdm1 = rohf._tag_rdm1

Grad = Gradients
