#!/usr/bin/env python

import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc.gto import _pbcintor

cell = gto.Cell()
cell.build(a = numpy.eye(3) * 2.5,
           atom = 'C',
           basis = 'ccpvdz')

pbcopt = _pbcintor.PBCOpt(cell)
pbcopt.init_rcut_cond(cell)
pbcopt.del_rcut_cond()
