from __future__ import print_function, division
from pyscf.nao import tddft_iter
from pyscf.nao.m_x_zip import x_zip, ee2dos, ee_xx_oo2dos
import numpy as np

c = tddft_iter(label='siesta', jcutoff=7, force_gamma=True, dealloc_hsx=False, verbosity=1)
c.build_x_zip()
