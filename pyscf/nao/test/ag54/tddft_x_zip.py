from __future__ import print_function, division
from pyscf.nao import tddft_iter
from pyscf.nao.m_x_zip import x_zip, ee2dos, ee_xx_oo2dos
import numpy as np

c = tddft_iter(label='siesta', nr=128, jcutoff=7, force_gamma=True, gen_pb=True, dealloc_hsx=False, verbosity=1, xc_code='RPA', x_zip=True)

#print(c.sm2e.shape)
#print(c.sma2x[0].shape)

print( c.comp_polariz_inter_ave( np.array([0.5+1j*0.01]) )  )



