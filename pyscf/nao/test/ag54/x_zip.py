from __future__ import print_function, division
from pyscf.nao import tddft_iter
from pyscf.nao.m_x_zip import x_zip, ee2dos, ee_xx_oo2dos
import numpy as np

c = tddft_iter(label='siesta', force_gamma=True, gen_pb=False, dealloc_hsx=False)

eps = 0.01
vst, i2w,i2dos, m2e, ma2x = x_zip(c.mo_energy[0,0], c.mo_coeff[0,0,:,:,0], eps)
print(vst)
print(m2e.size)
np.savetxt('i2dos.txt', np.column_stack((i2w.real,i2dos.real)))

i2w,i2dos = ee2dos(c.mo_energy[0,0], eps)
np.savetxt('n2dos.txt', np.column_stack((i2w.real,i2dos.real)))

ab2o = c.hsx.s4_csr.toarray()
i2w,i2dos = ee_xx_oo2dos(m2e, ma2x, ab2o, eps=eps)
np.savetxt('m2dos.txt', np.column_stack((i2w.real,i2dos.real)))
