from __future__ import print_function, division
from pyscf.nao import tddft_iter
from pyscf.nao.m_x_zip import x_zip
import numpy as np

c = tddft_iter(label='siesta', force_gamma=True, gen_pb=False, dealloc_hsx=False)

eps = 0.01
ee = c.mo_energy[0,0]
x_zip(ee, c.mo_coeff[0,0,:,:,0])


ww = np.arange(ee[0]*1.1,ee[-1]*1.1, eps/5.0)+1j*eps
dos = c.dos(ww)
np.savetxt('dos.txt', np.column_stack((ww.real*27.2114, dos)))

ii = [i for i in range(1,len(dos)-1) if dos[i-1]<dos[i] and dos[i]>dos[i+1]]

xx = ww.real
wwmx = []
for i in ii:
  aa = np.array([[xx[i-1]**2, xx[i-1], 1],[xx[i]**2, xx[i], 1],[xx[i+1]**2, xx[i+1], 1]])
  abc = np.linalg.solve(aa, np.array((dos[i-1], dos[i], dos[i+1])))
  wm = -abc[1]/2/abc[0]
  wwmx.append(wm)
  #print(wm)
wwmx = np.array(wwmx)
np.savetxt('sticks.txt', np.column_stack((wwmx*27.2114, np.ones(len(wwmx)))))

xro = np.zeros((len(wwmx), c.norbs))
for j,e in enumerate(c.mo_energy[0,0]):
  imx = np.argmin(abs(wwmx-e))
  xro[imx] += c.mo_coeff[0,0,j,:,0]
  #print(j,e, wwmx[imx])

dos = np.zeros(len(ww))
n2w = np.zeros(len(wwmx))
over = c.hsx.s4_csr.toarray()
for imx,a2x in enumerate(xro): n2w[imx] = np.dot( np.dot(a2x, over), a2x)

print(__name__, wwmx.shape)
for iw,zw in enumerate(ww):
  dos[iw] = (n2w/(zw - wwmx)).sum().imag
dos = -dos/np.pi
np.savetxt('dos_rough.txt', np.column_stack((ww.real*27.2114, dos)))

