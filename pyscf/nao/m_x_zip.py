from __future__ import print_function, division
import sys, numpy as np
from numpy import complex128, pi, arange, dot, zeros, einsum, array, savetxt, column_stack, concatenate, vstack
from timeit import default_timer as timer

def detect_maxima(ww, dos):
  """ Detects maxima of a function given on a grid and lists the arguments at maxima """
  assert dos.dtype == np.float64
  ii = [i for i in range(1,len(dos)-1) if dos[i-1]<dos[i] and dos[i]>dos[i+1]]
  xx = ww.real
  wwmx = []
  for i in ii:
    aa = array([[xx[i-1]**2, xx[i-1], 1],[xx[i]**2, xx[i], 1],[xx[i+1]**2, xx[i+1], 1]])
    abc = np.linalg.solve(aa, np.array((dos[i-1], dos[i], dos[i+1])))
    wm = -abc[1]/2/abc[0]
    wwmx.append(wm)
    #print(wm)
  wwmx = array(wwmx)
  return wwmx

def ee2dos(ee, eps):
  """ convert a set of energies to a grid and a density of states """
  assert eps>0.0
  assert len(ee.shape)==1
  i2w = arange(ee[0]-eps*20, ee[-1]+eps*20, eps/5.0)+1j*eps
  i2dos = zeros(len(i2w), dtype=complex128)
  for iw,zw in enumerate(i2w): i2dos[iw] = (1.0/(zw - ee)).sum()
  i2dos = i2dos/pi
  return i2w,i2dos

def ee_xx_oo2dos(m2e, ma2x, ab2o, eps):
  """ convert a set of energies, eigenvectors and overlap to a grid and a density of states """
  assert eps>0.0
  assert len(m2e.shape)==1
  i2w = arange(m2e[0]-eps*20, m2e[-1]+eps*20, eps/5.0)+1j*eps
  m2w = np.zeros(len(m2e))
  for m,a2x in enumerate(ma2x): m2w[m] = dot( dot(a2x, ab2o), a2x)

  i2dos = zeros(len(i2w), dtype=complex128)  
  for iw,zw in enumerate(i2w):
    i2dos[iw] = (m2w/(zw - m2e)).sum()
  i2dos = i2dos/np.pi
  return i2w,i2dos

def x_zip(n2e, na2x, eps, emax):
  """ Construct a 'presummed' set of eigenvectors, the effect must be a smaller number of eigenvectors """
  assert len(n2e.shape) == 1
  assert len(na2x.shape) == 2
  assert eps>0.0
  assert emax>0.0
  
  if len(np.where(n2e>emax)[0])==0: return n2e.size,[],[],n2e,na2x
    
  #print(__name__, max(n2e), emax)  
  vst = min(np.where(n2e>emax)[0])
  v2e = n2e[vst:]
  i2w,i2dos = ee2dos(v2e, eps) 
  j2e = detect_maxima(i2w, -i2dos.imag)
  nj = len(j2e)
  ja2x = np.zeros((len(j2e), na2x.shape[1]))
  for v,e in enumerate(n2e[vst:]):
    j = np.argmin(abs(j2e-e))

    # Sharing method: saved 16 out of 318 extra iterations caused by x-zip feature in case of Na20 chain.
    jp = j+1 if j<nj-1 else j
    jm = j-1 if j>0 else j
    if j2e[jm]<=e and e<=j2e[j]: 
      wj = (e-j2e[jm])/(j2e[j]-j2e[jm])
      ja2x[jm] += (1.0-wj)*na2x[vst+v]
      ja2x[j] += wj*na2x[vst+v]
      #print(v, 'share? -', jm, j, j2e[jm], e, j2e[j], wj, 1.0-wj)
    elif j2e[j]<=e and e<=j2e[jp]:
      wj = (j2e[jp]-e)/(j2e[jp]-j2e[j])
      ja2x[j] += wj*na2x[vst+v]
      ja2x[jp] += (1.0-wj)*na2x[vst+v]
      #print(v, 'share? + ', j, jp, j2e[j], e, j2e[jp], wj, 1.0-wj)
    else:
      #print(v, 'just sum', j2e[j], e)
      ja2x[j] += na2x[vst+v]

  m2e = concatenate((n2e[0:vst],j2e))
  ma2x = vstack((na2x[0:vst], ja2x))
  return vst,i2w,i2dos,m2e,ma2x
