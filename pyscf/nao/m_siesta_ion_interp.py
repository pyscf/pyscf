import numpy as np
from pyscf.nao.m_spline_diff2 import spline_diff2
from pyscf.nao.m_spline_interp import spline_interp
import sys

#
#
#
def siesta_ion_interp(rr, sp2ion, fj=1):
  """ Interpolation of orbitals given on linear grid in the ion dictionary  """
  nr = len(rr)
  assert(nr>2)
  nsp = len(sp2ion)
  nmultmax = max([len(sp2ion[sp]["paos"]["orbital"]) for sp in range(nsp)])

  smr2ro_log = [] #numpy.zeros((nsp,nmultmax,nr), dtype='float64', order='F')
  for sp,ion in enumerate(sp2ion):
    nmu = len(sp2ion[sp]["paos"]["orbital"])

    smr2ro_log.append(np.zeros((nmu,nr)))

    for mu,dat in enumerate(ion["paos"]["data"]):
      #print(__name__, 'dat.shape', dat.shape, dat[0:4,0], dat[0:4,1])
      j, h = ion["paos"]['orbital'][mu]['l'], ion["paos"]["delta"][mu]
      yy_diff2 = spline_diff2(h, dat[:, 1], 0.0, 1.0e301)
      for ir in range(nr): 
          smr2ro_log[sp][mu,ir] = spline_interp(h,dat[:, 1],yy_diff2,rr[ir])*(rr[ir]**(fj*j))

  return smr2ro_log
