import numpy
from pyscf.nao.m_spline_diff2 import spline_diff2
from pyscf.nao.m_spline_interp import spline_interp

#
#
#
def siesta_ion_interp(rr, sp2ion, fj=1):
  assert(len(rr)>2)
  nr = len(rr)
  nsp = len(sp2ion)
  nmultmax = max([len(sp2ion[sp]["orbital"]) for sp in range(nsp)])

  smr2ro_log = numpy.zeros((nsp,nmultmax,nr), dtype='float64', order='F')
  for sp in range(nsp):
    for mu in range(len(sp2ion[sp]["orbital"])):
      npts,j,h = len(sp2ion[sp]["data"][mu]),sp2ion[sp]["orbital"][mu]["l"],sp2ion[sp]["delta"][mu]
      yy = numpy.array([sp2ion[sp]["data"][mu][ir][1] for ir in range(npts)], dtype='float64')
      yy_diff2 = spline_diff2(h, yy, 0.0, 1.0e301)
      for ir in range(nr): smr2ro_log[sp,mu,ir] = spline_interp(h,yy,yy_diff2,rr[ir])*(rr[ir]**(fj*j))

  return(smr2ro_log)
