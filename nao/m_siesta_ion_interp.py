import numpy
from pyscf.nao.m_spline_diff2 import spline_diff2
from pyscf.nao.m_spline_interp import spline_interp
import sys

#
#
#
def siesta_ion_interp(rr, sp2ion, fj=1):
  """
    Interpolation of orbitals given on linear grid in the ion dictionary
  """
  nr = len(rr)
  assert(nr>2)
  nsp = len(sp2ion)
  nmultmax = max([len(sp2ion[sp]["paos"]["orbital"]) for sp in range(nsp)])

  smr2ro_log = [] #numpy.zeros((nsp,nmultmax,nr), dtype='float64', order='F')
  for sp,ion in enumerate(sp2ion):
    nmu = len(sp2ion[sp]["paos"]["orbital"])

    mr2ro_log = numpy.zeros((nmu,nr), dtype='float64')

    for mu,dat in enumerate(ion["paos"]["data"]):
      npts, j, h = dat.shape[0], ion["paos"]['orbital'][mu]['l'], ion["paos"]["delta"][mu]
      yy = numpy.array([dat[ir][1] for ir in range(npts)], dtype='float64')
      yy_diff2 = spline_diff2(h, yy, 0.0, 1.0e301)
      for ir in range(nr): mr2ro_log[mu,ir] = spline_interp(h,yy,yy_diff2,rr[ir])*(rr[ir]**(fj*j))

    smr2ro_log.append(mr2ro_log)

  return smr2ro_log
