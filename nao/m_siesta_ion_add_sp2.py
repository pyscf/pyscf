import numpy
import sys
import re

#
#
#
def _siesta_ion_add_sp2(self, sp2ion):
  '''
  adds fields sp2nmult, and sp_mu2j  to the gived object self, based on sp2ion
  '''
  self.nspecies = len(self.sp2ion)
  self.sp2nmult = numpy.empty((self.nspecies), dtype='int64', order='F') 
  self.sp2nmult[:] = list(len(self.sp2ion[sp]["orbital"]) for sp in range(self.nspecies))
  self.nmultmax = max(self.sp2nmult)
  self.sp_mu2j = numpy.empty((self.nspecies,self.nmultmax), dtype='int64', order='F')
  self.sp_mu2j.fill(-999)
  for sp in range(self.nspecies):
    nmu = self.sp2nmult[sp]
    for mu in range(nmu):
      self.sp_mu2j[sp,mu] = self.sp2ion[sp]["orbital"][mu]["l"]
