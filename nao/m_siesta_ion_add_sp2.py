import numpy as np
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
  self.sp2nmult = np.empty((self.nspecies), dtype='int64', order='F') 
  self.sp2nmult[:] = list(len(self.sp2ion[sp]["paos"]["orbital"]) for sp in range(self.nspecies))
  self.nmultmax = max(self.sp2nmult)
  self.sp_mu2j = np.empty((self.nspecies,self.nmultmax), dtype='int64', order='F')
  self.sp_mu2j.fill(-999)
  for sp in range(self.nspecies):
    nmu = self.sp2nmult[sp]
    for mu in range(nmu):
      self.sp_mu2j[sp,mu] = self.sp2ion[sp]["paos"]["orbital"][mu]["l"]

#
#
#
def _add_mu_sp2(self, sp2ion):
  """
    Adds fields sp2nmult, sp_mu2rcut and sp_mu2j  to the gived object self, based on sp2ion
  """
  self.nspecies = len(self.sp2ion)
  if self.nspecies <1: return

  self.sp2nmult = np.zeros((self.nspecies), dtype='int64', order='F')
  for sp, ion in enumerate(self.sp2ion):
    self.sp2nmult[sp] = ion["paos"]["npaos"]
  self.nmultmax = max(self.sp2nmult)

  self.sp_mu2j = np.zeros((self.nspecies, self.nmultmax), dtype='int64', order='F')
  self.sp_mu2rcut = np.zeros((self.nspecies, self.nmultmax), dtype='float64', order='F')
  self.sp_mu2j.fill(-999)

  for sp, ion in enumerate(self.sp2ion):
    npaos = ion["paos"]["npaos"]
    self.sp_mu2rcut[sp, 0:npaos] = ion["paos"]["cutoff"]
    for pao, orb in enumerate(ion["paos"]["orbital"]):
      self.sp_mu2j[sp, pao] = orb["l"]
