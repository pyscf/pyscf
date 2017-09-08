import numpy as np
from pyscf.nao.m_get_sp_mu2s import get_sp_mu2s

#
#
#
def _siesta_ion_add_sp2(self, sp2ion):
  """
    Adds fields sp2nmult, sp_mu2rcut and sp_mu2j  to the gived object self, based on sp2ion
  """
  self.nspecies = len(sp2ion)
  if self.nspecies<1: return

  self.sp2nmult = np.array([ion["paos"]["npaos"] for ion in sp2ion], dtype='int64')

  self.sp_mu2rcut = [np.array(ion["paos"]["cutoff"], dtype='float64') for ion in sp2ion]

  self.sp_mu2j = []
  for sp,ion in enumerate(sp2ion):
    npao = len(ion["paos"]["orbital"])
    self.sp_mu2j.append(np.array([ion["paos"]["orbital"][mu]['l'] for mu in range(npao)], dtype='int64'))
  
  self.sp_mu2s = get_sp_mu2s(self.sp2nmult, self.sp_mu2j)
