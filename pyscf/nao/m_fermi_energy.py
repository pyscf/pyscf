from __future__ import print_function, division
from scipy.optimize import bisect

def func1(x, i2e, nelec, telec, norm_const):
  from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
  nne = norm_const*(fermi_dirac_occupations(telec, i2e, x)).sum()-nelec
  return nne

#
#
#
def fermi_energy(ee, nelec, telec):
  """ Determine Fermi energy """
  from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
  
  d = len(ee.shape)
  if d==1 : # assuming state2energy
    fe = bisect(func1, ee.min(), ee.max(), args=(ee, nelec, telec, 2.0))
  elif d==2: # assuming spin_state2energy
    nspin = ee.shape[-2]
    assert nspin==1 or nspin==2
    fe = bisect(func1, ee.min(), ee.max(), args=(ee, nelec, telec, (3.0-nspin)))
  elif d==3: # assuming kpoint_spin_state2energy
    nspin = ee.shape[-2]
    nkpts = ee.shape[0]
    assert nspin==1 or nspin==2
    fe = bisect(func1, ee.min(), ee.max(), args=(ee, nelec, telec, (3.0-nspin)/nkpts))
  else: # how to interpret this ?
    raise RuntimeError('!impl?')

  return fe
  

