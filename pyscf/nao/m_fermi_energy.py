from __future__ import print_function, division
from scipy.optimize import newton

def func1(x, i2e, nelec, telec, norm_const):
  from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
  return norm_const*(fermi_dirac_occupations(telec, i2e, x)).sum()-nelec

#
#
#
def fermi_energy(ee, nelec, telec):
  """ Determine Fermi energy """
  from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
  
  d = len(ee.shape)
  if d==1 : # assuming state2energy
    fermi0 = (ee[int(nelec/2)-1]+ee[int(nelec/2)+1])/2.0
    fe = newton(func1, fermi0, args=(ee, nelec, telec, 2.0))
  elif d==2: # assuming spin_state2energy
    nspin = ee.shape[-2]
    assert nspin==1 or nspin==2
    fermi0 = (ee[0,int(nelec/(3.0-nspin))-1]+ee[0,int(nelec/(3.0-nspin))+1])/2.0
    fe = newton(func1, fermi0, args=(ee, nelec, telec, (3.0-nspin)))
  elif d==3: # assuming kpoint_spin_state2energy
    nspin = ee.shape[-2]
    nkpts = ee.shape[0]
    assert nspin==1 or nspin==2
    fermi0 = (ee[0,0,int(nelec/(3.0-nspin))-1]+ee[0,0,int(nelec/(3.0-nspin))+1])/2.0
    fe = newton(func1, fermi0, args=(ee, nelec, telec, (3.0-nspin)/nkpts))
  else: # how to interpret this ?
    raise RuntimeError('!impl?')

  return fe
  

