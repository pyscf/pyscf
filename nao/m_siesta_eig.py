
from pyscf.nao.m_siesta_ev2ha import siesta_ev2ha

def siesta_eig(label='siesta'):
  f = open(label+'.EIG', 'r')
  f.seek(0)
  Fermi_energy_eV = float(f.readline())
  Fermi_energy_Ha = Fermi_energy_eV * siesta_ev2ha
  f.close()
  return Fermi_energy_Ha
  
  
