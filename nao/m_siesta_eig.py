

def siesta_eig(fname='siesta.EIG'):
  f = open(fname, 'r')
  f.seek(0)
  Fermi_energy_eV = float(f.readline())
  Fermi_energy_Ha = Fermi_energy_eV / 27.2116
  f.close()
  return Fermi_energy_Ha
  
  
