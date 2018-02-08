from __future__ import print_function, division
import numpy as np

#
#
#
def fireball_get_eigen_dat(cd):
  """ Calls  """
  f = open(cd+'/eigen.dat', "r")
  s = f.readlines();
  f.close()
  nn = [nspin,norbs] = list(map(int, s[0].split()))
  if nspin>1 : raise RuntimeError("nspin>1 ==> don't know how to read this...")
  i2e = []
  for line in s[2:]:
    i2e += list(map(float, line.split()))
  return np.array(i2e)/27.2114
  
  


