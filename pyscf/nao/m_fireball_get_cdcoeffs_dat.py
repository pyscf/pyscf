from __future__ import print_function, division
import numpy as np

#
#
#
def fireball_get_cdcoeffs_dat(cd):
  """ the file does not seems to be complete (either number of k points or number of spins is missing)  """
  f = open(cd+'/cdcoeffs.dat', "r")
  s = f.readlines();
  f.close()
  norbs = int(s[0])
  ucell = [list(map(float, line.split())) for line in s[1:4] ]

  #print(__name__, 'ucell')
  #print(ucell)
  #norbs = int(s[4])
  #if nspin>1 : raise RuntimeError("nspin>1 ==> don't know how to read this...")

  #k2xyzw = []

  return ucell
  
#
#
#
def fireball_get_ucell_cdcoeffs_dat(cd):
  """ the file does not seems to be complete (either number of k points or number of spins is missing)  """
  f = open(cd+'/cdcoeffs.dat', "r")
  s = f.readlines();
  f.close()
  norbs = int(s[0])
  ucell = [list(map(float, line.split())) for line in s[1:4] ]
  return ucell


