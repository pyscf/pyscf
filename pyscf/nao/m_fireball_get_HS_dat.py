from __future__ import print_function, division
import numpy as np

#
#
#
def fireball_get_HS_dat(cd, fname='HS.dat'):
  """   """
  f = open(cd+'/'+fname, "r")
  nlines = int(f.readline())
  #print(nlines)
  s = f.readlines()
  assert nlines==len(s)
  i2aoao = np.zeros((nlines,4), dtype=int)
  i2h    = np.zeros((nlines))
  i2s    = np.zeros((nlines))
  i2x    = np.zeros((nlines,3))
  for i,line in enumerate(s):
    lspl = line.split()
    i2aoao[i] = list(map(int, lspl[0:4]))
    i2h[i]    = float(lspl[4])
    i2s[i]    = float(lspl[5])
    i2x[i]    = list(map(float, lspl[6:]))
    
  f.close()
  return i2aoao,i2h,i2s,i2x
