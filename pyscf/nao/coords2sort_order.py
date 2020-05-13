#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def read_xyz(fname):
  """ Reads xyz files """
  a2s  = np.loadtxt(fname, skiprows=2, usecols=[0], dtype=str)
  a2xyz = np.loadtxt(fname, skiprows=2, usecols=[1,2,3])
  assert len(a2s)==len(a2xyz)
  return a2s,a2xyz
  
def write_xyz(fname, s, ccc):
  """ Writes xyz files """
  assert len(s) == len(ccc)
  f = open(fname, "w")
  print(len(s), file=f)
  print(fname, file=f)
  for sym,xyz in zip(s,ccc): print("%2s %18.10f %18.10f %18.10f"%(sym, xyz[0],xyz[1],xyz[2]), file=f)
  f.close()
  return

def coords2sort_order(a2c):
  """ Delivers a list of atom indices which generates a near-diagonal overlap for a given set of atom coordinates """
  na  = a2c.shape[0]
  aa2d = squareform(pdist(a2c))
  mxd = np.amax(aa2d)+1.0
  a = 0
  lsa = []
  for ia in range(na):
    lsa.append(a)
    asrt = np.argsort(aa2d[a])
    for ja in range(1,na):
      b = asrt[ja]
      if b not in lsa: break
    aa2d[a,b] = aa2d[b,a] = mxd
    a = b
  return np.array(lsa)

if __name__=='__main__':
  import sys
  
  for fname in sys.argv[1:] :
    a2s,a2c = read_xyz(fname)
    lsa = coords2sort_order(a2c)
    b2s = np.copy(a2s)
    b2c = np.copy(a2c)
    na = len(b2s)
    b2s[range(na)] = a2s[lsa]
    b2c[range(na)] = a2c[lsa]
    write_xyz(fname+'.sort.xyz', b2s, b2c)
    print(fname,'->', fname+'.sort.xyz')

