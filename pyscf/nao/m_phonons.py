#!/usr/bin/env python
import numpy as np

def read_vibra_vectors(fname=None):
  """ Reads siesta.vectors file --- output of VIBRA utility """
  from io import StringIO   # StringIO behaves like a file object
  if fname is None: fname = 'siesta.vectors' 
  with open(fname, 'r') as content_file: content = content_file.readlines()
  
  na = -1
  na_prev = -1
  for i,l in enumerate(content):
    if 'Eigenmode (real part)' in l: l1 = i
    if 'Eigenmode (imaginary part)' in l: 
      na = i - l1 - 1
      if na!=na_prev and na_prev>0: raise RuntimeError('na!=na_prev')
      na_prev = na

  ma2xyz = list()
  m2e = list()
  for i,l in enumerate(content):
    if 'Frequency' in l: m2e.append(float(l.split()[2]))
    if 'Eigenmode (real part)' in l: ma2xyz.append(np.loadtxt(StringIO(''.join(str(k) for k in content[i+1:i+1+na]))))
  return np.array(m2e),np.array(ma2xyz)


def normal2cartesian(ma2xyz_nm, a2z):
  """ Converts from normal coordinates (multiplied with sqrt of atomic masses) to Cartesian coordinates """
  from pyscf.data.elements import MASSES
  n = a2z.size # number of atoms...
  a2m = np.array([MASSES[z] for z in a2z])
  ma2xyz = np.copy(ma2xyz_nm)
  ma2xyz = ma2xyz.reshape(3*n,n,3)
  for a in range(ma2xyz.shape[1]): ma2xyz[:,a,:] = ma2xyz[:,a,:] / np.sqrt(a2m[a])
  return ma2xyz


def read_xyz(fname):
  """ Reads xyz files """
  a2s  = np.loadtxt(fname, skiprows=2, usecols=[0], dtype=str)
  a2xyz = np.loadtxt(fname, skiprows=2, usecols=[1,2,3])
  assert len(a2s)==len(a2xyz)
  return a2s,a2xyz
  
def write_xyz(fname, s, ccc, line=None):
  """ Writes xyz file """
  assert len(s) == len(ccc)
  f = open(fname, "w")
  print(len(s), file=f)
  if line is None: 
    print(fname, file=f)
  else:
    print(line, file=f)
  for sym,xyz in zip(s,ccc): print("%2s %18.10f %18.10f %18.10f"%(sym, xyz[0],xyz[1],xyz[2]), file=f)
  f.close()
  return

def read_xv(fname):
  """ Reads siesta.XV file. All quantities are in Hartree atomic units"""
  from io import StringIO   # StringIO behaves like a file object
  with open(fname, 'r') as content_file: content = content_file.readlines()
  ucell_i2xyz = np.loadtxt(StringIO(''.join(str(k) for k in content[0:3])))
  natoms = int(content[3])
  dd = np.loadtxt(StringIO(''.join(str(k) for k in content[4:4+natoms])))
  a2sp,a2znuc = np.array(dd[:,0], dtype=int), np.array(dd[:,1], dtype=int)
  a2xyz,a2vel = dd[:,2:5],dd[:,5:]
  return ucell_i2xyz, a2sp, a2znuc, a2xyz, a2vel

if __name__ == '__main__':
  import sys
  T = 100.0
  n2e,na2xyz = read_siesta_vectors('siesta.vectors')
  a2s,xyz_equ = read_xyz('siesta_equilib.xyz')
  assert xyz_equ.shape[0]==na2xyz.shape[1]
  assert xyz_equ.shape[0]==na2xyz.shape[0]/3
  a2m = np.array([MASSES[ELEMENTS.index(s)] for s in a2s])
  for a in range(na2xyz.shape[1]): na2xyz[:,a,:] = na2xyz[:,a,:] / np.sqrt(a2m[a])
  for n in range(na2xyz.shape[0]):
    if n2e[n]<=0.0:
      na2xyz[n,:,:] = 0.0
    else:
      na2xyz[n,:,:] = na2xyz[n,:,:] * np.sqrt((T/2.0*0.6950303)/n2e[n]) # 0.6950303  --- conversion factor from K to cm^-1
  
  for jf,f in enumerate([-1.0, 1.0]):
    for m,xyz in enumerate(na2xyz):
      ccc = xyz_equ + f*xyz
      fname = 'phonon-%05d-%d-t%5.1fk.xyz'%(m,jf,T)
      print(fname)
      write_xyz(fname, a2s, ccc)
