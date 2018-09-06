from __future__ import print_function, division
import numpy as np

# Electron configuration:
configuration = {
'H' : [ 1, 0],            #  1  H, 1s1
'He': [ 2, 0],            #  2  He,1s2
'Li': [ 2, 1, 0],         #  3  Li,[He] 2s1
'Be': [ 2, 2, 0],         #  4  Be,[He] 2s2
'B' : [ 2, 2, 1],         #  5  B, [He] 2s2 2p1
'C' : [ 2, 2, 2],         #  6  C, [He] 2s2 2p2
'N' : [ 2, 2, 3],         #  7  N, [He] 2s2 2p3
'O' : [ 2, 2, 4],         #  8  O, [He] 2s2 2p4
'F' : [ 2, 2, 5],         #  9  F, [He] 2s2 2p5
'Ne': [ 2, 2, 6],         # 10  Ne,[He] 2s2 2p6
'Na': [10, 1, 0],         # 11  Na,[Ne] 3s1
'Mg': [10, 2, 0],         # 12  Mg,[Ne] 3s2
'Al': [10, 2, 1],         # 13  Al,[Ne] 3s2 3p1
'Si': [10, 2, 2],         # 14  Si,[Ne] 3s2 3p2
'P' : [10, 2, 3],         # 15  P, [Ne] 3s2 3p3
'S' : [10, 2, 4],         # 16  S, [Ne] 3s2 3p4
'Cl': [10, 2, 5],         # 17  Cl,[Ne] 3s2 3p5
'Ar': [10, 2, 6],         # 18  Ar,[Ne] 3s2 3p6
'K' : [18, 1, 0],         # 19  K, [Ar] 4s1
'Ca': [18, 2, 0],         # 20  Ca,[Ar] 4s2
'Sc': [18, 1, 2],         # 21  Sc,[Ar] 3d1 4s2
'Ti': [18, 2, 2],         # 22  Ti,[Ar] 3d2 4s2
'V' : [18, 3, 2],         # 23  V, [Ar] 3d3 4s2
'Cr': [18, 4, 2],         # 24  Cr,[Ar] 3d4 4s2
'Mn': [18, 5, 2],         # 25  Mn,[Ar] 3d5 4s2
'Fe': [18, 6, 2],         # 26  Fe,[Ar] 3d6 4s2
'Co': [18, 7, 2],         # 27  Co,[Ar] 3d7 4s2
'Ni': [18, 8, 2],         # 28  Ni,[Ar] 3d8 4s2
'Cu': [18, 9, 2],         # 29  Cu,[Ar] 3d9 4s2
'Zn': [18, 10, 2],        # 30  Zn,[Ar] 3d10 4s2
'Ga': [18, 10, 2, 1],     # 31  Ga,[Ar] 3d10 4s2 4p1
'Ge': [18, 10, 2, 2],     # 32  Ge,[Ar] 3d10 4s2 4p2
'As': [18, 10, 2, 3],     # 33  As,[Ar] 3d10 4s2 4p3
'Se': [18, 10, 2, 4],     # 34  Se,[Ar] 3d10 4s2 4p4
'Br': [18, 10, 2, 5],     # 35  Br,[Ar] 3d10 4s2 4p5
'Kr': [18, 10, 2, 6]      # 36  Kr,[Ar] 3d10 4s2 4p6
}

def n_valance (mol):
  import re
  a = re.split("[^a-zA-Z]*", mol)  #self.mol.atom
  v=0
  for x in range(len (a)):
      if a[x] in configuration.keys():
	      v+= sum(configuration[a[x]][-2:])
  return v


def get_start (atom,nspin,natm,frozen_core,**kw):
  
  if frozen_core == 'yes' or 'on':
    mol = atom
    val = n_valance (mol)
    if nspin == 2:
      if natm == 1: starting = np.array([val//2 , val//2])  
      else:
        if val <= 20 and val >= 4: starting = np.array([2 , 2])
        elif val <= 36 and val > 20: starting = np.array([10 , 10])
        elif val <= 36 and val > 36: starting = np.array([18 , 18])
  elif frozen_core == 'no' or 'off':
    starting = np.array([0 , 0])
  else:
    print('frozen_core', frozen_core)
    raise RuntimeError('Unknown yes-on or no-off is defined!')
  return starting
      
