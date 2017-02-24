from __future__ import print_function

#
#
#
def conv_yzx2xyz(mol, mat):
  from pyscf import gto
  n_mol = mol.nao_nr()
  n_mat = mat.shape[0]
  if(n_mol!=n_mat): raise SystemError('n_mol!=n_mat')
  print("dddd", n_mol, n_mat)
  labels = mol.spheric_labels()
  for i in range(len(labels)):
	  print(i, labels[i])
  
  o = -1
  for i in range(mol.nbas) :
	  j = mol.bas_angular(i)
	  for m in range(2*j+1):
		  o = o + 1
		  print(o, i,j,m)
		  
	  
  return mat
