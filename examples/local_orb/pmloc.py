# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Localization funciton: pmloc.loc(mol,mocoeff)
#
# -ZL@20151227- Add Boys. Note that to view LMOs
#		in jmol, 'orbital energies' need
#		to be defined and used to reorder
#		LMOs.
# -ZL@20151225- Pipek-Mezey localizaiton 
#
import math
import numpy
import scipy.linalg
from pyscf.tools import molden

#------------------------------------------------
# Initialization
#------------------------------------------------
def sqrtm(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())

#def scdmU(coeff,ova):
#   aux = numpy.identity(ova.shape[0])
#   #aux = lowdin(ova)
#   no = coeff.shape[1]	
#   ova = reduce(numpy.dot,(coeff.T,ova,aux))
#   # ova = no*nb
#   q,r,piv = scipy.linalg.qr(ova, pivoting=True)
#   # In fact, it is just "Lowdin-orthonormalized PAO".
#   bc = ova[:,piv[:no]]
#   ova = numpy.dot(bc.T,bc)
#   s12inv = lowdin(ova)
#   u = numpy.dot(bc,s12inv)
#   return u

#------------------------------------------------
# Boys/PM-Localization
#------------------------------------------------
def loc(mol,mocoeff,tol=1.e-6,maxcycle=1000,iop=0):
   partition = genAtomPartition(mol)
   ova = mol.intor_symmetric("cint1e_ovlp_sph")
   ierr,u = loc_kernel(mol,mocoeff,ova,partition,tol,maxcycle,iop)
   #ierr = 0
   #u = scdmU(mocoeff,ova)
   #if iop <= 2: 
   #   mocoeff2 = mocoeff.dot(u)
   #   ierr,u2 = loc_kernel(mol,mocoeff2,ova,partition,tol,maxcycle,iop)
   #   u = u.dot(u2)
   return ierr,u

def loc_kernel(mol,mocoeff,ova,partition,tol,maxcycle,iop):
   debug = False
   print
   print '[pm_loc_kernel]'
   print ' mocoeff.shape=',mocoeff.shape
   print ' tol=',tol
   print ' maxcycle=',maxcycle
   print ' partition=',len(partition),'\n',partition
   k = mocoeff.shape[0]
   n = mocoeff.shape[1]
   natom = len(partition)

   def genPaij(mol,mocoeff,ova,partition,iop):
      c = mocoeff.copy()
      # Mulliken matrix
      if iop == 0:
         cts = c.T.dot(ova)
         natom = len(partition)
         pija = numpy.zeros((natom,n,n))
         for iatom in range(natom):
            idx = partition[iatom]
            tmp = numpy.dot(cts[:,idx],c[idx,:])
            pija[iatom] = 0.5*(tmp+tmp.T)
      # Lowdin
      elif iop == 1:
         s12 = sqrtm(ova)
         s12c = s12.T.dot(c)
         natom = len(partition)
         pija = numpy.zeros((natom,n,n))
         for iatom in range(natom):
            idx = partition[iatom]
	    pija[iatom] = numpy.dot(s12c[idx,:].T,s12c[idx,:])
      # Boys
      elif iop == 2:
	 rmat = mol.intor_symmetric('cint1e_r_sph',3)
	 pija = numpy.zeros((3,n,n))
	 for icart in range(3):
	    pija[icart] = reduce(numpy.dot,(c.T,rmat[icart],c))
      # P[i,j,a]
      pija = pija.transpose(1,2,0).copy()
      return pija

   ## Initial from random unitary
   #iden = numpy.identity(n)
   #iden += 1.e-2*numpy.random.uniform(-1,1,size=n*n).reshape(n,n)
   #u,r = scipy.linalg.qr(iden)
   #mou = mocoeff.dot(u)
   #pija = genPaij(mol,mou,ova,partition,iop)
   u = numpy.identity(n)
   pija = genPaij(mol,mocoeff,ova,partition,iop)
   if debug: pija0 = pija.copy()

   # Start
   def funval(pija):
      return numpy.einsum('iia,iia',pija,pija)

   fun = funval(pija)
   print ' initial funval = ',fun
   #
   # Iteration
   #
   for icycle in range(maxcycle):
      delta = 0.0
      # i>j
      ijdx = []
      for i in range(n-1):
         for j in range(i+1,n):
	    bij = abs(numpy.sum(pija[i,j]*(pija[i,i]-pija[j,j])))
	    ijdx.append((i,j,bij))
      # Without pivoting
      #   6-31g: icycle= 184 delta= 5.6152945523e-09 fun= 54.4719738182
      #   avtz : icycle= 227 delta= 4.43639097128e-09 fun= 3907.60402435
      # With pivoting - significantly accelerated! 
      #   6-31g: icycle= 71 delta= 7.3566217445e-09 fun= 54.4719739144
      #   avdz : icycle= 28 delta= 2.31739493914e-10 fun= 3907.60402153
      # The delta value generally decay monotonically (adjoint diagonalization)
      ijdx = sorted(ijdx,key=lambda x:x[2],reverse=True)
      for i,j,bij in ijdx:
	 #
	 # determine angle
	 #
	 vij = pija[i,i]-pija[j,j] 
	 aij = numpy.dot(pija[i,j],pija[i,j]) \
	     - 0.25*numpy.dot(vij,vij)
	 bij = numpy.dot(pija[i,j],vij)
	 if abs(aij)<1.e-10 and abs(bij)<1.e-10: continue
	 p1 = math.sqrt(aij**2+bij**2)
	 cos4a = -aij/p1
	 sin4a = bij/p1
	 cos2a = math.sqrt((1+cos4a)*0.5)
	 sin2a = math.sqrt((1-cos4a)*0.5)
	 cosa  = math.sqrt((1+cos2a)*0.5)
	 sina  = math.sqrt((1-cos2a)*0.5)
	 # Why? Because we require alpha in [0,pi/2]
	 if sin4a < 0.0:
	    cos2a = -cos2a
	    sina,cosa = cosa,sina
	 # stationary condition
	 if abs(cosa-1.0)<1.e-10: continue
	 if abs(sina-1.0)<1.e-10: continue
	 # incremental value
	 delta += p1*(1-cos4a)
	 # 
	 # Transformation
	 #
	 if debug:
	    g = numpy.identity(n)
	    g[i,i] = cosa
	    g[j,j] = cosa
	    g[i,j] = -sina
	    g[j,i] = sina
	    ug = u.dot(g)
	    pijag = numpy.einsum('ik,jl,ija->kla',ug,ug,pija0)
	 # Urot
	 ui = u[:,i]*cosa+u[:,j]*sina
	 uj = -u[:,i]*sina+u[:,j]*cosa
	 u[:,i] = ui.copy() 
	 u[:,j] = uj.copy()
	 # Bra-transformation of Integrals
	 tmp_ip = pija[i,:,:]*cosa+pija[j,:,:]*sina
	 tmp_jp = -pija[i,:,:]*sina+pija[j,:,:]*cosa
 	 pija[i,:,:] = tmp_ip.copy() 
 	 pija[j,:,:] = tmp_jp.copy()
	 # Ket-transformation of Integrals
	 tmp_ip = pija[:,i,:]*cosa+pija[:,j,:]*sina
	 tmp_jp = -pija[:,i,:]*sina+pija[:,j,:]*cosa
 	 pija[:,i,:] = tmp_ip.copy()
 	 pija[:,j,:] = tmp_jp.copy()
         if debug:
	    diff1 = numpy.linalg.norm(u-ug)
	    diff2 = numpy.linalg.norm(pija-pijag)
            cu = numpy.dot(mocoeff,u)
            pija2 = genPaij(cu,ova,partition)
            fun2 = funval(pija2)
            diff = abs(fun+delta-fun2)
	    print 'diff(u/p/f)=',diff1,diff2,diff
	    if diff1>1.e-6:
	       print 'Error: ug',diff1
	       exit()
	    if diff2>1.e-6:
	       print 'Error: pijag',diff2
	       exit()
            if diff>1.e-6: 
               print 'Error: inconsistency in PMloc: fun/fun2=',fun+delta,fun2
               exit()

      fun = fun+delta
      print 'icycle=',icycle,'delta=',delta,'fun=',fun
      if delta<tol: break
   
   # Check 
   ierr = 0
   if delta<tol: 
      print 'CONG: PMloc converged!'
   else:
      ierr=1
      print 'WARNING: PMloc not converged'
   return ierr,u

def genAtomPartition(mol):
   part = {}
   for iatom in range(mol.natm):
      part[iatom]=[]
   ncgto = 0
   for binfo in mol._bas:
      atom_id = binfo[0]
      lang = binfo[1]
      ncntr = binfo[3]
      nbas = ncntr*(2*lang+1)
      part[atom_id]+=range(ncgto,ncgto+nbas)
      ncgto += nbas
   partition = []
   for iatom in range(mol.natm):
      partition.append(part[iatom])
   return partition

#
# Molden Format
#

if __name__ == '__main__':
  print 'PMlocal'
  
  from pyscf import gto,scf
  mol = gto.Mole()
  mol.verbose = 5 #6
  fac = 0.52917721092
  mol.atom = [['N',map(lambda x:x*fac,[0.0, -1.653532, 0.0])],
  	      ['N',map(lambda x:x*fac,[0.0, 1.653532,  0.0])],
  	      ['O',map(lambda x:x*fac,[-2.050381, -2.530377, 0.0])],
  	      ['O',map(lambda x:x*fac,[2.050381, -2.530377, 0.0])],
  	      ['O',map(lambda x:x*fac,[-2.050381, 2.530377, 0.0])],
  	      ['O',map(lambda x:x*fac,[2.050381, 2.530377, 0.0])]]
  mol.basis = 'aug-cc-pvdz'
  mol.charge = 0
  mol.spin = 0
  mol.build()
  
  mf = scf.RHF(mol)
  mf.init_guess = 'atom'
  mf.level_shift = 0.0
  mf.max_cycle = 100
  mf.conv_tol=1.e-20
  ehf=mf.scf()
 
  nocc = mol.nelectron/2
  ierr,uo = loc(mol,mf.mo_coeff[:,:nocc],iop=0)
  ierr,uv = loc(mol,mf.mo_coeff[:,nocc:],iop=0)
  u = scipy.linalg.block_diag(uo,uv)
  lmo = numpy.dot(mf.mo_coeff,u)

  fname = 'n2o4'
  with open(fname+'_cmo.molden','w') as thefile:
      molden.header(mol,thefile)
      molden.orbital_coeff(mol,thefile,mf.mo_coeff)
  with open(fname+'_lmo.molden','w') as thefile:
      molden.header(mol,thefile)
      molden.orbital_coeff(mol,thefile,lmo)
