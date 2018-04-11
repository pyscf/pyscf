#!/usr/bin/env python
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import tempfile
from functools import reduce
import numpy
import scipy.linalg
import h5py
from pyscf.lib import logger

'''
Extension to scipy.linalg module
'''

def davidson_nosymm(matvec,size,nroots,Adiag=None):
    '''Davidson diagonalization method to solve  a c = e c for a not Hermitian.  
    '''
    nroots = min(nroots,size)
    if Adiag == None:
       Adiag = matvec(numpy.ones(size))

    solver = eigenSolver()
    solver.ndim = size
    solver.neig = nroots
    solver.diag = matvec(numpy.ones(size)) 
    solver.matvec = matvec
    
    return solver.solve_iter()

VERBOSE = False 

def svd_cut(mat,thresh):
   if len(mat.shape) != 2:
      print("NOT A MATRIX in SVD_CUT !",mat.shape)
      exit(1) 
   d1, d2 = mat.shape	
   u, sig, v = scipy.linalg.svd(mat, full_matrices=False)
   r=len(sig)
   for i in range(r):
      if(sig[i]<thresh*1.01):
         r=i
         break
   # return empty list
   if r==0: return [[],[],[]]
   bdim=r
   rkep=r
   u2=numpy.zeros((d1,bdim))
   s2=numpy.zeros((bdim))
   v2=numpy.zeros((bdim,d2))
   u2[:,:rkep] = u[:,:rkep]
   s2[:rkep]   = sig[:rkep]
   v2[:rkep,:] = v[:rkep,:]
   return u2,s2,v2

def eigGenernal(Hmat):
    n = Hmat.shape[0]
    eig,vl,vr = scipy.linalg.eig(Hmat,left=True)
    order = numpy.argsort(eig.real)
    for i in range(n-1):
       rdiff = eig[order[i]].real-eig[order[i+1]].real
       idiff = eig[order[i]].imag-eig[order[i+1]].imag
       # swap to a-w,a+w
       if abs(rdiff) < 1.e-14 and idiff>0.0:
	  j = i + 1     
	  torder  =order[i]
	  order[i]=order[j]
	  order[j]=torder
    eig=eig[order]
    vl=vl[:,order]
    vr=vr[:,order]
    # Normalize
    for i in range(n):
       ova = vl[:,i].T.conj().dot(vr[:,i])
       vl[:,i] = vl[:,i]/ova.conj()
       #test
       #ova = vl[:,i].T.conj().dot(vr[:,i])
    # A=RwL^+T
    tmpH = reduce(numpy.dot,(vr,numpy.diag(eig),vl.T.conj()))
    diff = numpy.linalg.norm(tmpH-Hmat)
    if diff > 1.e-8:
       if VERBOSE: print('error: A=R*w*L^+ !',diff)
       #exit(1)
    return eig,vl,vr

def realRepresentation(vl,vr,nred):
    vbas = numpy.hstack((vl[:,:nred].real,vl[:,:nred].imag,\
                       vr[:,:nred].real,vl[:,:nred].imag))
    u,w,v=svd_cut(vbas,thresh=1.e-12)
    vbas =u.copy()
    return vbas

def mgs_ortho(vlst,rlst,crit_indp,iop=0):
    debug = False
    ndim = len(vlst)
    nres = len(rlst)
    maxtimes = 2
    # [N,n]
    vbas = numpy.array(vlst).transpose(1,0)
    rbas = numpy.array(rlst).transpose(1,0)
    # res=I-VVt*rlst
    for k in range(maxtimes):
       #rbas = rbas - reduce(numpy.dot,(vbas,vbas.T,rbas))
       tmp = numpy.dot(vbas.T,rbas)
       rbas -= numpy.dot(vbas,tmp)
    nindp = 0	
    vlst2 = []
    if iop == 1:
       u,w,v=svd_cut(rbas,thresh=1.e-12)
       nindp = len(w)
       if nindp == 0: return nindp,vlst2
       vbas = numpy.hstack((vbas,u))
       vlst2 = list(u.transpose(1,0))
    else:
       # orthogonalization 
       # - MORE STABLE since SVD sometimes does not converge !!!
       for k in range(maxtimes):
          for i in range(nres):
             rvec = rbas[:,i].copy()	    
             rii = numpy.linalg.norm(rbas[:,i])
             if debug: print(' ktime,i,rii=',k,i,rii)
             # TOO SMALL
             if rii <= crit_indp*10.0**(-k):
                if debug: print(' unable to normalize:',i,' norm=',rii,\
                	       ' thresh=',crit_indp)
                continue 
             # NORMALIZE
             rvec = rvec / rii
             rii = numpy.linalg.norm(rvec)
             rvec = rvec / rii
             nindp = nindp +1
             vlst2.append(rvec)
             # Substract all things
             # [N,n]
             vbas = numpy.hstack((vbas,rvec.reshape(-1,1)))
             for j in range(i,nres):
                #rbas[:,j]=rbas[:,j]-reduce(numpy.dot,(vbas,vbas.T,rbas[:,j]))
                tmp = numpy.dot(vbas.T,rbas[:,j])
                rbas[:,j] -= numpy.dot(vbas,tmp)
       # iden
    iden = vbas.T.dot(vbas)
    diff = numpy.linalg.norm(iden-numpy.identity(ndim+nindp))
    if diff > 1.e-10:
       if VERBOSE: print(' error in mgs_ortho: diff=',diff)
       if VERBOSE: print(iden)
       exit(1)
    else:
       if VERBOSE: print(' final nindp from mgs_ortho =',nindp,' diffIden=',diff)
    return nindp,vlst2 

class eigenSolver:
    def __init__(self):
        self.maxcycle =200
        self.crit_e   =1.e-7
        self.crit_vec =1.e-5
        self.crit_demo=1.e-10
        self.crit_indp=1.e-10
        # Basic setting
        self.iprt =1
        self.ndim =0
        self.neig =5
        self.matvec=None
        self.v0=None
        self.diag=None
        self.matrix=None
 
    def matvecs(self,vlst):
        n = len(vlst)
        wlst = [0]*n
        for i in range(n):
           wlst[i] = self.matvec(vlst[i])
        return wlst
 
    def genMatrix(self):
        v = numpy.identity(self.ndim)
        vlst = list(v)
        wlst = self.matvecs(vlst)
        Hmat = numpy.array(vlst).dot(numpy.array(wlst).T)
        self.matrix = Hmat
        return Hmat
 
    def solve_full(self):
        Hmat = self.matrix
        eig,vl,vr = eigGenernal(Hmat)
        return eig,vr
 
    def genV0(self):
        index = numpy.argsort(self.diag)[:self.neig]
        self.v0 = [0]*self.neig
        for i in range(self.neig):
  	 v = numpy.zeros(self.ndim)
  	 v[index[i]] = 1.0
  	 self.v0[i] = v.copy()
        return self.v0
 
    def solve_iter(self):
        if VERBOSE: print('\nDavdison solver for AX=wX')
        if VERBOSE: print(' ndim = ',self.ndim)
        if VERBOSE: print(' neig = ',self.neig)
	if VERBOSE: print(' maxcycle = ',self.maxcycle)
        #
        # Generate v0
        #
        vlst = self.genV0()
        wlst = self.matvecs(vlst)
        #
        # Begin to solve
        #
        ifconv= False
        neig  = self.neig
        iconv = [False]*neig
        ediff = 0.0
        eigs  = numpy.zeros(neig)
	ndim  = neig
	rlst  = []
        for niter in range(self.maxcycle):
           if self.iprt > 0: 
              if VERBOSE: print('\n --- niter=',niter,'ndim0=',self.ndim,\
			           'ndim=',ndim,'---')
           
           # List[n,N] -> Max[N,n]
           vbas = numpy.array(vlst).transpose(1,0) 
           wbas = numpy.array(wlst).transpose(1,0)
	   iden = vbas.T.dot(vbas)
	   diff = numpy.linalg.norm(iden-numpy.identity(ndim))
	   if diff > 1.e-10:
	      if VERBOSE: print('diff=',diff)
	      if VERBOSE: print(iden)
	      exit(1)
           tmpH = vbas.T.dot(wbas)
           eig,vl,vr = eigGenernal(tmpH)
           teig = eig[:neig]
  	 
  	   # Eigenvalue convergence
  	   nconv1 = 0
  	   for i in range(neig):
  	      tmp = abs(teig[i]-eigs[i])
  	      if VERBOSE: print(' i,eold,enew,ediff=',i,eigs[i],teig[i],tmp)
  	      if tmp <= self.crit_e: nconv1+=1
  	   if VERBOSE: print(' No. of converged eigval:',nconv1)
  	   if nconv1 == neig: 
              if VERBOSE: print(' Cong: all eignvalues converged ! ')
  	   eigs = teig.copy()
  
           # Full Residuals: Res[i]=Res'[i]-w[i]*X[i]
	   vr = vr[:,:neig].copy()
  	   jvec = vbas.dot(vr)
  	   rbas = wbas.dot(vr) - jvec.dot(numpy.diag(eigs))
  	   nconv2 = 0
  	   for i in range(neig):
  	      tmp = numpy.linalg.norm(rbas[:,i])
  	      if tmp <= self.crit_vec:
  	         nconv2 +=1
  	         iconv[i]=True
              else:
  	         iconv[i]=False   
  	      if VERBOSE: print(' i,norm=',i,tmp,iconv[i])
  	   if VERBOSE: print(' No. of converged eigvec:',nconv2)
  	   if nconv2 == neig: 
              if VERBOSE: print(' Cong: all eignvectors converged ! ')
  
           ifconv = (nconv1 == neig) or (nconv2 == neig)
  	   if ifconv:
  	      if VERBOSE: print(' Cong: ALL are converged !\n')
   	      break		
  
  	   # Rotated basis to minimal subspace that
  	   # can give the exact [neig] eigenvalues
	   nkeep = ndim #neig
           qbas = realRepresentation(vl,vr,nkeep)
           vbas = vbas.dot(qbas)
  	   wbas = wbas.dot(qbas)
    	   vlst = list(vbas.transpose(1,0))
  	   wlst = list(wbas.transpose(1,0))
  
           # New directions from residuals
  	   rlst = []
           for i in range(neig):
  	      if iconv[i] == True: continue
              for j in range(self.ndim):
      	          tmp = self.diag[j] - eigs[i]
                  if abs(tmp) < self.crit_demo:
                     rbas[j,i]=rbas[j,i]/self.crit_demo
                  else:
                     rbas[j,i]=rbas[j,i]/tmp
  	      rlst.append(rbas[:,i].real)
  	      rlst.append(rbas[:,i].imag)
  
           # Re-orthogonalization and get Nindp
     	   nindp,vlst2 = mgs_ortho(vlst,rlst,self.crit_indp)

           if nindp != 0:
              wlst2 = self.matvecs(vlst2)
              vlst  = vlst + vlst2
              wlst  = wlst + wlst2
              ndim  = len(vlst)
           else:
              if VERBOSE: print('Convergence Failure: Nindp=0 !')
              exit(1)
 
        if not ifconv:
           if VERBOSE: print('Convergence Failure: Out of Nmaxcycle !')
        
	return eigs,jvec
