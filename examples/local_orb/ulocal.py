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
# Localization based on ALPHA orbitals from UHF/UKS check files
#
import numpy
import scipy.linalg
from pyscf import tools,gto,scf,dft
import h5py
from pyscf.tools import molden

def sqrtm(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())

# Sort by <i|n|i>
def psort(ova,fav,pT,coeff):
   # Compute expectation value
   pTnew = 2.0*reduce(numpy.dot,(coeff.T,ova,pT,ova,coeff))
   nocc  = numpy.diag(pTnew)
   index = numpy.argsort(-nocc)
   ncoeff = coeff[:,index]
   nocc = nocc[index]
   enorb = numpy.diag(reduce(numpy.dot,(coeff.T,ova,fav,ova,coeff)))
   enorb = enorb[index]
   return ncoeff,nocc,enorb

def lowdinPop(mol,coeff,ova,enorb,occ):
   print '\nLowdin population for LMOs:'
   nb,nc = coeff.shape
   s12 = sqrtm(ova)
   lcoeff = s12.dot(coeff)
   diff = reduce(numpy.dot,(lcoeff.T,lcoeff)) - numpy.identity(nc)
   print 'diff=',numpy.linalg.norm(diff)
   pthresh = 0.05
   labels = mol.ao_labels(None)
   nelec = 0.0
   for iorb in range(nc):
      vec = lcoeff[:,iorb]**2
      idx = list(numpy.argwhere(vec>pthresh))
      print ' iorb=',iorb,' occ=',occ[iorb],' <i|F|i>=',enorb[iorb]
      for iao in idx:
         print '    iao=',labels[iao],' pop=',vec[iao]
      nelec += occ[iorb]
   print 'nelec=',nelec
   return 0

def scdm(coeff,ova,aux):
   no = coeff.shape[1]	
   ova = reduce(numpy.dot,(coeff.T,ova,aux))
   # ova = no*nb
   q,r,piv = scipy.linalg.qr(ova,pivoting=True)
   bc = ova[:,piv[:no]]
   ova2 = numpy.dot(bc.T,bc)
   s12inv = lowdin(ova2)
   cnew = reduce(numpy.dot,(coeff,bc,s12inv))
   return cnew

def dumpLMO(mol,fname,lmo):
   print 'Dump into '+fname+'.h5'
   f = h5py.File(fname+'.h5','w')
   f.create_dataset("lmo",data=lmo)
   f.close()
   print 'Dump into '+fname+'_lmo.molden'
   with open(fname+'_lmo.molden','w') as thefile:
       molden.header(mol,thefile)
       molden.orbital_coeff(mol,thefile,lmo)
   return 0

#=============================
# DUMP from chkfile to molden
#=============================
def dumpLocal(fname):
   chkfile = fname+'.chk'
   outfile = fname+'_cmo.molden'
   tools.molden.from_chkfile(outfile, chkfile)
   
   mol,mf = scf.chkfile.load_scf(chkfile)
   mo_coeff = mf["mo_coeff"]
   ova=mol.intor_symmetric("cint1e_ovlp_sph")
   nb = mo_coeff.shape[1]
   nalpha = (mol.nelectron+mol.spin)/2
   nbeta  = (mol.nelectron-mol.spin)/2
   print 'nalpha,nbeta,mol.spin,nb:',\
          nalpha,nbeta,mol.spin,nb
   # UHF-alpha/beta
   ma = mo_coeff[0]
   mb = mo_coeff[1]
   
   #=============================
   # Localization
   #=============================
   ma_c = ma[:,:nalpha].copy()
   ma_v = ma[:,nalpha:].copy()
   #--------------------
   # Occupied space: PM
   #--------------------
   import pmloc
   ierr,uc = pmloc.loc(mol,ma_c)
   mc = numpy.dot(ma_c,uc)
   #--------------------
   # Virtual space: PAO
   #--------------------
   from pyscf import lo
   aux = lo.orth_ao(mol,method='meta_lowdin')
   mv = scdm(ma_v,ova,aux)
 
   # P[dm] 
   pa = numpy.dot(ma[:,:nalpha],ma[:,:nalpha].T)
   pb = numpy.dot(mb[:,:nbeta],mb[:,:nbeta].T)
   pT = 0.5*(pa+pb)
   # E-SORT
   enorb = mf["mo_energy"]
   fa = reduce(numpy.dot,(ma,numpy.diag(enorb[0]),ma.T))
   fb = reduce(numpy.dot,(mb,numpy.diag(enorb[1]),mb.T))
   fav = 0.5*(fa+fb)
   mc,occ_c,ec = psort(ova,fav,pT,mc)
   mv,occ_v,ev = psort(ova,fav,pT,mv)
   #---Check---
   tij = reduce(numpy.dot,(mc.T,ova,ma_c))
   sig = scipy.linalg.svd(tij,compute_uv=False)
   print 'nc=',nalpha,numpy.sum(sig**2)
   assert abs(nalpha-numpy.sum(sig**2))<1.e-8
   tij = reduce(numpy.dot,(mv.T,ova,ma_v))
   sig = scipy.linalg.svd(tij,compute_uv=False)
   print 'nv=',nb-nalpha,numpy.sum(sig**2)
   assert abs(nb-nalpha-numpy.sum(sig**2))<1.e-8

   lmo = numpy.hstack([mc,mv])
   enorb = numpy.hstack([ec,ev])
   occ = numpy.hstack([occ_c,occ_v])
   lowdinPop(mol,lmo,ova,enorb,occ)
   dumpLMO(mol,fname,lmo)
   print 'nalpha,nbeta,mol.spin,nb:',\
          nalpha,nbeta,mol.spin,nb
   return 0

if __name__ == '__main__':   
   fname = 'hs_bp86' 
   dumpLocal(fname)
