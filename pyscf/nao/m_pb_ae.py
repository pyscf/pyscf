from __future__ import print_function, division
import numpy as np
from scipy.sparse import coo_matrix
from numpy import array, einsum, zeros, int64, sqrt
from timeit import default_timer as timer

def pb_ae(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0):
  """ It should work with GTOs as well."""
  from pyscf.nao import coulomb_am, get_atom2bas_s, conv_yzx2xyz_c, ls_part_centers, comp_coulomb_den
  from pyscf.nao.prod_log import prod_log
  from pyscf.nao.m_overlap_coo import overlap_coo
  from pyscf.nao.m_prod_biloc import prod_biloc_c
  from scipy.sparse import csr_matrix
  from pyscf import gto
    
  self.sv = sv
  self.tol_loc = tol_loc
  self.tol_biloc = tol_biloc
  self.ac_rcut_ratio = ac_rcut_ratio
  self.ac_rcut = ac_rcut_ratio*max(sv.ao_log.sp2rcut)
   
  self.prod_log = prod_log(ao_log=sv.ao_log, tol_loc=tol_loc) # local basis (for each specie) 
  self.hkernel_csr  = csr_matrix(overlap_coo(sv, self.prod_log, coulomb_am)) # compute local part of Coulomb interaction
  self.c2s = zeros((sv.natm+1), dtype=int64) # global product Center (atom) -> start in case of atom-centered basis
  for gc,sp in enumerate(sv.atom2sp): self.c2s[gc+1]=self.c2s[gc]+self.prod_log.sp2norbs[sp] # 
  c2s = self.c2s      # What is the meaning of this copy ?? ... This is a pointer to self.c2s
   
  self.bp2info   = [] # going to be some information including indices of atoms, list of contributing centres, conversion coefficients
  
  for ia1,n1 in enumerate(sv.atom2s[1:]-sv.atom2s[0:-1]):
    for ia2,n2 in enumerate(sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:-1]):
      ia2 += ia1+1
      z1pz2 = sv.atom_charge(ia1)+sv.atom_charge(ia2)
      mol2 = gto.Mole(atom=[sv._atom[ia1], sv._atom[ia2]], basis=sv.basis, unit='bohr', spin = z1pz2 % 2).build()
      bs = get_atom2bas_s(mol2._bas)
      ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
      eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape(n1,n2,n1,n2)
      eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape(n1*n2,n1*n2)
      ee,xx = np.linalg.eigh(eri)   # This the simplest way. TODO: diag in each m-channel separately
      mu2d = [domi for domi,eva in enumerate(ee) if eva>tol_biloc] # The choice of important linear combinations is here
      nprod=len(mu2d)
      if nprod<1: continue # Skip the rest of operations in case there is no large linear combinations.

      # add new vertex
      vrtx = zeros([nprod,n1,n2])
      for p,d in enumerate(mu2d): vrtx[p,:,:] = xx[:,d].reshape(n1,n2)
        
      #print(ia1,ia2,nprod,abs(einsum('pab,qab->pq', lambdx, lambdx).reshape(nprod,nprod)-np.identity(nprod)).sum())

      lc2c = ls_part_centers(sv, ia1, ia2, ac_rcut_ratio) # list of participating centers
      lc2s = zeros((len(lc2c)+1), dtype=int64) # local product center -> start for the current bilocal pair
      for lc,c in enumerate(lc2c): lc2s[lc+1]=lc2s[lc]+self.prod_log.sp2norbs[sv.atom2sp[c]]

      npbp = lc2s[-1] # size of the functions which will contribute to the given pair ia1,ia2
      hkernel_bp = np.zeros((npbp, npbp)) # this is local kernel for the current bilocal pair
      for lc1,c1 in enumerate(lc2c):
        for lc2,c2 in enumerate(lc2c):
          for i1 in range(lc2s[lc1+1]-lc2s[lc1]):
            for i2 in range(lc2s[lc2+1]-lc2s[lc2]):
              hkernel_bp[i1+lc2s[lc1],i2+lc2s[lc2]] = self.hkernel_csr[i1+c2s[c1],i2+c2s[c2]] # element-by-element construction here
      inv_hk = np.linalg.inv(hkernel_bp)

      llp = np.zeros((npbp, nprod))
      for c,s,f in zip(lc2c,lc2s,lc2s[1:]):
        n3 = sv.atom2s[c+1]-sv.atom2s[c]
        lcd = self.prod_log.sp2lambda[sv.atom2sp[c]]
        z1pz2pz3 = sv.atom_charge(ia1)+sv.atom_charge(ia2)+sv.atom_charge(c)
        mol3 = gto.Mole(atom=[sv._atom[ia1], sv._atom[ia2], sv._atom[c]], basis=sv.basis, unit='bohr', spin=z1pz2pz3 % 2).build()
        bs = get_atom2bas_s(mol3._bas)
        ss = (bs[2],bs[3], bs[2],bs[3], bs[0],bs[1], bs[1],bs[2])
        tci_ao = mol3.intor('cint2e_sph', shls_slice=ss).reshape(n3,n3,n1,n2)
        tci_ao = conv_yzx2xyz_c(mol3).conv_yzx2xyz_4d(tci_ao, 'pyscf2nao', ss)
        lp = einsum('lcd,cdp->lp', lcd,einsum('cdab,pab->cdp', tci_ao, vrtx))
        llp[s:f,:] = lp

      cc = einsum('ab,bc->ac', inv_hk, llp)
      pbiloc = prod_biloc_c(atoms=array([ia1,ia2]), vrtx=vrtx, cc2a=lc2c, cc2s=lc2s, cc=cc.T)
      
      self.bp2info.append(pbiloc)
      #print(ia1, ia2, len(mu2d), lc2c, hkernel_bp.sum(), inv_hk.sum())
  self.dpc2s,self.dpc2t,self.dpc2sp = self.init_c2s_domiprod() # dominant product's counting 
  return self

