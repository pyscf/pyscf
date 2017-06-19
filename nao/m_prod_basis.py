from __future__ import print_function, division
import numpy as np
from numpy import einsum

#
#
#
class prod_basis_c():
  '''
  Holder of local and bilocal product functions and vertices.
  Args:
    system_vars, i.e. holder of the geometry, and orbitals description
    tol : tolerance to keep the linear combinations
  Returns:
    For each specie returns a set of radial functions defining a product basis
    These functions are sufficient to represent the products of original atomic orbitals
    via a product vertex coefficients and conversion coefficients.
    
  Examples:
  '''
  def __init__(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0):
    """ First it should work with GTOs
        Variable belonging to the class prod_basis_c:
        From input:
            self.sv: copy of sv (system variable), probably not necessary??
            self.tol_loc: tolerance for local basis
            self.tol_biloc: tolerance for bilocal basis
            self.ac_rcut_ratio: ac rcut ratio??
        Output:
            self.prod_log: Holder of (local) product functions and vertices
            self.hkernel_csr: hartree kernel: local part of Coulomb interaction
            self.c2s: global product Center (atom) -> start in case of atom-centered basis
            self.bp2vertex: product vertex coefficients for each bilocal pair
            self.bp2info: some information including indices of atoms, list of contributing centres, conversion coefficients
                    Probably better a dictionary than a list, more clear.
            self.dpc2s, self.dpc2t, self.dpc2sp: product Center -> list of the size of the basis set in this center,of center's types,of product species
    """
    from pyscf.nao import coulomb_am, comp_overlap_coo, get_atom2bas_s, conv_yzx2xyz_c, prod_log_c, ls_part_centers, comp_coulomb_den
    from scipy.sparse import csr_matrix
    from pyscf import gto

    self.sv = sv
    self.tol_loc = tol_loc
    self.tol_biloc = tol_biloc
    self.ac_rcut_ratio = ac_rcut_ratio
    
    self.prod_log = prod_log_c(ao_log = sv.ao_log, tol_loc=tol_loc) # local basis (for each specie)
    self.hkernel_csr  = csr_matrix(comp_overlap_coo(sv, self.prod_log, coulomb_am)) # compute local part of Coulomb interaction
    self.c2s = np.zeros((sv.natm+1), dtype=np.int64) # global product Center (atom) -> start in case of atom-centered basis
    for gc,sp in enumerate(sv.atom2sp): self.c2s[gc+1]=self.c2s[gc]+self.prod_log.sp2norbs[sp] # 

    # What is the meaning of this copy ??
    c2s = self.c2s
    
    self.bp2vertex = [] # going to be the product vertex coefficients for each bilocal pair 
    self.bp2info   = [] # going to be some information including indices of atoms, list of contributing centres, conversion coefficients

    for ia1,n1 in enumerate(sv.atom2s[1:]-sv.atom2s[0:-1]):
      for ia2,n2 in enumerate(sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:-1]):
        ia2 += ia1+1

        mol2 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2]], basis=sv.basis, unit='bohr').build()
        bs = get_atom2bas_s(mol2._bas)
        ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
        eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape(n1,n2,n1,n2)
        eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape(n1*n2,n1*n2)
        ee,xx = np.linalg.eigh(eri)   # This the simplest way. TODO: diag in each m-channel separately
        mu2d = [domi for domi,eva in enumerate(ee) if eva>tol_biloc] # The choice of important linear combinations is here
        nprod=len(mu2d)
        if nprod<1: continue # Skip the rest of operations in case there is no large linear combinations.

        # add new vertex
        self.bp2vertex.append(np.zeros([nprod,n1,n2]))
        for p,d in enumerate(mu2d):
            self.bp2vertex[len(self.bp2vertex) -1][p,:,:] = xx[:,d].reshape(n1,n2)
        
        #print(ia1,ia2,nprod,abs(einsum('pab,qab->pq', lambdx, lambdx).reshape(nprod,nprod)-np.identity(nprod)).sum())

        lc2c = ls_part_centers(sv, ia1, ia2, ac_rcut_ratio) # list of participating centers
        lc2s = np.zeros((len(lc2c)+1), dtype=np.int64) # local product center -> start for the current bilocal pair
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
          mol3 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2], sv._atom[c]], basis=sv.basis, unit='bohr').build()
          bs = get_atom2bas_s(mol3._bas)
          ss = (bs[2],bs[3], bs[2],bs[3], bs[0],bs[1], bs[1],bs[2])
          tci_ao = mol3.intor('cint2e_sph', shls_slice=ss).reshape(n3,n3,n1,n2)
          tci_ao = conv_yzx2xyz_c(mol3).conv_yzx2xyz_4d(tci_ao, 'pyscf2nao', ss)
          lp = einsum('lcd,cdp->lp', lcd,einsum('cdab,pab->cdp', tci_ao, self.bp2vertex[len(self.bp2vertex) -1]))
          llp[s:f,:] = lp

        cc = einsum('ab,bc->ac', inv_hk, llp)
        self.bp2info.append([[ia1,ia2],lc2c,lc2s,cc])
        #print(ia1, ia2, len(mu2d), lc2c, hkernel_bp.sum(), inv_hk.sum())
    self.dpc2s,self.dpc2t,self.dpc2sp = self.get_c2s_domiprod() # dominant product's counting 

  def get_ad2cc_den(self):
    """ Returns Conversion Coefficients as dense matrix """
    nfdp,nfap = self.dpc2s[-1],self.c2s[-1]
    ad2cc = np.zeros((nfap,nfdp))
    for sd,fd,pt in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t):
      if pt==1: ad2cc[sd:fd,sd:fd] = np.identity(fd-sd)

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt==1: continue
      for c,ls,lf in zip(self.bp2info[spp][1],self.bp2info[spp][2],self.bp2info[spp][2][1:]): 
        ad2cc[self.c2s[c]:self.c2s[c+1],sd:fd] = self.bp2info[spp][3][ls:lf,:]
    return ad2cc

  def get_vertex_array(self):
    """ Returns the product vertex Coefficients as 3d array """
    nfap = self.c2s[-1]
    n = self.sv.atom2s[-1]
    pab2v = np.zeros((nfap,n,n))
    for atom,[sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=1: continue
      s,f = self.sv.atom2s[atom:atom+2]
      pab2v[sd:fd,s:f,s:f] = self.prod_log.sp2vertex[spp]

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt!=2: continue
      lab = einsum('ld,dab->lab', self.bp2info[spp][3], self.bp2vertex[spp])
      a,b = self.bp2info[spp][0][:]
      sa,fa,sb,fb = self.sv.atom2s[a],self.sv.atom2s[a+1],self.sv.atom2s[b],self.sv.atom2s[b+1]
      for c,ls,lf in zip(self.bp2info[spp][1],self.bp2info[spp][2],self.bp2info[spp][2][1:]): 
        pab2v[self.c2s[c]:self.c2s[c+1],sa:fa,sb:fb] = lab[ls:lf,:,:]
        pab2v[self.c2s[c]:self.c2s[c+1],sb:fb,sa:fa] = einsum('pab->pba', lab[ls:lf,:,:])
    return pab2v


  def get_c2s_domiprod(self):
    """Compute the array of start indices for dominant product basis set """
    c2n,c2t,c2sp = [],[],[] #  product Center -> list of the size of the basis set in this center,of center's types,of product species
    for atom,sp in enumerate(self.sv.atom2sp):
      c2n.append(self.prod_log.sp2vertex[sp].shape[0]); c2t.append(1); c2sp.append(sp);
    for ibp,vertex in enumerate(self.bp2vertex): 
      c2n.append(vertex.shape[0]); c2t.append(2); c2sp.append(ibp);

    ndpc = len(c2n)  # number of product centers in this vertex 
    c2s = np.zeros(ndpc+1, np.int64 ) # product Center -> Start index of a product function in a global counting for this vertex
    for c in range(ndpc): c2s[c+1] = c2s[c] + c2n[c]
    return c2s,c2t,c2sp

  
  def comp_moments(self):
    """ Computes the scalar and dipole moments for the all functions in the product basis """
    sp2mom0,sp2mom1 = self.prod_log.comp_moments()
    n = self.c2s[-1]
    mom0,mom1 = np.zeros(n), np.zeros((n,3))
    for a,[sp,coord,s,f] in enumerate(zip(self.sv.atom2sp,self.sv.atom2coord,self.c2s,self.c2s[1:])):
      mom0[s:f],mom1[s:f,:] = sp2mom0[sp], np.einsum('j,k->jk', sp2mom0[sp],coord)+sp2mom1[sp]
    return mom0,mom1

#
#
#
if __name__=='__main__':
  from pyscf.nao import prod_basis_c, system_vars_c, comp_overlap_coo
  from pyscf import gto
  import numpy as np
  
  mol = gto.M(atom='O 0 0 0; H 0 0 0.5; H 0 0.5 0', basis='ccpvdz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  print(sv.atom2s)
  s_ref = comp_overlap_coo(sv).todense()
  pb = prod_basis_c(sv)
  mom0,mom1=pb.comp_moments()
  pab2v = pb.get_vertex_array()
  s_chk = einsum('pab,p->ab', pab2v,mom0)
  print(abs(s_chk-s_ref).sum()/s_chk.size, abs(s_chk-s_ref).max())
