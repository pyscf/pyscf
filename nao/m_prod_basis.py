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
    """ First it should work with GTOs """
    from pyscf.nao import coulomb_am, comp_overlap_coo, get_atom2bas_s, conv_yzx2xyz_c, prod_log_c, ls_part_centers
    from scipy.sparse import csr_matrix
    from pyscf import gto

    self.sv = sv
    self.tol_loc = tol_loc
    self.tol_biloc = tol_biloc
    self.ac_rcut_ratio = ac_rcut_ratio
    
    self.prod_log = prod_log_c(sv.ao_log, tol_loc) # local basis (for each specie)
    self.hkernel_csr  = csr_matrix(comp_overlap_coo(sv, self.prod_log, coulomb_am))
    self.gc2s = np.zeros((sv.natm+1), dtype=np.int32) # global product center (atom) -> start in case of atom-centered basis
    for gc,sp in enumerate(sv.atom2sp): self.gc2s[gc+1]=self.gc2s[gc]+self.prod_log.sp2norbs[sp]
    gc2s = self.gc2s
    
    self.bp2vertex = [] # going to be the product vertex coefficients for each bilocal pair 
    self.bp2info   = [] # going to be some information including indices of atoms, list of contributing centres, conversion coefficients

    for ia1,n1 in zip(range(sv.natm), sv.atom2s[1:]-sv.atom2s[0:-1]):
      for ia2,n2 in zip(range(ia1+1,sv.natm+1), sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:-1]):

        mol2 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2]], basis=sv.basis).build()
        bs = get_atom2bas_s(mol2._bas)
        ss = (bs[0],bs[1], bs[1],bs[2], bs[0],bs[1], bs[1],bs[2])
        eri = mol2.intor('cint2e_sph', shls_slice=ss).reshape(n1,n2,n1,n2)
        eri = conv_yzx2xyz_c(mol2).conv_yzx2xyz_4d(eri, 'pyscf2nao', ss).reshape(n1*n2,n1*n2)
        ee,xx = np.linalg.eigh(eri)   # This the simplest way. TODO: diag in each m-channel separately
        mu2d = [domi for domi,eva in enumerate(ee) if eva>tol_biloc] # The choice of important linear combinations is here
        nprod=len(mu2d)
        if nprod<1: continue # Skip the rest of operations in case there is no large linear combinations.
        vertex = np.zeros([nprod,n1,n2])
        for p,d in enumerate(mu2d) : vertex[p,:,:] = xx[:,d].reshape(n1,n2)
        self.bp2vertex.append(vertex)

        lc2c = ls_part_centers(sv, ia1, ia2, ac_rcut_ratio) # list of participating centers
        lc2s = np.zeros((len(lc2c)+1), dtype=np.int32) # local product center -> start for the current bilocal pair
        for lc,c in enumerate(lc2c): lc2s[lc+1]=lc2s[lc]+self.prod_log.sp2norbs[sv.atom2sp[c]]

        npbp = lc2s[-1] # size of the functions which will contribute to the given pair ia1,ia2
        hkernel_bp = np.zeros((npbp, npbp)) # this is local kernel for the current bilocal pair
        for lc1,c1 in enumerate(lc2c):
          for lc2,c2 in enumerate(lc2c):
            for i1 in range(lc2s[lc1+1]-lc2s[lc1]):
              for i2 in range(lc2s[lc2+1]-lc2s[lc2]):
                hkernel_bp[i1+lc2s[lc1],i2+lc2s[lc2]] = self.hkernel_csr[i1+gc2s[c1],i2+gc2s[c2]] # element-by-element construction here
        inv_hk = np.linalg.inv(hkernel_bp)

        llp = np.zeros((npbp, nprod))
        for c,s,f in zip(lc2c,lc2s,lc2s[1:]):
          n3 = sv.atom2s[c+1]-sv.atom2s[c]
          lcd = self.prod_log.sp2lambda[sv.atom2sp[c]]
          mol3 = gto.Mole_pure(atom=[sv._atom[ia1], sv._atom[ia2], sv._atom[c]], basis=sv.basis).build()
          bs = get_atom2bas_s(mol3._bas)
          ss = (bs[2],bs[3], bs[2],bs[3], bs[0],bs[1], bs[1],bs[2])
          tci_ao = mol3.intor('cint2e_sph', shls_slice=ss).reshape(n3,n3,n1,n2)
          tci_ao = conv_yzx2xyz_c(mol3).conv_yzx2xyz_4d(tci_ao, 'pyscf2nao', ss)
          lp = einsum('lcd,cdp->lp', lcd,einsum('cdab,pab->cdp', tci_ao, vertex))
          llp[s:f,:] = lp

        cc = einsum('ab,bc->ac', inv_hk, llp)
        self.bp2info.append([[ia1,ia2],lc2c,lc2s,cc])
        #print(ia1, ia2, len(mu2d), lc2c, hkernel_bp.sum(), inv_hk.sum())

#
#
#
if __name__=='__main__':
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_prod_basis import prod_basis_c
  from pyscf import gto
  import numpy as np
  from timeit import default_timer as timer
  import matplotlib.pyplot as plt
  
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol)
  pb = prod_basis_c(sv)
  print(pb.prod_log.sp2norbs)
