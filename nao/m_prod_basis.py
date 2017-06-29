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
  def __init__(self):
    """ Variable belonging to the class prod_basis_c:
        From input:
            self.sv: copy of sv (system variable), probably not necessary??
            self.tol_loc: tolerance for local basis
            self.tol_biloc: tolerance for bilocal basis
            self.ac_rcut_ratio: ac rcut ratio??
            self.ac_npc_max: maximal number of participating centers
        Output:
            self.prod_log: Holder of (local) product functions and vertices
            self.hkernel_csr: hartree kernel: local part of Coulomb interaction
            self.c2s: global product Center (atom) -> start in case of atom-centered basis
            self.bp2vertex: product vertex coefficients for each bilocal pair
            self.bp2info: some information including indices of atoms, list of contributing centres, conversion coefficients
                    Probably better a dictionary than a list, more clear.
            self.dpc2s, self.dpc2t, self.dpc2sp: product Center -> list of the size of the basis set in this center,of center's types,of product species
    """
    
    return

  def init_pb_pp_libnao_apair(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0, ac_npc_max=8, jcutoff=14, metric_type=2):
    """ Talman's procedure should be working well with Pseudo-Potential starting point...
        This subroutine prepares the class for a later atom pair by atom pair generation 
        of the dominant product vertices and the conversion coefficients by calling 
        subroutines from the library libnao.
    """
    from pyscf.nao import coulomb_am, comp_overlap_coo, get_atom2bas_s, conv_yzx2xyz_c, prod_log_c, ls_part_centers, comp_coulomb_den
    from scipy.sparse import csr_matrix
    from pyscf.nao.m_libnao import libnao
    from ctypes import POINTER, c_double, c_int64
    
    self.sv = sv
    self.tol_loc,self.tol_biloc,self.ac_rcut_ratio,self.ac_npc_max = tol_loc, tol_biloc, ac_rcut_ratio, ac_npc_max
    self.jcutoff,self.metric_type = jcutoff, metric_type
    self.prod_log = prod_log_c().init_prod_log_dp(sv.ao_log, tol_loc) # local basis (for each specie)
    self.c2s = np.zeros((sv.natm+1), dtype=np.int64) # global product Center (atom) -> start in case of atom-centered basis
    for gc,sp in enumerate(sv.atom2sp): self.c2s[gc+1]=self.c2s[gc]+self.prod_log.sp2norbs[sp] # 
    self.sv_pbloc_data = self.chain_data_pb_pp_apair()
    libnao.sv_prod_log.argtypes = (
      POINTER(c_int64),      # ndat
      POINTER(c_double))     # dat(ndat)
    libnao.sv_prod_log(c_int64(len(self.sv_pbloc_data)), self.sv_pbloc_data.ctypes.data_as(POINTER(c_double)))
    
    return self
  
  def chain_data_pb_pp_apair(self):
    """ This subroutine creates a buffer of information to communicate with the library libnao """
    from numpy import zeros, concatenate as conc

    aos,sv,pl = self.sv.ao_log, self.sv, self.prod_log
    assert aos.nr==pl.nr
    assert aos.nspecies==pl.nspecies
    
    nr,nsp,nmt,nrt = aos.nr,aos.nspecies, sum(aos.sp2nmult),aos.nr*sum(aos.sp2nmult)
    nat,na1,tna,nms = sv.natoms,sv.natoms+1,3*sv.natoms,sum(aos.sp2nmult)+aos.nspecies
    nmtp,nrtp,nmsp = sum(pl.sp2nmult),pl.nr*sum(pl.sp2nmult),sum(pl.sp2nmult)+pl.nspecies
    nvrt = sum(aos.sp2norbs*aos.sp2norbs*pl.sp2norbs)
    
    ndat = 200 + 2*nr + 4*nsp + 2*nmt + nrt + nms + nat + 2*na1 + tna + \
      4*nsp + 2*nmtp + nrtp + nmsp + nvrt
      
    dat = np.zeros(ndat)
    
    # Simple parameters
    i = 0
    dat[i] = -999.0; i+=1 # pointer to the empty space in simple parameter
    dat[i] = aos.nspecies; i+=1
    dat[i] = aos.nr; i+=1
    dat[i] = aos.rmin;  i+=1;
    dat[i] = aos.rmax;  i+=1;
    dat[i] = aos.kmax;  i+=1;
    dat[i] = aos.jmx;   i+=1;
    dat[i] = conc(aos.psi_log).sum(); i+=1;
    dat[i] = conc(pl.psi_log).sum(); i+=1;
    dat[i] = sv.natoms; i+=1
    dat[i] = sv.norbs; i+=1
    dat[i] = sv.norbs_sc; i+=1
    dat[i] = sv.nspin; i+=1
    dat[i] = self.tol_loc; i+=1
    dat[i] = self.tol_biloc; i+=1
    dat[i] = self.ac_rcut_ratio; i+=1
    dat[i] = self.ac_npc_max; i+=1
    dat[i] = self.jcutoff; i+=1
    dat[i] = self.metric_type; i+=1
    dat[0] = i
    # Pointers to data
    i = 99
    s = 199
    dat[i] = s+1; i+=1; f=s+nr;  dat[s:f] = aos.rr; s=f; # pointer to rr
    dat[i] = s+1; i+=1; f=s+nr;  dat[s:f] = aos.pp; s=f; # pointer to pp
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = aos.sp2nmult; s=f; # pointer to sp2nmult
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = aos.sp2rcut;  s=f; # pointer to sp2rcut
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = aos.sp2norbs; s=f; # pointer to sp2norbs
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = aos.sp2charge; s=f; # pointer to sp2charge    
    dat[i] = s+1; i+=1; f=s+nmt; dat[s:f] = conc(aos.sp_mu2j); s=f; # pointer to sp_mu2j
    dat[i] = s+1; i+=1; f=s+nmt; dat[s:f] = conc(aos.sp_mu2rcut); s=f; # pointer to sp_mu2rcut
    dat[i] = s+1; i+=1; f=s+nrt; dat[s:f] = conc(aos.psi_log).reshape(nrt); s=f; # pointer to psi_log
    dat[i] = s+1; i+=1; f=s+nms; dat[s:f] = conc(aos.sp_mu2s); s=f; # pointer to sp_mu2s
    dat[i] = s+1; i+=1; f=s+nat; dat[s:f] = sv.atom2sp; s=f; # pointer to atom2sp
    dat[i] = s+1; i+=1; f=s+na1; dat[s:f] = sv.atom2s; s=f; # pointer to atom2s
    dat[i] = s+1; i+=1; f=s+na1; dat[s:f] = sv.atom2mu_s; s=f; # pointer to atom2mu_s
    dat[i] = s+1; i+=1; f=s+tna; dat[s:f] = conc(sv.atom2coord); s=f; # pointer to atom2coord
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = pl.sp2nmult; s=f; # sp2nmult of product basis
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = pl.sp2rcut; s=f; # sp2nmult of product basis
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = pl.sp2norbs; s=f; # sp2norbs of product basis
    dat[i] = s+1; i+=1; f=s+nsp; dat[s:f] = pl.sp2charge; s=f; # sp2norbs of product basis
    dat[i] = s+1; i+=1; f=s+nmtp; dat[s:f] = conc(pl.sp_mu2j); s=f; # pointer to sp_mu2j
    dat[i] = s+1; i+=1; f=s+nmtp; dat[s:f] = conc(pl.sp_mu2rcut); s=f; # pointer to sp_mu2rcut
    dat[i] = s+1; i+=1; f=s+nrtp; dat[s:f] = conc(pl.psi_log).reshape(nrtp); s=f; # pointer to psi_log
    dat[i] = s+1; i+=1; f=s+nmsp; dat[s:f] = conc(pl.sp_mu2s); s=f; # pointer to sp_mu2s
    dat[i] = s+1; i+=1; f=s+nvrt; dat[s:f] = conc([v.flatten() for v in pl.sp2vertex]); s=f; # pointer to sp2vertex
    dat[i] = s+1; # this is a terminator to simplify operation

    return dat
    
  def init_prod_basis_pp(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0):
    """ Talman's procedure should be working well with Pseudo-Potential starting point..."""
    from pyscf.nao import coulomb_am, comp_overlap_coo, get_atom2bas_s, conv_yzx2xyz_c, prod_log_c, ls_part_centers, comp_coulomb_den
    from scipy.sparse import csr_matrix

    self.sv = sv
    self.tol_loc = tol_loc
    self.tol_biloc = tol_biloc
    self.ac_rcut_ratio = ac_rcut_ratio

    self.prod_log = prod_log_c().init_prod_log_dp(sv.ao_log, tol_loc) # local basis (for each specie)

    self.hkernel_csr  = csr_matrix(comp_overlap_coo(sv, self.prod_log, coulomb_am)) # compute local part of Coulomb interaction

    self.c2s = np.zeros((sv.natm+1), dtype=np.int64) # global product Center (atom) -> start in case of atom-centered basis

    for gc,sp in enumerate(sv.atom2sp): self.c2s[gc+1]=self.c2s[gc]+self.prod_log.sp2norbs[sp] # 

    c2s = self.c2s 

    self.bp2vertex = [] # going to be the product vertex coefficients for each bilocal pair 
    self.bp2info   = [] # going to be some information including indices of atoms, list of contributing centres, conversion coefficients

    for ia1,n1 in enumerate(sv.atom2s[1:]-sv.atom2s[0:-1]):
      for ia2,n2 in enumerate(sv.atom2s[ia1+2:]-sv.atom2s[ia1+1:-1]):
        ia2+=ia1+1
        #print(ia1, ia2)


    self.dpc2s,self.dpc2t,self.dpc2sp = self.get_c2s_domiprod() # dominant product's counting 
    return self

  def init_prod_basis_gto(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0):
    """ It should work with GTOs as well."""
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

    c2s = self.c2s      # What is the meaning of this copy ?? ... This is a pointer to self.c2s
    
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
    return self

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
    """ Returns the product Vertex Coefficients as 3d array """
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
