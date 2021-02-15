from __future__ import print_function, division
import numpy as np
from scipy.sparse import coo_matrix
from pyscf.nao.lsofcsr import lsofcsr_c
from numpy import array, einsum, zeros, int64, sqrt, int32
from ctypes import POINTER, c_double, c_int64, byref
from pyscf.nao.m_libnao import libnao
from timeit import default_timer as timer

libnao.init_vrtx_cc_apair.argtypes = (POINTER(c_double), POINTER(c_int64))

libnao.init_vrtx_cc_batch.argtypes = (POINTER(c_double), POINTER(c_int64))

libnao.vrtx_cc_apair.argtypes = (
  POINTER(c_int64),   # sp12(1:2)     ! chemical species indices
  POINTER(c_double),  # rc12(1:3,1:2) ! positions of species
  POINTER(c_int64),   # lscc(ncc)     ! list of contributing centers
  POINTER(c_int64),   # ncc           ! number of contributing centers 
  POINTER(c_double),  # dout(nout)    ! vertex & converting coefficients
  POINTER(c_int64))   # nout          ! size of the buffer for vertex & converting coefficients

libnao.vrtx_cc_batch.argtypes = (
  POINTER(c_int64),   # npairs        ! chemical species indices
  POINTER(c_double),  # p2srncc       ! species indices, positions of species, number of cc, cc
  POINTER(c_int64),   # ncc           ! leading dimension of p2srncc
  POINTER(c_int64))   # p2ndp         ! pair -> number of dominant products in this pair

libnao.get_vrtx_cc_batch.argtypes = (
  POINTER(c_int64),   # ps            ! start pair 
  POINTER(c_int64),   # pf            ! finish pair
  POINTER(c_double),  # data          ! output data buffer
  POINTER(c_int64))   # ndat          ! size of data buffer

#
#
#
class prod_basis():
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
  def __init__(self, nao=None, **kw):
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
            self.bp2info: some information including indices of atoms, list of contributing centres, conversion coefficients
            self.dpc2s, self.dpc2t, self.dpc2sp: product Center -> list of the size of the basis set in this center,of center's types,of product species
    """
    
    self.nao = nao
    self.tol_loc = kw['tol_loc'] if 'tol_loc' in kw else 1e-5
    self.tol_biloc = kw['tol_biloc'] if 'tol_biloc' in kw else 1e-6
    self.tol_elim = kw['tol_elim'] if 'tol_elim' in kw else 1e-16
    self.ac_rcut_ratio = kw['ac_rcut_ratio'] if 'ac_rcut_ratio' in kw else 1.0
    self.ac_npc_max = kw['ac_npc_max'] if 'ac_npc_max' in kw else 8
    self.pb_algorithm = kw['pb_algorithm'].lower() if 'pb_algorithm' in kw else 'pp'
    self.jcutoff = kw['jcutoff'] if 'jcutoff' in kw else 14
    self.metric_type = kw['metric_type'] if 'metric_type' in kw else 2
    self.optimize_centers = kw['optimize_centers'] if 'optimize_centers' in kw else 0
    self.ngl = kw['ngl'] if 'ngl' in kw else 96
    
    if nao is None: return
    
    if self.pb_algorithm=='pp':
      self.init_prod_basis_pp_batch(nao, **kw)
    elif self.pb_algorithm=='fp':
      from pyscf.nao.m_pb_ae import pb_ae
      pb_ae(self, nao, self.tol_loc, self.tol_biloc, self.ac_rcut_ratio)
    else:
      print( 'self.pb_algorithm', self.pb_algorithm )
      raise RuntimeError('unknown pb_algorithm')
      
    return  

  def init_prod_basis_pp(self, sv, **kvargs):
    """ Talman's procedure should be working well with Pseudo-Potential starting point."""
    from pyscf.nao.m_prod_biloc import prod_biloc_c

    #t1 = timer()    
    self.init_inp_param_prod_log_dp(sv, **kvargs)
    data = self.chain_data()
    libnao.init_vrtx_cc_apair(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)))
    self.sv_pbloc_data = True
    
    #t2 = timer(); print(t2-t1); t1=timer();
    self.bp2info = [] # going to be some information including indices of atoms, list of contributing centres, conversion coefficients
    for ia1 in range(sv.natoms):
      rc1 = sv.ao_log.sp2rcut[sv.atom2sp[ia1]]
      for ia2 in range(ia1+1,sv.natoms):
        rc2,dist = sv.ao_log.sp2rcut[sv.atom2sp[ia2]], sqrt(((sv.atom2coord[ia1]-sv.atom2coord[ia2])**2).sum())
        if dist>rc1+rc2 : continue
        pbiloc = self.comp_apair_pp_libint(ia1,ia2)
        if pbiloc is not None : self.bp2info.append(pbiloc)
    
    self.dpc2s,self.dpc2t,self.dpc2sp = self.init_c2s_domiprod() # dominant product's counting
    self.npdp = self.dpc2s[-1]
    self.norbs = self.sv.norbs
    return self

  def init_prod_basis_pp_batch(self, nao, **kw):
    """ Talman's procedure should be working well with Pseudo-Potential starting point."""
    from pyscf.nao.prod_log import prod_log
    from pyscf.nao.m_prod_biloc import prod_biloc_c

    sv = nao
    t1 = timer()
    self.norbs = sv.norbs
    self.init_inp_param_prod_log_dp(sv, **kw)
    #t2 = timer(); print(' after init_inp_param_prod_log_dp ', t2-t1); t1=timer()
    data = self.chain_data()
    libnao.init_vrtx_cc_batch(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)))
    self.sv_pbloc_data = True

    aos = sv.ao_log
    p2srncc,p2npac,p2atoms = [],[],[]
    for a1,[sp1,ra1] in enumerate(zip(sv.atom2sp, sv.atom2coord)):
      rc1 = aos.sp2rcut[sp1]
      for a2,[sp2,ra2] in enumerate(zip(sv.atom2sp[a1+1:], sv.atom2coord[a1+1:])):
        a2+=a1+1
        rc2,dist = aos.sp2rcut[sp2], sqrt(((ra1-ra2)**2).sum())
        if dist>rc1+rc2 : continue
        cc2atom = self.ls_contributing(a1,a2)
        p2atoms.append([a1,a2])
        p2srncc.append([sp1,sp2]+list(ra1)+list(ra2)+[len(cc2atom)]+list(cc2atom))
        p2npac.append( sum([self.prod_log.sp2norbs[sv.atom2sp[ia]] for ia in cc2atom ]))

    #print(np.asarray(p2srncc))
    p2ndp = np.require( zeros(len(p2srncc), dtype=np.int64), requirements='CW')
    p2srncc_cp = np.require(  np.asarray(p2srncc), requirements='C')
    npairs = p2srncc_cp.shape[0]
    self.npairs = npairs
    self.bp2info = [] # going to be indices of atoms, list of contributing centres, conversion coefficients
    if npairs>0 : # Conditional fill of the self.bp2info if there are bilocal pairs (natoms>1)
      ld = p2srncc_cp.shape[1]
      if nao.verbosity>0:
        #print(__name__,'npairs {} and p2srncc_cp.shape is {}'.format(npairs, p2srncc_cp.shape))
        t2 = timer(); print(__name__,'\t====> Time for call vrtx_cc_batch: {:.2f} sec, npairs: {}'.format(t2-t1, npairs)); t1=timer()
      libnao.vrtx_cc_batch( c_int64(npairs), p2srncc_cp.ctypes.data_as(POINTER(c_double)), 
        c_int64(ld), p2ndp.ctypes.data_as(POINTER(c_int64)))
      
      if nao.verbosity>0:
        print(__name__,'\t====> libnao.vrtx_cc_batch is done!')
        t2 = timer(); print(__name__,'\t====> Time after vrtx_cc_batch:\t {:.2f} sec'.format(t2-t1)); t1=timer()
      nout = 0
      sp2norbs = sv.ao_log.sp2norbs
      for srncc,ndp,npac in zip(p2srncc,p2ndp,p2npac):
        sp1,sp2 = srncc[0],srncc[1]
        nout = nout + ndp*sp2norbs[sp1]*sp2norbs[sp2]+npac*ndp
      
      dout = np.require( zeros(nout), requirements='CW')
      if nao.verbosity>3:print(__name__,'\t====>libnao.get_vrtx_cc_batch is calling')
      libnao.get_vrtx_cc_batch(c_int64(0),c_int64(npairs),dout.ctypes.data_as(POINTER(c_double)),c_int64(nout))

      f = 0
      for srncc,ndp,npac,[a1,a2] in zip(p2srncc,p2ndp,p2npac,p2atoms):
        if ndp<1 : continue
        sp1,sp2,ncc = srncc[0],srncc[1],srncc[8]
        icc2a = array(srncc[9:9+ncc], dtype=int64)
        nnn = np.array((ndp,sp2norbs[sp2],sp2norbs[sp1]), dtype=int64)
        nnc = np.array([ndp,npac], dtype=int64)
        s = f;  f=s+np.prod(nnn); vrtx  = dout[s:f].reshape(nnn)
        s = f;  f=s+np.prod(nnc); ccoe  = dout[s:f].reshape(nnc)
        icc2s = zeros(len(icc2a)+1, dtype=int64)
        for icc,a in enumerate(icc2a): icc2s[icc+1] = icc2s[icc] + self.prod_log.sp2norbs[sv.atom2sp[a]]
        pbiloc = prod_biloc_c(atoms=array([a2,a1]),vrtx=vrtx,cc2a=icc2a,cc2s=icc2s,cc=ccoe)
        self.bp2info.append(pbiloc)

    #t2 = timer(); print('after loop ', t2-t1); t1=timer()
    self.dpc2s,self.dpc2t,self.dpc2sp = self.init_c2s_domiprod() # dominant product's counting
    self.npdp = self.dpc2s[-1]
    #t2 = timer(); print('after init_c2s_domiprod ', t2-t1); t1=timer()
    return self
  
  def init_inp_param_prod_log_dp(self, sv, tol_loc=1e-5, tol_biloc=1e-6, ac_rcut_ratio=1.0, ac_npc_max=8, jcutoff=14, metric_type=2, optimize_centers=0, ngl=96, **kw):
    """ Talman's procedure should be working well with a pseudo-potential hamiltonians.
        This subroutine prepares the class for a later atom pair by atom pair generation 
        of the dominant product vertices and the conversion coefficients by calling 
        subroutines from the library libnao.
    """
    from pyscf.nao.prod_log import prod_log
    from pyscf.nao.m_libnao import libnao
    
    self.sv = sv
    self.tol_loc,self.tol_biloc,self.ac_rcut_ratio,self.ac_npc_max = tol_loc, tol_biloc, ac_rcut_ratio, ac_npc_max
    self.jcutoff,self.metric_type,self.optimize_centers,self.ngl = jcutoff, metric_type, optimize_centers, ngl
    self.ac_rcut = ac_rcut_ratio*max(sv.ao_log.sp2rcut)    
    
    lload = kw['load_from_hdf5'] if 'load_from_hdf5' in kw else False 
    if lload :
      self.prod_log = prod_log(ao_log=sv.ao_log, sp2charge=sv.sp2charge, tol_loc=tol_loc, load=True) # tests Fortran input
    else :
      self.prod_log = prod_log(ao_log=sv.ao_log, tol_loc=tol_loc, **kw) # local basis (for each specie)
    
    self.c2s = zeros((sv.natm+1), dtype=int64) # global product Center (atom) -> start in case of atom-centered basis
    for gc,sp in enumerate(sv.atom2sp): self.c2s[gc+1]=self.c2s[gc]+self.prod_log.sp2norbs[sp] #
    return self
  
  def chain_data(self):
    """ This subroutine creates a buffer of information to communicate the system variables and the local product vertex to libnao. Later, one will be able to generate the bilocal vertex and conversion coefficient for a given pair of atom species and their coordinates ."""
    from numpy import concatenate as conc

    aos,sv,pl = self.sv.ao_log, self.sv, self.prod_log
    assert aos.nr==pl.nr
    assert aos.nspecies==pl.nspecies
    
    nr,nsp,nmt,nrt = aos.nr,aos.nspecies, sum(aos.sp2nmult),aos.nr*sum(aos.sp2nmult)
    nat,na1,tna,nms = sv.natoms,sv.natoms+1,3*sv.natoms,sum(aos.sp2nmult)+aos.nspecies
    nmtp,nrtp,nmsp = sum(pl.sp2nmult),pl.nr*sum(pl.sp2nmult),sum(pl.sp2nmult)+pl.nspecies
    nvrt = sum(aos.sp2norbs*aos.sp2norbs*pl.sp2norbs)
    
    ndat = 200 + 2*nr + 4*nsp + 2*nmt + nrt + nms + 3*3 + nat + 2*na1 + tna + \
      4*nsp + 2*nmtp + nrtp + nmsp + nvrt
      
    dat = zeros(ndat)
    
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
    dat[i] = self.optimize_centers; i+=1
    dat[i] = self.ngl; i+=1
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
    dat[i] = s+1; i+=1; f=s+3*3; dat[s:f] = conc(sv.ucell); s=f; # pointer to ucell (123,xyz) ?
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

  def comp_apair_pp_libint(self, a1,a2):
    """ Get's the vertex coefficient and conversion coefficients for a pair of atoms given by their atom indices """
    from operator import mul
    from pyscf.nao.m_prod_biloc import prod_biloc_c
    if not hasattr(self, 'sv_pbloc_data') : raise RuntimeError('.sv_pbloc_data is absent')
    assert a1>=0
    assert a2>=0
    
    t1 = timer()
    sv = self.sv
    aos = self.sv.ao_log
    sp12 = np.require( np.array([sv.atom2sp[a] for a in (a1,a2)], dtype=c_int64), requirements='C')
    rc12 = np.require( np.array([sv.atom2coord[a,:] for a in (a1,a2)]), requirements='C')
    icc2a = np.require( np.array(self.ls_contributing(a1,a2), dtype=c_int64), requirements='C')
    npmx = aos.sp2norbs[sv.atom2sp[a1]]*aos.sp2norbs[sv.atom2sp[a2]]
    npac = sum([self.prod_log.sp2norbs[sv.atom2sp[ia]] for ia in icc2a ])
    nout = c_int64(npmx**2+npmx*npac+10)
    dout = np.require( zeros(nout.value), requirements='CW')
    
    libnao.vrtx_cc_apair( sp12.ctypes.data_as(POINTER(c_int64)), rc12.ctypes.data_as(POINTER(c_double)), icc2a.ctypes.data_as(POINTER(c_int64)), c_int64(len(icc2a)), dout.ctypes.data_as(POINTER(c_double)), nout )    
    if dout[0]<1: return None
    
    nnn = np.array(dout[0:3], dtype=int)
    nnc = np.array([dout[8],dout[7]], dtype=int)
    ncc = int(dout[9])
    if ncc!=len(icc2a): raise RuntimeError('ncc!=len(icc2a)')
    s = 10; f=s+np.prod(nnn); vrtx  = dout[s:f].reshape(nnn)
    s = f;  f=s+np.prod(nnc); ccoe  = dout[s:f].reshape(nnc)
    icc2s = zeros(len(icc2a)+1, dtype=np.int64)
    for icc,a in enumerate(icc2a): icc2s[icc+1] = icc2s[icc] + self.prod_log.sp2norbs[sv.atom2sp[a]]
    pbiloc = prod_biloc_c(atoms=array([a2,a1]),vrtx=vrtx,cc2a=icc2a,cc2s=icc2s,cc=ccoe)
    
    return pbiloc


  def ls_contributing(self, a1,a2):
    """ Get the list of contributing centers """
    from pyscf.nao.m_ls_contributing import ls_contributing
    sp12 = np.array([self.sv.atom2sp[a] for a in (a1,a2)])
    rc12 = np.array([self.sv.atom2coord[a,:] for a in (a1,a2)])
    return ls_contributing(self, sp12, rc12)

  def get_da2cc_den(self, dtype=np.float64):
    """ Returns Conversion Coefficients as dense matrix """
    nfdp,nfap = self.dpc2s[-1],self.c2s[-1]
    da2cc = zeros((nfdp,nfap), dtype=dtype)
    for sd,fd,pt in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t):
      if pt==1: da2cc[sd:fd,sd:fd] = np.identity(fd-sd)

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt==1: continue
      inf = self.bp2info[spp]
      for c,ls,lf in zip(inf.cc2a, inf.cc2s, inf.cc2s[1:]): 
        da2cc[sd:fd, self.c2s[c]:self.c2s[c+1]] = inf.cc[:,ls:lf]
    return da2cc

  def get_da2cc_nnz(self):
    """ Computes the number of non-zero matrix elements in the conversion matrix ac <=> dp """
    nnz = 0
    for sd,fd,pt in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t):
      if pt==1: nnz = nnz + (fd-sd)

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt==1: continue
      inf = self.bp2info[spp]
      for c,ls,lf in zip(inf.cc2a, inf.cc2s, inf.cc2s[1:]): nnz = nnz + (fd-sd)*(lf-ls)
    return nnz

  # should we not keep only the sparse matrix and get rid of the original data ??
  def get_da2cc_sparse(self, dtype=np.float64, sparseformat=coo_matrix):
    """ Returns Conversion Coefficients as sparse COO matrix """

    nfdp,nfap = self.dpc2s[-1],self.c2s[-1]
    nnz = self.get_da2cc_nnz()
    irow,icol,data = zeros(nnz, dtype=int),zeros(nnz, dtype=int), zeros(nnz, dtype=dtype) # Start to construct coo matrix

    inz = 0
    for atom, [sd,fd,pt] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t)):
      if pt!=1: continue
      for d in range(sd,fd): 
        irow[inz],icol[inz],data[inz] = d,d,1.0
        inz+=1

    for atom, [sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt==1: continue
      inf = self.bp2info[spp]
      for c,ls,lf in zip(inf.cc2a, inf.cc2s, inf.cc2s[1:]): 
        for d in range(sd,fd):
          for a in range(self.c2s[c],self.c2s[c+1]):
            irow[inz],icol[inz],data[inz] = d,a,inf.cc[d-sd,a-self.c2s[c]+ls]
            inz+=1
    return sparseformat((data,(irow,icol)), dtype=dtype, shape=(nfdp, nfap))
    
  def get_ac_vertex_array(self, dtype=np.float64):
    """ Returns the product vertex coefficients as 3d array (dense table) """
    atom2so = self.sv.atom2s
    nfap = self.c2s[-1]
    n = self.sv.atom2s[-1]
    pab2v = np.require( zeros((nfap,n,n), dtype=dtype), requirements='CW')
    for atom,[sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=1: continue
      s,f = atom2so[atom:atom+2]
      pab2v[sd:fd,s:f,s:f] = self.prod_log.sp2vertex[spp]

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt!=2: continue
      inf= self.bp2info[spp]
      lab = einsum('dl,dab->lab', inf.cc, inf.vrtx)
      a,b = inf.atoms
      sa,fa,sb,fb = atom2so[a],atom2so[a+1],atom2so[b],atom2so[b+1]
      for c,ls,lf in zip(inf.cc2a, inf.cc2s, inf.cc2s[1:]):
        pab2v[self.c2s[c]:self.c2s[c+1],sa:fa,sb:fb] = lab[ls:lf,:,:]
        pab2v[self.c2s[c]:self.c2s[c+1],sb:fb,sa:fa] = einsum('pab->pba', lab[ls:lf,:,:])
    return pab2v

  def get_dp_vertex_array(self, dtype=np.float64):
    """ Returns the product vertex coefficients as 3d array for dominant products """
    atom2so = self.sv.atom2s
    nfdp = self.dpc2s[-1]
    n = self.sv.atom2s[-1]
    pab2v = np.require(zeros((nfdp,n,n), dtype=dtype), requirements='CW')
    for atom,[sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=1: continue
      s,f = atom2so[atom:atom+2]
      pab2v[sd:fd,s:f,s:f] = self.prod_log.sp2vertex[spp]

    for sd,fd,pt,spp in zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt!=2: continue
      inf= self.bp2info[spp]
      a,b = inf.atoms
      sa,fa,sb,fb = atom2so[a],atom2so[a+1],atom2so[b],atom2so[b+1]
      pab2v[sd:fd,sa:fa,sb:fb] = inf.vrtx
      pab2v[sd:fd,sb:fb,sa:fa] = einsum('pab->pba', inf.vrtx)
    return pab2v

  def get_dp_vertex_nnz(self):
    """ Number of non-zero elements in the dominant product vertex : can be speedup, but..."""
    nnz = 0
    for pt,spp in zip(self.dpc2t,self.dpc2sp):
      if pt==1: nnz += self.prod_log.sp2vertex[spp].size
      elif pt==2: nnz += 2*self.bp2info[spp].vrtx.size
      else:
        raise RuntimeError('pt?')
    return nnz


  def get_dp_vertex_doubly_sparse(self, dtype=np.float64, sparseformat=lsofcsr_c, axis=0):
    """ Returns the product vertex coefficients for dominant products as a one-dimensional array of sparse matrices """
    nnz = self.get_dp_vertex_nnz()
    i1,i2,i3,data = zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=dtype)
    atom2s,dpc2s,nfdp,n = self.sv.atom2s, self.dpc2s,self.dpc2s[-1],self.sv.atom2s[-1]

    inz = 0
    for s,f,sd,fd,pt,spp in zip(atom2s,atom2s[1:],dpc2s,dpc2s[1:],self.dpc2t,self.dpc2sp):
      size = self.prod_log.sp2vertex[spp].size
      lv = self.prod_log.sp2vertex[spp].reshape(size)
      dd,aa,bb = np.mgrid[sd:fd,s:f,s:f].reshape((3,size))
      i1[inz:inz+size],i2[inz:inz+size],i3[inz:inz+size],data[inz:inz+size] = dd,aa,bb,lv
      inz+=size

    for sd,fd,pt,spp in zip(dpc2s,dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt!=2: continue
      inf,(a,b),size = self.bp2info[spp],self.bp2info[spp].atoms,self.bp2info[spp].vrtx.size
      sa,fa,sb,fb = atom2s[a],atom2s[a+1],atom2s[b],atom2s[b+1]
      dd,aa,bb = np.mgrid[sd:fd,sa:fa,sb:fb].reshape((3,size))
      i1[inz:inz+size],i2[inz:inz+size],i3[inz:inz+size] = dd,aa,bb
      data[inz:inz+size] = inf.vrtx.reshape(size)
      inz+=size
      i1[inz:inz+size],i2[inz:inz+size],i3[inz:inz+size] = dd,bb,aa
      data[inz:inz+size] = inf.vrtx.reshape(size)
      inz+=size
    return sparseformat((data, (i1, i2, i3)), dtype=dtype, shape=(nfdp,n,n), axis=axis)

  def get_dp_vertex_doubly_sparse_loops(self, dtype=np.float64, sparseformat=lsofcsr_c, axis=0):
    """ Returns the product vertex coefficients for dominant products as a one-dimensional array of sparse matrices, slow version """
    nnz = self.get_dp_vertex_nnz()
    i1,i2,i3,data = zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=dtype)
    a2s,n,nfdp,lv = self.sv.atom2s,self.sv.atom2s[-1],self.dpc2s[-1],self.prod_log.sp2vertex # local "aliases"
    inz = 0
    for atom,[sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=1: continue
      s,f = a2s[atom:atom+2]
      for p in range(sd,fd):
        for a in range(s,f):
          for b in range(s,f):
            i1[inz],i2[inz],i3[inz],data[inz] = p,a,b,lv[spp][p-sd,a-s,b-s]
            inz+=1

    for atom, [sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=2: continue
      inf= self.bp2info[spp]
      a,b = inf.atoms
      sa,fa,sb,fb = a2s[a],a2s[a+1],a2s[b],a2s[b+1]
      for p in range(sd,fd):
        for a in range(sa,fa):
          for b in range(sb,fb):
            i1[inz],i2[inz],i3[inz],data[inz] = p,a,b,inf.vrtx[p-sd,a-sa,b-sb]; inz+=1;
            i1[inz],i2[inz],i3[inz],data[inz] = p,b,a,inf.vrtx[p-sd,a-sa,b-sb]; inz+=1;
    return sparseformat((data, (i1, i2, i3)), dtype=dtype, shape=(nfdp,n,n), axis=axis)


  def get_dp_vertex_sparse(self, dtype=np.float64, sparseformat=coo_matrix):
    """ Returns the product vertex coefficients as 3d array for dominant products, in a sparse format coo(p,ab) by default"""
    nnz = self.get_dp_vertex_nnz()                                                         #Number of non-zero elements in the dominant product vertex
    irow,icol,data = zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=dtype) # Start to construct coo matrix

    atom2s,dpc2s,nfdp,n = self.sv.atom2s, self.dpc2s,self.dpc2s[-1],self.sv.atom2s[-1]     #self.dpc2s, self.dpc2t, self.dpc2sp: product Center -> list of the size of the basis set in this center,of center's types,of product species

    inz = 0
    for s,f,sd,fd,pt,spp in zip(atom2s,atom2s[1:],dpc2s,dpc2s[1:],self.dpc2t,self.dpc2sp):
      size = self.prod_log.sp2vertex[spp].size
      lv = self.prod_log.sp2vertex[spp].reshape(size)
      dd,aa,bb = np.mgrid[sd:fd,s:f,s:f].reshape((3,size))
      irow[inz:inz+size],icol[inz:inz+size],data[inz:inz+size] = dd, aa+bb*n, lv
      inz+=size

    for sd,fd,pt,spp in zip(dpc2s,dpc2s[1:],self.dpc2t,self.dpc2sp):
      if pt!=2: continue
      inf,(a,b),size = self.bp2info[spp],self.bp2info[spp].atoms,self.bp2info[spp].vrtx.size
      sa,fa,sb,fb = atom2s[a],atom2s[a+1],atom2s[b],atom2s[b+1]
      dd,aa,bb = np.mgrid[sd:fd,sa:fa,sb:fb].reshape((3,size))
      irow[inz:inz+size],icol[inz:inz+size] = dd,aa+bb*n
      data[inz:inz+size] = inf.vrtx.reshape(size)
      inz+=size

      irow[inz:inz+size],icol[inz:inz+size] = dd,bb+aa*n
      data[inz:inz+size] = inf.vrtx.reshape(size)
      inz+=size
      
    return sparseformat((data, (irow, icol)), dtype=dtype, shape=(nfdp,n*n))


  def get_dp_vertex_sparse2(self, dtype=np.float64, sparseformat=coo_matrix):
    """ Returns the product vertex coefficients as 3d array for dominant products, in a sparse format coo(pa,b) by default"""
    import numpy as np
    from scipy.sparse import csr_matrix, coo_matrix
    nnz = self.get_dp_vertex_nnz()                                                           
    irow,icol,data = np.zeros(nnz, dtype=int), np.zeros(nnz, dtype=int), np.zeros(nnz, dtype=dtype)
    atom2s,dpc2s,nfdp,n = self.sv.atom2s, self.dpc2s, self.dpc2s[-1],self.sv.atom2s[-1]   

    inz = 0
    for s,f,sd,fd,pt,spp in zip(atom2s,atom2s[1:],dpc2s,dpc2s[1:],self.dpc2t,self.dpc2sp):
        size = self.prod_log.sp2vertex[spp].size
        lv = self.prod_log.sp2vertex[spp].reshape(size)
        dd,aa,bb = np.mgrid[sd:fd,s:f,s:f].reshape((3,size))
        irow[inz:inz+size],icol[inz:inz+size],data[inz:inz+size] = dd+aa*nfdp, bb, lv
        inz+=size
           
    for sd,fd,pt,spp in zip(dpc2s,dpc2s[1:],self.dpc2t, self.dpc2sp):
        if pt!=2: continue
        inf,(a,b),size = self.bp2info[spp],self.bp2info[spp].atoms,self.bp2info[spp].vrtx.size
        sa,fa,sb,fb = atom2s[a],atom2s[a+1],atom2s[b],atom2s[b+1]
        dd,aa,bb = np.mgrid[sd:fd,sa:fa,sb:fb].reshape((3,size))
        irow[inz:inz+size],icol[inz:inz+size] = dd+aa*nfdp, bb
        data[inz:inz+size] = inf.vrtx.reshape(size)
        inz+=size
     
        irow[inz:inz+size],icol[inz:inz+size] = dd+bb*nfdp, aa
        data[inz:inz+size] = inf.vrtx.reshape(size)
        inz+=size
    return coo_matrix((data, (irow, icol)), dtype=dtype, shape=(nfdp*n,n))


  def get_dp_vertex_sparse_loops(self, dtype=np.float64, sparseformat=coo_matrix):
    """ Returns the product vertex coefficients as 3d array for dominant products, in a sparse format coo(p,ab) slow version"""
    nnz = self.get_dp_vertex_nnz()
    irow,icol,data = zeros(nnz, dtype=int), zeros(nnz, dtype=int), zeros(nnz, dtype=dtype) # Start to construct coo matrix

    atom2so = self.sv.atom2s
    nfdp = self.dpc2s[-1]
    n = self.sv.atom2s[-1]
    inz = 0
    for atom,[sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=1: continue
      s,f = atom2so[atom:atom+2]
      for p in range(sd,fd):
        for a in range(s,f):
          for b in range(s,f):
            irow[inz],icol[inz],data[inz] = p,a+b*n,self.prod_log.sp2vertex[spp][p-sd,a-s,b-s]
            inz+=1

    for atom, [sd,fd,pt,spp] in enumerate(zip(self.dpc2s,self.dpc2s[1:],self.dpc2t,self.dpc2sp)):
      if pt!=2: continue
      inf= self.bp2info[spp]
      a,b = inf.atoms
      sa,fa,sb,fb = atom2so[a],atom2so[a+1],atom2so[b],atom2so[b+1]
      for p in range(sd,fd):
        for a in range(sa,fa):
          for b in range(sb,fb):
            irow[inz],icol[inz],data[inz] = p,a+b*n,inf.vrtx[p-sd,a-sa,b-sb]; inz+=1;
            irow[inz],icol[inz],data[inz] = p,b+a*n,inf.vrtx[p-sd,a-sa,b-sb]; inz+=1;
    return sparseformat((data, (irow, icol)), dtype=dtype, shape=(nfdp,n*n))


  def comp_fci_den(self, hk, dtype=np.float64):
    """ Compute the four-center integrals and return it in a dense storage """
    pab2v = self.get_ac_vertex_array(dtype=dtype)
    pcd = np.einsum('pq,qcd->pcd', hk, pab2v)
    abcd = np.einsum('pab,pcd->abcd', pab2v, pcd)
    return abcd
    
  def init_c2s_domiprod(self):
    """Compute the array of start indices for dominant product basis set """
    c2n,c2t,c2sp = [],[],[] #  product Center -> list of the size of the basis set in this center,of center's types,of product species
    for atom,sp in enumerate(self.sv.atom2sp):
      c2n.append(self.prod_log.sp2vertex[sp].shape[0]); c2t.append(1); c2sp.append(sp);
    for ibp,inf in enumerate(self.bp2info): 
      c2n.append(inf.vrtx.shape[0]); c2t.append(2); c2sp.append(ibp);

    ndpc = len(c2n)  # number of product centers in this vertex 
    c2s = zeros(ndpc+1, np.int64 ) # product Center -> Start index of a product function in a global counting for this vertex
    for c in range(ndpc): c2s[c+1] = c2s[c] + c2n[c]
    return c2s,c2t,c2sp

  def comp_moments(self, dtype=np.float64):
    """ Computes the scalar and dipole moments for the all functions in the product basis """
    sp2mom0, sp2mom1 = self.prod_log.comp_moments()
    n = self.c2s[-1]
    mom0 = np.require(zeros(n, dtype=dtype), requirements='CW')
    mom1 = np.require(zeros((n,3), dtype=dtype), requirements='CW')
    for a,[sp,coord,s,f] in enumerate(zip(self.sv.atom2sp,self.sv.atom2coord,self.c2s,self.c2s[1:])):
      mom0[s:f],mom1[s:f,:] = sp2mom0[sp], einsum('j,k->jk', sp2mom0[sp],coord)+sp2mom1[sp]
    return mom0,mom1

  def comp_coulomb_pack(self, **kw):
    """ Computes the packed version of the Hartree kernel """
    from pyscf.nao.m_comp_coulomb_pack import comp_coulomb_pack
    return comp_coulomb_pack(self.sv, self.prod_log, **kw)

  def comp_coulomb_den(self, **kw):
    """ Computes the dense (square) version of the Hartree kernel """
    from pyscf.nao.m_comp_coulomb_den import comp_coulomb_den
    return comp_coulomb_den(self.sv, self.prod_log, **kw)

  def comp_fxc_lil(self, **kw):
    """ Computes the sparse version of the xc kernel """
    from pyscf.nao.m_vxc_lil import vxc_lil
    return vxc_lil(self.sv, deriv=2, ao_log=self.prod_log, **kw)
  
  def comp_fxc_pack(self, **kw):
    """ Computes the packed version of the xc kernel """
    from pyscf.nao.m_vxc_pack import vxc_pack
    return vxc_pack(self.sv, deriv=2, ao_log=self.prod_log, **kw)
  
  def overlap_check(self, **kw):
    """ Our standard minimal check comparing with overlaps """
    sref = self.sv.overlap_coo(**kw).toarray()
    mom0,mom1 = self.comp_moments()
    vpab = self.get_ac_vertex_array()
    sprd = np.einsum('p,pab->ab', mom0,vpab)
    return [[abs(sref-sprd).sum()/sref.size, np.amax(abs(sref-sprd))]]

  def dipole_check(self, **kw):
    """ Our standard minimal check """
    dipcoo = self.sv.dipole_coo(**kw)
    mom0,mom1 = self.comp_moments()
    vpab = self.get_ac_vertex_array()
    xyz2err = []
    for i,dref in enumerate(dipcoo):
      dref = dref.toarray()
      dprd = np.einsum('p,pab->ab', mom1[:,i],vpab)
      xyz2err.append([abs(dprd-dref).sum()/dref.size, np.amax(abs(dref-dprd))])
    return xyz2err

  def get_nprod(self):
    """ Number of (atom-centered) products """
    return self.c2s[-1]

  def generate_png_spy_dp_vertex(self):
    """Produces pictures of the dominant product vertex in a common black-and-white way"""
    import matplotlib.pyplot as plt
    plt.ioff()
    dab2v = self.get_dp_vertex_doubly_sparse()
    for i,ab2v in enumerate(dab2v): 
      plt.spy(ab2v.toarray())
      fname = "spy-v-{:06d}.png".format(i)
      print(fname)
      plt.savefig(fname, bbox_inches='tight')
      plt.close()
    return 0
  
  def generate_png_chess_dp_vertex(self):
    """Produces pictures of the dominant product vertex a chessboard convention"""
    import matplotlib.pylab as plt
    plt.ioff()
    dab2v = self.get_dp_vertex_doubly_sparse()
    for i, ab in enumerate(dab2v): 
        fname = "chess-v-{:06d}.png".format(i)
        print('Matrix No.#{}, Size: {}, Type: {}'.format(i+1, ab.shape, type(ab)), fname)
        if type(ab) != 'numpy.ndarray': ab = ab.toarray()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(ab, interpolation='nearest', cmap=plt.cm.ocean)
        plt.colorbar()
        plt.savefig(fname)
        plt.close(fig)




  def reshape_COO(a, shape):
    """Reshape the sparse matrix (a) and returns a coo_matrix with favourable shape."""
    from scipy.sparse import coo_matrix
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b
#
#
#
if __name__=='__main__':
  from pyscf.nao import prod_basis_c, nao
  from pyscf.nao.m_overlap_coo import overlap_coo
  from pyscf import gto
  import numpy as np
  
  mol = gto.M(atom='O 0 0 0; H 0 0 0.5; H 0 0.5 0', basis='ccpvdz') # coordinates in Angstrom!
  sv = nao(gto=mol)
  print(sv.atom2s)
  s_ref = overlap_coo(sv).todense()
  pb = prod_basis_c()
  pb.init_prod_basis_pp_batch(sv)
  mom0,mom1=pb.comp_moments()
  pab2v = pb.get_ac_vertex_array()
  s_chk = einsum('pab,p->ab', pab2v,mom0)
  print(abs(s_chk-s_ref).sum()/s_chk.size, abs(s_chk-s_ref).max())
