from __future__ import print_function, division
import sys, numpy as np
from numpy import require
from timeit import default_timer as timer

from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix 

from pyscf.nao.m_color import color as bc
from pyscf.nao.m_system_vars_dos import system_vars_dos, system_vars_pdos
from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.ao_log import ao_log
from pyscf.nao.mesh_affine_equ import mesh_affine_equ
    
#
#
#
def get_orb2m(sv):
  orb2m = np.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom,sp in enumerate(sv.atom2sp):
    for mu,j in enumerate(sv.sp_mu2j[sp]):
      for m in range(-j,j+1): orb2m[orb],orb = m,orb+1
  return orb2m

#
#
#
def get_orb2j(sv):
  orb2j = np.empty(sv.norbs, dtype='int64')
  orb = 0
  for atom,sp in enumerate(sv.atom2sp):
    for mu,j in enumerate(sv.sp_mu2j[sp]):
      for m in range(-j,j+1): orb2j[orb],orb = j,orb+1
  return orb2j

#
#
#
def overlap_check(sv, tol=1e-5, **kvargs):
  over = sv.overlap_coo(**kvargs).tocsr()
  diff = abs(sv.hsx.s4_csr-over).sum()
  summ = abs(sv.hsx.s4_csr+over).sum()
  ac = diff/summ<tol
  if not ac: print(diff, summ)
  return ac

#
#
#
class nao():

  def __init__(self, **kw):
    """  Constructor of NAO class """
   
    self.dtype = kw['dtype'] if 'dtype' in kw else np.float64
    if self.dtype == np.float32:
      self.dtypeComplex = np.complex64
    elif self.dtype == np.float64:
      self.dtypeComplex = np.complex128
    else:
      raise ValueError("dtype can be only float32 or float64")

    import scipy
    if int(scipy.__version__[0])>0: 
      scipy_ver_def = 1;
    else:
      scipy_ver_def = 0
    self.scipy_ver = kw['scipy_ver'] if 'scipy_ver' in kw else scipy_ver_def

    try:
      import numba
      use_numba_def = True
    except:
      use_numba_def = False
    self.use_numba = kw['use_numba'] if 'use_numba' in kw else use_numba_def

    self.numba_parallel = kw["numba_parallel"] if "numba_parallel" in kw else True 
    
    self.verbosity = kw['verbosity'] if 'verbosity' in kw else 0
    self.verbose = self.verbosity

    if 'gto' in kw:
      self.init_gto(**kw)
      self.init_libnao_orbs()
    elif 'xyz_list' in kw:
      self.init_xyz_list(**kw)
    elif 'label' in kw:
      self.init_label(**kw)
      self.init_libnao_orbs()
    elif 'wfsx_fname' in kw: # init atomic orbitals with WFSX file from SIESTA output
      self.init_wfsx_fname(**kw)
      self.init_libnao_orbs()
    elif 'gpaw' in kw:
      self.init_gpaw(**kw)
      self.init_libnao_orbs()
    elif 'openmx' in kw:
      self.init_openmx(**kw)
      #self.init_libnao_orbs()
    elif 'fireball' in kw:
      self.init_fireball(**kw)
    else:
      print(__name__, kw.keys())
      raise RuntimeError('unknown init method')

    self.pseudo = hasattr(self, 'sp2ion') 
    self._keys = set(self.__dict__.keys())

    #print(kw)
    #print(dir(kw))

  #
  #
  #
  def init_gto(self, **kw):
    """Interpret previous pySCF calculation"""
    from pyscf.lib import logger

    gto = kw['gto']
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self.cart = False
    self._nelectron = gto.nelectron
    self._built = True
    self.max_memory = 20000

    self.spin = gto.spin
    #print(__name__, 'dir(gto)', dir(gto), gto.nelec)
    self.nspin = 1 if gto.spin==0 else 2 # this can be wrong and has to be redetermined at in the mean-field class (mf)
    self.label = kw['label'] if 'label' in kw else 'pyscf'
    self.mol=gto # Only some data must be copied, not the whole object. Otherwise, an eventual deepcopy(...) may fail.
    self.natm=self.natoms = gto.natm
    a2s = [gto.atom_symbol(ia) for ia in range(gto.natm) ]
    self.sp2symbol = sorted(list(set(a2s)))
    self.nspecies = len(self.sp2symbol)
    self.atom2sp = np.empty((gto.natm), dtype=np.int64)
    for ia,sym in enumerate(a2s): self.atom2sp[ia] = self.sp2symbol.index(sym)

    self.sp2charge = [-999]*self.nspecies
    for ia,sp in enumerate(self.atom2sp): self.sp2charge[sp]=gto.atom_charge(ia)
    self.ao_log = ao_log(**kw)
    self.atom2coord = np.zeros((self.natm, 3))
    for ia,coord in enumerate(gto.atom_coords()): self.atom2coord[ia,:]=coord # must be in Bohr already?
    self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]
    self.norbs = self.norbs_sc = self.atom2s[-1]
    self.ucell = 30.0*np.eye(3)
    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]
    self._atom = gto._atom
    self.basis = gto.basis
    ### implement when needed  self.init_libnao()
    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1
    self.sp_mu2j = self.ao_log.sp_mu2j
    self.nkpoints = 1
    return self

  #
  #
  #
  def init_xyz_list(self, **kw):
    """ This is simple constructor which only initializes geometry info """
    from pyscf.lib import logger
    from pyscf.lib.parameters import ELEMENTS as chemical_symbols
    self.verbose = logger.NOTE  # To be similar to Mole object...
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self.cart = False

    self.label = kw['label'] if 'label' in kw else 'pyscf'
    atom = kw['xyz_list']
    atom2charge = [atm[0] for atm in atom]
    self.atom2coord = np.array([atm[1] for atm in atom])
    self.sp2charge = list(set(atom2charge))
    self.sp2symbol = [chemical_symbols[z] for z in self.sp2charge]
    self.atom2sp = [self.sp2charge.index(charge) for charge in atom2charge]
    self.natm=self.natoms=len(self.atom2sp)
    self.atom2s = None
    self.nspin = 1
    self.nbas  = self.natm
    self.state = 'should be useful for something'
    return self

  #
  #
  #
  def init_fireball(self, **kw):
    from pyscf.nao.m_fireball_import import fireball_import
    from timeit import default_timer as timer
    """
      Initialise system var using only the fireball files (standard output in particular is needed)
      System variables:
      -----------------
        chdir (string): calculation directory
    """
    fireball_import(self, **kw)
    return self

  #
  #
  #
  def init_wfsx_fname(self, **kw):
    """  Initialise system var starting with a given WFSX file  """
    from pyscf.nao.m_tools import read_xyz
    from pyscf.nao.m_fermi_energy import fermi_energy
    from pyscf.nao.m_siesta_xml import siesta_xml
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao.m_siesta_hsx import siesta_hsx_c
    from timeit import default_timer as timer

    self.label = label = kw['label'] if 'label' in kw else 'siesta'
    self.cd = cd = kw['cd'] if 'cd' in kw else '.'

    fname = kw['wfsx_fname'] if 'wfsx_fname' in kw else None
    self.wfsx = siesta_wfsx_c(fname=fname, **kw)
    self.hsx = siesta_hsx_c(fname=cd+'/'+self.label+'.HSX', **kw)
    self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)
    self.natm = self.natoms = max(self.wfsx.orb2atm)
    self.norbs = len(self.wfsx.orb2atm)
    self.norbs_sc = self.norbs
    self.nspin = self.wfsx.nspin
    self.ucell = 30.0*np.eye(3)
    self.nkpoints  = self.wfsx.nkpoints

    self.sp2ion = []
    for sp in self.wfsx.sp2strspecie: 
        self.sp2ion.append(siesta_ion_xml(cd+'/'+sp+'.ion.xml'))
    _siesta_ion_add_sp2(self, self.sp2ion)
    #self.ao_log = ao_log_c().init_ao_log_ion(self.sp2ion, **kw)
    self.ao_log = ao_log(sp2ion=self.sp2ion, **kw)
    self.kb_log = ao_log(sp2ion=self.sp2ion, fname='kbs', **kw)
    
    strspecie2sp = {}
    # initialise a dictionary with species string as a key associated to the specie number
    for sp,strsp in enumerate(self.wfsx.sp2strspecie): strspecie2sp[strsp] = sp
    
    # list of atoms associated to them specie number
    self.atom2sp = np.empty((self.natm), dtype=np.int64)
    for o,atom in enumerate(self.wfsx.orb2atm):
      self.atom2sp[atom-1] = strspecie2sp[self.wfsx.orb2strspecie[o]]

    self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp):
        self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]

    # atom2mu_s list of atom associated to them multipletts (radial orbitals)
    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp):
        self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]

    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1


    #t1 = timer()
    orb2m = self.get_orb2m()
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)

    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.x[k,s,n,:,:])
    #t2 = timer(); print(t2-t1, 'rsh wfsx'); t1 = timer()

    # Get self.atom2coord via .xml or .xyz files
    try:
      self.xml_dict = siesta_xml(cd+'/'+self.label+'.xml')
      self.atom2coord = self.xml_dict['atom2coord']
      self.fermi_energy = self.xml_dict['fermi_energy']
    except:
      print(__name__, 'no siesta.xml file --> excepting with siesta.xyz file')
      a2sym,a2coord = read_xyz(cd+'/'+self.label+'.xyz')
      self.atom2coord = a2coord*1.8897259886
      # cross-check of loaded .xyz file:
      atom2sym_ref = np.array([self.sp2ion[sp]['symbol'].strip() for atom,sp in enumerate(self.atom2sp)])
      for ia,(a1,a2) in enumerate(zip(a2sym,atom2sym_ref)):
        if a1!=a2: raise RuntimeError('.xyz wrong? %d %s %s '% (ia, a1,a2))
      self.fermi_energy = fermi_energy(self.wfsx.ksn2e, self.hsx.nelec, self.hsx.telec)


    self.sp2symbol = [str(ion['symbol'].replace(' ', '')) for ion in self.sp2ion]
    self.sp2charge = self.ao_log.sp2charge

    # Trying to be similar to mole object from pySCF 
    self._xc_code   = 'LDA,PZ' # estimate how ? 
    self._nelectron = self.hsx.nelec
    self.cart = False
    self.spin = self.nspin-1
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self._built = True 
    self.max_memory = 20000
    self.incore_anyway = False        
    self._atom = [(self.sp2symbol[sp], list(self.atom2coord[ia,:])) for ia,sp in enumerate(self.atom2sp)]
    self.state = 'should be useful for something'      
    return self

  #
  #
  #
  def init_label(self, **kw):
    from pyscf.nao.m_siesta_xml import siesta_xml
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao.m_siesta_hsx import siesta_hsx_c
    """
      Initialise system var using only the siesta files (siesta.xml in particular is needed)

      System variables:
      -----------------
        label (string): calculation label
        chdir (string): calculation directory
        xml_dict (dict): information extracted from the xml siesta output, see m_siesta_xml
        wfsx: class use to extract the information about wavefunctions, see m_siesta_wfsx
        hsx: class to store a sparse representation of hamiltonian and overlap, see m_siesta_hsx
        norbs_sc (integer): number of orbital
        ucell (array, float): unit cell
        sp2ion (list): species to ions, list of the species associated to the information from the ion files, see m_siesta_ion_xml
        ao_log: Atomic orbital on an logarithmic grid, see m_ao_log
        atom2coord (array, float): array containing the coordinates of each atom.
        natm, natoms (integer): number of atoms
        norbs (integer): number of orbitals
        nspin (integer): number of spin
        nkpoints (integer): number of kpoints
        fermi_energy (float): Fermi energy
        atom2sp (list): atom to specie, list associating the atoms to their specie number
        atom2s: atom -> first atomic orbital in a global orbital counting
        atom2mu_s: atom -> first multiplett (radial orbital) in a global counting of radial orbitals
        sp2symbol (list): list associating the species to their symbol
        sp2charge (list): list associating the species to their charge
        state (string): this is an internal information on the current status of the class
    """

    #label='siesta', cd='.', verbose=0, **kvargs

    self.label = label = kw['label'] if 'label' in kw else 'siesta'
    self.cd = cd = kw['cd'] if 'cd' in kw else '.'
    self.xml_dict = siesta_xml(cd+'/'+self.label+'.xml')
    self.wfsx = siesta_wfsx_c(**kw)
    self.hsx = siesta_hsx_c(fname=cd+'/'+self.label+'.HSX', **kw)
    self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)
    if 'ucell' in kw:
      uc = kw['ucell']
      self.ucell = uc if type(uc)==np.ndarray else uc*np.eye(3)
      if self.verbosity>0: print(__name__, "ucell: \n{}".format(self.ucell))
      kw.pop('ucell')
    else:
      self.ucell = self.xml_dict["ucell"]

    self.atom2coord = self.xml_dict['atom2coord']
    self.natm=self.natoms=len(self.xml_dict['atom2sp'])
    orig = self.atom2coord.sum(axis=0)/self.natoms
    self.mesh3d = mesh_affine_equ(ucell=self.ucell, origin=orig, **kw)

    ##### The parameters as fields     
    self.sp2ion = []
    for sp in self.wfsx.sp2strspecie: self.sp2ion.append(siesta_ion_xml(cd+'/'+sp+'.ion.xml'))

    _siesta_ion_add_sp2(self, self.sp2ion)
    self.ao_log = ao_log(sp2ion=self.sp2ion, **kw)
    self.kb_log = ao_log(sp2ion=self.sp2ion, fname='kbs', rr=self.ao_log.rr, pp=self.ao_log.pp)

    self.norbs  = self.wfsx.norbs 
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints
    self.fermi_energy = self.xml_dict['fermi_energy']

    strspecie2sp = {}
    # initialise a dictionary with species string as key
    # associated to the specie number
    for sp,strsp in enumerate(self.wfsx.sp2strspecie): strspecie2sp[strsp] = sp
    
    # list of atoms associated to them specie number
    self.atom2sp = np.empty((self.natm), dtype=np.int64)
    for o,atom in enumerate(self.wfsx.orb2atm):
      self.atom2sp[atom-1] = strspecie2sp[self.wfsx.orb2strspecie[o]]

    self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp):
        self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]

    # atom2mu_s list of atom associated to them multipletts (radial orbitals)
    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp):
        self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]
    
    orb2m = self.get_orb2m()
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    #t1 = timer()
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.x[k,s,n,:,:])
    #t2 = timer(); print(t2-t1, 'rsh wfsx'); t1 = timer()

    
    self.sp2symbol = [str(ion['symbol'].replace(' ', '')) for ion in self.sp2ion]
    self.sp2charge = self.ao_log.sp2charge

    # Trying to be similar to mole object from pySCF 
    self._xc_code   = 'LDA,PZ' # estimate how ? 
    self._nelectron = self.hsx.nelec
    self.cart = False
    self.spin = self.nspin-1
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self._built = True 
    self.max_memory = 20000
    self.incore_anyway = False
    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1
    self._atom = [(self.sp2symbol[sp], list(self.atom2coord[ia,:])) for ia,sp in enumerate(self.atom2sp)]
    self.init_mo_coeff_label(**kw)
    self.spin = self.magnetization

    self.state = 'should be useful for something'
    return self

  def init_mo_coeff_label(self, **kw):
    """ Constructor a mean-field class from the preceeding SIESTA calculation """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_fermi_energy import fermi_energy
        
    self.mo_coeff = require(self.wfsx.x, dtype=self.dtype, requirements='CW')
    self.mo_energy = require(self.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else self.hsx.telec


    self.magnetization = kw['magnetization'] if 'magnetization' in kw else None #using this key for number of unpaired
    if self.nspin==1:
      self.nelec = kw['nelec'] if 'nelec' in kw else np.array([self.hsx.nelec])
    elif (self.nspin==2 and self.magnetization==None):
      self.nelec =kw['nelec'] if 'nelec' in kw else np.array([int(self.hsx.nelec/2), int(self.hsx.nelec/2)])
    elif (self.nspin==2 and self.magnetization!=None):
      if 'nelec' in kw: self.nelec = kw['nelec']
      else:
        ne = self.hsx.nelec
        nalpha = (ne + self.magnetization) // 2
        nbeta = nalpha - self.magnetization
        if nalpha + nbeta != ne:
          raise RuntimeError('Electron number %d and spin %d are not consistent\n'
                             'Note mol.spin = 2S = Nalpha - Nbeta, not 2S+1' % (ne, self.magnetization))
        self.nelec = np.array([nalpha, nbeta])


      if self.verbosity>0: print(__name__, 'not sure here: self.nelec', self.nelec)
    else:
      raise RuntimeError('0>nspin>2?')
    
    if 'fermi_energy' in kw: self.fermi_energy = kw['fermi_energy'] # possibility to redefine Fermi energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd
    nelec_occ = np.einsum('ksn->s', self.mo_occ)/self.nkpoints
    if not np.allclose(self.nelec, nelec_occ, atol=1e-4):
      fermi_guess = fermi_energy(self.wfsx.ksn2e, self.hsx.nelec, self.hsx.telec)
      np.set_printoptions(precision=2, linewidth=1000)
      raise RuntimeWarning(
      '''occupations?\n mo_occ: \n{}\n telec: {}\n nelec expected: {}
 nelec(occ): {}\n Fermi guess: {}\n Fermi: {}\n E_n:\n{}'''.format(self.mo_occ,
 self.telec, self.nelec, nelec_occ, fermi_guess, self.fermi_energy, self.mo_energy))
 
    if 'fermi_energy' in kw and self.verbosity>0:
      po = np.get_printoptions() 
      np.set_printoptions(precision=2, linewidth=1000)
      print(__name__, "mo_occ:\n{}".format(self.mo_occ))
      np.set_printoptions(**po)



      
  def make_rdm1(self, mo_coeff=None, mo_occ=None):
    # from pyscf.scf.hf import make_rdm1 -- different index order here
    if mo_occ is None: mo_occ = self.mo_occ[0,:,:]
    if mo_coeff is None: mo_coeff = self.mo_coeff[0,:,:,:,0]
    dm = np.zeros((1,self.nspin,self.norbs,self.norbs,1))
    for s in range(self.nspin):
      xocc = mo_coeff[s,mo_occ[s]>0,:]
      focc = mo_occ[s,mo_occ[s]>0]
      dm[0,s,:,:,0] = np.dot(xocc.T.conj() * focc, xocc)
    return dm

  def init_gpaw(self, **kw):
    """ Use the data from a GPAW LCAO calculations as input to initialize system variables. """
    try:
        import ase
        import gpaw
    except:
        raise ValueError("ASE and GPAW must be installed for using system_vars_gpaw")
    from pyscf.nao.m_system_vars_gpaw import system_vars_gpaw
    return system_vars_gpaw(self, **kw)

  #
  #
  #
  def init_openmx(self, **kw):
    from pyscf.nao.m_openmx_import_scfout import openmx_import_scfout
    from timeit import default_timer as timer
    """
      Initialise system var using only the OpenMX output (label.scfout in particular is needed)

      System variables:
      -----------------
        label (string): calculation label
        chdir (string): calculation directory
        xml_dict (dict): information extracted from the xml siesta output, see m_siesta_xml
        wfsx: class use to extract the information about wavefunctions, see m_siesta_wfsx
        hsx: class to store a sparse representation of hamiltonian and overlap, see m_siesta_hsx
        norbs_sc (integer): number of orbital
        ucell (array, float): unit cell
        sp2ion (list): species to ions, list of the species associated to the information from the pao files
        ao_log: Atomic orbital on an logarithmic grid, see m_ao_log
        atom2coord (array, float): array containing the coordinates of each atom.
        natm, natoms (integer): number of atoms
        norbs (integer): number of orbitals
        nspin (integer): number of spin
        nkpoints (integer): number of kpoints
        fermi_energy (float): Fermi energy
        atom2sp (list): atom to specie, list associating the atoms to their specie number
        atom2s: atom -> first atomic orbital in a global orbital counting
        atom2mu_s: atom -> first multiplett (radial orbital) in a global counting of radial orbitals
        sp2symbol (list): list soociating the species to their symbol
        sp2charge (list): list associating the species to their charge
        state (string): this is an internal information on the current status of the class
    """
    #label='openmx', cd='.', **kvargs
    openmx_import_scfout(self, **kw)
    self.state = 'must be useful for something already'
    return self

  # More functions for similarity with Mole
  def atom_symbol(self, ia): return self.sp2symbol[self.atom2sp[ia]]
  def atom_symbols(self): return np.array([self.sp2symbol[sp] for sp in self.atom2sp])
  def atom_charge(self, ia): return self.sp2charge[self.atom2sp[ia]]
  def atom_charges(self): return np.array([self.sp2charge[sp] for sp in self.atom2sp], dtype='int64')
  def atom_coord(self, ia): return self.atom2coord[ia,:]
  def atom_coords(self): return self.atom2coord
  def nao_nr(self): return self.norbs
  def atom_nelec_core(self, ia): return self.sp2charge[self.atom2sp[ia]]-self.ao_log.sp2valence[self.atom2sp[ia]]
  def ao_loc_nr(self): return self.mu2orb_s[0:self.natm]

  def intor_symmetric(self, intor, comp=None):
    print(__name__, intor, comp)
    if intor=='int1e_kin':
      return 0.5*self.laplace_coo().toarray()
    elif intor=='int1e_nuc':
      return self.vnucele_coo().toarray()
    else:
      raise RuntimeError('unknown intor '+intor)
    return 0

  def get_init_guess(self, key=None):
    """ Compute an initial guess for the density matrix. ???? """
    from pyscf.scf.hf import init_guess_by_minao
    if hasattr(self, 'mol'):
      dm = init_guess_by_minao(self.mol)
    else:
      dm = self.make_rdm1()  # the loaded ks orbitals will be used
      if dm.shape[0:2]==(1,1) and dm.shape[4]==1 : dm = dm.reshape((self.norbs,self.norbs))
    return dm
  
  init_guess_by_minao = get_init_guess

  # More functions for convenience (see PDoS)
  def get_orb2j(self): return get_orb2j(self)
  def get_orb2m(self): return get_orb2m(self)
    
  def overlap_coo(self, **kw):   # Compute overlap matrix for the molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    return overlap_coo(self, **kw)

  def overlap_lil(self, **kw):   # Compute overlap matrix in list of lists format
    from pyscf.nao.m_overlap_lil import overlap_lil
    return overlap_lil(self, **kw)

  def laplace_coo(self):   # Compute matrix of Laplace brakets for the whole molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    from pyscf.nao.m_laplace_am import laplace_am
    return overlap_coo(self, funct=laplace_am)

  def vnucele_coo(self, **kw):
    if self.pseudo:
      return self.vnucele_coo_pseudo(**kw)
    else:
      return self.vnucele_coo_coulomb(**kw)

  def vnucele_coo_coulomb(self, **kw):
    g = self.build_3dgrid_ae(**kw)
    vnuc = self.comp_vnuc_coulomb(g.coords)
    return self.matelem_int3d_coo(g, vnuc)

  def vnucele_coo_pseudo(self, **kw): # Compute matrix elements of attraction by forces from pseudo atom
    vna = self.vna_coo(**kw)
    vnl = self.vnl_coo()    
    return (vna+vnl).tocoo()

  def vnl_coo(self): 
    """  Non-local part of the Hamiltonian due to Kleinman-Bylander projectors  """
    from pyscf.nao.m_overlap_am import overlap_am
    from scipy.sparse import dia_matrix
    sop = self.overlap_coo(ao_log=self.ao_log, ao_log2=self.kb_log, funct=overlap_am).tocsr()
    nkb = sop.shape[1]
    vkb_dia = dia_matrix( ( self.get_vkb(), [0] ), shape = (nkb,nkb) )
    return ((sop*vkb_dia)*sop.T).tocoo()

  def get_vkb(self): 
    """ Compose the vector of Kleinman-Bylander energies v^p = v^KB_ln, where p is a global projector index """
    atom2s, kb = np.zeros((self.natm+1), dtype=int), self.kb_log
    for atom,sp in enumerate(self.atom2sp): 
        atom2s[atom+1]=atom2s[atom]+kb.sp2norbs[sp]
    vkb = np.zeros(atom2s[-1])
    for sp,gs in zip(self.atom2sp,atom2s):
        for v,s,f in zip(kb.sp_mu2vkb[sp], kb.sp_mu2s[sp], kb.sp_mu2s[sp][1:]): 
            vkb[gs+s:gs+f] = v
    return vkb
  
  def dipole_coo(self, **kw):   # Compute dipole matrix elements for the given system
    from pyscf.nao.m_dipole_coo import dipole_coo
    return dipole_coo(self, **kw)
  
  def overlap_check(self, **kw): # Works for pyscf and after init_siesta_xml()
      if hasattr(self, 'mol'):
          from pyscf import gto, scf                           
          tol = kw.get('tol', 1e-7)
          result = True
          ovlp_pyscf = self.get_ovlp()
          ovlp_nao = self.overlap_coo(**kw).toarray()
          diff = (abs(ovlp_nao - ovlp_pyscf)).sum()
          summ = (abs(ovlp_nao + ovlp_pyscf)).sum()
          if diff/summ > tol or diff/ovlp_nao.size > tol:
              result = False
          check = result,'tol:{}, MAX:{}'.format(tol,np.max(np.abs(ovlp_nao - ovlp_pyscf))), diff/summ
      else:
          tol = kw.get('tol', 1e-5)
          ovlp_nao = self.overlap_coo(**kw).tocsr()
          diff = (self.hsx.s4_csr - ovlp_nao).sum()
          summ = (self.hsx.s4_csr + ovlp_nao).sum()
          result = diff/summ < tol
          check = result, 'tol:{}'.format(tol), diff/summ
      return check

  def energy_nuc(self, charges=None, coords=None):
    """ Potential energy of electrostatic repulsion of point nuclei """
    from scipy.spatial.distance import cdist
    chrg = self.atom_charges() if charges is None else charges
    crds = self.atom_coords() if coords is None else coords
    identity = np.identity(len(chrg))
    return ((chrg[:,None]*chrg[None,:])*(1.0/(cdist(crds, crds)+identity)-identity)).sum()*0.5

  def build_3dgrid_pp(self, level=3):
    """ Build a global grid and weights for a molecular integration (integration in 3-dimensional coordinate space) """
    from pyscf import dft
    from pyscf.nao.m_gauleg import gauss_legendre
    grid = dft.gen_grid.Grids(self)
    grid.level = level # precision as implemented in pyscf
    grid.radi_method=gauss_legendre
    atom2rcut=np.zeros(self.natoms)
    for ia,sp in enumerate(self.atom2sp): atom2rcut[ia] = self.ao_log.sp2rcut[sp]
    grid.build(atom2rcut=atom2rcut)
    return grid

  def build_3dgrid_ae(self, level=3):
    """ Build a global grid and weights for a molecular integration (integration in 3-dimensional coordinate space) """
    from pyscf import dft
    grid = dft.gen_grid.Grids(self)
    grid.level = level # precision as implemented in pyscf
    grid.build()
    return grid

  def ucell_mom(self):  return (2*np.pi)*np.linalg.inv(self.ucell.T) # reciprocal unit cell

  def comp_aos_den(self, coords):
    """ Compute the atomic orbitals for a given set of (Cartesian) coordinates. """
    from pyscf.nao.m_aos_libnao import aos_libnao
    if not self.init_sv_libnao_orbs : raise RuntimeError('not self.init_sv_libnao')
    return aos_libnao(coords, self.norbs)

  def comp_aos_csr(self, coords, tol=1e-8, ram=160e6):
    """ 
          Compute the atomic orbitals for a given set of (Cartesian) coordinates.
        The sparse format CSR is used for output and the computation is organized block-wise.
        Thence, larger molecules can be tackled right away
          coords :: set of Cartesian coordinates
          tol :: tolerance for dropping the values 
          ram :: size of the allowed block (in bytes)
        Returns 
          co2v :: CSR matrix of shape (coordinate, atomic orbital) 
    """
    from pyscf.nao.m_aos_libnao import aos_libnao
    from pyscf import lib
    from scipy.sparse import csr_matrix
    if not self.init_sv_libnao_orbs : raise RuntimeError('not self.init_sv_libnao')
    assert coords.shape[-1] == 3
    nc,no = len(coords), self.norbs
    bsize = int(min(max(ram / (no*8.0), 1), nc))
    co2v = csr_matrix((nc,no))
    for s,f in lib.prange(0,nc,bsize):
      ca2o = aos_libnao(coords[s:f], no) # compute values of atomic orbitals
      ab = np.where(abs(ca2o)>tol)
      co2v += csr_matrix((ca2o[ab].reshape(-1), (ab[0]+s, ab[1])), shape=(nc,no))
    return co2v

  def comp_aos_py(self, coords):
    """ Compute the atomic orbitals for a given set of (Cartesian) coordinates. """
    res = np.zeros((len(coords), self.norbs))
    for sp,rc,s,f in zip(self.atom2sp, self.atom2coord,self.atom2s,self.atom2s[1:]):
      oc2v = self.ao_log.ao_eval(rc, sp, coords)
      res[:,s:f] = oc2v.T
    return res

  def comp_vnuc_coulomb(self, coords):
    ncoo = coords.shape[0]
    vnuc = np.zeros(ncoo)
    for R,sp in zip(self.atom2coord, self.atom2sp):
      dd, Z = cdist(R.reshape((1,3)), coords).reshape(ncoo), self.sp2charge[sp]
      vnuc = vnuc - Z / dd 
    return vnuc

  def vna(self, coords, **kw):
      """ Compute the neutral-atom potential V_NA(coords) for a set of Cartesian coordinates coords.
        The subroutine could be also used for computing the non-linear core corrections or some other atom-centered fields."""
      (sp2v,sp2rcut) = (kw['sp2v'],kw['sp2rcut']) if 'sp2v' in kw else (self.ao_log.sp2vna,self.ao_log.sp2rcut_vna)
      atom2coord = kw['atom2coord'] if 'atom2coord' in kw else self.atom2coord

      nc = coords.shape[0]
      vna = np.zeros(nc)
      for ia,(R,sp) in enumerate(zip(atom2coord, self.atom2sp)):
          if sp2v[sp] is None: # This can be done better via preparation of a special atom2sp excluding ghost atoms
              continue
          #print(__name__, ia, sp, sp2rcut[sp])
          dd = cdist(R.reshape((1,3)), coords).reshape(nc)
          vnaa = self.ao_log.interp_rr(sp2v[sp], dd, rcut=sp2rcut[sp])
          vna = vna + vnaa
      return vna

  def vna_coo(self, **kw):
    """ Compute matrix elements of a potential which is given as superposition of central fields from each nuclei """
    g = self.build_3dgrid_ae(**kw)
    vna = self.vna(g.coords, **kw)
    return self.matelem_int3d_coo(g, vna)

  def matelem_int3d_coo(self, g, v):
    """ Compute matrix elements of a potential v given on the 3d grid g using blocks along the grid """
    from pyscf import lib
    
    bsize = int(min(max(160e6 / (self.norbs*8.0), 1), g.size))
    #print(__name__, bsize, g.size*self.norbs*8)
    v_matelem = np.zeros((self.norbs, self.norbs))
    va = v.reshape(-1)
    wgts = g.weights if type(g.weights)==np.ndarray else np.repeat(g.weights, g.size)
    for s,f in lib.prange(0,g.size,bsize):
      ca2o = self.comp_aos_den(g.coords[s:f]) # compute values of atomic orbitals
      v_w = (wgts[s:f]*va[s:f]).reshape((f-s,1))
      cb2vo = ca2o*v_w
      v_matelem += np.dot(ca2o.T,cb2vo)
    return coo_matrix(v_matelem)

  def matelem_int3d_coo_ref(self, g, v):
    """ Compute matrix elements of a potential v given on the 3d grid g """
    ca2o = self.comp_aos_den(g.coords) # compute values of atomic orbitals
    v_w = (g.weights*v.reshape(g.size)).reshape((g.size,1))
    cb2vo = ca2o*v_w
    v_matelem = np.dot(ca2o.T,cb2vo)
    return coo_matrix(v_matelem)
    
  def init_libnao_orbs(self):
    """ Initialization of data on libnao site """
    from pyscf.nao.m_libnao import libnao
    from pyscf.nao.m_sv_chain_data import sv_chain_data
    from ctypes import POINTER, c_double, c_int64, c_int32, byref
    data = sv_chain_data(self)
    size_x = np.array([1,self.nspin,self.norbs,self.norbs,1], dtype=np.int32)
    libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
    libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
    self.init_sv_libnao_orbs = True

    libnao.init_aos_libnao.argtypes = (POINTER(c_int64), POINTER(c_int64))
    info = c_int64(-999)
    libnao.init_aos_libnao(c_int64(self.norbs), byref(info))
    if info.value!=0: raise RuntimeError("info!=0")
    return self

  @property
  def nelectron(self):
    if self._nelectron is None:
      return tot_electrons(self)
    else:
      return self._nelectron

  def get_symbols (self):
    atm_list = [ self.sp2symbol[sp] for sp in self.atom2sp ]
    return atm_list

#
# Example of reading pySCF orbitals.
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao import nao
  import matplotlib.pyplot as plt
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvtz') # coordinates in Angstrom!
  sv = nao(gto=mol, rcut_tol=1e-8, nr=512, rmin=1e-5)
  
  print('sv.ao_log.sp2norbs:',sv.ao_log.sp2norbs)
  print('sv.ao_log.sp2nmult:',sv.ao_log.sp2nmult)
  print('sv.ao_log.sp2rcut',sv.ao_log.sp2rcut)
  print('sv.ao_log.sp_mu2rcut',sv.ao_log.sp_mu2rcut)
  print('sv.ao_log.nr',sv.ao_log.nr)
  print('sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1]',sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1])
  print('sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape',sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape)

  sp = 0
  for mu,[ff,j] in enumerate(zip(sv.ao_log.psi_log[sp], sv.ao_log.sp_mu2j[sp])):
    nc = abs(ff).max()
    if j==0 : plt.plot(sv.ao_log.rr, ff/nc, '--', label=str(mu)+' j='+str(j))
    if j>0 : plt.plot(sv.ao_log.rr, ff/nc, label=str(mu)+' j='+str(j))

  plt.legend()
  #plt.xlim(0.0, 10.0)
  #plt.show()
