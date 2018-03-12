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

from __future__ import print_function, division
import sys, numpy as np

from pyscf.nao.m_color import color as bc
from pyscf.nao.m_system_vars_dos import system_vars_dos, system_vars_pdos
from pyscf.nao.m_siesta2blanko_csr import _siesta2blanko_csr
from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
from pyscf.nao.m_siesta_ion_add_sp2 import _siesta_ion_add_sp2
from pyscf.nao.m_ao_log import ao_log_c

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
def diag_check(sv, atol=1e-5, rtol=1e-4):
  from pyscf.nao.m_sv_diag import sv_diag 
  ksn2e = sv.xml_dict['ksn2e']
  ac = True
  for k,kvec in enumerate(sv.xml_dict["k2xyzw"]):
    for spin in range(sv.nspin):
      e,x = sv_diag(sv, kvec=kvec[0:3], spin=spin)
      eref = ksn2e[k,spin,:]
      acks = np.allclose(eref,e,atol=atol,rtol=rtol)
      ac = ac and acks
      if(not acks):
        aerr = sum(abs(eref-e))/len(e)
        print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
  return ac

#
#
#
def overlap_check(sv, tol=1e-5, **kvargs):
  over = sv.overlap_coo(**kvargs).tocsr()
  diff = (sv.hsx.s4_csr-over).sum()
  summ = (sv.hsx.s4_csr+over).sum()
  ac = diff/summ<tol
  if not ac: print(diff, summ)
  return ac


def tot_electrons(sv):
  return sv.hsx.nelec 

#
#
#
class system_vars_c():

  def __init__(self):
    """ 
      Constructor of system_vars class: so far can be initialized 
      with SIESTA orbitals and Hamiltonian and wavefunctions
    """
    self.state = 'call an initialize method...'

  #
  #
  #
  def init_xyzlike(self, atom, label='pyscf'):
    """ This is simple constructor which only initializes geometry info """
    from pyscf.lib import logger
    from pyscf.lib.parameters import ELEMENTS as chemical_symbols
    self.verbose = logger.NOTE  # To be similar to Mole object...
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self.cart = False

    self.label = label
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
  def init_pyscf_gto(self, gto, label='pyscf', verbose=0, **kvargs):
    """Interpret previous pySCF calculation"""
    from pyscf.lib import logger

    self.verbose = verbose
    self.stdout = sys.stdout
    self.symmetry = False
    self.symmetry_subgroup = None
    self.cart = False
    self._nelectron = gto.nelectron
    self._built = True
    self.max_memory = 20000

    self.label = label
    self.mol=gto # Only some data must be copied, not the whole object. Otherwise, an eventual deepcopy(...) may fail.
    self.natm=self.natoms = gto.natm
    a2s = [gto.atom_symbol(ia) for ia in range(gto.natm) ]
    self.sp2symbol = sorted(list(set(a2s)))
    self.nspecies = len(self.sp2symbol)
    self.atom2sp = np.empty((gto.natm), dtype=np.int64)
    for ia,sym in enumerate(a2s): self.atom2sp[ia] = self.sp2symbol.index(sym)

    self.sp2charge = [-999]*self.nspecies
    for ia,sp in enumerate(self.atom2sp): self.sp2charge[sp]=gto.atom_charge(ia)
    self.ao_log = ao_log_c().init_ao_log_gto_suggest_mesh(gto, self, **kvargs)
    self.atom2coord = np.zeros((self.natm, 3))
    for ia,coord in enumerate(gto.atom_coords()): self.atom2coord[ia,:]=coord # must be in Bohr already?
    self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]
    self.norbs = self.norbs_sc = self.atom2s[-1]
    self.spin = gto.spin if hasattr(gto, 'spin') else 0
    self.nspin = self.spin + 1
    self.ucell = 20.0*np.eye(3)
    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]
    self._atom = gto._atom
    self.basis = gto.basis
    self.init_libnao()
    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1
    self.state = 'should be useful for something'
    return self
    

  #
  #
  #
  def init_ase_atoms(self, Atoms, **kvargs):
    """ Initialise system vars using siesta file and Atom object from ASE."""
    from pyscf.nao.m_siesta_xml import siesta_xml
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao.m_siesta_hsx import siesta_hsx_c

    self.label = 'ase' if label is None else label
    self.xml_dict = siesta_xml(self.label)
    self.wfsx = siesta_wfsx_c(self.label)
    self.hsx = siesta_hsx_c(self.label, **kvargs)
    self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)

    try:
      import ase
    except:
      warn('no ASE installed: try via siesta.xml')
      self.init_siesta_xml(**kvargs)

    self.Atoms = Atoms
   
    ##### The parameters as fields     
    self.sp2ion = []
    species = []
    for sp in Atoms.get_chemical_symbols():
      if sp not in species:
        species.append(sp)
        self.sp2ion.append(siesta_ion_xml(sp+'.ion.xml'))
    
    _add_mu_sp2(self, self.sp2ion)
    self.sp2ao_log = ao_log_c(self.sp2ion)
  
    self.natm=self.natoms= Atoms.get_positions().shape[0]
    self.norbs  = self.wfsx.norbs
    self.nspin  = self.wfsx.nspin
    self.nkpoints  = self.wfsx.nkpoints

    strspecie2sp = {}
    for sp in range(len(self.wfsx.sp2strspecie)): strspecie2sp[self.wfsx.sp2strspecie[sp]] = sp
    
    self.atom2sp = np.empty((self.natoms), dtype='int64')
    for i, sp in enumerate(Atoms.get_chemical_symbols()):
      self.atom2sp[i] = strspecie2sp[sp]
    
    self.atom2s = np.zeros((sv.natm+1), dtype=np.int64)
    for atom,sp in enumerate(sv.atom2sp): atom2s[atom+1]=atom2s[atom]+self.ao_log.sp2norbs[sp]

    self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]

    orb2m = get_orb2m(self)
    _siesta2blanko_csr(orb2m, self.hsx.s4_csr, self.hsx.orb_sc2orb_uc)

    for s in range(self.nspin):
      _siesta2blanko_csr(orb2m, self.hsx.spin2h4_csr[s], self.hsx.orb_sc2orb_uc)
    
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.X[k,s,n,:,:])

    self.sp2symbol = [str(ion['symbol'].replace(' ', '')) for ion in self.sp2ion]
    self.sp2charge = self.ao_log.sp2charge
    self.nbas = self.atom2mu_s[-1] # total number of radial orbitals
    self.mu2orb_s = np.zeros((self.nbas+1), dtype=np.int64)
    for sp,mu_s in zip(self.atom2sp,self.atom2mu_s):
      for mu,j in enumerate(self.ao_log.sp_mu2j[sp]): self.mu2orb_s[mu_s+mu+1] = self.mu2orb_s[mu_s+mu] + 2*j+1
    self.state = 'should be useful for something'
    return self

  #
  #
  #
  def init_siesta_xml(self, label='siesta', cd='.', verbose=0, **kvargs):
    from pyscf.nao.m_siesta_xml import siesta_xml
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta_ion_xml import siesta_ion_xml
    from pyscf.nao.m_siesta_hsx import siesta_hsx_c
    from timeit import default_timer as timer
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

    self.label = label
    self.cd = cd
    self.xml_dict = siesta_xml(cd+'/'+self.label+'.xml')
    self.wfsx = siesta_wfsx_c(label, cd, **kvargs)
    self.hsx = siesta_hsx_c(cd+'/'+self.label+'.HSX', **kvargs)
    self.norbs_sc = self.wfsx.norbs if self.hsx.orb_sc2orb_uc is None else len(self.hsx.orb_sc2orb_uc)
    self.ucell = self.xml_dict["ucell"]
    ##### The parameters as fields     
    self.sp2ion = []
    for sp in self.wfsx.sp2strspecie: self.sp2ion.append(siesta_ion_xml(cd+'/'+sp+'.ion.xml'))

    _siesta_ion_add_sp2(self, self.sp2ion)
    self.ao_log = ao_log_c().init_ao_log_ion(self.sp2ion)

    self.atom2coord = self.xml_dict['atom2coord']
    self.natm=self.natoms=len(self.xml_dict['atom2sp'])
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
    self.init_libnao()
    self.state = 'should be useful for something'

    # Trying to be similar to mole object from pySCF 
    self._xc_code   = 'LDA,PZ' # estimate how ? 
    self._nelectron = self.hsx.nelec
    self.cart = False
    self.spin = self.nspin-1
    self.verbose = verbose
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
    return self

  def init_gpaw(self, calc, label="gpaw", cd='.', **kvargs):
    """
        use the data from a GPAW LCAO calculations as input to
        initialize system variables.

        Input parameters:
        -----------------
            calc: GPAW calculator
            label (optional, string): label used for the calculations
            chdir (optional, string): path to the directory in which are stored the
                data from gpaw
            kvargs (optional, dict): dictionary of optional arguments
                We may need a list of optional arguments!

        Example:
        --------
            from ase import Atoms
            from gpaw import GPAW
            fname = os.path.dirname(os.path.abspath(__file__))+'/h2o.gpw'
            if os.path.isfile(fname):
                # Import data from a previous gpaw calculations
                calc = GPAW(fname, txt=None) # read previous calculation if the file exists
            else:
                # Run first gpaw to initialize the calculator
                from gpaw import PoissonSolver
                atoms = Atoms('H2O', positions=[[0.0,-0.757,0.587], [0.0,+0.757,0.587], [0.0,0.0,0.0]])
                atoms.center(vacuum=3.5)
                convergence = {'density': 1e-7}     # Increase accuracy of density for ground state
                poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)     # Increase accuracy of Poisson Solver and apply multipole corrections up to l=1
                calc = GPAW(basis='dzp', xc='LDA', h=0.3, nbands=23, convergence=convergence, poissonsolver=poissonsolver, mode='lcao', txt=None)     # nbands must be equal to norbs (in this case 23)
                atoms.set_calculator(calc)
                atoms.get_potential_energy()    # Do SCF the ground state
                calc.write(fname, mode='all') # write DFT output

            from pyscf.nao import system_vars_c
            sv = system_vars_c().init_gpaw(calc)
    """
    try:
        import ase
        import gpaw
    except:
        raise ValueError("ASE and GPAW must be installed for using system_vars_gpaw")
    from pyscf.nao.m_system_vars_gpaw import system_vars_gpaw
    return system_vars_gpaw(self, calc, label="gpaw", chdir='.', **kvargs)
    
  #
  #
  #
  def init_openmx(self, label='openmx', cd='.', **kvargs):
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
    openmx_import_scfout(self, label, cd)
    self.state = 'must be useful for something already'
    return self

  def add_pb_hk(self, **kvargs):
    """ This is adding a product basis attribute to the class and making possible then to compute the matrix elements of Hartree potential or Fock exchange."""
    from pyscf.nao.m_prod_basis import prod_basis_c
    if hasattr(self, 'pb'):
      pb = self.pb
      hk = self.hkernel_den
    else:
      pb = self.pb = prod_basis_c().init_prod_basis_pp(self, **kvargs)
      hk = self.hkernel_den = pb.comp_coulomb_den(**kvargs)
    return pb,hk
    
  # More functions for similarity with Mole
  def atom_symbol(self, ia): return self.sp2symbol[self.atom2sp[ia]]
  def atom_charge(self, ia): return self.sp2charge[self.atom2sp[ia]]
  def atom_charges(self): return np.array([self.sp2charge[sp] for sp in self.atom2sp], dtype='int64')
  def atom_coord(self, ia): return self.atom2coord[ia,:]
  def atom_coords(self): return self.atom2coord
  def nao_nr(self): return self.norbs
  def atom_nelec_core(self, ia): return self.sp2charge[self.atom2sp[ia]]-self.ao_log.sp2valence[self.atom2sp[ia]]
  def ao_loc_nr(self): return self.mu2orb_s[0:self.natm]

  # More functions for convenience (see PDoS)
  def get_orb2j(self): return get_orb2j(self)
  def get_orb2m(self): return get_orb2m(self)
  def dos(self, zomegas): return system_vars_dos(self, zomegas)
  def pdos(self, zomegas): return system_vars_pdos(self, zomegas)

  def overlap_coo(self, **kvargs):   # Compute overlap matrix for the molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    return overlap_coo(self, **kvargs)

  def overlap_lil(self, **kvargs):   # Compute overlap matrix in list of lists format
    from pyscf.nao.m_overlap_lil import overlap_lil
    return overlap_lil(self, **kvargs)

  def laplace_coo(self):   # Compute matrix of Laplace brakets for the whole molecule
    from pyscf.nao.m_overlap_coo import overlap_coo
    from pyscf.nao.m_laplace_am import laplace_am
    return overlap_coo(self, funct=laplace_am)
  
  def vnucele_coo_coulomb(self, **kvargs): # Compute matrix elements of attraction by Coulomb forces from point nuclei
    from pyscf.nao.m_vnucele_coo_coulomb import vnucele_coo_coulomb
    return vnucele_coo_coulomb(self, **kvargs)

  def get_hamiltonian(self): # Returns the stored matrix elements of current hamiltonian 
    return self.hsx.spin2h4_csr

  def dipole_coo(self, **kvargs):   # Compute dipole matrix elements for the given system
    from pyscf.nao.m_dipole_coo import dipole_coo
    return dipole_coo(self, **kvargs)
  
  def overlap_check(self, tol=1e-5, **kvargs): # Works only after init_siesta_xml(), extend ?
    return overlap_check(self, tol=1e-5, **kvargs)

  def diag_check(self, atol=1e-5, rtol=1e-4, **kvargs): # Works only after init_siesta_xml(), extend ?
    return diag_check(self, atol, rtol, **kvargs)

  def vxc_lil(self, dm=None, xc_code=None, **kvargs):   # Compute exchange-correlation potentials
    from pyscf.nao.m_vxc_lil import vxc_lil
    dm1 = self.comp_dm() if dm is None else dm
    xc_code1 = self.xc_code if xc_code is None else xc_code
    return vxc_lil(self, dm1, xc_code1, deriv=1, **kvargs)

  @property
  def xc_code(self):
    if self._xc_code is None:
      return 'LDA,PZ' # just an estimate...
    else:
      return self._xc_code

  def exc(self, dm, xc_code, **kvargs):   # Compute exchange-correlation energies
    from pyscf.nao.m_exc import exc
    return exc(self, dm, xc_code, **kvargs)

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
    grid.radi_method = gauss_legendre
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
  
  def comp_dm(self):
    """ Computes the density matrix """
    from pyscf.nao.m_comp_dm import comp_dm
    return comp_dm(self.wfsx.x, self.get_occupations())

  def eval_ao(self, feval, coords, comp, shls_slice=None, non0tab=None, out=None):
    """ Computes the values of all atomic orbitals for a set of given Cartesian coordinates.
       This function should be similar to the pyscf's function eval_gto()... """
    from pyscf.nao.m_ao_eval_libnao import ao_eval_libnao as ao_eval
    assert feval=="GTOval_sph_deriv0"
    assert shls_slice is None
    assert non0tab is None
    assert comp==1
    aos = ao_eval(self.ao_log, np.zeros(3), 0, coords)
    return aos

  eval_gto = eval_ao
  eval_nao = eval_ao
 
  def dens_elec(self, coords, dm):
    """ Compute electronic density for a given density matrix and on a given set of coordinates """
    from pyscf.nao.m_dens_libnao import dens_libnao
    from pyscf.nao.m_init_dm_libnao import init_dm_libnao
    from pyscf.nao.m_init_dens_libnao import init_dens_libnao
    if not self.init_sv_libnao : raise RuntimeError('not self.init_sv_libnao')
    if init_dm_libnao(dm) is None : raise RuntimeError('init_dm_libnao(dm) is None')
    if init_dens_libnao()!=0 : raise RuntimeError('init_dens_libnao()!=0')
    return dens_libnao(coords, self.nspin)

  def comp_aos_den(self, coords):
    """ Compute the atomic orbitals for a given set of (Cartesian) coordinates. """
    from pyscf.nao.m_aos_libnao import aos_libnao
    if not self.init_sv_libnao : raise RuntimeError('not self.init_sv_libnao')
    return aos_libnao(coords, self.norbs)

  def comp_vnuc_coulomb(self, coords):
    from scipy.spatial.distance import cdist
    ncoo = coords.shape[0]
    vnuc = np.zeros(ncoo)
    for R,sp in zip(self.atom2coord, self.atom2sp):
      dd, Z = cdist(R.reshape((1,3)), coords).reshape(ncoo), self.sp2charge[sp]
      vnuc = vnuc - Z / dd 
    return vnuc
    
  def get_init_guess(self, key=None):
    """ Compute an initial guess for the density matrix. """
    from pyscf.nao.m_hf import init_guess_by_minao
    if hasattr(self, 'mol'):
      dm = init_guess_by_minao(self.mol)
    else:
      dm = self.comp_dm()  # the loaded ks orbitals will be used
      if dm.shape[0:2]==(1,1) and dm.shape[4]==1 : dm = dm.reshape((self.norbs,self.norbs))
    return dm

  def init_libnao(self, wfsx=None):
    """ Initialization of data on libnao site """
    from pyscf.nao.m_libnao import libnao
    from pyscf.nao.m_sv_chain_data import sv_chain_data
    from ctypes import POINTER, c_double, c_int64, c_int32, byref

    if wfsx is None:
        data = sv_chain_data(self)
        # (nkpoints, nspin, norbs, norbs, nreim)
        #print(' data ', sum(data))
        size_x = np.array([1, self.nspin, self.norbs, self.norbs, 1], dtype=np.int32)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True
    else:
        size_x = np.zeros(len(self.wfsx.x.shape), dtype=np.int32)
        for i, sh in enumerate(self.wfsx.x.shape):
            size_x[i] = sh

        data = sv_chain_data(self)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True

    libnao.init_aos_libnao.argtypes = (POINTER(c_int64), POINTER(c_int64))
    info = c_int64(-999)
    libnao.init_aos_libnao(c_int64(self.norbs), byref(info))
    if info.value!=0: raise RuntimeError("info!=0")
    return self

  def dens_elec_vec(self, coords, dm):
    """ Electronic density: python vectorized version """
    from m_dens_elec_vec import dens_elec_vec
    return dens_elec_vec(self, coords, dm)
  
  def get_occupations(self, telec=None, ksn2e=None, fermi_energy=None):
    """ Compute occupations of electron levels according to Fermi-Dirac distribution """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    Telec = self.hsx.telec if telec is None else telec
    ksn2E = self.wfsx.ksn2e if ksn2e is None else ksn2e
    Fermi = self.fermi_energy if fermi_energy is None else fermi_energy
    ksn2fd = fermi_dirac_occupations(Telec, ksn2E, Fermi)
    ksn2fd = (3.0-self.nspin)*ksn2fd
    return ksn2fd

  def get_eigenvalues(self):
    """ Returns mean-field eigenvalues """
    return self.wfsx.ksn2e

  def read_wfsx(self, fname, **kvargs):
    """ An occasional reading of the SIESTA's .WFSX file """
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    self.wfsx = siesta_wfsx_c(fname=fname, **kvargs)
    
    assert self.nkpoints == self.wfsx.nkpoints
    assert self.norbs == self.wfsx.norbs 
    assert self.nspin == self.wfsx.nspin
    orb2m = self.get_orb2m()
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.x[k,s,n,:,:])
    
    return self
  
  @property
  def nelectron(self):
    if self._nelectron is None:
      return tot_electrons(self)
    else:
      return self._nelectron


#
# Example of reading pySCF orbitals.
#
if __name__=="__main__":
  from pyscf import gto
  from pyscf.nao.m_system_vars import system_vars_c
  import matplotlib.pyplot as plt
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvtz') # coordinates in Angstrom!
  sv = system_vars_c(gto=mol, tol=1e-8, nr=512, rmin=1e-5)
  
  print(sv.ao_log.sp2norbs)
  print(sv.ao_log.sp2nmult)
  print(sv.ao_log.sp2rcut)
  print(sv.ao_log.sp_mu2rcut)
  print(sv.ao_log.nr)
  print(sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1])
  print(sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape)

  sp = 0
  for mu,[ff,j] in enumerate(zip(sv.ao_log.psi_log[sp], sv.ao_log.sp_mu2j[sp])):
    nc = abs(ff).max()
    if j==0 : plt.plot(sv.ao_log.rr, ff/nc, '--', label=str(mu)+' j='+str(j))
    if j>0 : plt.plot(sv.ao_log.rr, ff/nc, label=str(mu)+' j='+str(j))

  plt.legend()
  #plt.xlim(0.0, 10.0)
  #plt.show()
