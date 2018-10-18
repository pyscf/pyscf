from __future__ import print_function, division
import numpy as np

#
#
#
def fireball_import(self, **kw):
  """ Calls  """
  from pyscf.nao.ao_log import ao_log
  from pyscf.nao.m_fireball_get_cdcoeffs_dat import fireball_get_ucell_cdcoeffs_dat
  #label, cd
  self.label = label = kw['fireball'] if 'fireball' in kw else 'out'
  self.cd = cd = kw['cd'] if 'cd' in kw else '.'
  #print('self.label', self.label)
  fstdout = open(self.cd+'/'+self.label, "r")
  s = fstdout.readlines();
  fstdout.close()

  try :
    fl = [l for l in s if 'tempfe =' in l][0]
    self.telec = float(fl.split('=')[1])/11604.0/27.2114 # Electron temperature in Hartree
  except:
    raise RuntimeError("tempfe = is not found in Fireball's standard output. Stop...")

  try :
    fl = [l for l in s if 'Fermi Level = ' in l][-1]
    self.fermi_energy = float(fl.split('=')[1])/27.2114 # Fermi energy in Hartree
  except:
    raise RuntimeError("Fermi Level = is not found in Fireball's standard output. Stop...")

  # Read information about species from the standard output 
  ssll = [i for i,l in enumerate(s) if "Information for this species" in l] # starting lines in stdout
  self.nspecies  = len(ssll)
  self.sp2ion = []
  for sl in ssll:
    ion = {}
    scut = s[sl:sl+20]
    for iline,line in enumerate(scut):
      if "- Element" in line: ion["symbol"]=line.split()[0]
      elif "- Atomic energy" in line: ion["atomic_energy"]=float(line.split()[0])
      elif "- Nuclear Z" in line: ion["z"]=int(line.split()[0])
      elif "- Atomic Mass" in line: ion["mass"]=float(line.split()[0])
      elif "- Number of shells" in line and "(Pseudopotential)" not in line: 
        ion["npao"]=int(line.split()[0])
        ion["pao2j"] = list(map(int, scut[iline+1].split()))
      elif "- Number of shells (Pseudopotential)" in line:
        ion["nshpp"]=int(line.split()[0])
        ion["shpp2j"] = list(map(int, scut[iline+1].split()))
      elif "- Radial cutoffs (Pseudopotential)" in line:
        ion["rcutpp"] = float(line.split()[0])
        ion["pao2occ"] = list(map(float, scut[iline+1].split()))
        ion["pao2rcut"] = list(map(float, scut[iline+2].split()))
        ion["pao2fname"] = scut[iline+3].split()
      elif "============" in line: break
    self.sp2ion.append(ion)
  self.sp2charge = [ion["z"] for ion in self.sp2ion]
  self.sp2valence = [sum(ion["pao2occ"]) for ion in self.sp2ion]
  
  # Read the radial functions from the Fdata/basis/fname.*wf*
  for sp,ion in enumerate(self.sp2ion):
    pao2npts = []
    pao2delta = []
    pao2ff = []
    for pao,[fname,j,occp] in enumerate(zip(ion["pao2fname"],ion["pao2j"],ion["pao2occ"])):
      f = open(self.cd+'/Fdata/'+fname, "r"); ll = f.readlines(); f.close()
      #print(fname.split('/')[2].split(".")[0], ll[0].split('.')[0].strip())
      #print(fname.split('/')[2].split(".")[1], ll[0].split('.')[1].strip())
      assert fname.split('/')[2].split(".")[0]==ll[0].split('.')[0].strip()
      assert fname.split('/')[2].split(".")[1][0:3]==ll[0].split('.')[1].strip()[0:3]
      z = int(ll[1].split()[0])
      assert z == ion["z"]
      atom_name = ll[1].split()[1].strip()
      npts=int(ll[2])
      pao2npts.append(npts)
      [rc1,rc2,occ] = list(map(float, ll[3].split()))
      assert rc1>0.0
      assert rc2>0.0
      assert occ==occp
      delta = rc1/npts
      pao2delta.append(delta)
      assert int(ll[4]) == j
      ff = []
      for line in ll[5:]:
        ff += list(map(float, line.replace("D","E").split()))
      pao2ff.append(ff)
      #print(z, rc1, rc2, occ, atom_name, fname, ion["pao2rcut"][pao],delta)
    self.sp2ion[sp]["pao2npts"] = pao2npts
    self.sp2ion[sp]["pao2delta"] = pao2delta
    self.sp2ion[sp]["pao2ff"] = pao2ff
  
  # Convert to init_ao_log_ion-compatible
  for sp,ion in enumerate(self.sp2ion):
    ioncompat = ion
    data = []
    for i,[d,ff,npts,j] in enumerate(zip(ion["pao2delta"],ion["pao2ff"],ion["pao2npts"],ion["pao2j"])):
      rr = np.linspace(0.0, npts*d, npts)
      norm = (np.array(ff)**2*rr**2).sum()*(rr[1]-rr[0])
      ff = [ff[0]/np.sqrt(norm)]+[f/r/np.sqrt(norm) for f,r in zip(ff, rr) if r>0]
      data.append(np.array([rr, ff]).T)
      norm = (np.array(ff)**2*rr**2).sum()*(rr[1]-rr[0])
      #print(__name__, 'norm', norm)
    
    paos = {"npaos": ion["npao"], "delta": ion["pao2delta"], "cutoff": ion["pao2rcut"], "npts": ion["pao2npts"], "data": data, 
    "orbital": [ {"l": j, "population": occ} for occ,j in zip(ion["pao2occ"],ion["pao2j"])] }
    ioncompat["paos"] = paos
    ioncompat["valence"] = sum(ion["pao2occ"])
    self.sp2ion[sp] = ioncompat

  self.ao_log = ao_log(sp2ion=self.sp2ion, **kw)
  self.sp_mu2j = [mu2j for mu2j in self.ao_log.sp_mu2j ]
  
  #self.ao_log.view()

  f = open(self.cd+'/answer.bas', "r")
  lsa = f.readlines()
  f.close()
  self.natm=self.natoms = int(lsa[0]) # number of atoms
  atom2znuc = [int(l.split()[0]) for l in lsa[1:]] # atom --> nuclear charge
  self.atom2coord = [list(map(float, l.split()[1:])) for l in lsa[1:]] # coordinates
  self.atom2coord = np.array(self.atom2coord)
  assert self.natoms == len(atom2znuc)
  assert self.natoms == len(self.atom2coord)
  self.atom2sp = [self.sp2charge.index(znuc) for znuc in atom2znuc] # atom --> nuclear charge
  self.atom2s = np.zeros((self.natm+1), dtype=np.int64)
  for atom,sp in enumerate(self.atom2sp):
    self.atom2s[atom+1]=self.atom2s[atom]+self.ao_log.sp2norbs[sp]
  self.norbs = self.atom2s[-1]
  self.norbs_sc = self.norbs
  
  self.ucell = fireball_get_ucell_cdcoeffs_dat(self.cd)

  # atom2mu_s list of atom associated to them multipletts (radial orbitals)
  self.atom2mu_s = np.zeros((self.natm+1), dtype=np.int64)
  for atom,sp in enumerate(self.atom2sp):
    self.atom2mu_s[atom+1]=self.atom2mu_s[atom]+self.ao_log.sp2nmult[sp]

  nelec = 0.0
  for sp in self.atom2sp: nelec += self.sp2valence[sp]
  self._nelectron = nelec
  
  #print(self.atom2sp)
  #print(self.sp2charge)
  #print(self.atom2coord)
  #print(self.telec)
  #print(self.atom2s)
  #print(self.atom2mu_s)
  
  self.nspin = 1 # just a guess
  return self
