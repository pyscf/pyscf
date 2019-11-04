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

from __future__ import division, print_function
import numpy as np

#
#
#
def get_default_log_mesh_param4gto(gto, tol_in=None):
  rmin_gcs = 10.0
  rmax_gcs = -1.0
  akmx_gcs = -1.0

  tol = 1e-7 if tol_in is None else tol_in
  seen_species = [] # this is auxiliary to organize the loop over species 
  for ia in range(gto.natm):
    if gto.atom_symbol(ia) in seen_species: continue
    seen_species.append(gto.atom_symbol(ia))
    for sid in gto.atom_shell_ids(ia):
      for power,coeffs in zip(gto.bas_exp(sid), gto.bas_ctr_coeff(sid)):
        for coeff in coeffs:
          if coeff==0.0: continue
          rmin_gcs = min(rmin_gcs, np.sqrt( abs(np.log(1.0-tol)/power )))
          rmax_gcs = max(rmax_gcs, np.sqrt( abs(np.log(abs(coeff))-np.log(tol))/power ))
          akmx_gcs = max(akmx_gcs, np.sqrt( abs(np.log(abs(coeff))-np.log(tol))*4*power ))
  
  if rmin_gcs<1e-9 : print('rmin_gcs<1e-9', __name__)     # Last check 
  if rmax_gcs>1e+3 : print('rmax_gcs>1e+3', __name__)
  if akmx_gcs>1e+4 : print('akmx_gcs>1e+4', __name__)
  return 1024,rmin_gcs,rmax_gcs,akmx_gcs

#
#
#
def get_default_log_mesh_param4ion(sp2ion):
  from pyscf.nao.m_next235 import next235
  """ Determines the default (optimal) parameters for radial orbitals given on equidistant grid"""
  npts = max(max(ion["paos"]["npts"]) for ion in sp2ion)
  nr_def = next235( max(2.0*npts, 1024.0) )
  rmin_def = min(min(ion["paos"]["delta"]) for ion in sp2ion)
  rmax_def = 2.3*max(max(ion["paos"]["cutoff"]) for ion in sp2ion)
  kmax_def = 1.0/rmin_def/np.pi
  return nr_def,rmin_def,rmax_def,kmax_def

#
#
#
def get_default_log_mesh_param4gpaw(sp2dic):
  """ Determines the default (optimal) parameters for radial orbitals given on equidistant grid"""
  sp2key = sp2dic.keys()
  nr_def = 1024
  rmin_def = 1.0e100
  rmax_grid = -1.0e100
  for key in sp2key: 
    rmin_def = min(rmin_def, sp2dic[key].basis.rgd.r_g[1])
    rmax_grid = max(rmax_grid, sp2dic[key].basis.rgd.r_g[-1])
  rmax_def = 2.3*rmax_grid
  kmax_def = 1.0/rmin_def/np.pi
  return nr_def,rmin_def,rmax_def,kmax_def

#    sp2dic = setups.setups
#    print('dir(r_g) ', dir(sp2dic[sp2id[1]].basis.rgd.r_g))
#    print(sp2dic[sp2id[0]].basis.rgd.r_g.size)
#    print(sp2dic[sp2id[1]].basis.rgd.r_g.size)
        

#
#
#
def funct_log_mesh(nr, rmin, rmax, kmax=None):
  """
  Initializes log grid in real and reciprocal (momentum) spaces.
  These grids are used in James Talman's subroutines. 
  """
  assert(type(nr)==int and nr>2)
  rhomin=np.log(rmin)
  rhomax=np.log(rmax)
  kmax = 1.0/rmin/np.pi if kmax is None else kmax
  kapmin=np.log(kmax)-rhomax+rhomin
  
  rr=np.array(np.exp( np.linspace(rhomin, rhomax, nr)) )
  pp=np.array(rr*(np.exp(kapmin)/rr[0]))

  return rr, pp

#
#
#
class log_mesh():
  ''' Constructor of the log grid used with NAOs.'''
  def __init__(self, **kw):
    
    if 'gto' in kw:                 self.init_log_mesh_gto(**kw)
    elif 'sp2ion' in kw:            self.init_log_mesh_ion(**kw)
    elif 'setups' in kw:            self.init_log_mesh_gpaw(**kw)
    elif 'rr' in kw and 'pp' in kw: self.init_log_mesh(**kw)
    elif 'xyz_list' in kw: pass
    elif 'ao_log' in kw: pass
    else:
      print(kw.keys())
      raise RuntimeError('unknown init method')

  
  def init_log_mesh_gto(self, **kw):
    """ Initialize an optimal logarithmic mesh based on Gaussian orbitals from pySCF"""
    #self.gto = gto cannot copy GTO object here... because python3 + deepcopy in m_ao_log_hartree fails
    gto = kw['gto']
    self.rcut_tol = kw['rcut_tol'] if 'rcut_tol' in kw else 1e-7
    nr_def,rmin_def,rmax_def,kmax_def = get_default_log_mesh_param4gto(gto, self.rcut_tol)
    self.nr   = kw['nr'] if "nr" in kw else nr_def
    self.rmin = kw['rmin'] if "rmin" in kw else rmin_def
    self.rmax = kw['rmax'] if "rmax" in kw else rmax_def
    self.kmax = kw['kmax'] if "kmax" in kw else kmax_def
    assert(self.rmin>0.0); assert(self.kmax>0.0); assert(self.nr>2); assert(self.rmax>self.rmin);
    self.rr,self.pp = funct_log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    return self
    
  
  def init_log_mesh_ion(self, **kw):
    """ Initialize an optimal logarithmic mesh based on information from SIESTA ion files"""
    sp2ion = kw['sp2ion']
    self.sp2ion = sp2ion
    nr_def,rmin_def,rmax_def,kmax_def = get_default_log_mesh_param4ion(sp2ion)
    self.nr   = kw['nr'] if "nr" in kw else nr_def
    self.rmin = kw['rmin'] if "rmin" in kw else rmin_def
    self.rmax = kw['rmax'] if "rmax" in kw else rmax_def
    self.kmax = kw['kmax'] if "kmax" in kw else kmax_def
    assert(self.rmin>0.0); assert(self.kmax>0.0); assert(self.nr>2); assert(self.rmax>self.rmin);
    self.rr,self.pp = funct_log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    return self

  def init_log_mesh_gpaw(self, **kw):
    """ This initializes an optimal logarithmic mesh based on setups from GPAW"""

    #self.setups = setups same problem than in m_ao_log
    setups = kw['setups']
    nr_def,rmin_def,rmax_def,kmax_def = get_default_log_mesh_param4gpaw(setups.setups)
    self.nr   = kw['nr'] if "nr" in kw else nr_def
    self.rmin = kw['rmin'] if "rmin" in kw else rmin_def
    self.rmax = kw['rmax'] if "rmax" in kw else rmax_def
    self.kmax = kw['kmax'] if "kmax" in kw else kmax_def
    assert(self.rmin>0.0); assert(self.kmax>0.0); assert(self.nr>2); assert(self.rmax>self.rmin);
    self.rr,self.pp = funct_log_mesh(self.nr, self.rmin, self.rmax, self.kmax)
    return self

  def init_log_mesh(self, **kw):
    """ Taking over the given grid rr and pp"""
    rr, pp = kw['rr'], kw['pp']
    assert(len(pp)==len(rr))
    self.rr,self.pp = rr,pp
    self.nr = len(rr)
    self.rmin = rr[0]
    self.rmax = rr[-1]
    self.kmax = pp[-1]
    return self
