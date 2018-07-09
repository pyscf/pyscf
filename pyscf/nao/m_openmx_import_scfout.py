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
import numpy as np
from ctypes import c_char, c_int, sizeof, c_double

# Maybe even not necessary after all...
#from pyscf.lib import misc
#libopenmx = misc.load_library("libopenmx")
#libopenmx.openmx_unpack.argtypes = (c_int, c_char_p[])

#
#
#
def openmx_import_scfout(self, label, cd):
  """ Calls libopenmx to get the data and interpret it then """
  import struct
  from pyscf.nao.m_openmx_mat import openmx_mat_c
  
  self.label=label
  self.cd = cd
  
  fname = cd+'/'+label+'.scfout'

  with open(fname, mode='rb') as f: # b is important -> binary
    header = f.read(6*sizeof(c_int))
    (natoms,SpinP_switch,Catomnum,Latomnum,Ratomnum,TCpyCell) = struct.unpack("@6i", header)
    assert natoms>0
    assert SpinP_switch>-1
    #print(natoms,SpinP_switch,Catomnum,Latomnum,Ratomnum,TCpyCell)
    atv = np.fromfile(f, count=(TCpyCell+1)*4).reshape((TCpyCell+1,4))
    #print(atv)
    atv_ijk = np.fromfile(f, count=(TCpyCell+1)*4, dtype=c_int).reshape((TCpyCell+1,4))
    #print(atv_ijk)
    Total_NumOrbs = np.ones(natoms+1, dtype=c_int)
    Total_NumOrbs[1:] = np.fromfile(f, count=natoms, dtype=c_int)        
    #print('Total_NumOrbs')
    #print(Total_NumOrbs)
    FNAN = np.zeros(natoms+1, dtype=c_int)
    FNAN[1:] = np.fromfile(f, count=natoms, dtype=c_int)
    #print('FNAN')
    #print(FNAN, max(FNAN))
    natn = np.zeros((natoms+1,max(FNAN)+1), dtype=c_int)
    ncn  = np.zeros((natoms+1,max(FNAN)+1), dtype=c_int)
    for iatom,count in enumerate(FNAN[1:]): natn[iatom+1,:] = np.fromfile(f, count=count+1, dtype=c_int)
    for iatom,count in enumerate(FNAN[1:]): ncn[iatom+1,:] = np.fromfile(f, count=count+1, dtype=c_int)
    #print('natn ')
    #print(natn)
    #print('ncn  ')
    #print(ncn)
    tv  = np.zeros((4,4))
    for i in range(3): tv[i+1,:] = np.fromfile(f, count=4)
    rtv  = np.zeros((4,4))
    for i in range(3): rtv[i+1,:] = np.fromfile(f, count=4)
    #print('tv ', tv)
    #print('rtv ', rtv)
    Gxyz = np.fromfile(f, count=natoms*4).reshape((natoms,4))
    #print('Gxyz ')
    #print(Gxyz)

    omm = openmx_mat_c(natoms, Total_NumOrbs, FNAN, natn)

    Hks = np.zeros([SpinP_switch+1]+omm.get_dims())
    for spin in range(SpinP_switch+1): omm.fromfile(f, out=Hks[spin])
    
    if SpinP_switch==3:
      iHks = np.zeros([3]+omm.get_dims())
      for spin in range(3): omm.fromfile(f, out=iHks[spin])
    
    OLP = omm.fromfile(f)
    OLPx = omm.fromfile(f)
    OLPy = omm.fromfile(f)
    OLPz = omm.fromfile(f)

    DM = np.zeros([SpinP_switch+1]+omm.get_dims())
    for spin in range(SpinP_switch+1): omm.fromfile(f,out=DM[spin])

    solver = struct.unpack("@i", f.read(1*sizeof(c_int)))[0]    
    dipole_moment_core = np.zeros(3)
    dipole_moment_background = np.zeros(3)
    ChemP,E_Temp,\
      dipole_moment_core[0],dipole_moment_core[1],dipole_moment_core[2],\
      dipole_moment_background[0],dipole_moment_background[1],dipole_moment_background[2],\
      Valence_Electrons,Total_SpinS = struct.unpack("@10d", f.read(10*sizeof(c_double)))

    #print(solver)
    #print(ChemP,E_Temp)
    #print(dipole_moment_core)
    #print(dipole_moment_background)
    #print(Valence_Electrons)
    #print(Total_SpinS)
    nlines_input = struct.unpack("@i", f.read(1*sizeof(c_int)))[0]
    input_file = []
    for line in range(nlines_input):
      input_file.append(str(struct.unpack("@256s", f.read(256*sizeof(c_char)))[0]))

    self.natm=self.natoms=natoms
    self.nspin = SpinP_switch+1
    self.ucell = tv[1:4,1:4]
    self.atom2coord = Gxyz[:,0:3]
    self.atom2s = np.zeros((self.natm+1), dtype=np.int)
    for atom,norb in enumerate(Total_NumOrbs[1:]): self.atom2s[atom+1]=self.atom2s[atom]+norb
    self.norbs = self.atom2s[-1]
    self.nkpoints = 1
    #print(self.atom2s)
    #print(self.norbs)
    #print(self.nspin)
  return self
