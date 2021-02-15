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

def sv_chain_data(sv):
  """ This subroutine creates a buffer of information to communicate the system variables to libnao."""
  from numpy import zeros, concatenate as conc

  aos,sv = sv.ao_log, sv  
  nr,nsp,nmt,nrt = aos.nr,aos.nspecies, sum(aos.sp2nmult),aos.nr*sum(aos.sp2nmult)
  nat,na1,tna,nms = sv.natoms,sv.natoms+1,3*sv.natoms,sum(aos.sp2nmult)+aos.nspecies
  ndat = 200 + 2*nr + 4*nsp + 2*nmt + nrt + nms + 3*3 + nat + 2*na1 + tna + 4*nsp
    
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
  dat[i] = sv.natoms; i+=1
  dat[i] = sv.norbs; i+=1
  dat[i] = sv.norbs_sc; i+=1
  dat[i] = sv.nspin; i+=1
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
  dat[i] = s+1; # this is a terminator to simplify operation
  return dat
