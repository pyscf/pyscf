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

import numpy
import sys
import re
from io import BytesIO

#
#
#
class siesta_ion_c():

  def __init__(self, splabel):
    f = open(splabel + '.ion', 'r')
    f.seek(0)
    if( f.readline().strip()!="<preamble>"): raise SystemError('!<preamble>')
    dummy = f.readline() # !! <basis_specs>
    dummy = f.readline() # !! ============
    dummy = f.readline() # !! ============
    s = list(filter(None, re.split(r'\s+|=', f.readline()))) # O                    Z=   8    Mass=  16.000        Charge= 0.17977+309
    self.pp_name,self.atomic_number,self.atomic_mass = str(s[0]),int(s[2]),float(s[4])
    try: 
      self.charge      = float(s[6])
    except:
      self.charge        = 0.17977e+309
    s = list(filter(None, re.split(r'\s+|=', f.readline()))) # Lmxo=1 Lmxkb= 3    BasisType=split      Semic=F
    self.lmxo,self.lmxkb,self.basistype,self.semic = int(s[1]), int(s[3]), str(s[5]), (s[7].upper()!='F')
    self.l2rf_dat = []
#    di = dict()
#    di = {'L':1, 'Lkbmx':3}
#    print(di)
    
    for ilo in range(self.lmxo+1):
      s2 = list(filter(None, re.split(r'\s+|=', f.readline()))) # L=0  Nsemic=0  Cnfigmx=2
      lvalue, nsemic, cnfigmx = int(s2[1]),int(s2[3]),int(s2[5])
      lst2 = []
      for isemic in range(nsemic+1):
        s1 = list(filter(None, re.split(r'\s+|=', f.readline()))) #        n=1  nzeta=2  polorb=1
        n,nzeta,polorb=int(s2[1]),int(s2[3]),int(s2[5])
        print(isemic, ilo)
        di1 = dict()
        for idat in range(8):
          s0 = list(filter(None, re.split(r'\s+|=|:', f.readline())))
          di1[s0[0].lower()] = map(float, s0[1:])
          #print(ilo, isemic, idat, s2, s1, di1)
          if(s0[0].upper()=='LAMBDAS'): break
        lst2.append([n,nzeta,polorb, di1])
      self.l2rf_dat.append([lvalue,nsemic,cnfigmx, lst2])

    print(self.l2rf_dat)
    f.close()

