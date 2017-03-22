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
    self.Lmxo,self.Lmxkb,self.basistype,self.SemiC = int(s[1]), int(s[3]), str(s[5]), (s[7].upper()!='F')
    self.ilo2l = []
    self.ilo2nsemic = []
    self.ilo2cnfigmx = []
    for ilo in range(self.Lmxo+1):
      l = f.readline()
      s = list(filter(None, re.split(r'\s+|=', l))) # L=0  Nsemic=0  Cnfigmx=2
      print(ilo, s)
      self.ilo2l.append(int(s[1]))
      self.ilo2nsemic.append(int(s[3]))
      self.ilo2cnfigmx.append(int(s[5]))

  
    f.close()
    
