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

from pyscf.nao.m_color import color as bc
from pyscf.nao.m_siesta_ev2ha import siesta_ev2ha
from pyscf.nao.m_siesta_ang2bohr import siesta_ang2bohr, siesta_bohr2ang
import xml.etree.ElementTree as ET
import re
import numpy

pref = "{http://www.xml-cml.org/schema}"

def siesta_xml(fname="siesta.xml"):
  try :
    tree = ET.parse(fname)
  except:
    raise SystemError(fname+" cannot be parsed: calculation did not finish?")

  roo = tree.getroot()
  fin=roo.find(pref+"module[@title='Finalization']")
  mol=fin.find(pref+"molecule")
  coo=mol.find(pref+"atomArray")
  atoms = coo.findall(pref+"atom")
  natoms = len(atoms)
  atom2ref = [atom.attrib["ref"] for atom in atoms]
  atom2sp = [int(re.findall("\d+", ref)[0])-1 for ref in atom2ref]
  nspecies = len(set(atom2ref))
  atom2elem = [atom.attrib["elementType"] for atom in atoms]

  sp2elem = [None]*nspecies
  for a in range(len(atom2elem)): sp2elem[atom2sp[a]] = atom2elem[a]

  x3 = list(map(float, [atom.attrib["x3"] for atom in atoms]))
  y3 = list(map(float, [atom.attrib["y3"] for atom in atoms]))
  z3 = list(map(float, [atom.attrib["z3"] for atom in atoms]))
  atom2coord = numpy.empty((natoms,3), dtype='double')
  for a in range(natoms): atom2coord[a,0:3] = [x3[a],y3[a],z3[a]]
  #atom2coord = atom2coord*siesta_ang2bohr
  atom2coord = atom2coord/siesta_bohr2ang
  
  eigvals=fin.find(pref+"propertyList[@title='Eigenvalues']")

  lat=fin.find(pref+"lattice[@dictRef='siesta:ucell']")
  ucell = numpy.empty((3,3), dtype='double')
  for i in range(len(lat)): ucell[i,0:3] = list(map(float, filter(None, re.split(r'\s+|=', lat[i].text))))
  ucell = ucell * siesta_ang2bohr
  #print(len(lat), lat.attrib)
  #print(ucell)
  
  fermi_energy=float(eigvals.find(pref+"property[@dictRef='siesta:E_Fermi']")[0].text)*siesta_ev2ha
  
  nkp=int(eigvals.find(pref+"property[@dictRef='siesta:nkpoints']")[0].text)
  k2xyzw = numpy.empty((nkp,4), dtype='double')
    
  kpt_band=eigvals.findall(pref+"propertyList[@dictRef='siesta:kpt_band']")
  nspin=len(kpt_band)
  norbs = int(kpt_band[0].find(pref+"property[@dictRef='siesta:eigenenergies']")[0].attrib['size'])
  ksn2e = numpy.empty((nkp,nspin,norbs), dtype='double')

  for s in range(nspin):
    eigv = kpt_band[s].findall(pref+"property[@dictRef='siesta:eigenenergies']")
    kpnt = kpt_band[s].findall(pref+"kpoint")
    for k in range(nkp):
      ksn2e[k,s,0:norbs] = list(map(lambda x : siesta_ev2ha*float(x), filter(None, re.split(r'\s+|=', eigv[k][0].text))))
      k2xyzw[k,0:3] = list(map(float, filter(None, re.split(r'\s+|=', kpnt[k].attrib['coords']))))
      k2xyzw[k,3] = float(kpnt[k].attrib['weight'])

  d = dict({'fermi_energy':fermi_energy, 
            'atom2coord':atom2coord, 
            'atom2sp':atom2sp, 
            'sp2elem':sp2elem, 
            'k2xyzw':k2xyzw, 
            'ksn2e':ksn2e, 
            'ucell':ucell})
  return d
