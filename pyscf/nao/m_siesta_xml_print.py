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
import xml.etree.ElementTree as ET
from pyscf.nao.m_siesta_xml import pref

def siesta_xml_print(label="siesta"):
  fname = label+".xml"
  try :
    tree = ET.parse(fname)
  except:
    raise SystemError(fname+" cannot be parsed: calculation did not finish?")

  roo = tree.getroot()
  fin=roo.find(pref+"module[@title='Finalization']")
  mol=fin.find(pref+"molecule")
  coo=mol.find(pref+"atomArray")
  print(bc.RED+"children of roo[t]"+bc.ENDC)
  for child in roo:
    print(child.tag, child.attrib, child.text, len(child))

  print(bc.RED+"children of fin"+bc.ENDC)
  for child in fin:
    print(len(child), child.tag, child.attrib, child.text, len(child))

  print(bc.RED+"children of mol"+bc.ENDC)
  for child in mol:
    print(len(child), child.tag, child.attrib, child.text)
  
  print(bc.RED+"children of coo"+bc.ENDC+" (only attrib)")
  for child in coo:
    print(len(child), child.attrib)

  return 0
