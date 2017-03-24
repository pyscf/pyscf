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
