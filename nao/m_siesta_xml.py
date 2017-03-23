from pyscf.nao.m_color import bcolors as bc
from pyscf.nao.m_siesta_ev2ha import siesta_ev2ha
import xml.etree.ElementTree as ET
import re
import numpy

pref = "{http://www.xml-cml.org/schema}"

def siesta_xml(label="siesta"):
  fname = label+".xml"
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

  atom2xyz = [[atom.attrib["x3"],atom.attrib["y3"],atom.attrib["z3"]] for atom in atoms]
  atom2coord = numpy.array(atom2xyz, dtype='double', order='F')

  eigvals=fin.find(pref+"propertyList[@title='Eigenvalues']")
  print(bc.RED+"eigvals"+bc.ENDC)
  for child in eigvals:
    print(len(child), child.tag, child.attrib)
  print(bc.RED+"END of eigvals]"+bc.ENDC)

  Fermi_Energy=float(eigvals.find(pref+"property[@dictRef='siesta:E_Fermi']")[0].text)*siesta_ev2ha
  print(' Fermi_Energy ', Fermi_Energy)
  
  nkp=int(eigvals.find(pref+"property[@dictRef='siesta:nkpoints']")[0].text)
  print(' nkp          ', nkp)
  kp2coow = numpy.empty((4,nkp), dtype='double', order='F')
  
  kpt_band=eigvals.find(pref+"propertyList[@dictRef='siesta:kpt_band']")
  print(' kpt_band ', len(kpt_band))
  for c in kpt_band:
    print(len(c), c.attrib, c.tag)

  kpoints = kpt_band.find(pref+'kpoint') 
  print(kpoints.attrib)  
  return atom2coord, atom2sp, sp2elem
