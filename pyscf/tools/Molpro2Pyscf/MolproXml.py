# TODO: By PySCF-1.5 release
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
# 1. code style
#   * Indent space: 3 -> 4
#   * Function/method should be all lowercase
#   * Line wrap around 80 columns
#   * Use either double quote or single quote, not mix
# 
# 2. Conventions required by PySCF
#   * Use proper logger function for debug messages
#   * Add attribute ._keys for sanity check
#   * Class attributes should be all lowercase
#   * Use .verbose to control print level
# 
# 3. Use proper functions provided by PySCF
#

#  This file is adapted with permission from the wmme program of Gerald Knizia.
#  See http://sites.psu.edu/knizia/software/
#====================================================



"""Functions to read data from Molpro XML files generated via
   ...
   {put,xml,FileName; keepspherical; nosort}
(after a orbital-generating command, e.g., "{df-rks,pbe}")
"""
import wmme
import numpy as np
import xml.etree.ElementTree as ET

class FOrbitalInfo(object):
   def __init__(self, Coeffs, fEnergy, fOcc, iSym, iOrbInSym, Basis):
      self.Coeffs = np.array(Coeffs)
      assert(isinstance(Basis, wmme.FBasisSet))
      self.Basis = Basis
      self.fOcc = fOcc
      self.fEnergy = fEnergy
      self.iSym = iSym
      self.iOrbInSym = iOrbInSym
   @property
   def Desc(self):
      return "%i.%i [E=%.4f O=%.4f]" % (self.iOrbInSym, self.iSym, self.fEnergy, self.fOcc)
   @property
   def Name(self):
      return "%i.%i" % (self.iOrbInSym, self.iSym)


class FMolproXmlData(object):
   def __init__(self, Atoms, OrbBasis, Orbitals, FileName=None, Variables=None):
      self.Atoms = Atoms
      self.OrbBasis = OrbBasis
      self.Orbitals = Orbitals
      # make an orbital matrix by concatenating the individual
      # orbital coefficient arrays.
      nBf = self.OrbBasis.nFn
      nOrb = len(self.Orbitals)
      self.Variables = Variables
      if self.Variables is not None:
         self.ToAng = float(self.Variables["_TOANG"])

      if nOrb == 0:
         Orbs = np.zeros((nBf, nOrb))
      else:
         nBfOrb = len(Orbitals[0].Coeffs)
         Orbs = np.zeros((nBfOrb, nOrb))
         for i in range(nOrb):
            Orbs[:,i] = Orbitals[i].Coeffs
         # check if we need to convert from cartesian to spherical.
         nBfCa = self.OrbBasis.nFnCa
         if Orbs.shape[0] == nBf:
            # nope--already spherical.
            pass
         elif Orbs.shape[0] == nBfCa:
            # yes (old molpro version)
            ls = OrbBasis.GetAngmomList()
            Orbs = _Vec_Ca2Sh(Orbs, ls)
         else:
            raise Exception("MolproXml import: Import orbital matrix neither consistent with spherical nor cartesian basis functions.")

      self.Orbs = Orbs
      self.FileName = FileName

def remove_namespace(doc, namespace):
   """Remove namespace in the passed document in place."""
   ns = u'{%s}' % namespace
   nsl = len(ns)
   for elem in doc.getiterator():
      if elem.tag.startswith(ns):
         elem.tag = elem.tag[nsl:]
def remove_all_namespaces(doc):
   def remove_namespace1(s):
      if s.startswith("{"):
         return s[s.find("}")+1:]
      else:
         return s
   for elem in doc.getiterator():
      elem.tag = remove_namespace1(elem.tag)
      elem.attrib = dict([(remove_namespace1(k),v) for (k,v) in elem.attrib.items()])

def _ReadNodeArray(Node):
   return np.array(Node.text.split()).astype(float)


def _ReadAtoms(AtomNode, XmlToAng):
   Elements = []
   Positions = []
   IdToAtom = {}  # maps XML tag of atom to the index in the atom array.
   for XmlAtom in AtomNode:
      iAtom = len(Elements)
      Id = XmlAtom.attrib["id"] # it's a string
      Element = XmlAtom.attrib["elementType"]
      # input is in angstroems. (not AUs)
      x = float(XmlAtom.attrib['x3'])
      y = float(XmlAtom.attrib['y3'])
      z = float(XmlAtom.attrib['z3'])

      IdToAtom[Id] = iAtom
      Elements.append(Element)
      # convert output to bohr units.
      Positions.append((1./XmlToAng) * np.array([x,y,z]))
      #Positions.append(np.array([x,y,z]))
   return wmme.FAtomSet(np.array(Positions).T, Elements), IdToAtom

def _ReadBasisSet(BasisNode, Atoms, IdToAtom):
   IdToBf = {}
   BasisShells = []
   for ShellNode in BasisNode.findall("basisGroup"):
      #print "shell: %s -> %s " % (ShellNode.tag, ShellNode.attrib)
      nCo = int(ShellNode.attrib['contractions'])
      nExp = int(ShellNode.attrib['primitives'])
      MinL = int(ShellNode.attrib['minL'])
      MaxL = int(ShellNode.attrib['maxL'])
      if (MinL != MaxL):
         raise Exception("Basis sets with MinL != MaxL are not supported.")
      l = MinL
      AngularType = ShellNode.attrib['angular']
      if AngularType != "spherical":
         raise Exception("Only spherical harmonic basis sets are supported.")
      Id = ShellNode.attrib['id']
      Exps = _ReadNodeArray(ShellNode.find('basisExponents'))
      Cos = []
      for CoNode in ShellNode.findall('basisContraction'):
         Cos.append(_ReadNodeArray(CoNode))
      if (len(Exps) != nExp):
         raise Exception("Inconsistent basis declaration: nExp != len(Exp).")
      if (len(Cos) != nCo):
         raise Exception("Inconsistent basis declaration: nCo != len(Cos).")
      Bf = wmme.FBasisShell(l, np.array(Exps), np.array(Cos).T)
      BasisShells.append(Bf)
      IdToBf[Id] = Bf
      #print "Id = '%s' %s" % (Id, Bf)
   Associations = {}
   for AssocNode in BasisNode.findall("association"):
      def ReadLinks(Node, Type):
         #print "Node.attrib: %s" % Node.attrib
         xlink = Node.attrib["href"]
         s = Type + "["
         xlink = xlink[xlink.find(s)+len(s):xlink.rfind(']')]
         xlink = xlink.replace("@id=","").replace("'","").replace(" or ", " ")
         return xlink
      ShellLinks = ReadLinks(AssocNode.find('bases'), "basisGroup").split()
      AtomLinks = ReadLinks(AssocNode.find('atoms'), "atom").split()
      #print "atoms: %s\n   ->  shells: %s" % (AtomLinks, ShellLinks)
      for AtId in AtomLinks:
         for ShellId in ShellLinks:
            #print "   link: %s -> %s" % (AtId, ShellId)
            L = Associations.get(AtId, [])
            L.append(ShellId)
            Associations[AtId] = L

   #print "Associations: %s" % Associations
   AtomToId = dict((v,k) for (k,v) in IdToAtom.items())
   def GetAtomAssociations(AtomId):
      # Molpro2012 apparently sometimes linked to atom ids with names like "1"/"2" etc
      # even if the atom ids were actually defined as "a1"/"a2" etc.
      if AtomId in Associations:
         return Associations[AtomId]
      elif AtomId.startswith("a"):
         return Associations[AtomId[1:]]
      else:
         raise Exception("something went wrong in the association of basis sets and atoms.")
   # assemble the basis set in AtomSet order.
   Shells1 = []
   for (iAt,At) in enumerate(Atoms):
      #ShellIds = Associations[AtomToId[iAt]]
      ShellIds = GetAtomAssociations(AtomToId[iAt])
      for ShellId in ShellIds:
         Shells1.append(wmme.FBasisShell1(At, IdToBf[ShellId]))
   BasisSet = wmme.FBasisSet(Shells1, Atoms)
   if BasisSet.nFn != int(BasisNode.attrib["length"]):
      raise Exception("Expected size of basis and actual size of basis do not match.")
   return BasisSet

def _ReadOrbitals(OrbitalsNode, Atoms, OrbBasis, SkipVirtual):
   nBf = OrbBasis.nFn
   Orbitals = []
   count =0         #ELVIRA
   print "# orb       iSym    iOrbInSym"      #ELVIRA 

   nOrbInSym = np.array(8*[0])
   for OrbNode in OrbitalsNode.findall("orbital"):
      fOcc = float(OrbNode.attrib["occupation"])
      if SkipVirtual and fOcc == 0.0:
         continue
      fEnergy = float(OrbNode.attrib["energy"])
      iSym = int(OrbNode.attrib["symmetryID"])
      iOrbInSym = nOrbInSym[iSym-1] + 1  # 1-based.
      nOrbInSym[iSym-1] += 1
      Coeffs = _ReadNodeArray(OrbNode)
      #if len(Coeffs) != nBf:
         #raise Exception("Number of orbital coefficients differs from number of basis functions.")
      Orbitals.append(FOrbitalInfo(Coeffs, fEnergy, fOcc, iSym, iOrbInSym, OrbBasis))
      if fOcc != 0.0:
       print count , "       ", iSym, "          ", iOrbInSym      #ELVIRA 
      count +=1  #ElVIRA
   return Orbitals


def ReadMolproXml(FileName,SkipVirtual=False):
   XmlTree = ET.parse(FileName)
   Root = XmlTree.getroot()
   remove_all_namespaces(Root)
   Molecule = list(Root)[0]

   VariablesNode = Molecule.find("variables")
   if VariablesNode is None:
      Variables = None
   else:
      Variables = {}
      for VariableNode in VariablesNode:
         #print VariableNode.tag, VariableNode.attrib
         L = []
         for v in VariableNode.findall("value"):
            L.append(v.text)
         if len(L) == 1:
            L = L[0]
         Variables[VariableNode.attrib['name']] = L
   if Variables is not None and "_TOANG" in Variables:
      XmlToAng = float(Variables["_TOANG"])
   else:
      XmlToAng = wmme.ToAng


   # read atom declarations
   AtomArrayNode = Molecule.find("atomArray")
   if AtomArrayNode is None:
      # atomArray was put into cml:molecule at some point in time.
      CmlMoleculeNode = Molecule.find("molecule")
      AtomArrayNode = CmlMoleculeNode.find("atomArray")
   Atoms, IdToAtom = _ReadAtoms(AtomArrayNode, XmlToAng)

   # find the XML node describing the main orbital basis
   OrbBasisNode = None
   for Node in Molecule.findall("basisSet"):
      if Node.attrib["id"] == "ORBITAL":
         OrbBasisNode = Node
   assert(OrbBasisNode is not None)
   OrbBasis = _ReadBasisSet(OrbBasisNode, Atoms, IdToAtom)

   Orbitals = _ReadOrbitals(Molecule.find("orbitals"), Atoms, OrbBasis, SkipVirtual)
   #print "Number of orbitals read: %s" % len(Orbitals)
   return FMolproXmlData(Atoms, OrbBasis, Orbitals, FileName=FileName, Variables=Variables)


def _nCartY(l):
   return ((l+1)*(l+2))/2

def _Vec_CaMolden2CaMolpro(Orbs, ls):
   # transform from Molden cartesian component order to Molpro cartesian component order
   I = []
   iOffset = 0
   for l in ls:
      I0 = list(range(_nCartY(l)))
      if l == 3:
         # in IrImportTrafo.cpp:
         #  double c3 = pOrb[4]; double c4 = pOrb[5]; double c5 = pOrb[3]; double c6 = pOrb[8]; double c7 = pOrb[6]; double c8 = pOrb[7];
         #  pOrb[3] = c3; pOrb[4] = c4; pOrb[5] = c5; pOrb[6] = c6; pOrb[7] = c7; pOrb[8] = c8;
         # all others have equal cartesian component order.
         I0[3] = 4
         I0[4] = 5
         I0[5] = 3
         I0[6] = 8
         I0[7] = 6
         I0[8] = 7

      I += [(o + iOffset) for o in I0]
      iOffset += len(I0)
   I = np.array(I)
   return Orbs[I,:]

def _Vec_Ca2Sh(Ca, ls):
   # transformation orbital matrix from Molpro's obscure cartesian format to Molpro's equally obscure spherical format.
   # Input: Matrix nCartAo x nOrb
   # Returns: Matix nShAo x nOrb
   # note: Ported from IrImportTrafo.cpp.
   assert(len(Ca.shape) == 2)
   nCa = 0
   nSh = 0
   for l in ls:
      nCa += _nCartY(l)
      nSh += 2*l + 1
   if nCa != Ca.shape[0]:
      raise Exception("Expected first dimension of orbital matrix to have nCartAo dimension (which is %i), but it has %i rows" % (nCa, Ca.shape[0]))
   # allocate output matrix.
   Sh = np.zeros((nSh, Ca.shape[1]))

   sd0 = 5.e-01
   sd1 = 1.7320508075688772
   sd2 = 8.660254037844386e-01
   sd3 = 6.1237243569579458e-01
   sd4 = 2.4494897427831779
   sd5 = 1.5
   sd6 = 7.9056941504209488e-01
   sd7 = 2.3717082451262845
   sd8 = 3.872983346207417
   sd9 = 1.9364916731037085
   sda = 3.75e-01
   sdb = 7.5e-01
   sdc = 3.
   sdd = 1.1180339887498947
   sde = 6.7082039324993676
   sdf = 3.1622776601683791
   sd10 = 7.3950997288745202e-01
   sd11 = 4.4370598373247123
   sd12 = 5.5901699437494734e-01
   sd13 = 3.3541019662496838
   sd14 = 2.9580398915498081
   sd15 = 2.0916500663351889
   sd16 = 6.2749501990055672
   sd17 = 4.8412291827592718e-01
   sd18 = 9.6824583655185437e-01
   sd19 = 5.809475019311126
   sd1a = 2.5617376914898995
   sd1b = 5.1234753829797981
   sd1c = 5.2291251658379723e-01
   sd1d = 1.0458250331675947
   sd1e = 4.1833001326703778
   sd1f = 1.5687375497513918
   sd20 = 1.2549900398011134e+01
   sd21 = 8.8741196746494246
   sd22 = 2.2185299186623562
   sd23 = 1.3311179511974137e+01
   sd24 = 3.5078038001005702
   sd25 = 7.0156076002011396
   sd26 = 7.0156076002011403e-01
   sd27 = 1.8750000000000002
   sd28 = 3.7500000000000004
   sd29 = 5.
   sd2a = 1.0246950765959596e+01
   sd2b = 6.7169328938139616e-01
   sd2c = 1.0075399340720942e+01
   sd2d = 9.0571104663683977e-01
   sd2e = 1.8114220932736795
   sd2f = 1.4491376746189438e+01
   sd30 = 2.3268138086232857
   sd31 = 2.3268138086232856e+01
   sd32 = 1.1634069043116428e+01
   sd33 = 4.9607837082461076e-01
   sd34 = 2.4803918541230536
   sd35 = 4.9607837082461073
   sd36 = 2.9764702249476645e+01
   sd37 = 4.5285552331841988e-01
   sd38 = 7.245688373094719
   sd39 = 4.0301597362883772
   sd3a = 1.3433865787627923e+01
   sd3b = 2.7171331399105201
   sd3c = 5.434266279821041
   sd3d = 8.1513994197315611
   sd3e = 2.1737065119284161e+01
   sd3f = 1.984313483298443
   sd40 = 1.9843134832984429e+01
   sd41 = 3.125e-01
   sd42 = 9.375e-01
   sd43 = 5.625
   sd44 = 1.125e+01
   sd45 = 7.4999999999999991
   sd46 = 2.8641098093474002
   sd47 = 5.7282196186948005
   sd48 = 1.1456439237389599e+01
   sd49 = 4.5825756949558407

   iSh = 0
   iCa = 0
   for l in ls:
      if l == 0:
         Sh[iSh,:] = 1.0*Ca[iCa,:]
         iCa += 1
         iSh += 1
      elif l == 1:
         Sh[iSh+0,:] = (1.0*Ca[iCa+0,:])
         Sh[iSh+1,:] = (1.0*Ca[iCa+1,:])
         Sh[iSh+2,:] = (1.0*Ca[iCa+2,:])
         iCa += 3
         iSh += 3
      elif l == 2:
         Sh[iSh+0,:] = -(0.6666666666666666*Ca[iCa+0,:])*sd0 - (0.6666666666666666*Ca[iCa+1,:])*sd0 + (0.6666666666666666*Ca[iCa+2,:])
         Sh[iSh+1,:] = (0.5773502691896258*Ca[iCa+3,:])*sd1
         Sh[iSh+2,:] = (0.5773502691896258*Ca[iCa+4,:])*sd1
         Sh[iSh+3,:] = (0.6666666666666666*Ca[iCa+0,:])*sd2 - (0.6666666666666666*Ca[iCa+1,:])*sd2
         Sh[iSh+4,:] = (0.5773502691896258*Ca[iCa+5,:])*sd1
         iCa += 6
         iSh += 5
      elif l == 3:
         Sh[iSh+0,:] = -(0.29814239699997197*Ca[iCa+5,:])*sd3 + (0.29814239699997197*Ca[iCa+7,:])*sd4 - (0.4*Ca[iCa+0,:])*sd3
         Sh[iSh+1,:] = -(0.29814239699997197*Ca[iCa+3,:])*sd3 + (0.29814239699997197*Ca[iCa+8,:])*sd4 - (0.4*Ca[iCa+1,:])*sd3
         Sh[iSh+2,:] = -(0.29814239699997197*Ca[iCa+4,:])*sd5 - (0.29814239699997197*Ca[iCa+6,:])*sd5 + (0.4*Ca[iCa+2,:])
         Sh[iSh+3,:] = -(0.29814239699997197*Ca[iCa+5,:])*sd7 + (0.4*Ca[iCa+0,:])*sd6
         Sh[iSh+4,:] = (0.2581988897471611*Ca[iCa+9,:])*sd8
         Sh[iSh+5,:] = (0.29814239699997197*Ca[iCa+3,:])*sd7 - (0.4*Ca[iCa+1,:])*sd6
         Sh[iSh+6,:] = (0.29814239699997197*Ca[iCa+4,:])*sd9 - (0.29814239699997197*Ca[iCa+6,:])*sd9
         iCa += 10
         iSh += 7
      elif l == 4:
         Sh[iSh+0,:] = -(0.1301200097264711*Ca[iCa+10,:])*sdc - (0.1301200097264711*Ca[iCa+11,:])*sdc + (0.1301200097264711*Ca[iCa+9,:])*sdb + (0.22857142857142856*Ca[iCa+0,:])*sda + (0.22857142857142856*Ca[iCa+1,:])*sda + (0.22857142857142856*Ca[iCa+2,:])
         Sh[iSh+1,:] = (0.1126872339638022*Ca[iCa+14,:])*sde - (0.15118578920369088*Ca[iCa+3,:])*sdd - (0.15118578920369088*Ca[iCa+5,:])*sdd
         Sh[iSh+2,:] = -(0.1126872339638022*Ca[iCa+13,:])*sd7 - (0.15118578920369088*Ca[iCa+4,:])*sd7 + (0.15118578920369088*Ca[iCa+7,:])*sdf
         Sh[iSh+3,:] = -(0.1301200097264711*Ca[iCa+9,:])*sd11 + (0.22857142857142856*Ca[iCa+0,:])*sd10 + (0.22857142857142856*Ca[iCa+1,:])*sd10
         Sh[iSh+4,:] = -(0.1126872339638022*Ca[iCa+12,:])*sd7 - (0.15118578920369088*Ca[iCa+6,:])*sd7 + (0.15118578920369088*Ca[iCa+8,:])*sdf
         Sh[iSh+5,:] = (0.1301200097264711*Ca[iCa+10,:])*sd13 - (0.1301200097264711*Ca[iCa+11,:])*sd13 - (0.22857142857142856*Ca[iCa+0,:])*sd12 + (0.22857142857142856*Ca[iCa+1,:])*sd12
         Sh[iSh+6,:] = (0.15118578920369088*Ca[iCa+3,:])*sd14 - (0.15118578920369088*Ca[iCa+5,:])*sd14
         Sh[iSh+7,:] = -(0.1126872339638022*Ca[iCa+13,:])*sd16 + (0.15118578920369088*Ca[iCa+4,:])*sd15
         Sh[iSh+8,:] = (0.1126872339638022*Ca[iCa+12,:])*sd16 - (0.15118578920369088*Ca[iCa+6,:])*sd15
         iCa += 15
         iSh += 9
      elif l == 5:
         Sh[iSh+0,:] = -(0.04337333657549037*Ca[iCa+12,:])*sd19 + (0.05819143739626463*Ca[iCa+3,:])*sd18 - (0.05819143739626463*Ca[iCa+5,:])*sd19 + (0.0761904761904762*Ca[iCa+10,:])*sd17 + (0.0761904761904762*Ca[iCa+14,:])*sd8 + (0.12698412698412698*Ca[iCa+0,:])*sd17
         Sh[iSh+1,:] = -(0.04337333657549037*Ca[iCa+8,:])*sd19 - (0.05819143739626463*Ca[iCa+17,:])*sd19 + (0.05819143739626463*Ca[iCa+6,:])*sd18 + (0.0761904761904762*Ca[iCa+19,:])*sd8 + (0.0761904761904762*Ca[iCa+1,:])*sd17 + (0.12698412698412698*Ca[iCa+15,:])*sd17
         Sh[iSh+2,:] = -(0.05819143739626463*Ca[iCa+18,:])*sd1b + (0.05819143739626463*Ca[iCa+9,:])*sd1b + (0.0761904761904762*Ca[iCa+16,:])*sd1a - (0.0761904761904762*Ca[iCa+2,:])*sd1a
         Sh[iSh+3,:] = -(0.04337333657549037*Ca[iCa+12,:])*sd20 + (0.05819143739626463*Ca[iCa+3,:])*sd1d + (0.05819143739626463*Ca[iCa+5,:])*sd1e + (0.0761904761904762*Ca[iCa+10,:])*sd1f - (0.12698412698412698*Ca[iCa+0,:])*sd1c
         Sh[iSh+4,:] = -(0.05039526306789696*Ca[iCa+11,:])*sd21 + (0.05039526306789696*Ca[iCa+4,:])*sd21
         Sh[iSh+5,:] = (0.04337333657549037*Ca[iCa+8,:])*sd20 - (0.05819143739626463*Ca[iCa+17,:])*sd1e - (0.05819143739626463*Ca[iCa+6,:])*sd1d - (0.0761904761904762*Ca[iCa+1,:])*sd1f + (0.12698412698412698*Ca[iCa+15,:])*sd1c
         Sh[iSh+6,:] = -(0.04337333657549037*Ca[iCa+7,:])*sd23 + (0.0761904761904762*Ca[iCa+16,:])*sd22 + (0.0761904761904762*Ca[iCa+2,:])*sd22
         Sh[iSh+7,:] = -(0.05819143739626463*Ca[iCa+6,:])*sd25 + (0.0761904761904762*Ca[iCa+1,:])*sd24 + (0.12698412698412698*Ca[iCa+15,:])*sd26
         Sh[iSh+8,:] = (0.04337333657549037*Ca[iCa+7,:])*sd28 - (0.05819143739626463*Ca[iCa+18,:])*sd29 - (0.05819143739626463*Ca[iCa+9,:])*sd29 + (0.0761904761904762*Ca[iCa+16,:])*sd27 + (0.0761904761904762*Ca[iCa+2,:])*sd27 + (0.12698412698412698*Ca[iCa+20,:])
         Sh[iSh+9,:] = -(0.05819143739626463*Ca[iCa+3,:])*sd25 + (0.0761904761904762*Ca[iCa+10,:])*sd24 + (0.12698412698412698*Ca[iCa+0,:])*sd26
         Sh[iSh+10,:] = -(0.05039526306789696*Ca[iCa+11,:])*sd1b + (0.05039526306789696*Ca[iCa+13,:])*sd2a - (0.05039526306789696*Ca[iCa+4,:])*sd1b
         iCa += 21
         iSh += 11
      elif l == 6:
         Sh[iSh+0,:] = (0.026526119002773005*Ca[iCa+10,:])*sd2c - (0.026526119002773005*Ca[iCa+3,:])*sd2c + (0.06926406926406926*Ca[iCa+0,:])*sd2b - (0.06926406926406926*Ca[iCa+21,:])*sd2b
         Sh[iSh+1,:] = -(0.017545378532260507*Ca[iCa+17,:])*sd2f - (0.017545378532260507*Ca[iCa+8,:])*sd2f + (0.022972292920210562*Ca[iCa+19,:])*sd2f + (0.02353959545345999*Ca[iCa+6,:])*sd2e + (0.03828715486701761*Ca[iCa+15,:])*sd2d + (0.03828715486701761*Ca[iCa+1,:])*sd2d
         Sh[iSh+2,:] = -(0.017545378532260507*Ca[iCa+7,:])*sd31 + (0.022972292920210562*Ca[iCa+16,:])*sd32 + (0.03828715486701761*Ca[iCa+2,:])*sd30
         Sh[iSh+3,:] = -(0.015100657524077793*Ca[iCa+12,:])*sd36 + (0.026526119002773005*Ca[iCa+10,:])*sd34 + (0.026526119002773005*Ca[iCa+23,:])*sd35 + (0.026526119002773005*Ca[iCa+3,:])*sd34 + (0.026526119002773005*Ca[iCa+5,:])*sd35 - (0.06926406926406926*Ca[iCa+0,:])*sd33 - (0.06926406926406926*Ca[iCa+21,:])*sd33
         Sh[iSh+4,:] = -(0.017545378532260507*Ca[iCa+11,:])*sd31 + (0.022972292920210562*Ca[iCa+4,:])*sd32 + (0.03828715486701761*Ca[iCa+22,:])*sd30
         Sh[iSh+5,:] = -(0.026526119002773005*Ca[iCa+10,:])*sd37 + (0.026526119002773005*Ca[iCa+14,:])*sd38 + (0.026526119002773005*Ca[iCa+23,:])*sd38 - (0.026526119002773005*Ca[iCa+25,:])*sd38 + (0.026526119002773005*Ca[iCa+3,:])*sd37 - (0.026526119002773005*Ca[iCa+5,:])*sd38 + (0.06926406926406926*Ca[iCa+0,:])*sd37 - (0.06926406926406926*Ca[iCa+21,:])*sd37
         Sh[iSh+6,:] = -(0.02353959545345999*Ca[iCa+6,:])*sd3a + (0.03828715486701761*Ca[iCa+15,:])*sd39 + (0.03828715486701761*Ca[iCa+1,:])*sd39
         Sh[iSh+7,:] = -(0.017545378532260507*Ca[iCa+18,:])*sd3e + (0.017545378532260507*Ca[iCa+7,:])*sd3c + (0.022972292920210562*Ca[iCa+16,:])*sd3d + (0.02353959545345999*Ca[iCa+9,:])*sd38 - (0.03828715486701761*Ca[iCa+2,:])*sd3b
         Sh[iSh+8,:] = -(0.017545378532260507*Ca[iCa+17,:])*sd40 + (0.017545378532260507*Ca[iCa+8,:])*sd40 + (0.03828715486701761*Ca[iCa+15,:])*sd3f - (0.03828715486701761*Ca[iCa+1,:])*sd3f
         Sh[iSh+9,:] = (0.015100657524077793*Ca[iCa+12,:])*sd44 - (0.026526119002773005*Ca[iCa+10,:])*sd42 - (0.026526119002773005*Ca[iCa+14,:])*sd45 + (0.026526119002773005*Ca[iCa+23,:])*sd43 - (0.026526119002773005*Ca[iCa+25,:])*sd45 - (0.026526119002773005*Ca[iCa+3,:])*sd42 + (0.026526119002773005*Ca[iCa+5,:])*sd43 - (0.06926406926406926*Ca[iCa+0,:])*sd41 - (0.06926406926406926*Ca[iCa+21,:])*sd41 + (0.06926406926406926*Ca[iCa+27,:])
         Sh[iSh+10,:] = -(0.017545378532260507*Ca[iCa+11,:])*sd3c + (0.017545378532260507*Ca[iCa+13,:])*sd3e - (0.022972292920210562*Ca[iCa+4,:])*sd3d - (0.02353959545345999*Ca[iCa+24,:])*sd38 + (0.03828715486701761*Ca[iCa+22,:])*sd3b
         Sh[iSh+11,:] = (0.017545378532260507*Ca[iCa+11,:])*sd47 - (0.017545378532260507*Ca[iCa+13,:])*sd48 + (0.022972292920210562*Ca[iCa+4,:])*sd46 - (0.02353959545345999*Ca[iCa+24,:])*sd48 + (0.03828715486701761*Ca[iCa+22,:])*sd46 + (0.03828715486701761*Ca[iCa+26,:])*sd49
         Sh[iSh+12,:] = -(0.017545378532260507*Ca[iCa+18,:])*sd48 + (0.017545378532260507*Ca[iCa+7,:])*sd47 + (0.022972292920210562*Ca[iCa+16,:])*sd46 - (0.02353959545345999*Ca[iCa+9,:])*sd48 + (0.03828715486701761*Ca[iCa+20,:])*sd49 + (0.03828715486701761*Ca[iCa+2,:])*sd46
         iCa += 28
         iSh += 13
   return Sh









def _main():
   # read a file, including orbitals and basis sets, and test
   # if the output orbitals are orthogonal.
   def rmsd(a):
      return np.mean(a.flatten()**2)**.5
   FileName = "benzene.xml"
   #FileName = "/home/cgk/dev/xml-molpro/test1.xml"
   XmlData = ReadMolproXml(FileName,SkipVirtual=True)
   print "Atoms from file [a.u.]:\n%s" % XmlData.Atoms.MakeXyz(NumFmt="%20.15f",Scale=1/wmme.ToAng)
   OrbBasis = XmlData.OrbBasis
   #BasisLibs = ["def2-nzvpp-jkfit.libmol"]


   BasisLibs = []
   ic = wmme.FIntegralContext(XmlData.Atoms, XmlData.OrbBasis, FitBasis="univ-JKFIT", BasisLibs=BasisLibs)
   from wmme import mdot
   C = XmlData.Orbs
   S = ic.MakeOverlap()
   print "Orbital matrix shape: %s (loaded from '%s')" % (C.shape, FileName)
   print "Overlap matrix shape: %s (made via WMME)" % (S.shape,)
   np.set_printoptions(precision=4,linewidth=10000,edgeitems=3,suppress=False)
   SMo = mdot(C.T, S, C)
   print "Read orbitals:"
   for OrbInfo in XmlData.Orbitals:
      print "%30s" % OrbInfo.Desc
   print "MO deviation from orthogonality: %.2e" % rmsd(SMo - np.eye(SMo.shape[0]))

   pass

if __name__ == "__main__":
   _main()
# coding: utf-8





