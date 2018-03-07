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
#   * Indent: 3 -> 4
#   * Constant should be all uppercase
#   * Function/method should be all lowercase
#   * Line wrap around 80 columns
#   * Use either double quote or single quote, not mix
# 
# 2. Conventions required by PySCF
#   * Use PYSCF_TMPDIR to replace _TmpDir
# 
# 3. Use proper functions provided by PySCF
#


#  This file is adapted with permission from the wmme program of Gerald Knizia.
#  See http://sites.psu.edu/knizia/software/
#====================================================

from __future__ import print_function
import numpy as np
from numpy import dot, array
from os import path
from sys import version_info

def GetModulePath():
   # (hopefully) return the path of the .py file.
   # idea is to leave wmme.py in the same directory as the wmme executable,
   # and import invoke the scripts using it via, for example,
   #   PYTHONPATH=$HOME/dev/wmme:$PYTHONPATH python myscriptfile.py
   import inspect
   return path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))

if 0:
   # set executable/basis library directory explicitly.
   _WmmeDir = "/home/cgk/dev/wmme"
else:
   # set executable/basis library from path of wmme.py
   _WmmeDir = None

_TmpDir = None         # if None: use operating system default
_BasisLibDir = None    # if None: same as _WmmeDir/bases
#ToAng =     0.5291772108
ToAng =     0.529177209  # molpro default.

def ElementNameDummy():
   ElementNames = "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split()
   ElementNumbers = dict([(o,i) for (i,o) in enumerate(ElementNames)])
   return ElementNames, ElementNumbers
ElementNames, ElementNumbers = ElementNameDummy()

def mdot(*args):
   """chained matrix product: mdot(A,B,C,..) = A*B*C*...
   No attempt is made to optimize the contraction order."""
   r = args[0]
   for a in args[1:]:
      r = dot(r,a)
   return r

def dot2(A,B): return dot(A.flatten(),B.flatten())

def nCartY(l):
   return ((l+1)*(l+2))/2

class FAtom(object):
   def __init__(self, Element, Position, Index):
      self.Element = Element
      self.Pos = Position
      self.Index = Index
   @property
   def Label(self):
      # return element and center index combined.
      return "%2s%3s"%(self.Element,1 + self.Index)
   @property
   def iElement(self):
      return ElementNumbers[self.Element]
   def __str__(self):
      return "%s (%6.3f,%6.3f,%6.3f)"%(self.Label, self.Pos[0], self.Pos[1], self.Pos[2])


class FAtomSet(object):
   def __init__(self, Positions, Elements, Orientations=None, Name=None):
      """Positions: 3 x nAtom matrix. Given in atomic units (ABohr).
      Elements: element name (e.g., H) for each of the positions.
      Orientations: If given, a [3,3,N] array encoding the standard
      orientation of the given atoms (for replicating potentials!). For
      each atom there is a orthogonal 3x3 matrix denoting the ex,ey,ez
      directions."""
      self.Pos = Positions
      assert(self.Pos.shape[0] == 3 and self.Pos.shape[1] == len(Elements))
      self.Elements = Elements
      self.Orientations = Orientations
      self.Name = Name
   def MakeXyz(self,NumFmt = "%15.8f",Scale=1.):
      Lines = []
      for i in range(len(self.Elements)):
         Lines.append(" %5s {0} {0} {0}".format(NumFmt) % (\
            self.Elements[i], Scale*self.Pos[0,i], Scale*self.Pos[1,i], Scale*self.Pos[2,i]))
      return "\n".join(Lines)
   def nElecNeutral(self):
      """return number of electrons present in the total system if neutral."""
      return sum([ElementNumbers[o] for o in self.Elements])
   def fCoreRepulsion1(self, iAt, jAt):
      if iAt == jAt: return 0. # <- a core doesn't repulse itself.
      ChA, ChB = [ElementNumbers[self.Elements[o]] for o in [iAt, jAt]]
      return ChA * ChB / np.sum((self.Pos[:,iAt] - self.Pos[:,jAt])**2)**.5
   def fCoreRepulsion(self):
      N = len(self.Elements)
      Charges = array([ElementNumbers[o] for o in self.Elements])
      fCoreEnergy = 0
      for i in range(N):
         for j in range(i):
            fCoreEnergy += self.fCoreRepulsion1(i,j)
            #fCoreEnergy += Charges[i] * Charges[j] / np.sum((self.Pos[:,i] - self.Pos[:,j])**2)**.5
      return fCoreEnergy
   def __str__(self):
      Caption = "  %5s%15s %15s %15s" % ("ATOM", "POS/X", "POS/Y", "POS/Z")
      return Caption + "\n" + self.MakeXyz()
   def __len__(self):         return len(self.Elements)
   def __getitem__(self,key): return FAtom(self.Elements[key], self.Pos[:,key], key)
   def __iter__(self):
      for (iAt,(Type,Xyz)) in enumerate(zip(self.Elements, self.Pos.T)):
         #yield (Type,Xyz)
         yield FAtom(Type, Xyz, iAt)


class FBasisShell(object):
   """A generally contracted shell of spherical harmonic basis functions."""
   def __init__(self, l, Exp, Co):
      self.l = l
      assert(isinstance(l,int) and l >= 0 and l <= 8)
      self.Exp = np.array(Exp)
      assert(self.Exp.ndim == 1)
      self.Co = np.array(Co)
      assert(self.Co.ndim == 2 and self.Co.shape[0] == len(self.Exp))
      self.Element = None  # designated element for the basis function
      self.Comment = None  # comment on the basis function (e.g., literature reference)
   @property
   def nExp(self):
      return len(self.Exp)
   @property
   def nCo(self):
      return self.Co.shape[1]
   @property
   def nFn(self):
      return self.nCo * (2*self.l + 1)
   @property
   def nFnCa(self):
      return self.nCo * nCartY(self.l)
   @property
   def AngMom(self): return self.l

   def __str__(self):
      Lines = []
      Lines.append("BasisShell [l = %i, nExp = %i, nCo = %i]" % (self.l, self.nExp, self.nCo))
      def FmtA(L):
         return ", ".join("%12.5f" % o for o in L)
      Lines.append("  Exps   = [%s]" % FmtA(self.Exp))
      for iCo in range(self.nCo):
         Lines.append("  Co[%2i] = [%s]" % (iCo, FmtA(self.Co[:,iCo])))
      return "\n".join(Lines)

class FBasisShell1(object):
   """A FBasisShell which is placed on a concrete atom."""
   def __init__(self, Atom, ShellFn):
      self.Atom = Atom
      self.Fn = ShellFn
      assert(isinstance(self.Fn, FBasisShell))
   @property
   def Pos(self):
      return self.Atom.Pos
   @property
   def iAtom(self):
      return self.Atom.Index

   @property
   def l(self): return self.Fn.l
   @property
   def nExp(self): return self.Fn.nExp
   @property
   def Exp(self): return self.Fn.Exp
   @property
   def nCo(self): return self.Fn.nCo
   @property
   def Co(self): return self.Fn.Co
   @property
   def nFn(self): return self.Fn.nFn
   @property
   def nFnCa(self): return self.Fn.nFnCa


class FBasisSet(object):
   def __init__(self, Shells, Atoms):
      # list of FBasisShell1 objects.
      self.Shells = Shells
      self.Atoms = Atoms
   @property
   def nFn(self):
      n = 0
      for Sh in self.Shells:
         n += Sh.nFn
      return n
   @property
   def nFnCa(self):
      n = 0
      for Sh in self.Shells:
         n += Sh.nFnCa
      return n
   def __str__(self):
      Lines = []
      for o in self.Shells:
         Lines.append("Atom %s  %s" % (o.Atom, o.Fn))
      return "\n".join(Lines)
   def FmtCr(self):
      #f = 1./ToAng
      f = 1.
      Lines = []
      def Emit(s):
         Lines.append(s)
      def EmitArray(Name, A):
         #Emit("    " + Name + "<" + " ".join("%.16e"%o for o in A) + ">")
         Emit("    " + Name + "<" + " ".join("%r"%o for o in A) + ">")

      # collect all unique FBasisShell objects.
      BasisFns = []
      BasisFnIds = {}  # map id(BasisFn)->(index)
      for Shell in self.Shells:
         if id(Shell.Fn) not in BasisFnIds:
            BasisFnIds[id(Shell.Fn)] = len(BasisFns)
            BasisFns.append(Shell.Fn)
         pass

      Emit("Basis<Version<0.1> nFns<%i> nShells<%i>" % (len(BasisFns), len(self.Shells)))
      # store the function declarations...
      def EmitBasisFn(Fn):
         Emit("  Fn<Id<%i> l<%i> nExp<%i> nCo<%i>" % (
            BasisFnIds[id(Fn)], Fn.l, Fn.nExp, Fn.nCo))
         EmitArray("Exp", Fn.Exp)
         for Co in Fn.Co.T:
            EmitArray("Co", Co)
         Emit("  >")
         pass
      for Fn in BasisFns:
         EmitBasisFn(Fn)

      # ...and their distribution amongst atoms.
      def EmitShell(Sh):
         #Emit("  Shell<iAt<%i> x<%.16e> y<%.16e> z<%.16e> FnId<%i>>" % (
         Emit("  Shell<iAt<%i> x<%r> y<%r> z<%r> FnId<%i>>" % (
            Sh.Atom.Index, f*Sh.Atom.Pos[0], f*Sh.Atom.Pos[1], f*Sh.Atom.Pos[2], BasisFnIds[id(Sh.Fn)]))
         pass
      for Shell in self.Shells:
         EmitShell(Shell)
      Emit(">") # end of Basis
      return "\n".join(Lines)
   def GetAngmomList(self):
      # list of all basis function angular momenta in the basis, for converting basis function orders and types.
      ls = []
      for Shell in self.Shells:
         for iCo in range(Shell.nCo):
            ls.append(Shell.l)
      return ls



class FIntegralContext(object):
   """contains data describing how to evaluate quantum chemistry matrix
      elements on electronic system as defined by the given atoms and basis
      sets.

      Note: Basis sets must either be basis set names (i.e., library names)
      or FBasisSet objects.
      """
   def __init__(self, Atoms, OrbBasis, FitBasis=None, BasisLibs=None):
      self.Atoms = Atoms
      self.OrbBasis = OrbBasis
      self.FitBasis = FitBasis
      self.BasisLibs = BasisLibs

   def _InvokeBfint(self, Args, Outputs=None, Inputs=None, MoreBases=None):
      Bases = {}
      if self.OrbBasis: Bases['--basis-orb'] = self.OrbBasis
      if self.FitBasis: Bases['--basis-fit'] = self.FitBasis
      if MoreBases:
         Bases = dict(list(Bases.items()) + list(MoreBases.items()))

      return _InvokeBfint(self.Atoms, Bases, self.BasisLibs, Args, Outputs, Inputs)

   def MakeBaseIntegrals(self, Smh=True, MakeS=False):
      """Invoke bfint to calculate CoreEnergy (scalar), CoreH (nOrb x nOrb),
      Int2e_Frs (nFit x nOrb x nOrb), and overlap matrix (nOrb x nOrb)"""

      # assemble arguments to integral generation program
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
         # ^- calculate integrals in symmetrically orthogonalized AO basis
      Outputs = []
      Outputs.append(("--save-coreh", "INT1E"))
      Outputs.append(("--save-fint2e", "INT2E"))
      Outputs.append(("--save-overlap", "OVERLAP"))

      CoreH, Int2e, Overlap = self._InvokeBfint(Args, Outputs)

      nOrb = CoreH.shape[0]
      Int2e = Int2e.reshape((Int2e.shape[0], nOrb, nOrb))
      CoreEnergy = self.Atoms.fCoreRepulsion()

      if MakeS:
         return CoreEnergy, CoreH, Int2e, Overlap
      else:
         return CoreEnergy, CoreH, Int2e

   def MakeOverlaps2(self, OrbBasis2):
      """calculate overlap between current basis and a second basis, as
      described in OrbBasis2. Returns <1|2> and <2|2> matrices."""
      Args = []
      MoreBases = {'--basis-orb-2': OrbBasis2}
      Outputs = []
      Outputs.append(("--save-overlap-2", "OVERLAP_2"))
      Outputs.append(("--save-overlap-12", "OVERLAP_12"))
      #Outputs.append(("--save-overlap", "OVERLAP"))

      Overlap2, Overlap12 = self._InvokeBfint(Args, Outputs, MoreBases=MoreBases)
      return Overlap2, Overlap12

   def MakeOverlap(self, OrbBasis2=None):
      """calculate overlap within main orbital basis, and, optionally, between main
      orbital basis and a second basis, as described in OrbBasis2.
      Returns <1|1>, <1|2>, and <2|2> matrices."""
      Args = []
      Outputs = []
      Outputs.append(("--save-overlap", "OVERLAP_1"))
      if OrbBasis2 is not None:
         MoreBases = {'--basis-orb-2': OrbBasis2}
         Outputs.append(("--save-overlap-12", "OVERLAP_12"))
         Outputs.append(("--save-overlap-2", "OVERLAP_2"))
         return self._InvokeBfint(Args, Outputs, MoreBases=MoreBases)
      else:
         MoreBases = None
         Overlap, = self._InvokeBfint(Args, Outputs, MoreBases=MoreBases)
         return Overlap

   def MakeNuclearAttractionIntegrals(self, Smh=True):
      """calculate nuclear attraction integrals in main basis, for each individual atomic core.
      Returns nAo x nAo x nAtoms array."""
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
      Outputs = []
      Outputs.append(("--save-vnucN", "VNUC_N"))
      VNucN = self._InvokeBfint(Args, Outputs)[0]
      nOrb = int(VNucN.shape[0]**.5 + .5)
      assert(nOrb**2 == VNucN.shape[0])
      assert(VNucN.shape[1] == len(self.Atoms))
      return VNucN.reshape(nOrb, nOrb, VNucN.shape[1])

   def MakeNuclearSqDistanceIntegrals(self, Smh=True):
      """calculate <mu|(r-rA)^2|nu> integrals in main basis, for each individual atomic core.
      Returns nAo x nAo x nAtoms array."""
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
      Outputs = []
      Outputs.append(("--save-rsqN", "RSQ_N"))
      RsqN = self._InvokeBfint(Args, Outputs)[0]
      nOrb = int(RsqN.shape[0]**.5 + .5)
      assert(nOrb**2 == RsqN.shape[0])
      assert(RsqN.shape[1] == len(self.Atoms))
      return RsqN.reshape(nOrb, nOrb, RsqN.shape[1])

   def MakeKineticIntegrals(self, Smh=True):
      """calculate <mu|-1/2 Laplace|nu> integrals in main basis, for each individual atomic core.
      Returns nAo x nAo x nAtoms array."""
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
      Outputs = []
      Outputs.append(("--save-kinetic", "EKIN"))
      Op = self._InvokeBfint(Args, Outputs)[0]
      return Op

   def MakeDipoleIntegrals(self, Smh=True):
      """calculate dipole operator matrices <\mu|w|\nu> (w=x,y,z) in
      main basis, for each direction. Returns nAo x nAo x 3 array."""
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
      Outputs = []
      Outputs.append(("--save-dipole", "DIPN"))
      DipN = self._InvokeBfint(Args, Outputs)[0]
      nOrb = int(DipN.shape[0]**.5 + .5)
      assert(nOrb**2 == DipN.shape[0])
      assert(DipN.shape[1] == 3)
      return DipN.reshape(nOrb, nOrb, 3)

   def MakeOrbitalsOnGrid(self, Orbitals, Grid, DerivativeOrder=0):
      """calculate values of molecular orbitals on a grid of 3d points in space.
      Input:
         - Orbitals: nAo x nOrb matrix, where nAo must be compatible with
           self.OrbBasis. The AO dimension must be contravariant AO (i.e., not SMH).
         - Grid: 3 x nGrid array giving the coordinates of the grid points.
         - DerivativeOrder: 0: only orbital values,
                            1: orbital values and 1st derivatives,
                            2: orbital values and up to 2nd derivatives.
      Returns:
         - nGrid x nDerivComp x nOrb array. If DerivativeOrder is 0, the
           DerivComp dimension is omitted.
      """
      Args =    [("--eval-orbitals-dx=%s" % DerivativeOrder)]
      Inputs =  [("--eval-orbitals", "ORBITALS.npy", Orbitals)]\
              + [("--grid-coords", "GRID.npy", Grid)]
      Outputs = [("--save-grid-values", "ORBS_ON_GRID")]

      (ValuesOnGrid,) = self._InvokeBfint(Args, Outputs, Inputs)
      nComp = [1,4,10][DerivativeOrder]
      if nComp != 1:
         ValuesOnGrid = Values.reshape((Grid.shape[1], nComp, Orbitals.shape[1]))
      return ValuesOnGrid

   def MakeRaw2eIntegrals(self, Smh=True, Kernel2e="coulomb"):
      """compute Int2e_Frs (nFit x nOrb x nOrb) and fitting metric Int2e_FG (nFit x nFit),
      where the fitting metric is *not* absorbed into the 2e integrals."""

      # assemble arguments to integral generation program
      Args = []
      if Smh:
         Args.append("--orb-trafo=Smh")
         # ^- calculate integrals in symmetrically orthogonalized AO basis
      Args.append("--kernel2e='%s'" % Kernel2e)
      Args.append("--solve-fitting-eq=false")
      Outputs = []
      Outputs.append(("--save-fint2e", "INT2E_3IX"))
      Outputs.append(("--save-fitting-metric", "INT2E_METRIC"))

      Int2e_Frs, Int2e_FG = self._InvokeBfint(Args, Outputs)

      nOrb = int(Int2e_Frs.shape[1]**.5 + .5)
      assert(nOrb**2 == Int2e_Frs.shape[1])
      Int2e_Frs = Int2e_Frs.reshape((Int2e_Frs.shape[0], nOrb, nOrb))
      assert(Int2e_Frs.shape[0] == Int2e_FG.shape[0])
      assert(Int2e_FG.shape[0] == Int2e_FG.shape[1])
      return Int2e_FG, Int2e_Frs



def _InvokeBfint(Atoms, Bases, BasisLibs, BaseArgs, Outputs, Inputs=None):
   """Outputs: an array of tuples (cmdline-arguments,filename-base).
   We will generate arguments for each of them and try to read the
   corresponding files as numpy arrays and return them in order."""
   from tempfile import mkdtemp
   from shutil import rmtree
   #from commands import getstatusoutput
   from subprocess import check_output, CalledProcessError
   # make a directory to store our input/output in.
   BasePath = mkdtemp(prefix="wmme.", dir=_TmpDir)
   def Cleanup():
      rmtree(BasePath)
      pass

   BfIntDir = _WmmeDir
   if BfIntDir is None: BfIntDir = GetModulePath()
   BasisLibDir = _BasisLibDir
   if BasisLibDir is None:
      BasisLibDir = path.join(BfIntDir,"bases")
   MakeIntegralsExecutable = path.join(BfIntDir,"wmme")

   # assemble arguments to integral generation program
   FileNameXyz = path.join(BasePath, "ATOMS")

   Args = [o for o in BaseArgs]
   Args.append("--matrix-format=npy")
   for BasisLib in BasisLibs:
      Args.append("--basis-lib=%s" % path.join(BasisLibDir, BasisLib))
   Args.append("--atoms-au=%s" % FileNameXyz)
   iWrittenBasis = 0
   for (ParamName, BasisObj) in Bases.items():
      if BasisObj is None:
         continue
      if isinstance(BasisObj, FBasisSet):
         # basis is given as an explicit FBasisSet object.
         # Write the basis set to disk and supply the file name as argument
         BasisFile = path.join(BasePath, "BASIS%i" % iWrittenBasis)
         iWrittenBasis += 1
         with open(BasisFile, "w") as File:
            File.write(BasisObj.FmtCr())
         Args.append("%s='!%s'" % (ParamName, BasisFile))
      else:
         assert(isinstance(BasisObj, str))
         # it's just a basis set name: append the name to the arguments.
         # (set will be read from library by wmme itself)
         Args.append("%s=%s" % (ParamName, BasisObj))
      pass

   # make file names and arguments for output arrays
   FileNameOutputs = []
   for (ArgName,FileNameBase) in Outputs:
      FileName = path.join(BasePath, FileNameBase)
      FileNameOutputs.append(FileName)
      Args.append("%s='%s'" % (ArgName, FileName))

   XyzLines = "%i\n\n%s\n" % (len(Atoms), Atoms.MakeXyz("%24.16f"))
   # ^- note on the .16f: it actually does make a difference. I had .8f
   #    there before, and it lead to energy changes on the order of 1e-8
   #    when treating only non-redundant subsystem out of a symmetric
   #    arrangement.
   try:
      with open(FileNameXyz, "w") as File:
         File.write(XyzLines)

      # save input arrays if provided.
      if Inputs:
         for (ArgName,FileNameBase,Array) in Inputs:
            FileName = path.join(BasePath, FileNameBase)
            np.save(FileName,Array)
            Args.append("%s='%s'" % (ArgName, FileName))

      Cmd = "%s %s" % (MakeIntegralsExecutable, " ".join(Args))
      #print("!Invoking %s\n" % Cmd)
      #iErr, Output = getstatusoutput(Cmd)
      #if ( iErr != 0 ):
      try:
         Output = check_output(Cmd, shell=True)
         if (version_info) >= (3,0):
            # it returns a byte string in Python 3... which wouldn't be a problem
            # if not all OTHER literals were converted to unicode implicitly.
            Output = Output.decode("utf-8")
      except CalledProcessError as e:
         raise Exception("Integral calculation failed. Output was:\n%s\nException was: %s" % (e.output, str(e)))

      OutputArrays = []
      for FileName in FileNameOutputs:
         OutputArrays.append(np.load(FileName))
   except:
      Cleanup()
      raise
   # got everything we need. Delete the temporary directory.
   Cleanup()

   return tuple(OutputArrays)


def ReadXyzFile(FileName,Scale=1./ToAng):
   Text = open(FileName,"r").read()
   Lines = Text.splitlines()
   # allowed formats: <nAtoms> \n Desc \n <atom-list>
   #              or: <atom-list> (without any headers)
   # in the first case, only the first nAtoms+2 lines are read, in the
   # second case everything which does not look like a xyz line is
   # ignored.
   nAtoms = None
   r = 0,-1
   if ( len(Lines[0].split()) == 1 ):
      nAtoms = int(Lines[0].split()[0])
      r = 2,nAtoms+2
   Atoms = []
   Xyz = []
   for Line in Lines:
      ls = Line.split()
      try:
         Atom = ls[0]
         x,y,z = float(ls[1]), float(ls[2]), float(ls[3])
      except:
         continue
      Atom = Atom[0].upper() + Atom[1:].lower()
      # maybe we should allow for (and ignore) group numbers after the
      # elements?
      if Atom not in ElementNames:
         raise Exception("while reading '%s': unrecognized element '%s'." % (FileName,Atom))
      Atoms.append(Atom)
      Xyz.append((x,y,z))
   Xyz = Scale*array(Xyz).T
   if 0:
      print("*read '%s':\n%s" % (FileName, str(FAtomSet(Xyz, Atoms))))
   return Xyz, Atoms

def ReadAtomsFromXyzFile(FileName, Scale=1./ToAng):
   Xyz,Elements = ReadXyzFile(FileName, Scale)
   return FAtomSet(Xyz, Elements)
