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
# 2. Use proper functions provided by PySCF
#


# Developed by Elvira Sayfutyarova:
#  using wmme program provided by Gerald Knizia
#
# Comment: you need to get .xml file after running SCF or MCSCF in Molpro before running this program
#=====================================================================================================
import numpy as np
import wmme
import MolproXml
from wmme import mdot


def PrintMatrix(Caption, M):
   print("Matrix %s [%i x %i]:\n" % (Caption, M.shape[0], M.shape[1]))
   ColsFmt = M.shape[1] * " %11.5f"
   for iRow in range(M.shape[0]):
      print("  %s" % (ColsFmt % tuple(M[iRow,:])))

def ConvertMolproXmlToPyscfInput(XmlData):
# assemble a 'atoms' declaration compatible with PyScf.
# WARNING: currently all atoms of a given element must have the same basis set. May be fixed in future

   import pyscf.lib.parameters as param
   ToAngPyScf = param.BOHR

   AtomsPyscf = []
   for Atom in XmlData.Atoms:
      f = ToAngPyScf
      AtomsPyscf.append("%i %r %r %r" % (Atom.iElement, Atom.Pos[0]*f, Atom.Pos[1]*f, Atom.Pos[2]*f))
   AtomsPyscf = "\n".join(AtomsPyscf)

   iLastAtom = -1
   BasisPyscf = {}
   for iAtom in range(len(XmlData.Atoms)):
      Element = XmlData.Atoms[iAtom].Element
      if Element in BasisPyscf:
         # this doesn't work with multiple bases per atom for now
         continue

      # find all shells on the atom.
      AtShells = []
      for Shell in XmlData.OrbBasis.Shells:
         if Shell.iAtom == iAtom:
            AtShells.append(Shell)

      # convert to PyScf format: 
      AtBasisPyscf = []
      for Shell in AtShells:
         Sh = [Shell.l]
         for iExp in range(Shell.nExp):
            Sh.append([Shell.Exp[iExp]] + list(Shell.Co[iExp,:]))
         AtBasisPyscf.append(Sh)
      BasisPyscf[Element] =  AtBasisPyscf

   # convert orbitals from Molpro order to PyScf order.
   OrbsPyscf = _Vec_ShMolpro2ShPyscf(XmlData.Orbs, XmlData.OrbBasis.GetAngmomList())

   return AtomsPyscf, BasisPyscf, OrbsPyscf


# notes:
#   - pyscf order is ... -2 -1 0 +1 +2... (see cart2sph.c)
#   - Molpro -> PyScf trafo checked up to ang mom 'i'
_Molpro2PyscfBasisPermSph = {
   0: [0],
   1: [0, 1, 2],
   #  0  1   2   3   4
   # D0 D-2 D+1 D+2 D-1
   2: [1, 4, 0, 2, 3],
   #  0   1   2  3   4   5   6
   # F+1 F-1 F0 F+3 F-2 F-3 F+2
   3: [5, 4, 1, 2, 0, 6, 3],
   #  0   1   2   3   4   5   6   7   8
   # G0 G-2 G+1 G+4 G-1 G+2 G-4 G+3 G-3
   4: [6, 8, 1, 4, 0, 2, 5, 7, 3],
   #  0   1   2   3   4   5   6   7   8  9   10
   # H+1 H-1 H+2 H+3 H-4 H-3 H+4 H-5 H0 H+5 H-2
   5: [7, 4, 5, 10, 1, 8, 0, 2, 3, 6, 9],
   #  0   1   2   3   4   5   6   7   8   9  10  11  12
   # I+6 I-2 I+5 I+4 I-5 I+2 I-6 I+3 I-4 I0 I-3 I-1 I+1
   6: [6, 4, 8, 10, 1, 11, 9, 12, 5, 7, 3, 2, 0],
}

def _nCartY(l):
   return ((l+1)*(l+2))/2

def _Vec_ShMolpro2ShPyscf(Orbs, ls):
   I_MolproToPyscf = []
   iOff = 0
   for l in ls:
      I_MolproToPyscf += [(o + iOff) for o in _Molpro2PyscfBasisPermSph[l]]
      iOff += 2*l + 1
   I_MolproToPyscf = np.array(I_MolproToPyscf)
   return Orbs[I_MolproToPyscf, :]

def _rmsd(X):
   return np.mean(X.flatten()**2)**.5




def _run_with_pyscf(FileNameXml):
   from pyscf import gto
   from pyscf import scf

   print("\n* Reading: '%s'" % FileNameXml)
   XmlData = MolproXml.ReadMolproXml(FileNameXml, SkipVirtual=True)
   print("Atoms from file [a.u.]:\n%s" % XmlData.Atoms.MakeXyz(NumFmt="%20.15f",Scale=1/wmme.ToAng))


#  this gives you data from MolproXmlfile ready for use in PySCF
   Atoms, Basis, COrb = ConvertMolproXmlToPyscfInput(XmlData)

   mol = gto.Mole()
   mol.build(
      verbose = 0,
      atom = Atoms,
      basis = Basis,
   )

# compute overlap matrix with PySCF
   S = mol.intor_symmetric('int1e_ovlp_sph')
# compute overlap matrix of MO basis overlap, using the MOs imported from the XML,
# and the overlap matrix computed with PySCF to check that MO were imported properly.
   SMo = mdot(COrb.T, S, COrb)
   PrintMatrix("MO-Basis overlap (should be unity!)", SMo)
   print("RMSD(SMo-id): %8.2e" % _rmsd(SMo - np.eye(SMo.shape[0])))
   print


if __name__ == "__main__":
   _run_with_pyscf("pd_3d.xml")

