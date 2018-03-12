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
#   * Line wrap around 80 columns
# 
# 2. Move to pyscf/examples/tools
#


# Developed by Elvira Sayfutyarova:
#  using wmme program provided by Gerald Knizia
#
# Comment: you need to get .xml file after running SCF or MCSCF in Molpro before running this program
#=====================================================================================================
# Note, that right now the default value is  SkipVirtual =True, that means that Orbitals COrb include only those,
# which have non-zero occupation numbers in XML file. If you want virtual orbitals too, change SkipVirtual to False.


import numpy as np
from wmme import mdot
from pyscf import gto
from pyscf import scf
import wmme
import MolproXml
import MolproXmlToPyscf
from pyscf import ao2mo
from pyscf import mcscf
from functools import reduce
from pyscf import fci


def rmsd(X):
   return np.mean(X.flatten()**2)**.5


def PrintMatrix(Caption, M):
   print "Matrix %s [%i x %i]:\n" % (Caption, M.shape[0], M.shape[1])
   ColsFmt = M.shape[1] * " %11.5f"
   for iRow in range(M.shape[0]):
      print "  %s" % (ColsFmt % tuple(M[iRow,:]))


def _run_with_pyscf(FileNameXml):
   # read Molpro XML file exported via {put,xml,filename}
   print "\n* Reading: '%s'" % FileNameXml
   XmlData = MolproXml.ReadMolproXml(FileNameXml, SkipVirtual=True)
# Note, that right now the default value is  SkipVirtual =True, that means that Orbitals COrb include only those,
# which have non-zero occupation numbers in XML file. If you want virtual orbitals too, change SkipVirtual to False.
 
   print "Atoms from file [a.u.]:\n%s" % XmlData.Atoms.MakeXyz(NumFmt="%20.15f",Scale=1/wmme.ToAng)

   # convert data from XML file (atom positions, basis sets, MO coeffs) into format compatible
   # with PySCF.
   Atoms, Basis, COrb = MolproXmlToPyscf.ConvertMolproXmlToPyscfInput(XmlData)

   # make pyscf Mole object
   mol = gto.Mole()
   mol.build(
      verbose = 0,
      atom = Atoms,
      basis = Basis,
      spin = 0
   ) 

   
   # compute overlap matrix with PySCF
   S = mol.intor_symmetric('cint1e_ovlp_sph')
   # compute overlap matrix of MO basis overlap, using the MOs imported from the XML,
   # and the overlap matrix computed with PySCF to check that MO were imported properly.
   SMo = mdot(COrb.T, S, COrb)
   PrintMatrix("MO-Basis overlap (should be unity!)", SMo)
   print "RMSD(SMo-id): %8.2e" % rmsd(SMo - np.eye(SMo.shape[0]))
   print

def get_1e_integrals_in_MOs_from_Molpro_for_SOC(FileNameXml):
   # read Molpro XML file exported via {put,xml,filename}
   print "\n* Reading: '%s'" % FileNameXml
   XmlData = MolproXml.ReadMolproXml(FileNameXml, SkipVirtual=True)
   print "Atoms from file [a.u.]:\n%s" % XmlData.Atoms.MakeXyz(NumFmt="%20.15f",Scale=1/wmme.ToAng)

   # convert data from XML file (atom positions, basis sets, MO coeffs) into format compatible    # with PySCF.
   Atoms, Basis, COrb = MolproXmlToPyscf.ConvertMolproXmlToPyscfInput(XmlData)

   # make pyscf Mole object
   mol = gto.Mole()
   mol.build(
      verbose = 0,
      atom = Atoms,
     basis = Basis,
#      symmetry = 'D2h',
      spin = 0
   )

   natoms =1
   norb =11
   nelec =10


   all_orbs = len(COrb)
#  COrb is a list of orbs, you  get the info about orbs in the output  when reading orbs with a scheme above : 
#  # of an orb in COrb list, irrep in the point group used in Molpro calcs, # of orb in a given irrep in Molpro output

   Orblist = [6,7,8,9,10,18,19,24,25,27,28]

   ActOrb = np.zeros(shape=(all_orbs,norb))
   ActOrb2 = np.zeros(shape=(all_orbs,norb))

   for o1 in range(all_orbs):
     for o2 in range(norb):
       ActOrb[o1,o2]= COrb[o1, Orblist[o2]]

   print "================================================="
   print " Now print So1e integrals" 
   for id in range(natoms):
     chg = mol.atom_charge(id)
     mol.set_rinv_origin_(mol.atom_coord(id)) # set the gauge origin to first atom
     h1ao = abs(chg) *mol.intor('cint1e_prinvxp_sph', comp=3) # comp=3 for x,y,z directions
     h1 = []
     for i in range(3):
        h1.append(reduce(np.dot, (ActOrb.T, h1ao[i], ActOrb)))
     for i in range(3):
      for j in range(h1[i].shape[0]):
       for k in range(h1[i].shape[1]):
        print id, i+1, j+1, k+1, h1[i][j,k]



if __name__ == "__main__":
   _run_with_pyscf("pd_3d.xml")
   get_1e_integrals_in_MOs_from_Molpro_for_SOC("pd_3d.xml")
