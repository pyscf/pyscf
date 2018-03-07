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

from __future__ import print_function, division
import os
import sys
import numpy as np
from numpy import zeros, empty
from scipy.sparse import csr_matrix
import warnings

class gpaw_hsx_c():
  def __init__(self, sv, calc):
    """
        Gather information on the Hamiltonian
    """

    from gpaw.utilities import unpack
    assert calc.wfs.mode.lower()=='lcao'

    self.norbs    =  sv.norbs
    self.norbs_sc =  sv.norbs_sc
    self.nspin    =  sv.nspin
    self.telec = 0.001 # Don't know how to get it?? calc.atoms.get_temperature()
    self.nelec = calc.setups.nvalence
    #print(dir(calc.setups))
    #print("nelec = ", self.nelec, "telec = ", self.telec)
   # print(dir(calc.hamiltonian))

   # print(calc.hamiltonian)
   # print(calc.hamiltonian.vt_sG.shape)
   # print(calc.hamiltonian.vHt_g.shape)
   # print(calc.hamiltonian.vt_sg.shape)
   # for k in calc.hamiltonian.dH_asp.keys():
   #     print( k, unpack(calc.hamiltonian.dH_asp[k]))
    #self.nnz      =
   # self.is_gamma = (dat[i]>0); i=i+1;
   # self.nelec    =  dat[i]; i=i+1;
   # self.telec    =  dat[i]; i=i+1;
   #
   # self.h4 = np.reshape(dat[i:i+self.nnz*self.nspin], (self.nspin,self.nnz)); i=i+self.nnz*self.n
   # self.s4 = dat[i:i+self.nnz]; i = i + self.nnz;
   # self.x4 = np.reshape(dat[i:i+self.nnz*3], (self.nnz,3)); i = i + self.nnz*3;
   # self.row_ptr = np.array(dat[i:i+self.norbs+1]-1, dtype='int'); i = i + self.norbs+1;
   # self.col_ind = np.array(dat[i:i+self.nnz]-1, dtype='int'); i = i + self.nnz;
   # self.spin2h4_csr = []
   # for s in range(self.nspin):
   #     self.spin2h4_csr.append(csr_matrix((self.h4[s,:], self.col_ind, self.row_ptr), dtype=np.floa
   # self.s4_csr = csr_matrix((self.s4, self.col_ind, self.row_ptr), dtype=np.float32)
   #
    #self.orb_sc2orb_uc=None
    #if(i<len(dat)):
    #    if(self.is_gamma): raise SystemError('i<len(dat) && gamma')
    #    self.orb_sc2orb_uc = np.array(dat[i:i+self.norbs_sc]-1, dtype='int'); i = i + self.norbs_sc

    """
    Comment about the overlap matrix in Gpaw:
    Marc:
    > From what I see the overlap matrix is not written in the GPAW output. Do
    > there is an option to write it or it is not implemented?

    Ask answer (asklarsen@gmail.com):
    It is not implemented.  Just write it manually from the calculation script.

    If you use ScaLAPACK (or band/orbital parallelization) the array will
    be distributed, so each core will have only a chunk.  One needs to
    call a function to collect it on master then.  But probably you won't
    need that.

    The array will exist after you call calc.set_positions(atoms), in case
    you want to generate it without triggering a full calculation.
    """

    self.S_qMM = calc.wfs.S_qMM
    if calc.wfs.S_qMM is None:
        # sv.overlap_coo return csr sparce matrix
        self.s4_csr = sv.overlap_coo()
    else:
        self.s4_csr = csr_matrix(calc.wfs.S_qMM[0, :, :])

        if calc.wfs.S_qMM.shape[0] >1:
          warnings.warn("""
            GPAW overlaps has more than one kpts
            """, UserWarning)

  def check_overlaps(self, pyscf_overlaps):

      # overlap not in gpaw output, this routine can be used only
      # after a direct call to gpaw calculator
      if self.S_qMM is None: return -1

      s4 = self.s4_csr.toarray()

      if s4.shape != pyscf_overlaps.shape:
          warnings.warn("""
            Gpaw and Pyscf overlaps have different shapes.
            Something should be wrong!
            """, UserWarning)
          print("Shape: overlaps gpaw: ", s4.shape)
          print("Shape: overlaps pyscf: ", pyscf_overlaps)
          return -1

      return np.sum(abs(s4-pyscf_overlaps))
