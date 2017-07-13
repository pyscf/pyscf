from __future__ import print_function, division
import os
import sys
import numpy as np
from numpy import zeros, empty 
import warnings

class gpaw_wfsx_c():
  
  def __init__(self, calc):
    """
        Gathers the information on the available wavefunctions
        (Kohn-Sham or Hartree-Fock orbitals)
    """
    assert calc.wfs.mode.lower()=='lcao'

    self.nreim = 1 # Only real part? because wavefunctions from gpaw are complex
    self.nspin = calc.get_number_of_spins()
    self.norbs = calc.setups.nao
    self.nbands= calc.parameters['nbands']
    self.k2xyz = calc.parameters['kpts']
    self.nkpoints = len(self.k2xyz)

    self.ksn2e = np.zeros((self.nkpoints, self.nspin, self.nbands))
    for ik in range(self.nkpoints):
      for spin in range(self.nspin):
        self.ksn2e[ik, spin, :] = calc.wfs.collect_eigenvalues(spin,ik)

    # Import wavefunctions from GPAW calculator
    self.x = np.zeros((self.nkpoints, self.nspin, self.nbands, self.norbs, self.nreim))
    for k in range(calc.wfs.kd.nibzkpts):
        for s in range(calc.wfs.nspins):
            C_nM = calc.wfs.collect_array('C_nM', k, s)
            self.x[k, s, :, :, 0] = C_nM.real

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
    if calc.wfs.S_qMM is None:
        self.overlaps = None
    else:
        self.overlaps = calc.wfs.S_qMM[0, :, :]

        if calc.wfs.S_qMM.shape[0] >1:
          warnings.warn("""
            GPAW overlaps has more than one kpts
            """, UserWarning)

  def check_overlaps(self, pyscf_overlaps):

      # overlap not in gpaw output, this routine can be used only
      # after a direct call to gpaw calculator
      if self.overlaps is None: return -1

      if self.overlaps.shape != pyscf_overlaps.shape:
          warnings.warn("""
            Gpaw and Pyscf overlaps have different shapes.
            Something should be wrong!
            """, UserWarning)
          print("Shape: overlaps gpaw: ", self.overlaps.shape)
          print("Shape: overlaps pyscf: ", pyscf_overlaps)
          return -1

      return np.sum(abs(self.overlaps-pyscf_overlaps))
