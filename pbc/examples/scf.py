import numpy as np

from pyscf.pbc import scf as pbchf
from pyscf.pbc import dft as pbcdft

import ase
import ase.lattice
import ase.dft.kpoints

def run_hf(cell, exxdiv=None):
    """Run a gamma-point Hartree-Fock calculation."""
    mf = pbchf.RHF(cell, exxdiv=exxdiv)
    mf.verbose = 7
    print mf.scf()
    return mf

def run_dft(cell):
    """Run a gamma-point DFT (LDA) calculation."""
    mf = pbcdft.RKS(cell)
    mf.xc = 'lda,vwn'
    mf.verbose = 7
    print mf.scf()
    return mf

def run_khf(cell, nmp=[1,1,1], gamma=False, kshift=np.zeros(3), exxdiv=None):
    """Run a k-point-sampling Hartree-Fock calculation."""
    scaled_kpts = ase.dft.kpoints.monkhorst_pack(nmp)
    if gamma:
        for i in range(3):
            if nmp[i] % 2 == 0:
                scaled_kpts[:,i] += 0.5/nmp[i]
    # Move first kpt to the Gamma pt
    scaled_kpts -= scaled_kpts[0,:]
    # Shift by kshift
    scaled_kpts += kshift
    # Put back in BZ
    print "Before shifting back"
    print scaled_kpts
    scaled_kpts -= 1.0*np.round(scaled_kpts/1.0)
    print "After shifting back"
    print scaled_kpts
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
    kmf = pbchf.KRHF(cell, abs_kpts, exxdiv=exxdiv)
    kmf.verbose = 7
    print kmf.scf()
    return kmf

def run_kdft(cell, nmp=[1,1,1], gamma=False):
    """Run a k-point-sampling DFT (LDA) calculation."""
    scaled_kpts = ase.dft.kpoints.monkhorst_pack(nmp)
    if gamma:
        for i in range(3):
            if nmp[i] % 2 == 0:
                scaled_kpts[:,i] += 0.5/nmp[i]
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
    kmf = pbcdft.KRKS(cell, abs_kpts)
    kmf.xc = 'lda,vwn'
    kmf.verbose = 7
    print kmf.scf()
    return kmf

if __name__ == '__main__':
    from helpers import get_ase_diamond_primitive, build_cell
    ase_atom = get_ase_diamond_primitive() 
    cell = build_cell(ase_atom)
    run_hf(cell)
    run_dft(cell)
    run_khf(cell)
    run_kdft(cell)
