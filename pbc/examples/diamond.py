import sys
import numpy as np

from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc import dft as pbcdft
import pyscf.pbc.tools.pyscf_ase as pyscf_ase

import ase
import ase.lattice
import ase.dft.kpoints

def run_hf(ase_atom, ngs, nmp=1):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell

    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    #cell.verbose = 4
    cell.build()

    print "-- The Gamma point calculation -----------"
    mf = pbchf.RHF(cell)
    mf.verbose = 7
    print mf.scf()

def run_lda(ase_atom, ngs, nmp=1):
    cell = pbcgto.Cell()
    cell.unit = 'A'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])

    #cell.verbose = 4
    cell.build()

    print "-- The Gamma point calculation -----------"
    mf = pbcdft.RKS(cell)
    mf.xc = 'lda,vwn'
    #mf.verbose = 7
    print mf.scf()

    print "-- The k-point sampling calculation ------"
    scaled_kpts = ase.dft.kpoints.monkhorst_pack((nmp,nmp,nmp))
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
    kmf = pbcdft.KRKS(cell, abs_kpts)
    kmf.analytic_int = False
    kmf.xc = 'lda,vwn'
    #kmf.verbose = 7
    print kmf.scf()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print 'usage: n' 
        sys.exit(1)
    n = int(args[0])

    # ------------------------------------------------#
    # -- HARTREE-FOCK --------------------------------#
    # ------------------------------------------------#
    # PRIMITIVE
    # Gamma point
    # n = 8  : converged SCF energy = 
    # n = 10 : converged SCF energy = 
    from ase.lattice import bulk
    ase_atom = bulk('C', 'diamond', a=3.5668)
    run_hf(ase_atom, n, 2)

    xxxx
    
    # CUBIC
    # Gamma point
    # n = 8  : converged SCF energy = -44.8952124954005 
    # n = 10 : converged SCF energy = -44.8811199335566
    # n = 14 : converged SCF energy = -44.8804889926916
    # (2x2x2) MP 
    # n = 8  : converged SCF energy = -45.4292039673842
    # n = 10 : converged SCF energy = -45.4166583035505
    # n = 14 : converged SCF energy = -45.4161732738524
    from ase.lattice.cubic import Diamond
    ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
    run_lda(ase_atom, n, 2)

    # PRIMITIVE
    # Gamma point
    # n = 8  : converged SCF energy = -10.2214263103746
    # n = 10 : converged SCF energy = -10.2213629403609
    # (2x2x2) MP
    # n = 8  : converged SCF energy = -11.3536435234899
    # n = 10 : converged SCF energy = -11.3536187856902
    from ase.lattice import bulk
    ase_atom = bulk('C', 'diamond', a=3.5668)
    run_lda(ase_atom, n, 2)

