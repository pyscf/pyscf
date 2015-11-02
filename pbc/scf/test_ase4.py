import scipy
import numpy as np
import ase
import pyscf.pbc.tools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf.kscf

import ase.lattice
from ase.lattice import bulk
from ase.lattice.cubic import Diamond
import ase.dft.kpoints

pi=np.pi

def test():

    ANALYTIC=False
    ase_atom=bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h=ase_atom.cell
    cell.verbose = 7

    # gth-szv for C
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'

    cell.gs = np.array([4,1,1])
    cell.build(None, None)

    print "N images: ", cell.nimgs
    cell.nimgs = [10,1,1]
    repcell = pyscf.pbc.tools.replicate_cell(cell, (3,1,1))
    print "REPCELL", repcell.pseudo

    mf = pbcdft.RKS(repcell)
    mf.verbose = 5
    mf.diis = False
    mf.analytic_int=ANALYTIC
    mf.xc = 'lda,vwn'
    mf.max_cycle = 1
    mf.init_guess = '1e'
    mf.scf()

    scaled_kpts=np.array(ase.dft.kpoints.monkhorst_pack((3,1,1)))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)

    cell.gs = np.array([1,1,1])
    cell.nimgs = [14,1,1]
    cell.build(None, None)

    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=ANALYTIC
    kmf.diis = False
    kmf.verbose = 5
    kmf.xc = 'lda,vwn'
    kmf.max_cycle = 1
    kmf.init_guess = '1e'
    kmf.scf()
