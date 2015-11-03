import scipy
import numpy as np
import ase
import pyscf.pbc.tools
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf.kscf

import ase.lattice
from ase.lattice.cubic import Diamond
import ase.dft.kpoints

pi=np.pi

def test_cubic_diamond_C():
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    ase_atom=Diamond(symbol='C', latticeconstant=3.5668)
    print ase_atom.get_volume()

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h=ase_atom.cell
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"

    cell.gs = np.array([6,6,6])
    # cell.verbose = 7
    cell.build(None, None)
    mf = pbcdft.RKS(cell)
    mf.analytic_int=False
    mf.xc = 'lda,vwn'
    print mf.scf()
    # Diamond cubic: -45.0015722647692

    # mf = pbcdft.RKS(cell)
    # mf.analytic_int=True
    # mf.xc = 'lda,vwn'
    # print mf.scf()
    # Diamond cubic (analytic): -44.9501094353535

    # K pt calc
    scaled_kpts=ase.dft.kpoints.monkhorst_pack((2,2,2))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)

    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=False
    kmf.xc = 'lda,vwn'
    kmf.verbose = 7
    print kmf.scf()
    # Diamond cubic (2x2x2): -18.1970946252
    #                      = -4.5492736563 / cell

def test_diamond_C():
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, get_bandpath
    C = bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(C)
    cell.h=C.cell

    #cell.basis = 'gth-szv'
    # cell.basis = {'C': [[0, (4.3362376436, 0.1490797872), (1.2881838513, -0.0292640031), (0.4037767149, -0.688204051), (0.1187877657, -0.3964426906)], [1, (4.3362376436, -0.0878123619), (1.2881838513, -0.27755603), (0.4037767149, -0.4712295093), (0.1187877657, -0.4058039291)]]}
    # Easier basis for quick testing
    cell.basis = {'C': [[0, (1.2881838513, -0.0292640031), (0.4037767149, -0.688204051)], [1, (1.2881838513, -0.27755603), (0.4037767149, -0.4712295093) ]]}

    # Cell used for K-points
    cell.pseudo = 'gth-pade'
    # cell.pseudo = 'gth-pade'
    cell.gs=np.array([1,1,1])
    cell.nimgs = [4,4,4]
    cell.verbose=7
    cell.build(None,None)

    # Replicate cell NxNxN for comparison
    repcell = pyscf.pbc.tools.replicate_cell(cell, (3,1,1))
    repcell.gs = np.array([4,1,1]) # for 3 replicated cells, then
                                   # ngs must be of the form [3gs0 + 1, gs1, gs2]

    repcell.nimgs = [4,2,2]
    repcell.build()

    # Replicated MF calc
    mf = pbcdft.RKS(repcell)
    mf.analytic_int=True
    mf.xc = 'lda,vwn'
    mf.max_cycle = 3
    mf.init_guess = '1e'
    mf.diis = True # when turned off, this agrees with k-pt calc below precisely
    mf.scf()
    
    # K-pt calc
    scaled_kpts=ase.dft.kpoints.monkhorst_pack((3,1,1))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)
    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=True
    kmf.diis=True # when turned off, this agrees with above replicated cell precisely
    kmf.init_guess = '1e'
    kmf.xc = 'lda,vwn'
    kmf.max_cycle = 3
    kmf.scf() 
