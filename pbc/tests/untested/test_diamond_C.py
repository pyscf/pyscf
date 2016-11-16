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

def test_cubic_diamond_C():
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
    print "Cell volume =", ase_atom.get_volume()

    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell.T
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"

    cell.gs = np.array([10,10,10])
    # cell.verbose = 7
    cell.build(None, None)
    mf = pbcdft.RKS(cell)
    mf.analytic_int = False
    mf.xc = 'lda,vwn'
    print mf.scf()

    # Gamma point gs: 10x10x10: -11.220279983393
    #             gs: 14x14x14: -11.220122248175
    # K pt calc
    scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))
    abs_kpts = cell.get_abs_kpts(scaled_kpts)

    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int = False
    kmf.xc = 'lda,vwn'
    kmf.verbose = 7
    print kmf.scf()
    # Diamond cubic (Kpt 2x2x2) gs: 10x10x10: -11.354164575875
    #                           gs: 14x14x14: -11.354043318198075
    
def test_diamond_C():
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, get_bandpath
    C = bulk('C', 'diamond', a=3.5668)

    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(C)
    cell.h = C.cell.T

    # cell.basis = 'gth-tzvp'
    cell.basis = 'gth-szv'
    # cell.basis = {'C': [[0, (4.3362376436, 0.1490797872), (1.2881838513, -0.0292640031), (0.4037767149, -0.688204051), (0.1187877657, -0.3964426906)], [1, (4.3362376436, -0.0878123619), (1.2881838513, -0.27755603), (0.4037767149, -0.4712295093), (0.1187877657, -0.4058039291)]]}
    # Easier basis for quick testing
    #cell.basis = {'C': [[0, (1.2881838513, -0.0292640031), (0.4037767149, -0.688204051)], [1, (1.2881838513, -0.27755603), (0.4037767149, -0.4712295093) ]]}

    # Cell used for K-points
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([8,8,8])
    cell.verbose = 7
    cell.build(None,None)

    # Replicate cell NxNxN for comparison
    # repcell = pyscf.pbc.tools.replicate_cell(cell, (2,2,2))
    # repcell.gs = np.array([12,12,12]) # for 3 replicated cells, then
    # #                                 # ngs must be of the form [3gs0 + 1, gs1, gs2]
    # repcell.build()
    # # # Replicated MF calc
    # mf = pbcdft.RKS(repcell)
    # # mf.analytic_int = True
    # mf.xc = 'lda,vwn'
    # mf.init_guess = '1e'
    # mf.diis = True
    # mf.scf()
    
    # K-pt calc
    scaled_kpts = ase.dft.kpoints.monkhorst_pack((2,2,2))

    # shift if 2x2x2 includes Gamma point
    # shift = np.array([1./4., 1./4., 1./4.])
    # scaled_kpts += shift
    abs_kpts = cell.get_abs_kpts(scaled_kpts)
    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int = False
    kmf.diis = True
    kmf.init_guess = '1e'
    kmf.xc = 'lda,vwn'
    print "N imgs", cell.nimgs
    print "Cutoff for 400 eV", pyscf.pbc.tools.cutoff_to_gs(cell.lattice_vectors(), 400/27.2)

    #kmf.max_cycle = 3
    kmf.scf() 

    # Default sets nimg = [5,5,5] for this cell
    #
    # 2x2x2 replicated 12x12x12 gs, szv: -11.241426956044675 (*8)
    #
    # 1x1x1 Kpt  8x8x8 gs, szv  : -10.2214263103132
    #            8x8x8 gs, dzvp : -10.3171863686858
    #            analytic       : -10.3171890900077
    #            8x8x8 gs, tzvp : -10.3310789041567 (ovlp singular warning)
    #            analytic       :  ovlp not positive definite
    # 2x2x2 Kpt, 8x8x8 gs, szv : -11.3536437382296 (16 atoms)
    #            8x8x8 gs dzvp : -11.4183859541816
    #         14x14x14 gs, szv: -11.353612722046 (16 atoms)
    # 2x2x2 shift Kpt (includes Gamma) : -11.240852789145 (agrees with rep. cell) 
    # 3x3x3 Kpt, 8x8x8 gs, szv: -11.337727688022  (54 atoms) 
    # 4x4x4 Kpt, 8x8x8 gs, szv: -11.3565363884127 (128 atoms)
