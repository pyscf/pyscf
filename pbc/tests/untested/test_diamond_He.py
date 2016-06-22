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

def test_cubic_diamond_He():
    """
    Take ASE Diamond structure, input into PySCF and run
    """
    ase_atom=Diamond(symbol='He', latticeconstant=3.5)
    print ase_atom.get_volume()

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h=ase_atom.cell
    cell.basis = {"He" : [[0, (1.0, 1.0)], [0, (0.8, 1.0)]] }

    cell.gs = np.array([15,15,15])
    # cell.verbose = 7
    cell.build(None, None)
    mf = pbcdft.RKS(cell)
    mf.analytic_int=False
    mf.xc = 'lda,vwn'
    print mf.scf()
    # Diamond cubic: -18.2278622592408

    mf = pbcdft.RKS(cell)
    mf.analytic_int=True
    mf.xc = 'lda,vwn'
    print mf.scf()
    # Diamond cubic (analytic): -18.2278622592283
    #                         = -4.556965564807075 / cell              

    # K pt calc
    scaled_kpts=ase.dft.kpoints.monkhorst_pack((2,2,2))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)

    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=False
    kmf.xc = 'lda,vwn'
    print kmf.scf()
    # Diamond cubic (2x2x2): -18.1970946252
    #                      = -4.5492736563 / cell

def test_diamond_He():
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, get_bandpath
    He = bulk('He', 'diamond', a=3.5)

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(He)
    cell.h=He.cell

    cell.basis = {"He" : [[0, (1.0, 1.0)], [0, (0.8, 1.0)]] }

    # cell.pseudo = 'gth-pade'
    cell.gs=np.array([15,15,15])
    cell.verbose=7
    cell.build(None,None)

    # replicate cell NxNxN
    repcell = pyscf.pbc.tools.replicate_cell(cell, (2,2,2))
    mf = pbcdft.RKS(repcell)
    mf.analytic_int=False
    mf.xc = 'lda,vwn'
    # 1x1x1 Gamma -4.59565988176
    # 2x2x2 Gamma [10x10x10 grid] -36.4239485658445
    #                          =  -4.552993570730562 / cell          
    #             [20x20x20 grid] -4.55312928715 / cell
    #             
    # 3x3x3 Gamma [15x15x15 grid] -4.553523556740741 / cell
    # Diamond cubic: -4.56518345190625
    
    scaled_kpts=ase.dft.kpoints.monkhorst_pack((3,3,3))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)

    kmf = pyscf.pbc.scf.kscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=False
    kmf.xc = 'lda,vwn'
    kmf.scf()
    # 2x2x2 K-pt: [15x15x15] grid -4.54824938484
    # 3x3x3 K-pt: [15x15x15] grid -4.55034210881
    #
    # See test_pyscf_He (for diamond Cubic)
    # Diamond cubic (2x2x2): -18.1970946252
    #                      = -4.5492736563 / cell
