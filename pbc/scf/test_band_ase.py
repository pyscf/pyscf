import numpy as np
import ase
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.scf.hf as hf
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft
import pyscf.pbc.scf.hf as pbchf
import pyscf.pbc.scf.kscf as pbckscf
import pyscf.pbc.scf.scfint as scfint

import ase.lattice
from ase.lattice.cubic import Diamond
import ase.dft.kpoints

import matplotlib.pyplot as plt

def test_band_ase():
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, get_bandpath
    c = bulk('C', 'diamond', a=3.5668)
    print c.get_volume()
    points = ibz_points['fcc']
    G = points['Gamma']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    band_kpts, x, X = get_bandpath([L, G, X, W, K, G], c.cell, npoints=30)

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(c)
    cell.h=c.cell

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs=np.array([5,5,5])
    cell.verbose=7
    cell.build(None,None)
    
    mf = pbcdft.RKS(cell)
    mf.analytic_int=False
    mf.xc = 'lda,vwn'
    print mf.scf()
 
    e_kn = []
    for kpt in band_kpts:
        fb, sb=mf.get_band_fock_ovlp(mf.get_hcore()+mf.get_veff(),
                                     mf.get_ovlp(), kpt)
        # fb, sb=mf.get_band_fock_ovlp(mf.get_hcore(),
        #                              mf.get_ovlp(), kpt)
        e, c=hf.eig(fb, sb)
        print kpt, e
        e_kn.append(e)
    
    emin = -1 
    emax = 2

    plt.figure(figsize=(5, 6))
    nbands = cell.nao_nr()
    for n in range(nbands):
        plt.plot(x, [e_kn[i][n] for i in range(len(x))])
    for p in X:
        plt.plot([p, p], [emin, emax], 'k-')
    plt.plot([0, X[-1]], [0, 0], 'k-')
    plt.xticks(X, ['$%s$' % n for n in ['L', 'G', 'X', 'W', 'K', r'\Gamma']])
    plt.axis(xmin=0, xmax=X[-1], ymin=emin, ymax=emax)
    plt.xlabel('k-vector')

    plt.show()


def test_band_ase_kpts():
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, get_bandpath
    c = bulk('C', 'diamond', a=3.5668)
    print c.get_volume()
    points = ibz_points['fcc']
    G = points['Gamma']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    band_kpts, x, X = get_bandpath([L, G, X, W, K, G], c.cell, npoints=30)

    cell = pbcgto.Cell()
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(c)
    cell.h=c.cell

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs=np.array([5,5,5])
    cell.verbose=7
    cell.build(None,None)
    
    scaled_kpts=ase.dft.kpoints.monkhorst_pack((1,1,1))
    abs_kpts=cell.get_abs_kpts(scaled_kpts)

    kmf = pbckscf.KRKS(cell, abs_kpts)
    kmf.analytic_int=False
    kmf.xc = 'lda,vwn'
    print kmf.scf()
 
    e_kn = []
    for kpt in band_kpts:
        fb, sb=kmf.get_band_fock_ovlp(kmf.get_hcore()+kmf.get_veff(),
                                      kmf.get_ovlp(), kpt)
        e, c=hf.eig(fb, sb)
        print kpt, e
        e_kn.append(e)
    
    emin = -1 
    emax = 2

    plt.figure(figsize=(5, 6))
    nbands = cell.nao_nr()
    for n in range(nbands):
        plt.plot(x, [e_kn[i][n] for i in range(len(x))])
    for p in X:
        plt.plot([p, p], [emin, emax], 'k-')
    plt.plot([0, X[-1]], [0, 0], 'k-')
    plt.xticks(X, ['$%s$' % n for n in ['L', 'G', 'X', 'W', 'K', r'\Gamma']])
    plt.axis(xmin=0, xmax=X[-1], ymin=emin, ymax=emax)
    plt.xlabel('k-vector')

    plt.show()



    # kmf=pbckscf.KRKS(cell)
    # kmf.xc='lda,vwn'
    # print mf.scf()
