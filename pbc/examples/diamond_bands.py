import numpy as np
import matplotlib.pyplot as plt

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.scf.hf as hf
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

ANG2BOHR = 1.889725989

def plot_bands():
    # Set-up the unit cell
    from ase.lattice import bulk
    from ase.dft.kpoints import ibz_points, kpoint_convert, get_bandpath
    ase_atom = bulk('C', 'diamond', a=3.5668*ANG2BOHR)
    print "Cell volume =", ase_atom.get_volume(), "Bohr^3"

    # Set-up the band-path via special points
    points = ibz_points['fcc']
    G = points['Gamma']
    X = points['X']
    W = points['W']
    K = points['K']
    L = points['L']
    band_kpts, x, X = get_bandpath([L, G, X, W, K, G], ase_atom.cell, npoints=30)
    abs_kpts = kpoint_convert(ase_atom.cell, skpts_kc=band_kpts) 

    # Build the cell
    cell = pbcgto.Cell()
    cell.unit = 'B'
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.h = ase_atom.cell
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([5,5,5])
    cell.verbose = 7
    cell.build(None,None)

    # Perform the gamma-point SCF
    mf = pbcdft.RKS(cell)
    mf.analytic_int = False
    mf.xc = 'lda,vwn'
    mf.scf()

    # Proceed along k-point band-path
    e_kn = []
    for kpt in abs_kpts:
        fb, sb = mf.get_band_fock_ovlp(band_kpt=kpt)
        e, c = hf.eig(fb, sb)
        print kpt, e
        e_kn.append(e)

    # Write the bands to stdout 
    #for kk, ek in zip(x, e_kn):
    #    print "%0.6f "%(kk),
    #    for ekn in ek:
    #        print "%0.6f "%(ekn),
    #    print ""

    # Plot the band structure via matplotlib
    emin = -0.4 
    emax = 1.4
    plt.figure(figsize=(8, 4))
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

if __name__ == '__main__':
    plot_bands()

