import sys
import numpy as np
import matplotlib.pyplot as plt

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbchf
import pyscf.pbc.dft as pbcdft

import ase
import ase.dft.kpoints

ANG2BOHR = 1.889725989

def plot_bands(scftype, basis, ngs, nmp=None):
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
    cell.basis = 'gth-%s'%(basis)
    #cell.basis = 'gth-szv'
    #cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.gs = np.array([ngs,ngs,ngs])
    cell.verbose = 7
    cell.build(None,None)

    # Perform the gamma-point SCF
    if scftype == 'dft':
        mf = pbcdft.RKS(cell)
        mf.xc = 'lda,vwn'
    elif scftype == 'hf':
        mf = pbchf.RHF(cell)
    else: 
        scaled_mp_kpts = ase.dft.kpoints.monkhorst_pack((nmp,nmp,nmp))
        abs_mp_kpts = cell.get_abs_kpts(scaled_mp_kpts)
        if scftype == 'kdft':
            mf = pbcdft.KRKS(cell, abs_mp_kpts)
            mf.xc = 'lda,vwn'
        else:
            mf = pbchf.KRHF(cell, abs_mp_kpts)

    mf.analytic_int = False
    mf.scf()
    print "SCF evals =", mf.mo_energy

    # Proceed along k-point band-path
    e_kn = []
    efermi = -99
    for kpt in abs_kpts:
        e, c = mf.get_bands(kpt)
        print kpt, e
        e_kn.append(e)
        if e[4-1] > efermi:
            efermi = e[4-1]
    for k, ek in enumerate(e_kn):
        e_kn[k] = ek-efermi

    # Write the bands to stdout 
    for kk, ek in zip(x, e_kn):
        print "%0.6f "%(kk),
        for ekn in ek:
            print "%0.6f "%(ekn),
        print ""

    # Plot the band structure via matplotlib
    emin = -1.0
    emax = 1.0
    plt.figure(figsize=(8, 4))
    nbands = cell.nao_nr()
    for n in range(nbands):
        plt.plot(x, [e_kn[i][n] for i in range(len(x))])
    for p in X:
        plt.plot([p, p], [emin, emax], 'k-')
    plt.plot([0, X[-1]], [0, 0], 'k-')
    plt.xticks(X, ['$%s$' % n for n in ['L', r'\Gamma', 'X', 'W', 'K', r'\Gamma']])
    plt.axis(xmin=0, xmax=X[-1], ymin=emin, ymax=emax)
    plt.xlabel('k-vector')
    plt.ylabel('Energy [au]')

    #plt.show()
    if nmp is None:
        plt.savefig('bands_%s_%s_%d.png'%(scftype,basis,ngs))
    else:
        plt.savefig('bands_%s_%s_%d_%d.png'%(scftype,basis,ngs,nmp))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 3 and len(args) != 4:
        print 'usage: diamond_bands.py (k)dft/hf basis ngs nmp' 
        sys.exit(1)
    scftype = args[0]
    assert scftype in ['dft','hf', 'kdft', 'khf']
    basis = args[1]
    assert basis in ['szv', 'dzvp']
    ngs = int(args[2])
    if len(args) == 4:
        nmp = int(args[3])
        plot_bands(scftype, basis, ngs, nmp)
    else:
        plot_bands(scftype, basis, ngs)

