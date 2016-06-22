import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft.rks as pbcrks
import pyscf.pbc.scf.kscf as kscf
import scipy.linalg

def test_kscf_gamma(atom, ncells):
    import numpy as np
    # import warnings

    cell = pbcgto.Cell()
    cell.unit = 'B'
    # As this is increased, we can see convergence between the molecule
    # and cell calculation
    Lunit = 2
    Ly = Lz = 2
    Lx = ncells*Lunit
    cell.h = np.diag([Lx,Ly,Lz])
    # place atom in middle of big box
    for i in range(ncells):
        cell.atom.extend([[atom, ((.5+i)*Lunit, 0.5*Ly, 0.5*Lz)]])
    cell.basis = { atom: [[0, (1.0, 1.0)]] }

    n = 40 
    cell.gs = np.array([n*ncells,n,n])
    cell.nimgs = [2,2,2]
    cell.verbose = 7
    cell.build()

    #warnings.simplefilter("error", np.ComplexWarning)

    kmf = pbcrks.RKS(cell)
    kmf.init_guess = "atom"
    return kmf.scf()

def test_kscf_kpoints(atom, ncells):
    import numpy as np
   # import warnings

    cell = pbcgto.Cell()
    cell.unit = 'B'
    Lunit = 2
    Ly = Lz = 2
    Lx = Lunit
    cell.h = np.diag([Lx,Ly,Lz])
    # place atom in middle of big box
    cell.atom.extend([[atom, (0.5*Lunit, 0.5*Ly, 0.5*Lz)]])
    cell.basis = { atom: [[0, (1.0, 1.0)]] }

    n = 40
    cell.gs = np.array([n,n,n])
    cell.nimgs = [2,2,2]
    cell.verbose = 7
    cell.build()

    # make Monkhorst-Pack (equally spaced k points along x direction)
    invhT = scipy.linalg.inv(np.asarray(cell._h).T)
    kGvs = []
    for i in range(ncells):
        kGvs.append(i*1./ncells*2*np.pi*np.dot(invhT,(1,0,0)))
    kpts = np.vstack(kGvs)

    kmf = kscf.KRKS(cell, kpts)
    kmf.init_guess = "atom"
    return kmf.scf()

def test_kscf_kgamma():
    # tests equivalence of gamma supercell and kpoints calculations

    emf_gamma = []
    emf_kpt = []

    # TODO: currently works only if atom has an even #of electrons
    # *only* reason is that Mole checks this against spin,
    # all other parts of the code
    # works so long as cell * nelectron is even, e.g. 4 unit cells of H atoms
    for ncell in range(1,3):
        emf_gamma.append(test_kscf_gamma("He", ncell)/ncell)
        emf_kpt.append(test_kscf_kpoints("He", ncell))
        print "COMPARISON", emf_gamma[-1], emf_kpt[-1] # should be the same up to integration error

    # each entry should be the same up to integration error (abt 5 d.p.)
    print "ALL ENERGIES, GAMMA", emf_gamma
    print "ALL ENERGIES, KPT", emf_kpt

if __name__ == '__main__':
    test_kscf_kgamma()
