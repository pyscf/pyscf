from __future__ import division
import numpy as np
try:
    import gpaw
    print(gpaw.__version__)
except:
    raise ValueError("GPAW need to be installed to use the gpaw input!")
from gpaw.io import Reader
from gpaw import restart


class gpaw_reader():
    """
    GPAW reader class. Read DFT input from the GPAW LCAO calculator.

    example:
    from ase import Atoms
    from gpaw import PoissonSolver, GPAW

    # Sodium dimer
    atoms = Atoms('Na2', positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    atoms.center(vacuum=3.5)

    # Increase accuragy of density for ground state
    convergence = {'density': 1e-7}

    # Increase accuracy of Poisson Solver and apply multipole corrections up to l=1
    poissonsolver = PoissonSolver(eps=1e-14, remove_moment=1 + 3)

    # nbands must be equal to norbs (in this case 10)
    calc = GPAW(setups={'Na': '1'}, basis='dzp', xc='LDA', h=0.3, nbands=10,
            convergence=convergence, poissonsolver=poissonsolver,
                    mode='lcao')

    atoms.set_calculator(calc)

    # Relax the ground state
    atoms.get_potential_energy()

    # write DFT output
    calc.write('Na2.gpw', mode='all')
    """

    def __init__(self, filename):

        self.atoms, self.calc = restart(filename)
        reader = Reader(filename)
        print(reader.keys())

        self.Read_WaveFunctions(reader.wave_functions)
        self.dataset_path = gpaw.setup_paths[0]

    #
    #
    #
    def Read_WaveFunctions(self, wf):
        self.nspin, self.nkpoints, self.nbands, self.norbs = wf.coefficients.shape
        if self.nbands != self.norbs:
            raise ValueError("nbands != norbs")
        self.nreim = 2

        self.orb2atm = np.zeros((self.norbs), dtype=np.int)
        self.orb2ao = np.zeros((self.norbs), dtype=np.int)
        self.orb2n = np.zeros((self.norbs), dtype=np.int)
        self.orb2strspecie = []
        self.orb2strsym = []
        self.ksn2e = wf.eigenvalues
        self.k2xyz = wf.kpts.ibzkpts
        self.X = np.empty((self.nreim,self.norbs,self.norbs,self.nspin,self.nkpoints), dtype='float32')

        strsym  = ['s', ['Ppy', 'Ppz', 'Ppx']] # need to add higher orders
        ni = 0
        nf = 0

        for ia, atm in enumerate(self.atoms):
            ni = nf
            setup = self.calc.wfs.setups[ia]
            setup.l_j.sort()
            setup.n_j.sort()
            for l, n in zip(setup.l_j, setup.n_j):
                nf += 2 * l + 1
                self.orb2strsym.extend(strsym[l])
            self.orb2atm[ni:nf] = ia + 1
            self.orb2ao[ni:nf] = np.arange(1, nf-ni + 1, dtype=np.int)
            self.orb2n[ni:nf] = np.ones((nf-ni), dtype=np.int)*np.max(setup.n_j)
            self.orb2strspecie.extend([atm.symbol]*(nf-ni))

        self.sp2strspecie = list(set(self.atoms.get_chemical_symbols()))

        for spin in range(self.nspin):
            for kpt in range(self.nkpoints):
                self.X[0, :, :, spin, kpt] = wf.coefficients[spin, kpt, :, :].real
                self.X[1, :, :, spin, kpt] = wf.coefficients[spin, kpt, :, :].imag
