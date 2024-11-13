import time

import numpy as np

from gpaw import GPAW, PW
from gpaw.utilities import unpack_hermitian

from pyscf.pbc.scf.khf import KRHF

import pyscf.pbc.gto as pbcgto
from pyscf.pbc.tools import pyscf_ase
from pyscf import __config__
from pyscf.scf import hf as mol_hf
from pyscf import lib
from pyscf.pbc.scf import hf as pbchf

from ase.lattice.cubic import SimpleCubic
from ase.dft.kpoints import monkhorst_pack

from functools import partial

def init_gpaw_calc(system, kpts, nbands, e_cut=350, name='He'):
    calc = GPAW(mode=PW(e_cut), kpts=kpts, nbands=nbands,
                txt=f'gpaw-{name}.txt')

    calc.atoms = system.copy()

    # set up the objects needed for a calculation: Density, Hamiltonian, WaveFunctions, Setups
    calc.initialize(calc.atoms)

    # Update the positions of the atoms and initialize wave functions, density
    calc.set_positions(calc.atoms)

    return calc

def apply_overlap(wfs, u, calculate_P_ani=True, psit_nG=None):
    """
    ***Adapated from gpaw.overlap.py***
    Apply the overlap operator to a wave function (specified by
    the basis and the expansion coefficient).

    Parameters
    ==========
    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    calculate_P_ani: bool
        When True, the integrals of projector times vectors
        P_ani = <p_ai | psit_nG> for a specific u are calculated.
        When False, existing P_ani are used
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """
    # psi_t at u-th kpoint, u is the combined spin and kpoint index
    kpt = wfs.kpt_u[u]

    # expansion coefficient
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG

    # b_xG is the resulting expansion coefficient after applying S
    Spsit_nG = np.copy(psit_nG)

    # (taken from GPAW)
    # random initialization of a dictionary with signature
    # {atom: array of len(projectors)}
    shape = psit_nG.shape[0]
    P_ani = wfs.pt.dict(shape)

    if calculate_P_ani:
        # the original function does not update P_ani
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
    else:
        for a, P_ni in kpt.P_ani.items():
            P_ani[a][:] = P_ni

    for a, P_ni in P_ani.items():
        P_ani[a] = np.dot(P_ni, wfs.setups[a].dO_ii)
        # gemm(1.0, wfs.setups[a].dO_ii, P_xi, 0.0, P_xi, 'n')
    wfs.pt.add(Spsit_nG, P_ani, kpt.q)  # b_xG += sum_ai pt^a_i P_ani

    return Spsit_nG

def apply_pseudo_hamiltonian(wfs, u, ham, psit_nG=None):
    """Apply the pseudo Hamiltonian (without PAW correction) to
    wavefunction (specified by the basis and the expansion coefficient).

    Parameters:

    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    ham: Hamiltonian
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """

    kpt = wfs.kpt_u[u]
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG

    Htpsit_nG = np.zeros_like(psit_nG)

    # # compare gpaw and my implementation
    # psit = kpt.psit.new(buf=psit_nG)
    # tmp = psit.new(buf=wfs.work_array)
    # H = wfs.work_matrix_nn
    # P2 = kpt.projections.new()
    # Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)
    # psit.matrix_elements(operator=Ht, result=tmp, out=H,
    #                      symmetric=True, cc=True)

    # this function can only be used if we use GPAW's wfs
    # maybe we don't want to do that...
    wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, Htpsit_nG)

    return Htpsit_nG

def apply_PAW_correction(wfs, u, ham, calculate_P_ani=True, psit_nG=None):
    """
    Apply PAW correction to the wavefunction.

    Parameters
    ==========
    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    ham: Hamiltonian
    calculate_P_ani: bool
        When True, the integrals of projector times vectors
        P_ani = <p_ai | psit_nG> for the specific u are calculated.
        When False, existing P_ani are used
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """
    # psi_t at u-th kpoint, u is the combined spin and kpoint index
    kpt = wfs.kpt_u[u]

    # expansion coefficient
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG

    # b_xG is the resulting expansion coefficient after applying dH
    dHpsit_nG = np.zeros_like(psit_nG)

    # (taken from GPAW)
    # random initialization of a dictionary with signature
    # {atom: array of len(projectors)}
    shape = psit_nG.shape[0]
    P_ani = wfs.pt.dict(shape)

    if calculate_P_ani:  # TODO calculate_P_ani=False is experimental
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
    else:
        for a, P_ni in kpt.P_ani.items():
            P_ani[a][:] = P_ni

    for a, P_ni in P_ani.items():
        dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
        P_ani[a] = np.dot(P_ni, dH_ii)
    wfs.pt.add(dHpsit_nG, P_ani, kpt.q)

    return dHpsit_nG

def Ht_example(wfs, u, ham, psit_nG = None, new=False, scalewithocc=True):
    '''
    Some example I found in GPAW of applying PAW hamiltonian to PW wfs:
    H |psit_nG>. I have verified that this function give the same result
    as the functions I implemented
    '''
    kpt = wfs.kpt_u[u]
    psit_nG = psit_nG if psit_nG is not None else kpt.psit_nG

    nbands = wfs.bd.mynbands
    Hpsi_nG = wfs.empty(nbands, q=kpt.q)
    wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, Hpsi_nG)

    c_axi = {}
    if new:
        dH_asii = ham.potential.dH_asii
        for a, P_xi in kpt.P_ani.items():
            dH_ii = dH_asii[a][kpt.s]
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi
    else:
        for a, P_xi in kpt.P_ani.items():
            dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

    # not sure about this:
    # ham.xc.add_correction(
    #     kpt, kpt.psit_nG, Hpsi_nG, kpt.P_ani, c_axi, n_x=None,
    #     calculate_change=False)
    # add projectors to the H|psi_i>

    wfs.pt.add(Hpsi_nG, c_axi, kpt.q)
    # # scale with occupation numbers
    # if scalewithocc:
    #     for i, f in enumerate(kpt.f_n):
    #         Hpsi_nG[i] *= f
    return psit_nG, Hpsi_nG


def apply_to_single_u(ham, wfs, u):
    '''
    Apply the Hamiltonian and Overlap to wavefunction for a 
    single (spin, k) point.
    '''
    Spsit_nG = apply_overlap(wfs, u)

    Htpsit_nG = apply_pseudo_hamiltonian(wfs, u, ham)
    dHpsit_nG = apply_PAW_correction(wfs, u, ham)

    return Htpsit_nG + dHpsit_nG, Spsit_nG

def main():
    # Initialize ASE atom
    ase_atom=SimpleCubic(symbol='He', latticeconstant=8)
    print(ase_atom.get_volume())

    # Initialize PySCF cell
    cell = pbcgto.Cell()
    cell.verbose = 5
    cell.atom=pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a=ase_atom.cell
    cell.basis = 'gth-tzvp'     # TODO: import smooth Gaussian basis from DOI: 10.1039/d0cp05229a
    cell.pseudo = 'gth-pade'    # TODO: should not need this with PAW
    cell.build()

    # Initialize kpoints
    nk = [1,1,1]  # gamma point only for now...
    kpts = cell.make_kpts(nk)

    # TODO: PySCF kpoint is in 1/Bohr, need to convert to fractional
    # coords wrt reciprocal lattice vector before passing into GPAW
    # maybe the other way around is better: generate fractional kpts
    # in gpaw; then convert and pass into pyscf.

    # initialize GPAW calculator
    calc = init_gpaw_calc(ase_atom, kpts, cell.nao, e_cut=350)

    pawkrhf = PAWKRHF(cell, calc, kpts)
    # pawkrhf.get_h_matrix()

    # psit_nG = pawkrhf.gto2pw[0].T.copy()
    # Htpsitng = apply_pseudo_hamiltonian(calc.wfs, 0, calc.hamiltonian, psit_nG=psit_nG)
    # dHpsitng = apply_PAW_correction(calc.wfs, 0, calc.hamiltonian, psit_nG=psit_nG, calculate_P_ani=False)
    # psit_nG2, Htpsitng1 = Ht_example(calc.wfs, 0, calc.hamiltonian, psit_nG=psit_nG)
    # h = pawkrhf.get_h_matrix()
    print(pawkrhf.kernel())
    print('Hooray')


class PAWKRHF(KRHF):
    def __init__(self, cell, calc,
                 kpts=np.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        super().__init__(cell, kpts, exxdiv)
        self.calc = calc    # GPAW calculator
        self.cell = cell
        
        # construct the basis transformation matrix from GTO to PW
        # self.expand_GTO_in_PW()
        t1 = time.time()
        self.expand_GTO_in_PW_by_fft()
        t2 = time.time()
        # self.expand_GTO_in_PW_by_grid_int()
        # t3 = time.time()

        print(f'FFT takes {t2-t1} seconds')
        # print(f'Grid int takes {t3-t2} seconds')


    def expand_GTO_in_PW_by_grid_int(self):
        '''
        Calculate the expansion coefficient of each GTO in
        the auxillary PW basis; outputs a matrix of size
        (n_PW, n_GTO)
        '''
        mesh = self.calc.wfs.gd.N_c
        coords = self.cell.get_uniform_grids(mesh)

        # list of (nx*ny*nz, nao) arrays of length nkpts
        GTOs = self.cell.pbc_eval_gto("GTOval_sph", coords, kpts=self.kpts)

        self.gto2pw_grid = list()
        for k in range(len(self.kpts)):
            pd = self.calc.wfs.pd

            # get all the reciprocal vectors for the plane wave basis
            # i.e. (k+G) for each e^（-i(k+G)*r）, this is a (n_PW, 3) vector
            kplusG = pd.get_reciprocal_vectors(k, add_q=True)

            # e^-i(k+G)*r: an (n_PW, nx*ny*nz) array
            expmikGr = np.exp(-1j*(kplusG @ coords.T))

            # get each all GTOs for some k-point on a real sapce grid
            # representation; this should give (nx*ny*nz, nao) array
            GTO = GTOs[k]

            # integrate and normalize
            assert(expmikGr.shape[1] == GTO.shape[0])
            ngrid = expmikGr.shape[1]
            self.gto2pw_grid.append(expmikGr @ GTO / ngrid)

    def expand_GTO_in_PW_by_fft(self):
        '''
        Calculate the expansion coefficient of each GTO in
        the auxillary PW basis using FFT; outputs a matrix of size
        (n_PW, n_GTO)
        '''
        # TODO: figure out beyond gamma point (multiple kpts) case.
        
        mesh = self.calc.wfs.gd.N_c
        coords = self.cell.get_uniform_grids(mesh)
        GTOs = self.cell.pbc_eval_gto("GTOval_sph", coords, kpts=self.kpts)

        self.gto2pw = list()
        for k in range(len(self.kpts)):
            npw, nao = self.calc.wfs.ng_k[k], self.cell.nao
            gto2pw_k = np.zeros((npw, nao), dtype=np.complex128)
            for j in range(nao):
                ao_1d = GTOs[k][:, j]
                ao_3d = ao_1d.reshape(mesh)
                gto2pw_k[:, j] = self.calc.wfs.pd.fft(ao_3d, q=k) / ao_1d.shape[0]
            self.gto2pw.append(gto2pw_k)

    # def GTO2PW(coeff, mat): 
    #     '''
    #     Takes coefficient under GTO basis and map to coefficient
    #     under the plane wave basis (n_GTO, 1)->(n_PW, 1)
    #     '''
    #     return mat @ coeff

    def get_h_matrix(self):
        '''
        This should return (kpts, nao, nao) size array
        '''
        nkpts = len(self.kpts)
        nao = self.cell.nao
        calc = self.calc
        h = np.zeros((nkpts, nao, nao), dtype=np.complex128)

        for k in range(nkpts):
            psit_nG = self.gto2pw[k].T.copy()
            Htpsit_nG = apply_pseudo_hamiltonian(calc.wfs, k, calc.hamiltonian, psit_nG=psit_nG).T
            dHpsit_nG = apply_PAW_correction(calc.wfs, k, calc.hamiltonian, psit_nG=psit_nG, calculate_P_ani=True).T
            Hpsit_nG = Htpsit_nG + dHpsit_nG    # H |psit_nG>
            h[k, :, :] = self.gto2pw[k].conj().T @ Hpsit_nG
            # for j in range(nao):
            #     psit_nG = self.gto2pw[k].T.copy # j-th GTO in PW representation
            #     Htpsit_nG = apply_pseudo_hamiltonian(calc.wfs, k, calc.hamiltonian, psit_nG=ket)
            #     dHpsit_nG = apply_PAW_correction(calc.wfs, k, calc.hamiltonian, psit_nG=ket)
            #     Hpsit_nG = Htpsit_nG + dHpsit_nG    # H |psi_j>
            #     for i in range(nao):
            #         bra = self.gto2pw[k][:, i].conj()
            #         h[k, i, j] = bra.dot(Hpsit_nG)     # <psi_i| H | psi_j>

        return h
    
    def update_calc(self, wfs, ham, dens):
        # get occ and mo_coeff
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
        assert(len(self.mo_coeff) == len(self.mo_occ))

        # TODO: update wfs using occ and mo_coeff
        # self.mo_coeff is (kpt, nao, nao) list of 2D array
        # self.mo_occ is (kpt, nao) list of 1D array
        for k in len(self.mo_coeff):
            mo_coeff_pw = self.gto2pw @ self.mo_coeff[k]
            ### TODO: make this work...
            psit = kpt.psit.new(buf=psit_nG)
            wfs.kpt_u[k].psit.array = mo_coeff_pw.T # this won't work
            ####
            wfs.kpt_u[k].f_n = self.mo_occ[k] # aliasing?

        # update dens and ham
        dens.update(wfs)
        ham.update(dens)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                fock_last=None):
        
        h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
        # Parts to modify
        #################
        # if h1e_kpts is None: h1e_kpts = mf.get_hcore()
        # if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
        # f_kpts = h1e_kpts + vhf_kpts
        # TODO: update calc hamiltonian
        calc = self.calc
        if cycle > 0:
            self.update_calc(calc.wfs, calc.ham, calc.dens)
        f_kpts = self.get_h_matrix()
        #################

        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f_kpts

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        if s_kpts is None: s_kpts = self.get_ovlp()
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            f_kpts = [mol_hf.damping(f, f_prev, damp_factor) for f,f_prev in zip(f_kpts,fock_last)]
        if diis and cycle >= diis_start_cycle:
            f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, self, h1e_kpts, vhf_kpts, f_prev=fock_last)
        if abs(level_shift_factor) > 1e-4:
            f_kpts = [mol_hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                    for k, s in enumerate(s_kpts)]
        return lib.asarray(f_kpts)

    def get_ovlp(self, cell=None, kpts=None):
        '''Get the overlap AO matrices at sampled k-points.

        Args:
            kpts : (nkpts, 3) ndarray

        Returns:
            ovlp_kpts : (nkpts, nao, nao) ndarray
        '''
        calc = self.calc
        nao = self.cell.nao
        nkpts = len(self.kpts)
        result = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        
        for k in range(nkpts):
            # the copy is somehow necessary to avoid bug when doing integral in gpaw
            psit_nG = self.gto2pw[k].T.copy()
            Spsit_nG = apply_overlap(calc.wfs, k, psit_nG=psit_nG).T
            result[k, :, :] = self.gto2pw[k].conj().T @ Spsit_nG
            # for j in range(nao):
            #     ket = self.gto2pw[k][:,j] # j-th GTO in PW representation
            #     Spsit_nG = apply_overlap(calc.wfs, k, psit_nG=ket)
            #     for i in range(nao):
            #         bra = self.gto2pw[:, i].conj()
            #         result[k, i, j] = bra.dot(Spsit_nG)
        
        return result


if __name__ == '__main__':
    main()