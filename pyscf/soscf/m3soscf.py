'''

author: Linus Bjarne Dittmer

'''

import numpy
import scipy
import pyscf.scf
from pyscf.dft import uks
import pyscf.symm
import pyscf.soscf.newton_ah as newton_ah
import pyscf.soscf.sigma_utils as sigutils

from pyscf.lib import logger


class M3SOSCF:
    '''
    Attributes for M3SOSCF:
        mf: Instance of SCF class
            SCF Object that is to be converged.
        agents: int > 0
            Number of theoretically parallel agents that is to be used.
            Generally increases speed
        purge_solvers: float, optional
            Partition of solvers that is to be purged and each iteration.
            Between 0 and 1
        convergence: int
            Convergence Threshold for Trust is 10**-convergence
        init_scattering: float
            Initial statistical distribution of subconverger guesses. The
            original initial guess (e.g. minao, huckel, 1e, ...) is always
            conserved as the 0th guess, the rest is scattered around with
            uniform radius distribution and uniform angular distribution on
            a box
        trust_scale_range: float[3]
            Scaling array used for adaptive scaling of trust values. The 3
            floats correspond to minimum, maximum and power of the
            scaling calculation.
        init_guess: String or ndarray
            Initial Guess for the SCF iteration. Either one of the string
            aliases supported by pyscf.scf.SCF or an appropriately sized
            custom matrix can be used.
        stepSize: float
            Step size of the NR algorithm.

        Examples:

        >>> mol = gto.M('C 0.0 0.0 0.0; O 0.0 0.0 1.1')
        >>> mf = scf.RHF(mol)
        >>> agents = 5
        >>> m3 = M3SOSCF(agents)
        >>> m3.kernel()
    '''

    def __init__(self, mf, agents, purge_solvers=0.5, convergence=8,
                 init_scattering=0.3, trust_scale_range=(0.05, 0.5, 0.5),
                 init_guess='minao', stepsize=0.2):

        # SCF object, currently implemented: RHF, ROHF, UHF, RKS, ROKS, UKS
        self.mf = mf

        if self.method in ('uhf', 'uks'):
            # MO coeffs of each subconverger (agents,n,n).
            self.mo_coeffs = numpy.zeros((agents, 2, self.mf.mol.nao_nr(),
                                         self.mf.mol.nao_nr()))
        else:
            self.mo_coeffs = numpy.zeros((agents, self.mf.mol.nao_nr(),
                                         self.mf.mol.nao_nr()))
        # Number of subconvergers
        self.agents = agents
        # Array of current trusts (agents,)
        self.current_trusts = numpy.zeros(agents)
        # Array of current energies (agents,)
        self.current_energies = numpy.zeros(agents)
        # MO coeffs used for initalisation (n,n)
        self.mo_basis_coeff = numpy.zeros(self.mo_coeffs[0].shape)
        # Array of subconverger objects
        self.subconvergers = []
        # Manager for subconverger reassignment
        self.subconverger_rm = SubconvergerReassigmentManager(self)
        self.subconverger_rm.trust_scale_range = trust_scale_range
        # Initial scattering of Subconvergers
        self.init_scattering = init_scattering
        # Stepsize of parent Newton-Raphson solver
        self.nr_stepsize = stepsize

        # self.mo_cursor = 0
        # Maximum M3 cycles
        self.max_cycle = self.mf.max_cycle

        for _ in range(agents):
            self.subconvergers.append(Subconverger(self))

        # Reference array for finding solutions and trusts
        # subconverger_indices[(Converger)] =
        # (Index of Converger in current_trusts/mo_coeffs)
        self.subconverger_indices = numpy.arange(len(self.subconvergers))
        # Portion of purged subconvergers in each step
        self.purge_subconvergers = purge_solvers
        # Convergence threshold for trust
        self.convergence_thresh = 10**-convergence


        if not isinstance(init_guess, numpy.ndarray):
            self.init_dm_with_roothaan_step(self.mf.get_init_guess(key=init_guess))
        else:
            self.init_dm_directly(init_guess)

        # Debugging for turning off Differential Evolution
        self._ignore_de = False


    @property
    def method(self):
        if isinstance(self.mf, pyscf.dft.uks.UKS):
            return 'uks'
        if isinstance(self.mf, pyscf.dft.rks.KohnShamDFT):
            return 'rks'
        if isinstance(self.mf, pyscf.scf.uhf.UHF):
            return 'uhf'
        if isinstance(self.mf, pyscf.scf.rohf.ROHF):
            return 'rohf'
        if isinstance(self.mf, pyscf.scf.hf.RHF):
            return 'rhf'
        return None


    def get_degrees_of_freedom(self):
        '''
        Returns the number of Degrees Of Freedom: N(N-1)/2
        '''
        return int(0.5 * len(self.mo_basis_coeff[0]) * (len(self.mo_basis_coeff[0])-1))

    def init_dm_directly(self, idc):
        self.mo_basis_coeff = idc
        mo_pe = None
        if self.method in ('uks', 'uhf'):
            mo_pe = numpy.array((numpy.arange(len(idc[0])),
                                 numpy.arange(len(idc[0]))))
        else:
            mo_pe = numpy.array(numpy.arange(len(idc[0])))
        mo_occ = self.mf.get_occ(mo_pe, idc)
        self.mf.mo_occ = mo_occ

    def init_dm_with_roothaan_step(self, idm=None):
        '''
        Initialises the M3SOSCF-Solver with a given density matrix. One
        Rothaan step is performed afterwards to ensure DM properties.

        Arguments:
            idm: ndarray
                Density Matrix used for initialisation
        '''

        mf = self.mf
        it_num = 1
        mo_coeff = None
        if self.method == 'rohf':
            it_num = 2
        for _ in range(it_num):
            fock = mf.get_fock(dm=idm)
            mo_energy, mo_coeff = mf.eig(fock, mf.get_ovlp())
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            self.mf.mo_occ = mo_occ
            idm = mf.make_rdm1(mo_coeff, mo_occ)
        self.mo_basis_coeff = mo_coeff

    def set(self, purge_solvers=-1, convergence=-1, init_scattering=-1, trust_scale_range=None,
            mo_coeffs=None, stepsize=-1):
        if purge_solvers >= 0:
            self.purge_subconvergers = purge_solvers
        if convergence >= 0:
            self.convergence_thresh = 10**-convergence
        if init_scattering >= 0:
            self.init_scattering = init_scattering
        if isinstance(trust_scale_range, list):
            self.subconverger_rm.trust_scale_range = trust_scale_range
        if isinstance(mo_coeffs, numpy.ndarray):
            self.mo_basis_coeff = mo_coeffs
        if stepsize >= 0:
            self.nr_stepsize = stepsize

    def kernel(self, purge_solvers=0.5, convergence=8, init_scattering=0.1,
               trust_scale_range=(0.01, 0.2, 8), dm0=None):
        self.purge_subconvergers = purge_solvers
        self.convergence_thresh = 10**(-convergence)
        self.init_scattering = init_scattering
        self.subconverger_rm.trust_scale_range = trust_scale_range

        if isinstance(dm0, numpy.ndarray):
            self.init_dm_directly(dm0)
        elif dm0 is not None:
            raise TypeError('Illegal initial matrix: dm0 is not a '
                            'numpy.ndarray.')

        return self.converge()

    def converge(self):
        '''
        Starts the SCF procedure.

        Returns:
            scf_conv: boolean
                Whether the SCF managed to converge within the set amount of
                cycles to the given precision.
            final_energy: float
                Total SCF energy of the converged solution.
            final_mo_energy: ndarray
                Orbital energies of the converged MOs.
            final_mo_coeffs: ndarray
                MO coefficient matrix of the converged MOs.
            mo_occs: ndarray
                Absolute occupancies of the converged MOs.

        Examples:
        >>> mol = gto.M('H 0.0 0.0 0.0; F 0.0 0.0 1.0', basis='6-31g')
        >>> mf = scf.RHF(mol)
        >>> agents = 5
        >>> m3 = scf.M3SOSCF(mf, agents)
        >>> result = m3.converge()
        >>> print(result[1]) # Print SCF energy
        -99.9575044930158
        '''
        log = logger.new_logger(self.mf, self.mf.mol.verbose)
        cpu_timer0 = (logger.process_clock(), logger.perf_counter())

        if numpy.einsum('i->', self.mo_coeffs.flatten()) == 0:
            for sc in self.subconvergers:
                sc.mo_coeffs = self.mo_basis_coeff

        self.subconvergers[0].mo_coeffs = self.mo_basis_coeff
        self.mo_coeffs[0] = self.mo_basis_coeff

        cpu_timer0 = logger.timer(self.mf, 'First M3 Initialisation',
                                  *cpu_timer0)
        if self.agents >= 2 or self._ignore_de:
            for j in range(1 * (not self._ignore_de), self.agents):
                if self.method in ('uhf', 'uks'):
                    mo_pert_a = numpy.random.random(1)[0] * self.init_scattering * \
                            sigutils.vec_to_matrix(numpy.random.uniform(low=-0.5, high=0.5,
                            size=(self.get_degrees_of_freedom(),)))
                    mo_pert_b = numpy.random.random(1)[0] * self.init_scattering * \
                            sigutils.vec_to_matrix(numpy.random.uniform(low=-0.5, high=0.5,
                            size=(self.get_degrees_of_freedom(),)))
                    mo_coeffs_l = numpy.array((self.mo_basis_coeff[0]@scipy.linalg.expm(mo_pert_a),
                            self.mo_basis_coeff[1] @ scipy.linalg.expm(mo_pert_b)))
                    self.subconvergers[j].mo_coeffs = mo_coeffs_l
                    self.mo_coeffs[j] = mo_coeffs_l
                else:
                    mo_pert = numpy.random.random(1)[0] * self.init_scattering * \
                            sigutils.vec_to_matrix(numpy.random.uniform(low=-0.5, high=0.5,
                            size=(self.get_degrees_of_freedom(),)))
                    mo_coeffs_l = self.mo_basis_coeff @ scipy.linalg.expm(mo_pert)
                    self.subconvergers[j].mo_coeffs = mo_coeffs_l
                    self.mo_coeffs[j] = mo_coeffs_l

        cpu_timer0 = logger.timer(self.mf, 'Generate Initial MO Coeffs', *cpu_timer0)

        total_cycles = self.max_cycle
        final_energy = 0.0
        scf_conv = False
        final_mo_coeffs = None
        final_mo_energy = None

        s1e = self.mf.get_ovlp()
        h1e = self.mf.get_hcore()

        mo_occs = self.mf.mo_occ

        guess_energy = self.mf.energy_elec(self.mf.make_rdm1(self.mo_coeffs[0], mo_occs))[0]
        log.info("Guess energy: " + str(guess_energy))

        cpu_timer1 = logger.timer(self.mf, 'Second M3 Initialisation phase', *cpu_timer0)
        for cycle in range(self.max_cycle):

            # handle MO Coefficient cursor for I/O
            # edit subconverges according to solution density
            # a certain number of subconvergers get purged each iteration
            # purge = 0.3 - 0.8

            log.info("Iteration: " + str(cycle))

            purge_indices = None

            if cycle > 0 and len(self.subconvergers) > 1:
                sorted_indices = numpy.argsort(self.current_trusts)
                purge_indices = sorted_indices[0:int(min(int(len(sorted_indices) *
                                               self.purge_subconvergers), len(sorted_indices)))]
                uniqueindices = numpy.unique(self.current_trusts, return_index=True)[1]
                nonuniqueindices = []

                for i in range(self.agents):
                    if i not in uniqueindices:
                        nonuniqueindices.append(i)

                nui = numpy.array(nonuniqueindices, dtype=numpy.int32)
                zero_indices = numpy.where(self.current_trusts <= self.convergence_thresh)[0]
                purge_indices = numpy.unique(numpy.concatenate((purge_indices, nui, zero_indices)))
                purge_indices = numpy.sort(purge_indices)

                if purge_indices[0] == 0 and self.current_trusts[0] > 0.0:
                    purge_indices = purge_indices[1:]

                log.info(f"Purge Indices: {purge_indices}")
                log.info(f"Purging: {len(purge_indices)} / {len(self.subconvergers)}")
                new_shifts = self.subconverger_rm.gen_new_shifts(
                    self.current_trusts, self.mo_coeffs, len(purge_indices), log)

                for j, purge_index in enumerate(purge_indices):
                    self.mo_coeffs[purge_index] = new_shifts[j]
                    self.current_energies[purge_index] = numpy.finfo(dtype=numpy.float32).max


            for j, subconverger in enumerate(self.subconvergers):
                subconverger.mo_coeffs = self.mo_coeffs[self.subconverger_indices[j]]

            # generate local solutions and trusts

            # buffer array for new mocoeffs
            new_mo_coeffs = numpy.copy(self.mo_coeffs)

            sorted_trusts = numpy.zeros(1, dtype=numpy.int32)
            if len(self.subconvergers) > 1:
                sorted_trusts = numpy.argsort(self.current_trusts[1:]) + 1

            cpu_timer2 = logger.timer(self.mf, 'Purging MO Coefficients', *cpu_timer1)
            for j, subconverger in enumerate(self.subconvergers):

                sol, trust, etot = subconverger.get_local_sol_and_trust()

                #numpy.set_printoptions(linewidth=500, precision=2)
                log.info("J: " + str(j) + " Trust: " + str(trust))

                if trust == 0:
                    continue

                write_trust_index = 0
                if j > 0:
                    write_trust_index = sorted_trusts[j-1]

                # update trust and solution array

                mc_threshold = 1 - self.current_trusts[write_trust_index] + trust

                if j == 0 and self.current_trusts[j] > 0.0 or len(self.subconvergers) == 1:
                    self.current_trusts[j] = trust
                    new_mo_coeffs[j] = sol
                    self.subconverger_indices[j] = j


                elif numpy.random.rand(1) < mc_threshold:
                    self.current_trusts[write_trust_index] = trust
                    new_mo_coeffs[write_trust_index] = sol
                    self.current_energies[write_trust_index] = etot
                    self.subconverger_indices[j] = write_trust_index


                cpu_timer2 = logger.timer(self.mf, f'Solving for Local Sol and Trust, Iteration '
                                          f'{cycle}, Subconverger {j}', *cpu_timer2)

            # update moCoeff array with buffer
            self.mo_coeffs = numpy.copy(new_mo_coeffs)

            # check for convergence

            highest_trust_index = numpy.argmax(self.current_trusts)
            log.info("Highest Trust Index: " + str(highest_trust_index))
            log.info("Lowest Energy: " + str(numpy.min(self.current_energies)))
            log.info("Lowest Energy Index: " + str(numpy.argmin(self.current_energies)))
            # current energy (redundant)

            for j, ej in enumerate(self.current_energies):
                log.info("ENERGY (" + str(j) + "): " + str(ej))
            log.info("")


            scf_tconv =  1 - self.current_trusts[highest_trust_index]**4 < self.convergence_thresh
            current_energy = numpy.min(self.current_energies)
            log.info(f"Lowest Energy: {current_energy}")
            log.info(f"Current Highest Trust Energy: {self.current_energies[highest_trust_index]}")
            log.info(f"Energy Difference: "
                    f"{current_energy-self.current_energies[highest_trust_index]}")
            log.info(f"Convergence Thresh: {self.convergence_thresh}")
            if scf_tconv and current_energy - self.current_energies[highest_trust_index] \
                    < -self.convergence_thresh:
                del_array1 = numpy.where(self.current_energies >=
                        self.current_energies[highest_trust_index])[0]
                del_array2 = numpy.where(1 - self.current_trusts**4 < self.convergence_thresh)[0]
                log.info(f"Deletion Array 1 (Too High Energy): {del_array1}")
                log.info(f"Deletion Array 2 (Converged): {del_array2}")
                log.info(f"Intersected Deletion Array: {numpy.intersect1d(del_array1, del_array2)}")
                self.current_trusts[numpy.intersect1d(del_array1, del_array2)] = 0.0
                log.info("### DISREGARDING SOLUTION DUE TO NON VARIATIONALITY ###")
                scf_tconv = False

            log.info("Trust converged: " + str(scf_tconv))


            if scf_tconv:
                dm0 = self.mf.make_rdm1(self.mo_coeffs[highest_trust_index], self.mf.mo_occ)

                final_fock = self.mf.get_fock(dm=dm0, h1e=h1e, s1e=s1e)
                final_mo_coeffs = self.mo_coeffs[highest_trust_index]
                final_mo_energy = self.calcuate_orbital_energies(final_mo_coeffs, final_fock, s1e)
                final_energy = self.mf.energy_tot(dm0, h1e=h1e)
                total_cycles = cycle+1

                self.mf.mo_energy = final_mo_energy
                self.mf.mo_coeff = final_mo_coeffs
                self.mf.e_tot = final_energy
                self.mf.converged = True

                scf_conv = True
                break

            cpu_timer2 = logger.timer(self.mf, 'Checking convergence', *cpu_timer2)
            cpu_timer1 = logger.timer(self.mf, 'Total Cycle time', *cpu_timer1)

        log.info("Final Energy: " + str(final_energy) + " ha")
        log.info("Cycles: " + str(total_cycles))

        cpu_timer0 = logger.timer(self.mf, 'Total SCF time', *cpu_timer0)
        self.dump_info(log, total_cycles)

        return scf_conv, final_energy, final_mo_energy, final_mo_coeffs, mo_occs


    def calcuate_orbital_energies(self, mo_coefficients, fock, s1e):
        # Oribtal energies calculated from secular equation

        mo_energies = None

        if self.method in ('uhf', 'uks'):
            s1e_inv = numpy.linalg.inv(s1e)
            f_eff_a = s1e_inv @ fock[0]
            f_eff_b = s1e_inv @ fock[1]
            mo_energies_a = numpy.diag(numpy.linalg.inv(mo_coefficients[0]) @ f_eff_a
                                       @ mo_coefficients[0])
            mo_energies_b = numpy.diag(numpy.linalg.inv(mo_coefficients[1]) @ f_eff_b
                                       @ mo_coefficients[1])

            mo_energies = numpy.array((mo_energies_a, mo_energies_b))

        else:
            f_eff = numpy.linalg.inv(s1e) @ fock
            mo_energies = numpy.diag(numpy.linalg.inv(mo_coefficients) @ f_eff @ mo_coefficients)

        return mo_energies

    def dump_info(self, log, cycles):
        log.info("\n==== INFO DUMP ====\n")
        log.info(f"Number of Cycles:         {cycles}")
        log.info(f"Final Energy:             {self.mf.e_tot}")
        log.info(f"Converged:                {self.mf.converged}")

        aux_mol = pyscf.gto.M(
            atom=self.mf.mol.atom,
            basis=self.mf.mol.basis,
            spin=self.mf.mol.spin,
            charge=self.mf.mol.charge,
            symmetry=1
        )
        log.info(f"Point group:              {aux_mol.topgroup} (Supported: {aux_mol.groupname})")

        homo_index, lumo_index = None, None
        occs = numpy.where(self.mf.mo_occ[0, :] > 0.5)[0] if self.method in \
                           ['uhf', 'uks'] else numpy.where(self.mf.mo_occ > 0.5)[0]
        no_occs = numpy.where(self.mf.mo_occ[0, :] < 0.5)[0] if self.method in \
                           ['uhf', 'uks'] else numpy.where(self.mf.mo_occ < 0.5)[0]

        homo_index = (0, occs[numpy.argmax(self.mf.mo_energy[0, occs])]) if self.method in \
                      ['uhf', 'uks'] else occs[numpy.argmax(self.mf.mo_energy[occs])]
        lumo_index = (0, no_occs[numpy.argmin(self.mf.mo_energy[0, no_occs])]) if self.method in \
                      ['uhf', 'uks'] else no_occs[numpy.argmin(self.mf.mo_energy[no_occs])]

        log.info(f"HOMO Index:               {homo_index}")
        log.info(f"LUMO Index:               {lumo_index}")

        homo_energy = self.mf.mo_energy[homo_index] if self.method in ['uhf', 'uks'] \
                      else self.mf.mo_energy[homo_index]
        lumo_energy = self.mf.mo_energy[lumo_index] if self.method in ['uhf', 'uks'] \
                      else self.mf.mo_energy[lumo_index]

        log.info(f"HOMO Energy:              {homo_energy}")
        log.info(f"LUMO Energy:              {lumo_energy}")
        log.info(f"Aufbau solution:          {homo_energy < lumo_energy}")

        if self.method in ['uhf', 'uks']:
            ss = self.mf.spin_square()
            log.info(f"Spin-Square:              {ss[0]}")
            log.info(f"Multiplicity:             {ss[1]}")

        if not self.mf.converged:
            return

        irreps = ['-'] * len(self.mf.mo_coeff[0])
        forced_irreps = False
        symm_overlap = numpy.ones(len(self.mf.mo_coeff[0]))

        if self.method in ['uhf', 'uks']:
            irreps = [irreps, irreps]
            symm_overlap = [symm_overlap, symm_overlap]

        try:
            if self.method in ['uhf', 'uks']:
                irreps_a = pyscf.symm.addons.label_orb_symm(
                    aux_mol, aux_mol.irrep_name, aux_mol.symm_orb, self.mf.mo_coeff[0])
                irreps_b = pyscf.symm.addons.label_orb_symm(
                    aux_mol, aux_mol.irrep_name, aux_mol.symm_orb, self.mf.mo_coeff[1])
                if isinstance(irreps_a, numpy.ndarray) and isinstance(irreps_b, numpy.ndarray):
                    irreps = [irreps_a, irreps_b]
            else:
                irreps1 = pyscf.symm.addons.label_orb_symm(
                    aux_mol, aux_mol.irrep_name, aux_mol.symm_orb, self.mf.mo_coeff)
                if isinstance(irreps1, numpy.ndarray):
                    irreps = irreps1
        except Exception:
            if self.method in ['uhf', 'uks']:
                mo_coeff_symm_a = pyscf.symm.addons.symmetrize_orb(aux_mol, self.mf.mo_coeff[0])
                mo_coeff_symm_b = pyscf.symm.addons.symmetrize_orb(aux_mol, self.mf.mo_coeff[1])
                symm_overlap_a = numpy.diag(mo_coeff_symm_a.conj().T @ self.mf.get_ovlp()
                                            @ self.mf.mo_coeff[0])
                symm_overlap_b = numpy.diag(mo_coeff_symm_b.conj().T @ self.mf.get_ovlp()
                                            @ self.mf.mo_coeff[1])
                irreps_a = pyscf.symm.addons.label_orb_symm(
                    aux_mol, aux_mol.irrep_name, aux_mol.symm_orb, mo_coeff_symm_a)
                irreps_b = pyscf.symm.addons.label_orb_symm(
                    aux_mol, aux_mol.irrep_name, aux_mol.symm_orb, mo_coeff_symm_b)
                if isinstance(irreps_a, numpy.ndarray) or isinstance(irreps_b, numpy.ndarray):
                    irreps = [irreps_a, irreps_b]
                    forced_irreps = True
                    symm_overlap = numpy.array([symm_overlap_a, symm_overlap_b])
            else:
                mo_coeff_symm = pyscf.symm.addons.symmetrize_orb(aux_mol, self.mf.mo_coeff)
                symm_overlap = numpy.diag(mo_coeff_symm.conj().T @ self.mf.get_ovlp()
                                          @ self.mf.mo_coeff)
                irreps1 = pyscf.symm.addons.label_orb_symm(aux_mol, aux_mol.irrep_name,
                                                           aux_mol.symm_orb, mo_coeff_symm)
                if isinstance(irreps1, numpy.ndarray):
                    irreps = irreps1
                    forced_irreps = True

        log.info("\n\nORIBTAL SUMMARY:\n")
        log.info("Index:        Energy [ha]:                        Occupation:    Symmetry:")
        if self.method in ['uhf', 'uks']:
            for index, mo_e in enumerate(self.mf.mo_energy[0]):
                label = "A  " * (self.mf.mo_occ[0,index] > 0)
                s = f"{index} (A){(9 - len(str(index))) * ' '}{mo_e}{' ' * (36 - len(str(mo_e)))}"
                s += f"{label}"
                s += f"{' ' * (65 - len(s))}{irreps[0][index]}"
                s +=f"{(f' (FORCED, Overlap: {round(symm_overlap[0][index], 5)})') * forced_irreps}"
                log.info(s)

                mo_e = self.mf.mo_energy[1, index]
                label = "  B" * (self.mf.mo_occ[1,index] > 0)
                s = f"{index} (B){(9 - len(str(index))) * ' '}{mo_e}{' ' * (36 - len(str(mo_e)))}"
                s += f"{label}"
                s += f"{' ' * (65 - len(s))}{irreps[1][index]}"
                s +=f"{(f' (FORCED, Overlap: {round(symm_overlap[1][index], 5)})') * forced_irreps}"
                log.info(s)
        else:
            for index, mo_e in enumerate(self.mf.mo_energy):
                label = "A B" if self.mf.mo_occ[index] > 1 else "A" if self.mf.mo_occ[index] > 0 \
                        else ""
                s = f"{index}{(13 - len(str(index))) * ' '}{mo_e}{' ' * (36 - len(str(mo_e)))}"
                s += f"{label}"
                s += f"{' ' * (65 - len(s))}{irreps[index]}"
                s += f"{(f' (FORCED, Overlap: {round(symm_overlap[index], 5)})') * forced_irreps}"
                log.info(s)

###

# GENERAL SUBCONVERGER/SLAVE. FUNCTIONS AS AN EVALUATOR OF TRUST AND LOCAL SOL, FROM WHICH SOL DENSITY IS CONSTRUCTED

###


class Subconverger:
    '''
    Subconverger class used in M3SOSCF. This class calculates local solutions via a local NR step,
    which are then returned to the Master M3SOSCF class together with a trust value and the local
    gradient. Instances of this class should not be created in the code and interaction with the
    code herein should only be performed via the superlevel M3SOSCF instance.

    Arguments:
        m3: M3SOSCF
            Master M3SOSCF class
    '''


    def __init__(self, m3):
        self.m3 = m3
        self.mo_coeffs = None
        molv = self.m3.mf.mol.verbose
        self.m3.mf.mol.verbose = 0
        self.newton = newton_ah.newton(self.m3.mf)
        self.m3.mf.mol.verbose = molv
        self.newton.max_cycle = 1
        self.newton.verbose = 0
        self.trust_scale = 10**-4
        self.newton.max_stepsize = self.m3.nr_stepsize

    def get_local_sol_and_trust(self):
        '''
        This method is directly invoked by the master M3SOSCF. It solves the local NR step from
        the previously assigned base MO coefficients and returns the solution as well as a trust
        and the gradient.

        Arguments:
            h1e: ndarray, optional, default None
                Core Hamiltonian. Inclusion not necessary, but improves performance
            s1e: ndarray, optional, default None
                Overlap matrix. Inclusion not necessary, but improves performance

        Returns:
            sol: ndarray
                local solution as an anti-hermitian MO rotation matrix.
            egrad: ndarray
                local gradient calculated at the base MO coefficients and used in the NR step as a
                compressed vector that can be expanded to an anti-hermitian matrix via contraction
                with the canonical basis.
            trust: float
                Trust value of the local solution, always between 0 and 1. 0 indicates that the
                solution is infinitely far away and 1 indicates perfect convergence.
        '''

        old_dm = self.m3.mf.make_rdm1(self.mo_coeffs, self.m3.mf.mo_occ)
        esol, _, etot = self.solve_for_local_sol()
        new_dm = self.m3.mf.make_rdm1(esol, self.m3.mf.mo_occ)

        trust = self.get_trust(old_dm, new_dm)

        sol = esol
        return sol, trust, etot


    def solve_for_local_sol(self):
        '''
        This method indirectly solves the NR step. In the current implementation, this solution is
        exported to the xCIAH module intrinsic to PySCF. This is done by executing one SCF step
        with the CIAH module.

        Returns:
            localSol: ndarray
                Local Solution to the NR step.

        '''

        etot = self.newton.kernel(mo_coeff=numpy.copy(self.mo_coeffs), mo_occ=self.m3.mf.mo_occ)
        return self.newton.mo_coeff, self.newton.converged, etot

    def get_trust(self, dm0, dm1):
        '''
        Calculates the trust of a given solution from the old and new density matrix as well as
        the energy difference.

        Arguments:
            dm0: ndarray
                Old density matrix
            dm1: ndarray
                New density matrix
            converged: bool
                Whether the NR solver considers the NR iteration to be converged.

        Returns:
            trust: float
                The trust of the given solution.
        '''

        e1 = self.m3.mf.energy_elec(dm1)[0]
        e0 = self.m3.mf.energy_elec(dm0)[0]
        denergy = e1 - e0

        l = numpy.linalg.norm(dm1-dm0) * self.trust_scale
        return 1.0 / (l + 1.0)**2 * self.auxilliary_xi(denergy)


    def auxilliary_xi(self, x):
        '''
        Auxilliary function that is used to define the effects of energy difference on the trust.

        Arguments:
            x: float
                function input value (here: the difference in energy)

        Returns:
            y: float
                function output value (here: scalar effect on trust)
        '''
        if x <= 0:
            return 1
        return numpy.e**(-x)






###

# Manages Reassignment of Subconvergers via Trust densities during the SCF procedures

###



class SubconvergerReassigmentManager:
    '''
    This class regulates the reassignment of subconvergers after each iteration. If a subconverger
    is either redundant or gives a solution with a low trust, it is reassigned to another place in
    the electronic phase space where it generates more valuable local solutions.

    Arguments:
        m3: M3SOSCF
            master M3SOSCF instance
    '''


    def __init__(self, m3):
        self.m3 = m3
        self.trust_scale_range = (0.01, 0.2, 8)
        self.alpha = 1.0 / self.trust_scale_range[1]

    def gen_new_shifts(self, trusts, sols, total, log):
        '''
        This method is directly invoked by the master M3SOSCF class at the start of each SCF
        iteration to generate new, useful positions for the reassigned subconvergers.

        Arguments:
            trusts: ndarray
                Array of trusts of each subconverger. trusts[i] belongs to the same subconverger
                as sols[i]
            sols: ndarray
                Array of the current MO coefficents of each subconverger. sols[i] belongs to the
                same subconverger as trusts[i]
            total: int
                Total number of subconvergers to be reassigned, therefore total number of new
                shifts that need to be generated
            cursor: int
                Current position of the cursor in the solution buffer
            log: Logger
                Instance of Logger object

        Returns:
            shifts: ndarray
                Array of new MO coefficient for each subconverger that is to be reassigned. The
                shape of shifts is ( total, n, n )

        '''

        max_trust = numpy.max(trusts.flatten())
        self.alpha = 1.0 / ((self.trust_scale_range[1] - self.trust_scale_range[0]) *
                (1 - max_trust)**(self.trust_scale_range[2]) + self.trust_scale_range[0])
        log.info("Current Trust Scaling: " + str(self.alpha))
        trustmap = numpy.argsort(trusts)
        normed_trusts = trusts[trustmap] / sum(trusts)

        def inverse_cdf(x):
            for i, nt in enumerate(normed_trusts):
                if x < nt:
                    return i
                x -= nt
            return len(normed_trusts)-1

        selected_indices = numpy.zeros(total, dtype=numpy.int32)

        for i, _ in enumerate(selected_indices):
            rand = numpy.random.random(1)
            selected_indices[i] = trustmap[inverse_cdf(rand)]

        # generate Points in each trust region

        if self.m3.method in ('uhf', 'uks'):
            shifts = numpy.zeros((total, 2, len(self.m3.mo_basis_coeff[0]),
                                  len(self.m3.mo_basis_coeff[0])))
        else:
            shifts = numpy.zeros((total, len(self.m3.mo_basis_coeff[0]),
                                  len(self.m3.mo_basis_coeff[0])))

        for i, si in enumerate(selected_indices):
            if self.m3.method in ('uhf', 'uks'):
                p_a = self.gen_trust_region_points(trusts[si], 1)[0]
                p_b = self.gen_trust_region_points(trusts[si], 1)[0]
                shifts[i,0] = sols[si,0] @ scipy.linalg.expm(sigutils.vec_to_matrix(p_a))
                shifts[i,1] = sols[si,1] @ scipy.linalg.expm(sigutils.vec_to_matrix(p_b))
            else:
                p = self.gen_trust_region_points(trusts[si], 1)[0]
                shifts[i] = sols[si] @ scipy.linalg.expm(sigutils.vec_to_matrix(p))

        return shifts

    def gen_sphere_point(self):
        '''
        This method generates a single point on any dimensional sphere. See gen_sphere_points for
        more details.

        Returns:
            point: ndarray
                Random point on a sphere
        '''

        return self.gen_sphere_points(1)[0]

    def gen_sphere_points(self, num):
        '''
        Generate multiple random points on any dimensional sphere. This method utilises the
        spherical symmetry of the normal distribution by generating a point whose cartesian
        coordinates are normally distributed and subsequently normalising said point.

        Arguments:
            num: int
                number of random points to be generated

        Returns:
            points: ndarray
                array of random points on the sphere.
        '''
        points = numpy.random.normal(0.0, 1.0, size=(num, self.m3.get_degrees_of_freedom()))

        for i, p in enumerate(points):
            points[i] /= numpy.linalg.norm(p)

        return points

    def gen_trust_region_points(self, trust, num):
        '''
        Generates random points in a specific trust region.

        Arguments:
            trust: float
                The trust of the trust region, in which the points should be generated
            num: int
                The number of points to be generated

        Returns:
            points: ndarray
                The random points generated
        '''
        def inverse_cdf(x):
            # Exponential Distribution
            return numpy.log(1-x) / (- self.alpha * trust)
            # Gaussian Distribution
            #return scipy.special.erfinv(x) / (self.alpha * trust)


        radii = inverse_cdf(numpy.random.rand(num))

        spoints = self.gen_sphere_points(num)
        dpoints = numpy.zeros(spoints.shape)

        for i, spoint in enumerate(spoints):
            dpoints[i] = spoint * radii[i]

        return dpoints

