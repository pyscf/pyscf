#!/usr/bin/env python
#
# Author: George Booth <george.booth24@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import numpy
import pyscf.tools
import pyscf.lib.logger as logger
import pyscf.ao2mo
import pyscf.symm
import pyscf.symm.param as param
from subprocess import call

try:
    from pyscf.fciqmcscf import settings
except ImportError:
    msg = '''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py')
    sys.stderr.write(msg)

try:
    import settings
except ImportError:
    import os, sys
    msg = '''settings.py not found.  Please create %s
''' % os.path.join(os.path.dirname(__file__), 'settings.py')
    sys.stderr.write(msg)

IRREP_MAP = {'D2h': (1,         # Ag
                     4,         # B1g
                     6,         # B2g
                     7,         # B3g
                     8,         # Au
                     5,         # B1u
                     3,         # B2u
                     2),        # B3u
             'C2v': (1,         # A1
                     4,         # A2
                     2,         # B1
                     3),        # B2
             'C2h': (1,         # Ag
                     4,         # Bg
                     2,         # Au
                     3),        # Bu
             'D2' : (1,         # A
                     4,         # B1
                     3,         # B2
                     2),        # B3
             'Cs' : (1,         # A'
                     2),        # A"
             'C2' : (1,         # A
                     2),        # B
             'Ci' : (1,         # Ag
                     2),        # Au
             'C1' : (1,)}

class FCIQMCCI(object):
    def __init__(self, mol):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout

        self.executable = settings.FCIQMCEXE
        # Shouldn't need scratch dir settings.BLOCKSCRATCHDIR
        self.scratchDirectory = ''

        self.integralFile = "FCIDUMP"
        self.configFile = "neci.inp"
        self.outputFileRoot = "neci.out"
        self.outputFileCurrent = self.outputFileRoot
        self.maxwalkers = 10000
        self.maxIter = -1
        self.RDMSamples = 5000
        self.restart = False
        self.time = 10
        self.tau = -1.0
        self.seed = 7
        self.AddtoInit = 3
        self.orbsym = []
        self.state_weights = [1.0]
        # This is the number of orbitals to freeze in the neci calculation.
        # Note that if you do this for a CASSCF calculation, it will freeze in the active space.
        self.nfreezecore = 0
        self.nfreezevirt = 0
        if mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info('******** FCIQMC options ********')
        log.info('Number of walkers = %s', self.maxwalkers)
        log.info('Maximum number of iterations = %d', self.maxIter)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        nstates = len(self.state_weights)

        #Normalize the state_weights vector to conserve number of electrons
        norm = sum(self.state_weights)

        two_pdm = numpy.zeros( (norb, norb, norb, norb) )

        for irdm in range(nstates):
            if self.state_weights[irdm] != 0.0:
                dm_filename = 'spinfree_TwoRDM.' + str(irdm+1)
                temp_dm = read_neci_two_pdm(self, dm_filename, norb, self.scratchDirectory)
                two_pdm += (self.state_weights[irdm]/norm)*temp_dm

        one_pdm = one_from_two_pdm(two_pdm, nelectrons)

        return one_pdm, two_pdm

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec, link_index, **kwargs)[0]

    def kernel(self, h1e, eri, norb, nelec, fci_restart=None, **kwargs):
        if fci_restart is None:
            fci_restart = self.restart
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else:
            neleca, nelecb = nelec

        write_integrals_file(h1e, eri, norb, neleca, nelecb, self)
        write_fciqmc_config_file(self, neleca, nelecb, fci_restart)
        if self.verbose >= logger.DEBUG1:
            # os.path.join(self.scratchDirectory,self.configFile)
            in_file = self.configFile
            logger.debug1(self, 'FCIQMC Input file')
            logger.debug1(self, open(in_file, 'r').read())
        execute_fciqmc(self)
        if self.verbose >= logger.DEBUG1:
            # os.path.join(self.scratchDirectory,self.outputFile)
            out_file = self.outputFileCurrent
            logger.debug1(self, open(out_file))
        rdm_energy = read_energy(self)

        return rdm_energy, None

def run_standalone(fciqmcci, mo_coeff, restart = None):
    '''Run a neci calculation standalone for the molecule listed in the
    FCIQMCCI object. The basis to run this calculation in is given by the
    mo_coeff array.
    '''
    
    tol = 1e-9
    nmo = mo_coeff.shape[1]
    nelec = fciqmcci.mol.nelectron
    fciqmcci.dump_flags(verbose=5)

    with open(fciqmcci.integralFile, 'w') as fout:
        if fciqmcci.mol.symmetry:
            if fciqmcci.groupname == 'Dooh':
                logger.info(fciqmcci, 'Lower symmetry from Dooh to D2h')
                raise RuntimeError('''Lower symmetry from Dooh to D2h''')
            elif fciqmcci.groupname == 'Coov':
                logger.info(fciqmcci, 'Lower symmetry from Coov to C2v')
                raise RuntimeError('''Lower symmetry from Coov to C2v''')
            else:
                # We need the AO basis overlap matrix to calculate the
                # symmetries.
                s = fciqmcci.mol.intor_symmetric('cint1e_ovlp_sph')
                fciqmcci.orbsym = pyscf.symm.label_orb_symm(fciqmcci.mol, 
                        fciqmcci.mol.irrep_name, fciqmcci.mol.symm_orb,
                        mo_coeff, s=s)
                orbsym = [param.IRREP_ID_TABLE[fciqmcci.groupname][i]+1 for
                          i in fciqmcci.orbsym]
                pyscf.tools.fcidump.write_head(fout, nmo, nelec,
                                               fciqmcci.mol.spin, orbsym)
        else:
            pyscf.tools.fcidump.write_head(fout, nmo, nelec, fciqmcci.mol.spin)

        eri = pyscf.ao2mo.outcore.full_iofree(fciqmcci.mol, mo_coeff, verbose=0)
        pyscf.tools.fcidump.write_eri(fout, pyscf.ao2mo.restore(8,eri,nmo),
                                      nmo, tol=tol)

        # Lookup and return the relevant 1-electron integrals, and print out
        # the FCIDUMP file.
        t = fciqmcci.mol.intor_symmetric('cint1e_kin_sph')
        v = fciqmcci.mol.intor_symmetric('cint1e_nuc_sph')
        h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))
        pyscf.tools.fcidump.write_hcore(fout, h, nmo, tol=tol)
        fout.write(' %.16g  0  0  0  0\n' % fciqmcci.mol.energy_nuc())

    # The number of alpha and beta electrons.
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else:
        neleca, nelecb = nelec

    write_fciqmc_config_file(fciqmcci, neleca, nelecb, restart)

    if fciqmcci.verbose >= logger.DEBUG1:
        # os.path.join(self.scratchDirectory,self.configFile)
        in_file = fciqmcci.configFile
        logger.debug1(fciqmcci, 'FCIQMC Input file')
        logger.debug1(fciqmcci, open(in_file, 'r').read())

    execute_fciqmc(fciqmcci)

    if fciqmcci.verbose >= logger.DEBUG1:
        # os.path.join(self.scratchDirectory,self.outputFile)
        out_file = fciqmcci.outputFileCurrent
        logger.debug1(fciqmcci, open(out_file))

    rdm_energy = read_energy(fciqmcci)

    return rdm_energy


def write_fciqmc_config_file(fciqmcci, neleca, nelecb, restart):
    config_file = fciqmcci.configFile
    nstates = len(fciqmcci.state_weights)

    f = open(config_file, 'w')

    f.write('title\n')
    f.write('\n')
    f.write('system read noorder\n')
    f.write('symignoreenergies\n')
    f.write('freeformat\n')
    f.write('electrons %d\n' % (neleca+nelecb))
    f.write('nonuniformrandexcits 4ind-weighted\n')
    f.write('hphf 0\n')
    f.write('nobrillouintheorem\n')
    if nstates > 1:
        f.write('system-replicas %d\n' % (2*nstates))
    f.write('endsys\n')
    f.write('\n')
    f.write('calc\n')
    f.write('methods\n')
    f.write('method vertex fcimc\n')
    f.write('endmethods\n')
    f.write('time %d\n' % fciqmcci.time)
    f.write('memoryfacpart 2.0\n')
    f.write('memoryfacspawn 1.0\n')
    f.write('totalwalkers %d\n' % fciqmcci.maxwalkers)
    f.write('nmcyc %d\n' % fciqmcci.maxIter)
    f.write('seed %d\n' % fciqmcci.seed)
    if (restart):
        f.write('readpops')
    else:
        f.write('startsinglepart 500\n')
        f.write('diagshift 0.1\n')
    f.write('rdmsamplingiters %d\n' % fciqmcci.RDMSamples)
    f.write('shiftdamp 0.05\n')
    if (fciqmcci.tau != -1.0):
        f.write('tau 0.01\n')
    f.write('truncinitiator\n')
    f.write('addtoinitiator %d\n' % fciqmcci.AddtoInit)
    f.write('allrealcoeff\n')
    f.write('realspawncutoff 0.4\n')
    f.write('semi-stochastic\n')
    #f.write('cas-core 6 6\n')
    f.write('mp1-core 1000\n')
    #f.write('fci-core\n')
    #f.write('trial-wavefunction 5\n')
    f.write('jump-shift\n')
    f.write('proje-changeref 1.5\n')
    f.write('stepsshift 10\n')
    f.write('maxwalkerbloom 3\n')
    if nstates > 1:
        f.write('orthogonalise-replicas\n')
        f.write('doubles-init\n')
        f.write('multi-ref-shift\n')
    f.write('endcalc\n')
    f.write('\n')
    f.write('integral\n')
    f.write('freeze {},{}\n'.format(fciqmcci.nfreezecore, fciqmcci.nfreezevirt))
    f.write('endint\n')
    f.write('\n')
    f.write('logging\n')
    f.write('popsfiletimer 60.0\n')
    f.write('binarypops\n')
    f.write('calcrdmonfly 3 200 500\n')
    f.write('write-spin-free-rdm\n') 
    f.write('endlog\n')
    f.write('end\n')

    f.close()
    #no reorder
    #f.write('noreorder\n')


def write_integrals_file(h1eff, eri_cas, ncas, neleca, nelecb, fciqmcci):
    integralFile = os.path.join(fciqmcci.scratchDirectory,fciqmcci.integralFile)
    # Ensure 4-fold symmetry.
    eri_cas = pyscf.ao2mo.restore(4, eri_cas, ncas)
    if fciqmcci.mol.symmetry and fciqmcci.orbsym:
        orbsym = [IRREP_MAP[fciqmcci.groupname][i] for i in fciqmcci.orbsym]
    else:
        orbsym = []
    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                       neleca+nelecb, ms=abs(neleca-nelecb),
                                       orbsym=orbsym, tol=1e-10)


def execute_fciqmc(fciqmcci):
    in_file = os.path.join(fciqmcci.scratchDirectory, fciqmcci.configFile)
    outfiletmp = fciqmcci.outputFileRoot
    files = os.listdir(fciqmcci.scratchDirectory + '.')
    i = 1
    while outfiletmp in files:
        outfiletmp = fciqmcci.outputFileRoot + '_{}'.format(i)
        i += 1
    logger.info(fciqmcci, 'FCIQMC output file: %s', outfiletmp)
    fciqmcci.outputFileCurrent = outfiletmp
    out_file = os.path.join(fciqmcci.scratchDirectory, outfiletmp)

    if fciqmcci.executable == 'external':
        logger.info(fciqmcci, 'External FCIQMC calculation requested from '
                              'dumped integrals.')
        logger.info(fciqmcci, 'Waiting for density matrices and output file '
                              'to be returned.')
        try:
            raw_input("Press Enter to continue with calculation...")
        except:
            input("Press Enter to continue with calculation...")
    else:
        call("%s  %s > %s" % (fciqmcci.executable, in_file, out_file), shell=True)


def read_energy(fciqmcci):
    out_file = open(os.path.join(fciqmcci.scratchDirectory,
                 fciqmcci.outputFileCurrent), "r")

    for line in out_file:
        # Lookup the RDM energy from the output.
        if "*TOTAL ENERGY* CALCULATED USING THE" in line:
            rdm_energy = float(line.split()[-1])
            break
    logger.info(fciqmcci, 'Total energy from FCIQMC: %.15f', rdm_energy)
    out_file.close()

    return rdm_energy

def read_neci_one_pdm(fciqmcci, filename, norb, nelec, directory='.'):
    '''Obtain the spin-free 1RDM from neci by reading in the spin free 2RDM. 
    If core orbitals have been indicated as frozen in neci, this core contribution 
    will be explicitly added back in to the RDM. Therefore, the norb parameter 
    should be the total number of orbitals passed to neci (inc. frozen), while 
    nelec is the total number of electrons (inc. frozen), but not inactive if running
    through CASSCF.
    '''
    two_pdm = read_neci_two_pdm(fciqmcci, filename, norb, directory)
    return one_from_two_pdm(two_pdm, nelec)


def read_neci_two_pdm(fciqmcci, filename, norb, directory='.'):
    '''Read in a spin-free 2RDM from neci. Note that the RDMs in neci are stored
    in chemical notation, so that RDM_ijkl = < a^+_i a^+_k a_l a_j >. In pyscf,
    the 2RDM_ijkl = < a^+_i a^+_j a_l a_k >. If core orbitals have been
    indicated as frozen in neci, this core contribution will be explicitly
    added back in to the RDM. Therefore, the norb parameter should be the
    unfrozen number of orbitals passed to neci, but not inactive if running
    through CASSCF.
    '''

    f = open(os.path.join(directory, filename), 'r')

    # It is necessary to zero the array in case we have frozen virtual orbitals.
    two_pdm = numpy.zeros( (norb, norb, norb, norb) )
    for line in f.readlines():
        linesp = line.split()

        if(int(linesp[0]) != -1):
            # Arrays from neci are '1' indexed
            ind1 = int(linesp[0]) - 1 + fciqmcci.nfreezecore
            ind2 = int(linesp[2]) - 1 + fciqmcci.nfreezecore
            ind3 = int(linesp[1]) - 1 + fciqmcci.nfreezecore
            ind4 = int(linesp[3]) - 1 + fciqmcci.nfreezecore
            assert(int(ind1) < norb - fciqmcci.nfreezevirt)
            assert(int(ind2) < norb - fciqmcci.nfreezevirt)
            assert(int(ind3) < norb - fciqmcci.nfreezevirt)
            assert(int(ind4) < norb - fciqmcci.nfreezevirt)

            two_pdm[ind1, ind2 , ind3, ind4] = float(linesp[4])

    f.close()

    # Add on frozen core contribution, assuming that the core orbitals are
    # doubly occupied
    for i in range(fciqmcci.nfreezecore):
        for j in range(fciqmcci.nfreezecore):
            two_pdm(i,j,i,j) = 1.0
            two_pdm(i,j,j,i) = -1.0

    return two_pdm


def one_from_two_pdm(two_pdm, nelec):
    one_pdm = numpy.einsum('ikjj->ik', two_pdm)
    one_pdm /= (nelec-1)
    return one_pdm


def calc_dipole(mol, mo_coeff, one_pdm, ncore=0):
    '''Calculate the dipole moment, given the molecule, the mo coefficent
    array, the 1RDM. Optionally, also specify that the first ncore orbitals
    are doubly occupied. This will be required if you are taking an RDM from
    a CASSCF calculation, where the core contribution from inactive orbitals 
    has not been included explicitly. Note that if core orbitals are just 
    frozen with fciqmcci.nfreezecore, then this is already included.
    '''

    assert(one_pdm.shape[0] == one_pdm.shape[1])
    norb = mo_coeff.shape[1]
    nsizerdm = one_pdm.shape[0]

    logger.info('Calculating dipole moments of molecule')
    logger.info('Dimension of passed in density matrix: {} x {}'.   \
            format(nsizerdm,nsizerdm))
    logger.info('Number of orbitals: {}'.format(norb))
    logger.info('Number of doubly occupied orbitals not in density matrix: {}'. \
            format(ncore))
    
    # Add core first
    one_pdm_ = numpy.zeros( (norb, norb) )
    for i in range(ncore):
        one_pdm_(i,i) = 1.0

    # Add the rest of the density matrix
    one_pdm_[ncore:ncore+nsizerdm,ncore:ncore+nsizerdm] = one_pdm[:,:]

    # Call the integral generator for r integrals in the AO basis. There
    # are 3 dimensions for x, y and z components.
    aodmints = mol.intor('cint1e_r_sph', comp=3)
    # modmints will hold the MO transformed integrals.
    modmints = numpy.empty_like(aodmints)
    # For each component, transform integrals into the MO basis.
    for i in range(aodmints.shape[0]):
        modmints[i] = reduce(numpy.dot, (mo_coeff.T, aodmints[i], mo_coeff))

    # Contract with MO r integrals for electronic contribution.
    elec_dipmom = []
    for i in range(modmints.shape[0]):
        elec_dipmom.append( -numpy.trace( numpy.dot( one_pdm_, modmints[i])) )

    # Nuclear contribution.
    nuc_dipmom = [0.0, 0.0, 0.0]
    for i in range(mol.natm):
        for j in range(aodmints.shape[0]):
            nuc_dipmom[j] += mol.atom_charge(i)*mol.atom_coord(i)[j]

    tot_dipmom = [a+b for (a,b) in zip(elec_dipmom, nuc_dipmom)]

    return tot_dipmom, elec_dipmom, nuc_dipmom

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    from pyscf.tools import molden

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = None, #'out-fciqmc',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True, 
# fciqmc cannot handle Dooh currently, so reduce the point group if full group is infinite.
        symmetry_subgroup = 'D2h',
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    mc.fcisolver = FCIQMCCI(mol)
    mc.fcisolver.tau = 0.01
    mc.fcisolver.RDMSamples = 1000 
    mc.max_cycle_macro = 10
    mc.natorb = True    #Return natural orbitals from mc2step in casscf_mo
    emc_1, e_ci, fcivec, casscf_mo = mc.mc2step(m.mo_coeff)

# Write orbitals to molden output
    with open( 'output.molden', 'w' ) as fout:
        molden.header(mol, fout)
        molden.orbital_coeff(mol, fout, casscf_mo)

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver =  FCIQMCCI(mol)
    mc.fcisolver.tau = 0.01
    mc.fcisolver.RDMSamples = 1000
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = None, #'out-casscf',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True, 
        symmetry_subgroup = 'D2h',
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('FCIQMCCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('FCIQMCSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))

