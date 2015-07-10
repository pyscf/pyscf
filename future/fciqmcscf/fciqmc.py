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
        self.nstates = 1
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

    def make_rdm12(self, rdm_label, fcivec, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0

        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        rdm_filename = 'spinfree_TwoRDM.' + str(rdm_label)
        f = open(os.path.join(self.scratchDirectory, rdm_filename), 'r')

        two_pdm = numpy.zeros( (norb, norb, norb, norb) )
        for line in f.readlines():
            linesp = line.split()

            if(int(linesp[0]) != -1):

                assert(int(linesp[0]) <= norb)
                assert(int(linesp[1]) <= norb)
                assert(int(linesp[2]) <= norb)
                assert(int(linesp[3]) <= norb)

                ind1 = int(linesp[0])-1
                ind2 = int(linesp[2])-1
                ind3 = int(linesp[1])-1
                ind4 = int(linesp[3])-1

                two_pdm[ind1, ind2 , ind3, ind4] = float(linesp[4])

        one_pdm = numpy.einsum('ikjj->ik', two_pdm)
        one_pdm /= (nelectrons-1)

        return one_pdm, two_pdm

    def make_rdm1(self, rdm_label, fcivec, norb, nelec, link_index=None, **kwargs):
        return self.make_rdm12(rdm_label, fcivec, norb, nelec, link_index, **kwargs)[0]

    def dipoles(self, rdm_label, mo_coeff, fcivec, norb, nelec, link_index=None):

        # Call the integral generator for r integrals in the AO basis. There
        # are 3 dimensions for x, y and z components.
        aodmints = self.mol.intor('cint1e_r_sph', comp=3)
        # modmints holds the MO transformed integrals.
        modmints = numpy.empty_like(aodmints)
        # For each component, transform integrals into the MO basis.
        for i in range(aodmints.shape[0]):
            modmints[i] = reduce(numpy.dot, (mo_coeff.T, aodmints[i], mo_coeff))

        # Obtain 1-RDM from NECI.
        one_pdm = self.make_rdm1(rdm_label, fcivec, norb, nelec, link_index)

        # Contract with MO r integrals for electronic contribution.
        dipmom = []
        for i in range(modmints.shape[0]):
            dipmom.append( -numpy.trace( numpy.dot( one_pdm, modmints[i])) )
        
        elec_comp_str = 'Electronic component to dipole moment: %.15g %.15g %.15g'
        logger.info(self, elec_comp_str, dipmom[0], dipmom[1], dipmom[2])

        # Nuclear contribution.
        for i in range(self.mol.natm):
            for j in range(aodmints.shape[0]):
                dipmom[j] += self.mol.atom_charge(i)*self.mol.atom_coord(i)[j]

        full_str = 'Full dipole moment: %.15g %.15g %.15g'
        logger.info(self, full_str, dipmom[0], dipmom[1], dipmom[2])

        return dipmom

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

def run_standalone(qmc_obj, mo_coeff, restart = None):
    '''Run a neci calculation standalone for the molecule listed in the
    FCIQMCCI object. The basis to run this calculation in is given by the
    mo_coeff array.
    '''
    
    tol = 1e-9
    nmo = mo_coeff.shape[1]
    nelec = qmc_obj.mol.nelectron
    qmc_obj.dump_flags(verbose=5)

    with open(qmc_obj.integralFile, 'w') as fout:
        if qmc_obj.mol.symmetry:
            if qmc_obj.groupname == 'Dooh':
                logger.info(qmc_obj, 'Lower symmetry from Dooh to D2h')
                raise RuntimeError('''Lower symmetry from Dooh to D2h''')
            elif qmc_obj.groupname == 'Coov':
                logger.info(qmc_obj, 'Lower symmetry from Coov to C2v')
                raise RuntimeError('''Lower symmetry from Coov to C2v''')
            else:
                # We need the AO basis overlap matrix to calculate the
                # symmetries.
                s = qmc_obj.mol.intor_symmetric('cint1e_ovlp_sph')
                qmc_obj.orbsym = pyscf.symm.label_orb_symm(qmc_obj.mol, 
                        qmc_obj.mol.irrep_name, qmc_obj.mol.symm_orb,
                        mo_coeff, s=s)
                orbsym = [param.IRREP_ID_TABLE[qmc_obj.groupname][i]+1 for
                          i in qmc_obj.orbsym]
                pyscf.tools.fcidump.write_head(fout, nmo, nelec,
                                               qmc_obj.mol.spin, orbsym)
        else:
            pyscf.tools.fcidump.write_head(fout, nmo, nelec, qmc_obj.mol.spin)

        eri = pyscf.ao2mo.outcore.full_iofree(qmc_obj.mol, mo_coeff, verbose=0)
        pyscf.tools.fcidump.write_eri(fout, pyscf.ao2mo.restore(8,eri,nmo),
                                      nmo, tol=tol)

        # Lookup and return the relevant 1-electron integrals, and print out
        # the FCIDUMP file.
        t = qmc_obj.mol.intor_symmetric('cint1e_kin_sph')
        v = qmc_obj.mol.intor_symmetric('cint1e_nuc_sph')
        h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))
        pyscf.tools.fcidump.write_hcore(fout, h, nmo, tol=tol)
        fout.write(' %.16g  0  0  0  0\n' % qmc_obj.mol.energy_nuc())

    # The number of alpha and beta electrons.
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else:
        neleca, nelecb = nelec

    write_fciqmc_config_file(qmc_obj, neleca, nelecb, restart)

    if qmc_obj.verbose >= logger.DEBUG1:
        # os.path.join(self.scratchDirectory,self.configFile)
        in_file = qmc_obj.configFile
        logger.debug1(qmc_obj, 'FCIQMC Input file')
        logger.debug1(qmc_obj, open(in_file, 'r').read())

    execute_fciqmc(qmc_obj)

    if qmc_obj.verbose >= logger.DEBUG1:
        # os.path.join(self.scratchDirectory,self.outputFile)
        out_file = qmc_obj.outputFileCurrent
        logger.debug1(qmc_obj, open(out_file))

    rdm_energy = read_energy(qmc_obj)

    return rdm_energy


def write_fciqmc_config_file(qmc_obj, neleca, nelecb, restart):
    config_file = qmc_obj.configFile

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
    if qmc_obj.nstates > 1:
        f.write('system-replicas %d\n' % (2*qmc_obj.nstates))
    f.write('endsys\n')
    f.write('\n')
    f.write('calc\n')
    f.write('methods\n')
    f.write('method vertex fcimc\n')
    f.write('endmethods\n')
    f.write('time %d\n' % qmc_obj.time)
    f.write('memoryfacpart 2.0\n')
    f.write('memoryfacspawn 1.0\n')
    f.write('totalwalkers %d\n' % qmc_obj.maxwalkers)
    f.write('nmcyc %d\n' % qmc_obj.maxIter)
    f.write('seed %d\n' % qmc_obj.seed)
    if (restart):
        f.write('readpops')
    else:
        f.write('startsinglepart 500\n')
        f.write('diagshift 0.1\n')
    f.write('rdmsamplingiters %d\n' % qmc_obj.RDMSamples)
    f.write('shiftdamp 0.05\n')
    if (qmc_obj.tau != -1.0):
        f.write('tau 0.01\n')
    f.write('truncinitiator\n')
    f.write('addtoinitiator %d\n' % qmc_obj.AddtoInit)
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
    if qmc_obj.nstates > 1:
        f.write('orthogonalise-replicas\n')
        f.write('doubles-init\n')
        f.write('multi-ref-shift\n')
    f.write('endcalc\n')
    f.write('\n')
    f.write('integral\n')
    f.write('freeze 0,0\n')
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


def write_integrals_file(h1eff, eri_cas, ncas, neleca, nelecb, qmc_obj):
    integralFile = os.path.join(qmc_obj.scratchDirectory,qmc_obj.integralFile)
    # Ensure 4-fold symmetry.
    eri_cas = pyscf.ao2mo.restore(4, eri_cas, ncas)
    if qmc_obj.mol.symmetry and qmc_obj.orbsym:
        orbsym = [IRREP_MAP[qmc_obj.groupname][i] for i in qmc_obj.orbsym]
    else:
        orbsym = []
    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                       neleca+nelecb, ms=abs(neleca-nelecb),
                                       orbsym=orbsym, tol=1e-10)


def execute_fciqmc(qmc_obj):
    in_file = os.path.join(qmc_obj.scratchDirectory, qmc_obj.configFile)
    outfiletmp = qmc_obj.outputFileRoot
    files = os.listdir(qmc_obj.scratchDirectory+'.')
    i = 1
    while outfiletmp in files:
        outfiletmp = qmc_obj.outputFileRoot + '_{}'.format(i)
        i += 1
    logger.info(qmc_obj,'fciqmc outputfile: %s', outfiletmp)
    qmc_obj.outputFileCurrent = outfiletmp
    out_file = os.path.join(qmc_obj.scratchDirectory, outfiletmp)

    if qmc_obj.executable == 'external':
        logger.info(qmc_obj,'External FCIQMC calculation requested from '
                             'dumped integrals.')
        logger.info(qmc_obj,'Waiting for density matrices and output file '
                             'to be returned.')
        try:
            raw_input("Press Enter to continue with calculation...")
        except:
            input("Press Enter to continue with calculation...")
    else:
        call("%s  %s > %s" % (qmc_obj.executable, in_file, out_file), shell=True)


def read_energy(qmc_obj):
    out_file = open(os.path.join(qmc_obj.scratchDirectory,
                 qmc_obj.outputFileCurrent),"r")

    for line in out_file:
        # Lookup the RDM energy from the output.
        if "*TOTAL ENERGY* CALCULATED USING THE" in line:
            rdm_energy = float(line.split()[-1])
            break
    logger.info(qmc_obj, 'total energy from fciqmc: %.15f', rdm_energy)
    out_file.close()

    return rdm_energy

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
    with open( 'molden.out', 'w' ) as fout:
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

