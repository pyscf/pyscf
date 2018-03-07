#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: George Booth
#          Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
from functools import reduce
import numpy
import pyscf.tools
import pyscf.lib.logger as logger
import pyscf.ao2mo
import pyscf.symm
import pyscf.fci
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
        # Shouldn't need scratch dir settings.BLOCKSCRATCHDIR.
        self.scratchDirectory = ''

        self.generate_neci_input = True
        self.integralFile = "FCIDUMP"
        self.configFile = "neci.inp"
        self.outputFileRoot = "neci.out"
        self.outputFileCurrent = self.outputFileRoot
        self.maxwalkers = 10000
        self.maxIter = -1
        self.InitShift = 0.1
        self.RDMSamples = 5000
        self.restart = False
        self.time = 10
        self.tau = -1.0
        self.seed = 7
        self.AddtoInit = 3
        self.orbsym = []
        self.pg_symmetry = 1
        self.state_weights = [1.0]
        # This is the number of spin orbitals to freeze in the NECI calculation.
        # Note that if you do this for a CASSCF calculation, it will freeze in
        # the active space.
        self.nfreezecore = 0
        self.nfreezevirt = 0
        self.system_options = ''
        self.calc_options = ''
        self.logging_options = ''

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

        # If norm != 1 then the state weights will need normalising.
        norm = sum(self.state_weights)

        two_pdm = numpy.zeros( (norb, norb, norb, norb) )

        for irdm in range(nstates):
            if self.state_weights[irdm] != 0.0:
                dm_filename = 'spinfree_TwoRDM.' + str(irdm+1)
                temp_dm = read_neci_two_pdm(self, dm_filename, norb,
                                            self.scratchDirectory)
                two_pdm += (self.state_weights[irdm]/norm)*temp_dm

        one_pdm = one_from_two_pdm(two_pdm, nelectrons)

        return one_pdm, two_pdm

    def make_rdm1(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return self.make_rdm12(fcivec, norb, nelec, link_index, **kwargs)[0]

    def kernel(self, h1e, eri, norb, nelec, fci_restart=None, ecore=0, **kwargs):
        if fci_restart is None:
            fci_restart = self.restart
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else:
            neleca, nelecb = nelec

        write_integrals_file(h1e, eri, norb, neleca, nelecb, self, ecore)
        if self.generate_neci_input:
            write_fciqmc_config_file(self, neleca, nelecb, fci_restart)
        if self.verbose >= logger.DEBUG1:
            in_file = self.configFile
            logger.debug1(self, 'FCIQMC Input file')
            logger.debug1(self, open(in_file, 'r').read())
        execute_fciqmc(self)
        if self.verbose >= logger.DEBUG1:
            out_file = self.outputFileCurrent
            logger.debug1(self, open(out_file))
        rdm_energy = read_energy(self)

        return rdm_energy, None

def calc_energy_from_rdms(mol, mo_coeff, one_rdm, two_rdm):
    '''From the full density matrices, calculate the energy.

    Args:
        mol : An instance of :class:`Mole`
            The molecule to calculate
        mo_coeff: ndarray
            The MO orbitals in which the RDMs are calculated
        one_rdm: ndarray
            The 1RDM
        two_rdm: ndarray
            The 2RDM as RDM_ijkl = < a^+_is a^+_kt a_lt a_js >.
    '''

    nmo = mo_coeff.shape[1]
    eri = pyscf.ao2mo.full(mol, mo_coeff, verbose=0)
    eri = pyscf.ao2mo.restore(1, eri, nmo)
    t = mol.intor_symmetric('cint1e_kin_sph')
    v = mol.intor_symmetric('cint1e_nuc_sph')
    h = reduce(numpy.dot, (mo_coeff.T, t+v, mo_coeff))

    two_e = numpy.einsum('ijkl,ijkl->', eri, two_rdm) * 0.5
    one_e = numpy.einsum('ij,ij->', h, one_rdm)

    return two_e + one_e + mol.energy_nuc()

def run_standalone(fciqmcci, scf_obj, orbs=None, restart=None):
    '''Run a standalone NECI calculation for the molecule listed in the
    FCIQMCCI object. The basis to run this calculation in is given by the
    orbs array.

    Args:
        fciqmcci : an instance of :class:`FCIQMCCI`
            FCIQMC calculation containing parameters of NECI calculation to
            run.
        mo_coeff : ndarray
            Orbital coefficients. Each column is one orbital.
        restart : bool
            Is this a restarted NECI calculation?

    Returns:
        rdm_energy : float
            Final RDM energy obtained from the NECI output file.
    '''

    if orbs is None:
        orbs = scf_obj.mo_coeff
    tol = 1e-9
    if isinstance(orbs,tuple):
        # Assume UHF
        print('uhf orbitals detected')
        nmo = orbs[0].shape[1]
        tUHF = True
    else:
        print('rhf orbitals detected')
        nmo = orbs.shape[1]
        tUHF = False
    nelec = fciqmcci.mol.nelectron
    fciqmcci.dump_flags(verbose=5)

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
            if tUHF:
                fciqmcci.orbsym = pyscf.symm.label_orb_symm(fciqmcci.mol,
                        fciqmcci.mol.irrep_name, fciqmcci.mol.symm_orb,
                        orbs[0])
                tmp_orblist = fciqmcci.orbsym.tolist()
                tmp_orblist += pyscf.symm.label_orb_symm(fciqmcci.mol,
                        fciqmcci.mol.irrep_name, fciqmcci.mol.symm_orb,
                        orbs[1]).tolist()
                fciqmcci.orbsym = numpy.array(tmp_orblist)
                orbsym = [param.IRREP_ID_TABLE[fciqmcci.groupname][i]+1 for
                          i in fciqmcci.orbsym]
            else:
                fciqmcci.orbsym = pyscf.symm.label_orb_symm(fciqmcci.mol,
                        fciqmcci.mol.irrep_name, fciqmcci.mol.symm_orb,
                        orbs)
                orbsym = [param.IRREP_ID_TABLE[fciqmcci.groupname][i]+1 for
                          i in fciqmcci.orbsym]
#            pyscf.tools.fcidump.write_head(fout, nmo, nelec,
#                                           fciqmcci.mol.spin, orbsym)
    else:
        orbsym = []

#    eri = pyscf.ao2mo.outcore.full(fciqmcci.mol, orbs, verbose=0)
    # Lookup and return the relevant 1-electron integrals, and print out
    # the FCIDUMP file.
    if tUHF:
        write_uhf_integrals_neci(fciqmcci,scf_obj,nmo,nelec,orbs,orbsym,tol=tol)
    else:
        eri = pyscf.ao2mo.incore.general(scf_obj._eri, (orbs,)*4, compact=False)
        h_core = scf_obj.get_hcore(fciqmcci.mol)
#        t = fciqmcci.mol.intor_symmetric('cint1e_kin_sph')
#        v = fciqmcci.mol.intor_symmetric('cint1e_nuc_sph')
        h = reduce(numpy.dot, (orbs.T, h_core, orbs))

        pyscf.tools.fcidump.from_integrals(fciqmcci.integralFile, h, 
                pyscf.ao2mo.restore(8,eri,nmo), nmo, nelec, fciqmcci.mol.energy_nuc(),
                fciqmcci.mol.spin, orbsym, tol=tol)

#    pyscf.tools.fcidump.write_eri(fout, pyscf.ao2mo.restore(8,eri,nmo),
#                                  nmo, tol=tol)
#    pyscf.tools.fcidump.write_hcore(fout, h, nmo, tol=tol)
#    fout.write(' %.16g  0  0  0  0\n' % fciqmcci.mol.energy_nuc())

    # The number of alpha and beta electrons.
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else:
        neleca, nelecb = nelec

    if fciqmcci.generate_neci_input:
        write_fciqmc_config_file(fciqmcci, neleca, nelecb, restart, tUHF)

    if fciqmcci.verbose >= logger.DEBUG1:
        in_file = fciqmcci.configFile
        logger.debug1(fciqmcci, 'FCIQMC Input file')
        logger.debug1(fciqmcci, open(in_file, 'r').read())

    execute_fciqmc(fciqmcci)

    if fciqmcci.verbose >= logger.DEBUG1:
        out_file = fciqmcci.outputFileCurrent
        logger.debug1(fciqmcci, open(out_file))

    rdm_energy = read_energy(fciqmcci)

    return rdm_energy

def write_uhf_integrals_neci(fciqmcci,scf_obj,nmo,nelec,orbs,orbsym,tol=1e-15):
    ''' nmo is number of MO orbitals per spin channel
        note that ordering is abababa...   '''

    eri_aaaa = pyscf.ao2mo.restore(8,pyscf.ao2mo.incore.general(scf_obj._eri, (orbs[0],orbs[0],orbs[0],orbs[0]), compact=False),nmo)
    eri_bbbb = pyscf.ao2mo.restore(8,pyscf.ao2mo.incore.general(scf_obj._eri, (orbs[1],orbs[1],orbs[1],orbs[1]), compact=False),nmo)
    eri_aabb = pyscf.ao2mo.restore(8,pyscf.ao2mo.incore.general(scf_obj._eri, (orbs[0],orbs[0],orbs[1],orbs[1]), compact=False),nmo)
    eri_bbaa = pyscf.ao2mo.restore(8,pyscf.ao2mo.incore.general(scf_obj._eri, (orbs[1],orbs[1],orbs[0],orbs[0]), compact=False),nmo)
    h_core = scf_obj.get_hcore(fciqmcci.mol)
#    t = fciqmcci.mol.intor_symmetric('cint1e_kin_sph')
#    v = fciqmcci.mol.intor_symmetric('cint1e_nuc_sph')
    h_aa = reduce(numpy.dot, (orbs[0].T, h_core, orbs[0]))
    h_bb = reduce(numpy.dot, (orbs[1].T, h_core, orbs[1]))
    nuc = fciqmcci.mol.energy_nuc()
    float_format = ' %.16g'

    # Stupidly, NECI wants its orbitals as a,b,a,b,a,b rather than aaaabbbb
    # Reorder things so this is the case
    assert(len(orbsym) % 2 == 0)
    orbsym_reorder = [i for tup in zip(orbsym[:len(orbsym)/2], orbsym[len(orbsym)/2:]) for i in tup]
    a_inds = [i*2+1 for i in range(orbs[0].shape[1])]
    b_inds = [i*2+2 for i in range(orbs[1].shape[1])]

    with open(fciqmcci.integralFile, 'w') as fout:
        if not isinstance(nelec, (int, numpy.number)):
            ms = abs(nelec[0] - nelec[1])
            nelec = nelec[0] + nelec[1]
        else: ms=0
        fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo*2, nelec, ms))
        if orbsym is not None and len(orbsym_reorder) > 0:
            fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym_reorder]))
        else:
            fout.write('  ORBSYM=%s\n' % ('1,' * 2*nmo))
        fout.write('  ISYM=1, UHF=TRUE\n')
        fout.write(' &END\n')
        # Assume 8-fold symmetry
        npair = nmo*(nmo+1)//2
        output_format = float_format + ' %4d %4d %4d %4d\n'
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri_aaaa[ijkl]) > tol:
                                fout.write(output_format % (eri_aaaa[ijkl], a_inds[i], a_inds[j], a_inds[k], a_inds[l]))
                            if abs(eri_bbbb[ijkl]) > tol:
                                fout.write(output_format % (eri_bbbb[ijkl], b_inds[i], b_inds[j], b_inds[k], b_inds[l]))
                            if abs(eri_aabb[ijkl]) > tol:
                                fout.write(output_format % (eri_aabb[ijkl], a_inds[i], a_inds[j], b_inds[k], b_inds[l]))
                            if abs(eri_bbaa[ijkl]) > tol:
                                fout.write(output_format % (eri_bbaa[ijkl], b_inds[i], b_inds[j], a_inds[k], a_inds[l]))
                            ijkl += 1
                        kl += 1
                ij += 1
        h_aa = h_aa.reshape(nmo,nmo)
        h_bb = h_bb.reshape(nmo,nmo)
        output_format = float_format + ' %4d %4d  0  0\n'
        for i in range(nmo):
            for j in range(0, i+1):
                if abs(h_aa[i,j]) > tol:
                    fout.write(output_format % (h_aa[i,j], a_inds[i], a_inds[j]))
                if abs(h_bb[i,j]) > tol:
                    fout.write(output_format % (h_bb[i,j], b_inds[i], b_inds[j]))
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)
    return 

def write_fciqmc_config_file(fciqmcci, neleca, nelecb, restart, tUHF=False):
    '''Write an input file for a NECI calculation.

    Args:
        fciqmcci : an instance of :class:`FCIQMCCI`
            Contains all the parameters used to create the input file.
        neleca : int
            The number of alpha electrons.
        nelecb : int
            The number of beta electrons.
        restart : bool
            Is this a restarted NECI calculation?
    '''

    config_file = fciqmcci.configFile
    nstates = len(fciqmcci.state_weights)

    f = open(config_file, 'w')

    f.write('title\n')
    f.write('\n')

    f.write('system read noorder\n')
    f.write('symignoreenergies\n')
    f.write('freeformat\n')
    f.write('electrons %d\n' % (neleca+nelecb))
    # fci-core requires these two options.
    f.write('spin-restrict %d\n' %(-fciqmcci.mol.spin))
    f.write('sym %d 0 0 0\n' % (fciqmcci.pg_symmetry-1))
    f.write('nonuniformrandexcits 4ind-weighted\n')
    if not (tUHF or fciqmcci.mol.spin != 0):
        f.write('hphf 0\n')
    f.write('nobrillouintheorem\n')
    if nstates > 1:
        f.write('system-replicas %d\n' % (2*nstates))
    if fciqmcci.system_options:
        f.write(fciqmcci.system_options + '\n')
    f.write('endsys\n')
    f.write('\n')

    f.write('calc\n')
    f.write('methods\n')
    f.write('method vertex fcimc\n')
    f.write('endmethods\n')
    f.write('time %f\n' % fciqmcci.time)
    f.write('memoryfacpart 2.0\n')
    f.write('memoryfacspawn 1.0\n')
    f.write('totalwalkers %d\n' % fciqmcci.maxwalkers)
    f.write('nmcyc %d\n' % fciqmcci.maxIter)
    f.write('seed %d\n' % fciqmcci.seed)
    if (restart):
        f.write('readpops')
    else:
        f.write('startsinglepart\n')
        f.write('diagshift %f\n' % fciqmcci.InitShift)
    f.write('rdmsamplingiters %d\n' % fciqmcci.RDMSamples)
    f.write('shiftdamp 0.05\n')
    if (fciqmcci.tau != -1.0):
        f.write('tau 0.01\n')
    f.write('truncinitiator\n')
    f.write('addtoinitiator %d\n' % fciqmcci.AddtoInit)
    f.write('allrealcoeff\n')
    f.write('realspawncutoff 0.4\n')
    f.write('semi-stochastic\n')
    f.write('mp1-core 2000\n')
#    f.write('fci-core\n')
#    f.write('trial-wavefunction 5\n')
    f.write('jump-shift\n')
    f.write('proje-changeref 1.5\n')
    f.write('stepsshift 10\n')
    f.write('maxwalkerbloom 3\n')
# Dynamic load-balancing is incompatible with semi-stochastic.
# Ok if restarting from a semi-stochastic popsfile,
# (where it will do one redistribution) but not otherwise.
    f.write('load-balance-blocks off\n')
    if nstates > 1:
        f.write('orthogonalise-replicas\n')
        f.write('doubles-init\n')
        f.write('multi-ref-shift\n')
#        f.write('fci-init\n')
    if fciqmcci.calc_options:
        f.write(fciqmcci.calc_options + '\n')
    f.write('endcalc\n')
    f.write('\n')

    f.write('integral\n')
    f.write('freeze %d %d\n' % (fciqmcci.nfreezecore, fciqmcci.nfreezevirt))
    f.write('endint\n')
    f.write('\n')

    f.write('logging\n')
    f.write('popsfiletimer 60.0\n')
    f.write('binarypops\n')
    f.write('calcrdmonfly 3 500 500\n')
    f.write('write-spin-free-rdm\n')
    f.write('printonerdm\n')
    if fciqmcci.logging_options:
        f.write(fciqmcci.logging_options + '\n')
    f.write('endlog\n')
    f.write('end\n')

    f.close()


def write_integrals_file(h1e, eri, norb, neleca, nelecb, fciqmcci, ecore=0):
    '''Write an integral dump file, based on the integrals provided.

    Args:
        h1e : 2D ndarray
            Core Hamiltonian.
        eri : 2D ndarray
            Two-electron integrals.
        norb : int
            Number of orbitals.
        neleca : int
            Number of alpha electrons.
        nelecb : int
            Number of beta electrons
        fciqmcci : an instance of :class:`FCIQMCCI`
            FCIQMC calculation, used to access the integral dump file name and
            some symmetry properties.
    '''

    integralFile = os.path.join(fciqmcci.scratchDirectory,fciqmcci.integralFile)
    # Ensure 4-fold symmetry.
    eri = pyscf.ao2mo.restore(4, eri, norb)
    if fciqmcci.mol.symmetry and fciqmcci.orbsym is not []:
        orbsym = [IRREP_MAP[fciqmcci.groupname][i] for i in fciqmcci.orbsym]
    else:
        orbsym = []
    pyscf.tools.fcidump.from_integrals(integralFile, h1e, eri, norb,
                                       neleca+nelecb, ecore, ms=abs(neleca-nelecb),
                                       orbsym=orbsym, tol=1e-10)


def execute_fciqmc(fciqmcci):
    '''Call the external FCIQMC program.

    Args:
        fciqmcci : an instance of :class:`FCIQMCCI`
            Specifies the FCIQMC calculation.
    '''

    in_file = os.path.join(fciqmcci.scratchDirectory, fciqmcci.configFile)
    outfiletmp = fciqmcci.outputFileRoot
    files = os.listdir(fciqmcci.scratchDirectory + '.')
    # Search for an unused output file.
    i = 1
    while outfiletmp in files:
        outfiletmp = fciqmcci.outputFileRoot + '_' + str(i)
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
    '''Read and return the final RDM energy from a NECI output file.

    Args:
        fciqmcci : an instance of :class:`FCIQMCCI`
            Specifies the FCIQMC calculation. Used to locate the FCIQMC output
            file.

    Returns:
        rdm_energy : float
            The final RDM energy printed to the output file.
    '''

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
    one_pdm = one_from_two_pdm(two_pdm, nelec)
    return one_pdm

def read_neci_1dms(fciqmcci, norb, nelec, filename='OneRDM.1', directory='.'):
    ''' Read spinned rdms, as they are in the neci output '''

    f = open(os.path.join(directory, filename),'r')

    dm1a = numpy.zeros((norb,norb))
    dm1b = numpy.zeros((norb,norb))
    for line in f.readlines():
        linesp = line.split()
        i, j = int(linesp[0]), int(linesp[1])
        assert((i % 2) == (j % 2))
        if i % 2 == 1:
            # alpha
            assert(all(x<norb for x in (i/2,j/2)))
            dm1a[i/2,j/2] = float(linesp[2])
            dm1a[j/2,i/2] = float(linesp[2])
        else:
            assert(all(x<norb for x in (i/2 - 1,j/2 - 1)))
            dm1b[i/2 - 1,j/2 - 1] = float(linesp[2])
            dm1b[j/2 - 1,i/2 - 1] = float(linesp[2])

    f.close()
    assert(numpy.allclose(dm1a.trace()+dm1b.trace(),sum(nelec)))
    return dm1a, dm1b

def read_neci_2dms(fciqmcci, norb, nelec, filename_aa='TwoRDM_aaaa.1', 
    filename_abba='TwoRDM_abba.1', filename_abab='TwoRDM_abab.1', directory='.', reorder=True,
    dm1a=None,dm1b=None):
    ''' Find spinned RDMs (assuming a/b symmetry). Return in pyscf form that
    you would get from the e.g. direct_spin1.make_rdm12s routine, with Reorder=True.
    
    This means (assuming reorder = True):
    dm2ab[i,j,k,l] = < i_a* k_b* l_b j_a >
    dm2aa[i,j,k,l] = < i_a* k_a* l_a j_a > 

    to get the dm2abba matrix (see spin_op.make_rdm2_abba) from this (assuming rhf), then you need
    dm2abba = -dm2ab.transpose(2,1,0,3)
    
    if reorder = False:

        dm2aa[:,k,k,:] += dm1a
        dm2bb[:,k,k,:] += dm1b
        dm2ab unchanged

    Note that the spin-free RDMs are just dm2aa + dm2bb + 2*dm2ab if reorder = True
    '''

    f = open(os.path.join(directory, filename_aa),'r')
    dm2aa = numpy.zeros((norb,norb,norb,norb))
    for line in f.readlines():
        linesp = line.split()
        i,j,k,l = (int(linesp[0])-1, int(linesp[1])-1, int(linesp[3])-1, int(linesp[2])-1)
        val = float(linesp[4])
        assert(all(x<norb for x in (i,j,k,l)))
        # Stored as 1* 2* 4 3
        dm2aa[i,j,k,l] = val 
        # Other permutations
        dm2aa[j,i,k,l] = -val
        dm2aa[i,j,l,k] = -val
        dm2aa[j,i,l,k] = val
        # Hermitian conjugate symmetry, assuming real orbitals
        dm2aa[l,k,j,i] = val 
        # Other permutations
        dm2aa[l,k,i,j] = -val
        dm2aa[k,l,j,i] = -val
        dm2aa[k,l,i,j] = val 

    f.close()
    dm2bb = dm2aa.copy()    #spin symmetry
    
    # dm2ab initially (before reordering) stores [a,b,b,a]
    dm2ab = numpy.zeros((norb,norb,norb,norb))
    f_abba = open(os.path.join(directory, filename_abba),'r')
    f_abab = open(os.path.join(directory, filename_abab),'r')
    for line in f_abba.readlines():
        linesp = line.split()
        i,j,k,l = (int(linesp[0])-1, int(linesp[1])-1, int(linesp[2])-1, int(linesp[3])-1)
        val = float(linesp[4])
        assert(all(x<norb for x in (i,j,k,l)))
        assert(numpy.allclose(dm2ab[i,j,k,l],-val) or dm2ab[i,j,k,l] == 0.0)
        dm2ab[i,j,k,l] = -val 
        # Hermitian conjugate
        assert(numpy.allclose(dm2ab[l,k,j,i],-val) or dm2ab[l,k,j,i] == 0.0)
        dm2ab[l,k,j,i] = -val
        # Time reversal sym
#        print(i,j,k,l,val,dm2ab[j,i,l,k],
#        numpy.allclose(dm2ab[j,i,l,k],-val), dm2ab[j,i,l,k] == 0.0)
        assert(numpy.allclose(dm2ab[j,i,l,k],-val) or dm2ab[j,i,l,k] == 0.0)
        dm2ab[j,i,l,k] = -val
        assert(numpy.allclose(dm2ab[k,l,i,j],-val) or dm2ab[k,l,i,j] == 0.0)
        dm2ab[k,l,i,j] = -val

    for line in f_abab.readlines():
        linesp = line.split()
        i,j,k,l = (int(linesp[0])-1, int(linesp[1])-1, int(linesp[2])-1, int(linesp[3])-1)
        val = float(linesp[4])
        assert(all(x<norb for x in (i,j,k,l)))
        assert(numpy.allclose(dm2ab[i,j,l,k],val) or dm2ab[i,j,l,k] == 0.0)
        dm2ab[i,j,l,k] = val
        # Hermitian conjugate
        assert(numpy.allclose(dm2ab[k,l,j,i],val) or dm2ab[k,l,j,i] == 0.0)
        dm2ab[k,l,j,i] = val
        # Time reversal symmetry
        assert(numpy.allclose(dm2ab[j,i,k,l],val) or dm2ab[j,i,k,l] == 0.0)
        dm2ab[j,i,k,l] = val
        assert(numpy.allclose(dm2ab[l,k,i,j],val) or dm2ab[l,k,i,j] == 0.0)
        dm2ab[l,k,i,j] = val

    f_abab.close()
    f_abba.close()

    # i.e. I want the last index to go second 
    dm2aa = dm2aa.transpose(0,3,1,2)
    dm2bb = dm2bb.transpose(0,3,1,2)
    dm2ab = dm2ab.transpose(0,3,1,2)

    if not reorder:
        # We need to undo the reordering routine in rdm.py
        if dm1a is None:
            pdmfile = filename_aa.split('.')
            pdmfile = 'OneRDM.'+pdmfile[1]
            dm1a, dm1b = read_neci_1dms(fciqmcci, norb, nelec, filename=pdmfile, directory=directory)
        for k in range(norb):
            dm2aa[:,k,k,:] += dm1a
            dm2bb[:,k,k,:] += dm1b
    
    return dm2aa, dm2ab, dm2bb

def add_spinned_core_rdms(mf, ncore, dm1a_act, dm1b_act, dm2aa_act, dm2ab_act, dm2bb_act, reorder=True):
    ''' Add an RHF core to the rdms in the MO basis to the 1 and 2 RDMs'''

    norb = ncore + dm1a_act.shape[0]
    dm1a = numpy.zeros((norb,norb))
    dm1b = numpy.zeros((norb,norb))
    dm2aa = numpy.zeros((norb,norb,norb,norb))
    dm2ab = numpy.zeros((norb,norb,norb,norb))
    dm2bb = numpy.zeros((norb,norb,norb,norb))

    if not reorder:
        # Assume that the ordering of the active rdms is 'False'.
        # Switch before including (back to 'true')
        dm1a_act_, dm2aa_act_ = pyscf.fci.rdm.reorder_rdm(dm1a_act, dm2aa_act, inplace=False)
        dm1b_act_, dm2bb_act_ = pyscf.fci.rdm.reorder_rdm(dm1b_act, dm2bb_act, inplace=False)
    else:
        dm1a_act_ = dm1a_act
        dm1b_act_ = dm1b_act
        dm2aa_act_ = dm2aa_act
        dm2bb_act_ = dm2bb_act
    
    # Always add the core to the 'reorder=True' ordering of the rdms
    dm1a[ncore:,ncore:] = dm1a_act_
    dm1b[ncore:,ncore:] = dm1b_act_
    for i in range(ncore):
        dm1a[i,i] = 1.0
        dm1b[i,i] = 1.0

    dm2aa[ncore:,ncore:,ncore:,ncore:] = dm2aa_act_
    dm2bb[ncore:,ncore:,ncore:,ncore:] = dm2bb_act_
    dm2ab[ncore:,ncore:,ncore:,ncore:] = dm2ab_act

    for i in range(ncore):
        for j in range(ncore): 
            dm2aa[i,i,j,j] += 1.0
            dm2aa[j,i,i,j] += -1.0
            dm2bb[i,i,j,j] += 1.0
            dm2bb[j,i,i,j] += -1.0
            dm2ab[i,i,j,j] += 1.0
        for p in range(ncore,norb):
            for q in range(ncore,norb):
                dm2aa[p,q,i,i] +=  dm1a[p,q]
                dm2aa[i,i,p,q] +=  dm1a[p,q]
                dm2aa[i,q,p,i] += -dm1a[p,q]
                dm2aa[p,i,i,q] += -dm1a[p,q]
                dm2bb[p,q,i,i] +=  dm1b[p,q]
                dm2bb[i,i,p,q] +=  dm1b[p,q]
                dm2bb[i,q,p,i] += -dm1b[p,q]
                dm2bb[p,i,i,q] += -dm1b[p,q]
                dm2ab[p,q,i,i] +=  dm1a[p,q]
                dm2ab[i,i,p,q] +=  dm1b[p,q]

    if not reorder:
        # Change back to the 'non-reordered' ordering!
        for k in range(norb):
            dm2aa[:,k,k,:] += dm1a
            dm2bb[:,k,k,:] += dm1b

    return dm1a, dm1b, dm2aa, dm2ab, dm2bb

def read_neci_two_pdm(fciqmcci, filename, norb, directory='.'):
    '''Read a spin-free 2-rdm output from a NECI calculation, and return it in
    a form supported by pyscf. Note that the RDMs in neci are written in
    as RDM_ijkl = < a^+_is a^+_jt a_lt a_ks >. In pyscf, the correlated _after
    reordering_ is 2RDM_ijkl = < a^+_is a^+_kt a_lt a_js >, where s and t are spin
    indices to be summed over. Therefore, the middle two indices need to be swapped.
    If core orbitals have been indicated as frozen in neci, this core contribution
    will be explicitly added back in to the RDM. Therefore, the norb parameter
    should be the unfrozen number of orbitals passed to neci, but not inactive
    if running through CASSCF.

    Args:
        filename : str
            Name of the file to read the 2-rdm from.
        norb : int
            The number of orbitals inc. frozen in neci, and therefore the
            number of values each 2-rdm index can take.
        directory : str
            The directory in which to search for the 2-rdm file.

    Returns:
        two_pdm : ndarray
            The read-in 2-rdm.
    '''

    f = open(os.path.join(directory, filename), 'r')

    nfrzorb = fciqmcci.nfreezecore//2

    norb_active = norb - nfrzorb
    two_pdm_active = numpy.zeros( (norb_active, norb_active, norb_active, norb_active) )
    for line in f.readlines():
        linesp = line.split()

        if(int(linesp[0]) != -1):
            # Arrays from neci are '1' indexed
            # We reorder from D[i,j,k,l] = < i^+ j^+ l k >
            # to              D[i,j,k,l] = < i^+ k^+ l j > to match pyscf
            # Therefore, all we need to do is to swap the middle two indices.
            ind1 = int(linesp[0]) - 1
            ind2 = int(linesp[2]) - 1
            ind3 = int(linesp[1]) - 1
            ind4 = int(linesp[3]) - 1
            assert(int(ind1) < norb_active)
            assert(int(ind2) < norb_active)
            assert(int(ind3) < norb_active)
            assert(int(ind4) < norb_active)
            assert(ind1 >= 0)
            assert(ind2 >= 0)
            assert(ind3 >= 0)
            assert(ind4 >= 0)

            two_pdm_active[ind1, ind2, ind3, ind4] = float(linesp[4])

    f.close()

    # In order to add any frozen core, we first need to find the spin-free
    # 1-RDM in the active space.
    one_pdm_active = one_from_two_pdm(two_pdm_active,fciqmcci.mol.nelectron-fciqmcci.nfreezecore)

    # Copy the 2RDM part of the active space.
    two_pdm = numpy.zeros( (norb, norb, norb, norb) )
    actstart = nfrzorb
    actend = norb - fciqmcci.nfreezevirt/2
    two_pdm[actstart:actend, actstart:actend, actstart:actend, actstart:actend] = two_pdm_active

    # Interaction between frozen and active space.
    for p in range(nfrzorb):
        # p loops over frozen spatial orbitals.
        for r in range(actstart,actend):
            for s in range(actstart,actend):
                two_pdm[p,p,r,s] += 2.0*one_pdm_active[r-nfrzorb,s-nfrzorb]
                two_pdm[r,s,p,p] += 2.0*one_pdm_active[r-nfrzorb,s-nfrzorb]
                two_pdm[p,r,s,p] -= one_pdm_active[r-nfrzorb,s-nfrzorb]
                two_pdm[r,p,p,s] -= one_pdm_active[r-nfrzorb,s-nfrzorb]

    # Add on frozen core contribution, assuming that the core orbitals are
    # doubly occupied.
    for i in range(nfrzorb):
        for j in range(nfrzorb):
            two_pdm[i,i,j,j] += 4.0
            two_pdm[i,j,j,i] += -2.0

    return two_pdm

def one_from_two_pdm(two_pdm, nelec):
    '''Return a 1-rdm, given a 2-rdm to contract.

    Args:
        two_pdm : ndarray
            A (spin-free) 2-particle reduced density matrix.
        nelec: int
            The number of electrons contributing to the RDMs.

    Returns:
        one_pdm : ndarray
            The (spin-free) 1-particle reduced density matrix.
    '''

    # Last two indices refer to middle two second quantized operators in the 2RDM
    one_pdm = numpy.einsum('ikjj->ik', two_pdm)
    one_pdm /= (numpy.sum(nelec)-1)
    return one_pdm

def find_full_casscf_12rdm(fciqmcci, mo_coeff, filename, norbcas, neleccas, directory='.'):
    '''Return the 1 and 2 full RDMs after a CASSCF calculation, by adding
    on the contributions from the inactive spaces. Requires the cas space to
    be given, as we as a set of mo coefficients in the complete space.
    '''

    two_pdm = read_neci_two_pdm(fciqmcci, filename, norbcas, directory)
    one_pdm = one_from_two_pdm(two_pdm, neleccas)

    return add_inactive_space_to_rdm(fciqmcci.mol, mo_coeff, one_pdm, two_pdm)

def add_inactive_space_to_rdm(mol, mo_coeff, one_pdm, two_pdm):
    '''If a CASSCF calculation has been done, the final RDMs from neci will
    not contain the doubly occupied inactive orbitals. This function will add
    them and return the full density matrices.
    '''

    # Find number of inactive electrons by taking the number of electrons
    # as the trace of the 1RDM, and subtracting from the total number of
    # electrons
    ninact = (mol.nelectron - int(round(numpy.trace(one_pdm)))) / 2
    norb = mo_coeff.shape[1]
    nsizerdm = one_pdm.shape[0]

    one_pdm_ = numpy.zeros( (norb, norb) )
    # Add the core first.
    for i in range(ninact):
        one_pdm_[i,i] = 2.0

    # Add the rest of the density matrix.
    one_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = one_pdm[:,:]

    two_pdm_ = numpy.zeros( (norb, norb, norb, norb) )

    # Add on frozen core contribution, assuming that the inactive orbitals are
    # doubly occupied.
    for i in range(ninact):
        for j in range(ninact):
            two_pdm_[i,i,j,j] += 4.0
            two_pdm_[i,j,j,i] += -2.0

    # Inactve-Active elements.
    for p in range(ninact):
        for r in range(ninact,ninact+nsizerdm):
            for s in range(ninact,ninact+nsizerdm):
                two_pdm_[p,p,r,s] += 2.0*one_pdm_[r,s]
                two_pdm_[r,s,p,p] += 2.0*one_pdm_[r,s]
                two_pdm_[p,r,s,p] -= one_pdm_[r,s]
                two_pdm_[r,p,p,s] -= one_pdm_[r,s]

    # Add active space.
    two_pdm_[ninact:ninact+nsizerdm,ninact:ninact+nsizerdm, \
             ninact:ninact+nsizerdm,ninact:ninact+nsizerdm] = \
             two_pdm[:,:]

    return one_pdm_, two_pdm_

def calc_dipole(mol, mo_coeff, one_pdm):
    '''Calculate and return the dipole moment for a given molecule, set of
    molecular orbital coefficients and a 1-rdm.

    Args:
        mol : an instance of :class:`Mole`
            Specifies the molecule.
        mo_coeff : ndarray
            Orbital coefficients. Each column is one orbital.
        one_pdm : ndarray
            1-rdm.

    Returns:
        tot_dipmom : list of float
            The total dipole moment of the system in each dimension.
        elec_dipmom : list of float
            The electronic component of the dipole moment in each dimension.
        nuc_dipmom : list of float
            The nuclear component of the dipole moment in each dimension.
    '''

    assert(one_pdm.shape[0] == one_pdm.shape[1])
    norb = mo_coeff.shape[1]
    nsizerdm = one_pdm.shape[0]
    if nsizerdm != norb:
        raise RuntimeError('''Size of 1RDM is not the same size as number of
                orbitals. Have you correctly included the external space if
                running from CASSCF??''')

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
        elec_dipmom.append( -numpy.trace( numpy.dot( one_pdm, modmints[i])) )

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
        # fciqmc cannot handle Dooh currently, so reduce the point group if
        # full group is infinite.
        symmetry_subgroup = 'D2h',
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    mc.fcisolver = FCIQMCCI(mol)
    mc.fcisolver.tau = 0.01
    mc.fcisolver.RDMSamples = 1000
    mc.max_cycle_macro = 10
    # Return natural orbitals from mc2step in casscf_mo.
    mc.natorb = True
    emc_1, e_ci, fcivec, casscf_mo, mo_energy = mc.mc2step(m.mo_coeff)

    # Write orbitals to molden output.
    with open( 'output.molden', 'w' ) as fout:
        molden.header(mol, fout)
        molden.orbital_coeff(mol, fout, casscf_mo)

    # Now, calculate the full RDMs for the full energy.
    one_pdm, two_pdm = find_full_casscf_12rdm(mc.fcisolver, casscf_mo,
            'spinfree_TwoRDM.1', 4, 4)
    e = calc_energy_from_rdms(mol, casscf_mo, one_pdm, two_pdm)
    print('Energy from rdms and CASSCF should be the same: ',e,emc_1)

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver =  FCIQMCCI(mol)
    mc.fcisolver.tau = 0.01
    mc.fcisolver.RDMSamples = 1000
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = None,
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

