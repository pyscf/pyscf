#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import numpy
import pyscf.tools
import pyscf.lib.logger as logger

'''
DMRG solver for CASSCF.

DMRGCI.kernel function has a kerword fciRestart to control the Block solver
restart from previous calculation
'''

try:
    from pyscf.dmrgscf import settings
except ImportError:
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


class DMRGCI(object):
    def __init__(self, mol):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout

        self.executable = settings.BLOCKEXE
        self.scratchDirectory = settings.BLOCKSCRATCHDIR

        self.integralFile = "FCIDUMP"
        self.configFile = "dmrg.conf"
        self.outputFile = "dmrg.out"
        self.maxIter = 20
        self.twodot_to_onedot = 15
        self.tol = 1e-12
        self.maxM = 1000
        self.restart = False
        self.scheduleSweeps = [0, 1, 10, 16]
        self.scheduleMaxMs  = [self.maxM] * 4
        self.scheduleTols   = [1e-5, 1e-5, 1e-6, self.tol/10]
        self.scheduleNoises = [10, 1e-4, 1e-5, 0]

        self.orbsym = []
        if mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info('******** Block flags ********')
        log.info('scratchDirectory = %s', self.scratchDirectory)
        log.info('integralFile = %s', self.integralFile)
        log.info('configFile = %s', self.configFile)
        log.info('outputFile = %s', self.outputFile)
        log.info('maxIter = %d', self.maxIter)
        log.info('scheduleSweeps = %s', str(scheduleSweeps))
        log.info('scheduleMaxMs = %s', str(scheduleMaxMs))
        log.info('scheduleTols = %s', str(scheduleTols))
        log.info('scheduleNoises = %s', str(scheduleNoises))
        log.info('twodot_to_onedot = %d', self.twodot_to_onedot)
        log.info('tol = %g', self.tol)
        log.info('maxM = %d', self.maxM)

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, int):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        import os
        f = open(os.path.join(self.scratchDirectory, "spatial_twopdm.0.0.txt"), 'r')

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        norb_read = int(f.readline().split()[0])
        assert(norb_read == norb)

        for line in f.readlines():
            linesp = line.split()
            twopdm[int(linesp[0]),int(linesp[3]),int(linesp[1]),int(linesp[2])] = 2.0*float(linesp[4])

        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)

        return onepdm, twopdm

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, **kwargs):
        if fciRestart is None:
            fciRestart = self.restart
        if isinstance(nelec, int):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else :
            neleca, nelecb = nelec

        writeIntegralFile(h1e, eri, norb, neleca, nelecb, self)
        writeDMRGConfFile(neleca, nelecb, fciRestart, self)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.scratchDirectory,self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.scratchDirectory,self.outputFile)
            logger.debug1(self, open(outFile))
        calc_e = readEnergy(self)

        return calc_e, None

def make_schedule(sweeps, Ms, tols, noises):
    if len(sweeps) == len(Ms) == len(tols) == len(noises):
        schedule = ['schedule']
        for i, s in enumerate(sweeps):
            schedule.append('%d %6d  %8.4e  %8.4e' % (s, Ms[i], tols[i], noises[i]))
        schedule.append('end')
        return '\n'.join(schedule)
    else:
        return 'schedule default'

def writeDMRGConfFile(neleca, nelecb, Restart, DMRGCI):
    confFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.configFile)

    f = open(confFile, 'w')
    f.write('nelec %i\n'%(neleca+nelecb))
    f.write('spin %i\n' %(neleca-nelecb))

    if (not Restart):
        #f.write('schedule\n')
        #f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 10.0))
        #f.write('1 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 1e-4))
        #f.write('10 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-6, 1e-5))
        #f.write('16 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10.0, 0e-6))
        #f.write('end\n')
        schedule = make_schedule(DMRGCI.scheduleSweeps,
                                 DMRGCI.scheduleMaxMs,
                                 DMRGCI.scheduleTols,
                                 DMRGCI.scheduleNoises)
        f.write('%s\n' % schedule)
        f.write('twodot_to_onedot %i\n'%DMRGCI.twodot_to_onedot)
    else :
        f.write('schedule\n')
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10.0, 0e-6))
        f.write('end\n')
        f.write('fullrestart\n')
        f.write('onedot \n')

    if DMRGCI.mol.symmetry:
        f.write('sym %s\n' % DMRGCI.groupname.lower())
    f.write('orbitals %s\n' % os.path.join(DMRGCI.scratchDirectory,
                                           DMRGCI.integralFile))
    f.write('maxiter %i\n'%DMRGCI.maxIter)
    f.write('sweep_tol %8.4e\n'%DMRGCI.tol)
    f.write('outputlevel 2\n')
    f.write('hf_occ integral\n')
    f.write('twopdm\n')
    f.write('prefix  %s\n'%DMRGCI.scratchDirectory)
    f.close()
    #no reorder
    #f.write('noreorder\n')

def writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb, DMRGCI):
    integralFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.integralFile)
# ensure 4-fold symmetry
    eri_cas = pyscf.ao2mo.restore(4, eri_cas, ncas)
    if DMRGCI.mol.symmetry and DMRGCI.orbsym:
        orbsym = [IRREP_MAP[DMRGCI.groupname][i] for i in DMRGCI.orbsym]
    else:
        orbsym = []
    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                       neleca+nelecb, ms=abs(neleca-nelecb),
                                       orbsym=orbsym)

#    f = open(integralFile, 'w')
#    f.write(' &FCI NORB= %i,NELEC= %i,MS2= %i,\n' %(ncas, neleca+nelecb, neleca-nelecb))
#    f.write(' ORBSYM=%s\n')
#    for i in range(ncas):
#        f.write('1 ')
#
#    f.write('\nISYM=1\n')
#    f.write('&END\n')
#    index1 = 0
#    for i in range(ncas):
#        for j in range(i+1):
#            index2=0
#            for k in range(ncas):
#                for l in range(k+1):
#                    f.write('%18.10e %3i  %3i  %3i  %3i\n' %(eri_cas[index1,index2], i+1, j+1, k+1, l+1))
#                    index2=index2+1
#            index1=index1+1
#    for i in range(ncas):
#        for j in range(i+1):
#            f.write('%18.10e %3i  %3i  %3i  %3i\n' %(h1eff[i,j], i+1, j+1, 0, 0))
#
#    f.close()


def executeBLOCK(DMRGCI):
    inFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.configFile)
    outFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.outputFile)
    from subprocess import call
    call("%s  %s > %s"%(DMRGCI.executable, inFile, outFile), shell=True)

def readEnergy(DMRGCI):
    import struct, os
    file1 = open(os.path.join(DMRGCI.scratchDirectory, "dmrg.e"),"rb")
    calc_e = struct.unpack('d', file1.read(8))[0]
    file1.close()

    return calc_e



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'out-dmrgci',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True,
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    mc.fcisolver.tol = 1e-9
    emc_1 = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True,
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('DMRGCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('DMRGSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))

