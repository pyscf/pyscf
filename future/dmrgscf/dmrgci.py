#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Sheng Guo <shengg@princeton.edu>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import time
import tempfile
import numpy
import pyscf.tools
import pyscf.lib.logger as logger
from pyscf import mcscf
from pyscf.dmrgscf import dmrg_sym
from pyscf.dmrgscf.dmrg_sym import IRREP_MAP
from pyscf.dmrgscf.nevpt_mpi import write_chk

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
    raise ImportError



class DMRGCI(object):
    def __init__(self, mol, maxM=None, tol=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.outputlevel = 2

        self.executable = settings.BLOCKEXE
        self.scratchDirectory = settings.BLOCKSCRATCHDIR
        self.mpiprefix = settings.MPIPREFIX

        self.integralFile = "FCIDUMP"
        self.configFile = "dmrg.conf"
        self.outputFile = "dmrg.out"
        self.twopdm = True
        self.maxIter = 20
        self.approx_maxIter = 4
        self.twodot_to_onedot = 15
        self.dmrg_switch_tol = 1e-3
        self.nroots = 1
        self.nevpt_state_num = 0
        self.weights = []
        self.wfnsym = 1

        if tol is None:
            self.tol = 1e-8
        else:
            self.tol = tol/10
        if maxM is None:
            self.maxM = 1000
        else:
            self.maxM = maxM
        self.startM =  None
        self.restart = False
        self.force_restart = False
        self.nonspinAdapted = False
        self.mps_nevpt = False
        self.scheduleSweeps = []
        self.scheduleMaxMs  = []
        self.scheduleTols   = []
        self.scheduleNoises = []

        self.orbsym = []
        if mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

        self.generate_schedule()
        self.has_threepdm = False
        self.has_nevpt = False
        self.onlywriteIntegral = False
        self.extraline = []
        self.dmrg_switch_tol = 1.0e-3

        self._keys = set(self.__dict__.keys())


    def generate_schedule(self):

        if self.startM == None:
            self.startM = 25
        if len(self.scheduleSweeps) == 0:
            startM = self.startM
            N_sweep = 0
            if self.restart or self.force_restart :
                Tol = self.tol/10.0
            else:
                Tol = 1.0e-4
            Noise = Tol
            while startM < self.maxM:
                self.scheduleSweeps.append(N_sweep)
                N_sweep +=2
                self.scheduleMaxMs.append(startM)
                startM *=2
                self.scheduleTols.append(Tol) 
                self.scheduleNoises.append(Noise) 
            while Tol > self.tol:
                self.scheduleSweeps.append(N_sweep)
                N_sweep +=2
                self.scheduleMaxMs.append(self.maxM)
                self.scheduleTols.append(Tol)
                Tol /=10.0
                self.scheduleNoises.append(0.0)
            self.scheduleSweeps.append(N_sweep)
            N_sweep +=2
            self.scheduleMaxMs.append(self.maxM)
            self.scheduleTols.append(self.tol)
            self.scheduleNoises.append(0.0)
            self.twodot_to_onedot = N_sweep+2
            self.maxIter = self.twodot_to_onedot+20


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
        log.info('scheduleSweeps = %s', str(self.scheduleSweeps))
        log.info('scheduleMaxMs = %s', str(self.scheduleMaxMs))
        log.info('scheduleTols = %s', str(self.scheduleTols))
        log.info('scheduleNoises = %s', str(self.scheduleNoises))
        log.info('twodot_to_onedot = %d', self.twodot_to_onedot)
        log.info('tol = %g', self.tol)
        log.info('maxM = %d', self.maxM)
        log.info('fullrestart = %s', str(self.restart or self.force_restart))
        log.info('dmrg switch tol =%s', self.dmrg_switch_tol)
        log.info('wfnsym = %s', self.wfnsym)

    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        import os
        f = open(os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_twopdm.%d.%d.txt" %(state, state)), 'r')

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        norb_read = int(f.readline().split()[0])
        assert(norb_read == norb)

        for line in f.readlines():
            linesp = line.split()
            twopdm[int(linesp[0]),int(linesp[3]),int(linesp[1]),int(linesp[2])] = 2.0*float(linesp[4])

        onepdm = numpy.einsum('ikjj->ik', twopdm)
        onepdm /= (nelectrons-1)

        return onepdm

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        import os
        f = open(os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_twopdm.%d.%d.txt" %(state, state)), 'r')

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        norb_read = int(f.readline().split()[0])
        assert(norb_read == norb)

        for line in f.readlines():
            linesp = line.split()
            twopdm[int(linesp[0]),int(linesp[3]),int(linesp[1]),int(linesp[2])] = 2.0*float(linesp[4])

        onepdm = numpy.einsum('ijkk->ij', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def make_rdm123(self, state, norb, nelec,  link_index=None, **kwargs):
        import os
        if self.has_threepdm == False:
            self.twopdm = False
            self.extraline.append('restart_threepdm')
            if isinstance(nelec, (int, numpy.integer)):
                neleca = nelec//2 + nelec%2
                nelecb = nelec - neleca
            else :
                neleca, nelecb = nelec
            writeDMRGConfFile(neleca, nelecb, True, self)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True
            self.extraline.pop()

        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        f = open(os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.txt" %(state, state)), 'r')

        threepdm = numpy.zeros( (norb, norb, norb, norb, norb, norb) )
        norb_read = int(f.readline().split()[0])
        assert(norb_read == norb)

        for line in f.readlines():
            linesp = line.split()
            threepdm[int(linesp[0]),int(linesp[1]),int(linesp[2]),int(linesp[3]),int(linesp[4]),int(linesp[5])] = float(linesp[6])

        twopdm = numpy.einsum('ijkklm->ijlm',threepdm)
        twopdm /= (nelectrons-2)
        onepdm = numpy.einsum('ijjk->ik', twopdm)
        onepdm /= (nelectrons-1)

        threepdm = numpy.einsum('jk,lm,in->ijklmn',numpy.identity(norb),numpy.identity(norb),onepdm)\
                 + numpy.einsum('jk,miln->ijklmn',numpy.identity(norb),twopdm)\
                 + numpy.einsum('lm,kijn->ijklmn',numpy.identity(norb),twopdm)\
                 + numpy.einsum('jm,kinl->ijklmn',numpy.identity(norb),twopdm)\
                 + numpy.einsum('mkijln->ijklmn',threepdm)

        twopdm = numpy.einsum('iklj->ijkl',twopdm) + numpy.einsum('il,jk->ijkl',onepdm,numpy.identity(norb))\

        return onepdm, twopdm, threepdm

    def nevpt_intermediate(self, type, norb, nelec, state, **kwargs):
        import os

        if self.has_nevpt == False:
            self.twopdm = False
            self.extraline.append('restart_nevpt2_npdm')
            if isinstance(nelec, (int, numpy.integer)):
                neleca = nelec//2 + nelec%2
                nelecb = nelec - neleca
            else :
                neleca, nelecb = nelec
            writeDMRGConfFile(neleca, nelecb, True, self)
            if self.verbose >= logger.DEBUG1:
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                inFile = self.configFile
                logger.debug1(self, 'Block Input conf')
                logger.debug1(self, open(inFile, 'r').read())
            executeBLOCK(self)
            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_nevpt = True
            self.extraline.pop()

        f = open(os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "%s_matrix.%d.%d.txt" %(type,state, state)), 'r')

        a16 = numpy.zeros( (norb, norb, norb, norb, norb, norb) )
        norb_read = int(f.readline().split()[0])
        assert(norb_read == norb)

        for line in f.readlines():
            linesp = line.split()
            a16[int(linesp[0]),int(linesp[1]),int(linesp[2]),int(linesp[3]),int(linesp[4]),int(linesp[5])] = float(linesp[6])

        return a16

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, **kwargs):
        if self.nroots==1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self.force_restart
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else :
            neleca, nelecb = nelec

        writeIntegralFile(h1e, eri, norb, neleca, nelecb, self)
        writeDMRGConfFile(neleca, nelecb, fciRestart, self)
        if self.verbose >= logger.DEBUG1:
            inFile = self.configFile
            #inFile = os.path.join(self.scratchDirectory,self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral :
            logger.info(self,'Only write integral')
            try:
                calc_e = readEnergy(self)
            except IOError:
                if self.nroots ==1:
                    calc_e = 0.0
                else :
                    calc_e = [0.0] * self.nroots
            return calc_e, roots
            
        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = self.outputFile
            #outFile = os.path.join(self.scratchDirectory,self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        return calc_e, roots

    def approx_kernel(self, h1e, eri, norb, nelec, fciRestart=None, **kwargs):
        fciRestart = True 
        if isinstance(nelec, (int, numpy.integer)):
            neleca = nelec//2 + nelec%2
            nelecb = nelec - neleca
        else :
            neleca, nelecb = nelec

        writeIntegralFile(h1e, eri, norb, neleca, nelecb, self)
        writeDMRGConfFile(neleca, nelecb, fciRestart, self, True)
        if self.verbose >= logger.DEBUG1:
            inFile = self.configFile
            #inFile = os.path.join(self.scratchDirectory,self.configFile)
            logger.debug1(self, 'Block Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeBLOCK(self)
        if self.verbose >= logger.DEBUG1:
            outFile = self.outputFile
            #outFile = os.path.join(self.scratchDirectory,self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        if self.nroots==1:
            roots = 0
        else:
            roots = range(self.nroots)
        return calc_e, roots

    def restart_scheduler_(self):
        def callback(envs):
            if (envs['norm_gorb'] < self.dmrg_switch_tol or
                ('norm_ddm' in envs and envs['norm_ddm'] < self.dmrg_switch_tol*10)):
                self.restart = True
            else :
                self.restart = False
        return callback


def make_schedule(sweeps, Ms, tols, noises, twodot_to_onedot):
    if len(sweeps) == len(Ms) == len(tols) == len(noises):
        schedule = ['schedule']
        for i, s in enumerate(sweeps):
            schedule.append('%d %6d  %8.4e  %8.4e' % (s, Ms[i], tols[i], noises[i]))
        schedule.append('end')
        if (twodot_to_onedot != 0):
            schedule.append('twodot_to_onedot %i'%twodot_to_onedot)
        return '\n'.join(schedule)
    else:
        
        return 'schedule default\nmaxM %s'%Ms[-1]

def writeDMRGConfFile(neleca, nelecb, Restart, DMRGCI, approx= False):
    confFile = DMRGCI.configFile
    #confFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.configFile)

    f = open(confFile, 'w')
    f.write('nelec %i\n'%(neleca+nelecb))
    f.write('spin %i\n' %(neleca-nelecb))
    if isinstance(DMRGCI.wfnsym, str):
        wfnsym = dmrg_sym.irrep_name2id(DMRGCI.mol.groupname, DMRGCI.wfnsym)
    else:
        wfnsym = DMRGCI.wfnsym
    f.write('irrep %i\n' % wfnsym)

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
                                 DMRGCI.scheduleNoises,
                                 DMRGCI.twodot_to_onedot)
        f.write('%s\n' % schedule)
    else :
        f.write('schedule\n')
        #if approx == True :
        #    f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol*10.0, 0e-6))
        #else :
        #    f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol, 0e-6))
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10, 0e-6))
        f.write('end\n')
        f.write('fullrestart\n')
        f.write('onedot \n')

    if DMRGCI.mol.symmetry:
        if DMRGCI.groupname.lower() == 'dooh':
            f.write('sym d2h\n' )
        elif DMRGCI.groupname.lower() == 'cooh':
            f.write('sym c2h\n' )
        else:
            f.write('sym %s\n' % DMRGCI.groupname.lower())
    f.write('orbitals %s\n' % DMRGCI.integralFile)
    #f.write('orbitals %s\n' % os.path.join(DMRGCI.scratchDirectory,
    #                                       DMRGCI.integralFile))
    if approx == True :
        f.write('maxiter %i\n'%DMRGCI.approx_maxIter)
    else :
        f.write('maxiter %i\n'%DMRGCI.maxIter)
    f.write('sweep_tol %8.4e\n'%DMRGCI.tol)

    f.write('outputlevel %s\n'%DMRGCI.outputlevel)
    f.write('hf_occ integral\n')
    if(DMRGCI.twopdm):
        f.write('twopdm\n')
    if(DMRGCI.nonspinAdapted):
        f.write('nonspinAdapted\n')
    if(DMRGCI.scratchDirectory):
        f.write('prefix  %s\n'%DMRGCI.scratchDirectory)
    if (DMRGCI.nroots !=1):
        f.write('nroots %d\n'%DMRGCI.nroots)
        if (DMRGCI.weights==[]):
            DMRGCI.weights= [1.0/DMRGCI.nroots]* DMRGCI.nroots
        f.write('weights ')
        for weight in DMRGCI.weights:
            f.write('%f '%weight)
        f.write('\n')
    for line in DMRGCI.extraline:
        f.write('%s\n'%line)
    f.close()
    #no reorder
    #f.write('noreorder\n')

def writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb, DMRGCI):
    #integralFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.integralFile)
    integralFile = DMRGCI.integralFile
# ensure 4-fold symmetry
    eri_cas = pyscf.ao2mo.restore(4, eri_cas, ncas)
    if DMRGCI.mol.symmetry and DMRGCI.orbsym:
        if DMRGCI.groupname.lower() == 'dooh':
            orbsym = [IRREP_MAP['D2h'][i % 10] for i in DMRGCI.orbsym]
        elif DMRGCI.groupname.lower() == 'cooh':
            orbsym = [IRREP_MAP['C2h'][i % 10] for i in DMRGCI.orbsym]
        else:
            orbsym = [IRREP_MAP[DMRGCI.groupname][i] for i in DMRGCI.orbsym]
    else:
        orbsym = []
    if not os.path.exists(DMRGCI.scratchDirectory):
        os.makedirs(DMRGCI.scratchDirectory)
    f = open(integralFile, 'w+')
    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri_cas, ncas,
                                       neleca+nelecb, ms=abs(neleca-nelecb),
                                       orbsym=orbsym)

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

    inFile = DMRGCI.configFile
    outFile = DMRGCI.outputFile
    #inFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.configFile)
    #outFile = os.path.join(DMRGCI.scratchDirectory,DMRGCI.outputFile)
    from subprocess import check_call
    try:
        output = check_call("%s  %s  %s > %s"%(DMRGCI.mpiprefix, DMRGCI.executable, inFile, outFile), shell=True)
    except ValueError:
        print(output)
        exit()

def readEnergy(DMRGCI):
    import struct, os
    file1 = open(os.path.join('%s/%s/'%(DMRGCI.scratchDirectory,"node0"), "dmrg.e"),"rb")
    format = ['d']*DMRGCI.nroots
    format = ''.join(format)
    calc_e = struct.unpack(format, file1.read())
    file1.close()
    if DMRGCI.nroots ==1:
        return calc_e[0]
    else:
        return list(calc_e)


def DMRGSCF(mf, norb, nelec, *args, **kwargs):
    '''Wrapper for DMRG-SCF, to setup CASSCF object using the DMRGCI solver'''
    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = DMRGCI(mf.mol)
    mc.callback = mc.fcisolver.restart_scheduler_()
    mc.chkfile = tempfile.mktemp(dir=settings.BLOCKSCRATCHDIR)
    return mc


def DMRG_MPS_NEVPT(mc, root=0, fcisolver=None, maxm = 500, tol =1e-6, parallel= True):
    
    if (isinstance(mc, basestring)):
        fh5 = h5py.File(mc,'r')

        mol = eval(fh5['mol'].value)
        ncas = fh5['mc/ncas'].value
        ncore = fh5['mc/ncore'].value
        nvirt = fh5['mc/nvirt'].value
        nelecas = fh5['mc/nelecas'].value
        fh5.close()
        mc_chk = mc
    else :
        mol = mc.mol
        ncas = mc.ncas
        ncore = mc.ncore
        nvirt = mc.mo_coeff.shape[1] - mc.ncas-mc.ncore
        nelecas = mc.nelecas
        mc_chk = 'mc_chkfile'
        write_chk(mc,root,mc_chk)
        



    if fcisolver is None:
        fcisolver = DMRGCI(mol, maxm, tol)
    fcisolver.twopdm = False
    fcisolver.nroots = mc.fcisolver.nroots
    scratch = fcisolver.scratchDirectory
    fcisolver.scratchDirectory = ''
    #if (not parallel):
    #    ci.extraline.append('restart_mps_nevpt %d %d %d'%(ncas,ncore, nvirt))


    fcisolver.extraline.append('fullrestart')
    fcisolver.extraline.append('nevpt_state_num %d'%root)
    
    writeDMRGConfFile(nelecas[0], nelecas[1], False, fcisolver)
    fcisolver.scratchDirectory = scratch

    if fcisolver.verbose >= logger.DEBUG1:
        inFile = fcisolver.configFile
        #inFile = os.path.join(self.scratchDirectory,self.configFile)
        logger.debug1(fcisolver, 'Block Input conf')
        logger.debug1(fcisolver, open(inFile, 'r').read())

    t0 = (time.clock(), time.time())

    from subprocess import check_call
    import os
    full_path = os.path.realpath(__file__)
    check_call('%s %s/nevpt_mpi.py %s %s %s %s %s'%(fcisolver.mpiprefix, os.path.dirname(full_path), mc_chk, fcisolver.executable, fcisolver.configFile,fcisolver.outputFile, fcisolver.scratchDirectory), shell=True)

    if fcisolver.verbose >= logger.DEBUG1:
        logger.debug1(fcisolver, open(os.path.join(fcisolver.scratchDirectory, '0/dmrg.out')).read())

    import h5py
    fh5 = h5py.File('Perturbation_%d'%root,'r')
    Vi_e  =  fh5['Vi/energy'].value      
    Vi_n  =  fh5['Vi/norm'].value        
    Vr_e  =  fh5['Vr/energy'].value      
    Vr_n  =  fh5['Vr/norm'].value        
    fh5.close()
    logger.note(fcisolver,'Nevpt Energy:')
    logger.note(fcisolver,'Sr Subspace: Norm = %s, E = %s'%(Vr_n, Vr_e))
    logger.note(fcisolver,'Si Subspace: Norm = %s, E = %s'%(Vi_n, Vi_e))

    logger.timer(fcisolver,'MPS NEVPT calculation time', *t0)

    #if (parallel):
    #    from subprocess import check_call
    #    check_call('/home/shengg/opt/pyscf/future/dmrgscf/nevpt_mpi.py mc_chk %s %s %s'%(ci.executable, ci.configFile,ci.outputFile), shell=True)
    #    #check_call('%s /home/shengg/opt/pyscf/future/dmrgscf/nevpt_mpi.py mc_chk %s %s %s'%(ci.mpiprefix, ci.executable, ci.configFile,ci.outputFile))
    #else:
    #    nevpt_integral(mc)
    #    executeBLOCK(ci)



 #   if ci.verbose >= logger.DEBUG1:
 #       outFile = ci.outputFile
 #       logger.debug1(ci, open(outFile).read())
    #calc_e = readEnergy(self)

    #return calc_e, None


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    settings.MPIPREFIX =''
    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-dmrgci',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = DMRGSCF(m, 4, 4)
    mc.fcisolver.tol = 1e-9
    emc_1 = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    mc.fcisolver.scheduleSweeps = []
    emc_0 = mc.casci()[0]

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-casscf',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 4, 4)
    emc_1ref = mc.mc2step()[0]

    mc = mcscf.CASCI(m, 4, 4)
    emc_0ref = mc.casci()[0]

    print('DMRGCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('DMRGSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))

