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
# Author: Sandeep Sharma <sanshar@gmail.com>
#
'''
VMC solver for CASCI and CASSCF.
'''

from functools import reduce
import ctypes
import os
import sys
import struct
import time
import tempfile
import warnings
from subprocess import check_call
from subprocess import CalledProcessError

from pyscf.lo import pipek, boys, edmiston, iao, ibo
from pyscf import ao2mo, tools, scf, mcscf, lo, gto
import numpy 
import scipy
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf import mcscf
ndpointer = numpy.ctypeslib.ndpointer

# Settings
try:
    from pyscf.vmcscf import settings
except ImportError:
    from pyscf import __config__
    settings = lambda: None
    settings.VMCEXE = getattr(__config__, 'vmc_VMCEXE', None)
    settings.VMCSCRATCHDIR = getattr(__config__, 'vmc_VMCSCRATCHDIR', None)
    settings.VMCRUNTIMEDIR = getattr(__config__, 'vmc_VMCRUNTIMEDIR', None)
    settings.MPIPREFIX = getattr(__config__, 'vmc_MPIPREFIX', None)
    if (settings.VMCEXE is None or settings.VMCSCRATCHDIR is None):
        import sys
        sys.stderr.write('settings.py not found for module vmcscf.  Please create %s\n'
                         % os.path.join(os.path.dirname(__file__), 'settings.py'))
        raise ImportError('settings.py not found')

# Libraries
from pyscf.lib import load_library
libE3unpack = load_library('libicmpspt')
# TODO: Organize this better.
vmcLib = load_library('libshciscf')


writeIntNoSymm = vmcLib.writeIntNoSymm
writeIntNoSymm.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_int,
    ndpointer(ctypes.c_int)
]

fcidumpFromIntegral = vmcLib.fcidumpFromIntegral
fcidumpFromIntegral.restype = None
fcidumpFromIntegral.argtypes = [
    ctypes.c_char_p,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_double,
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), ctypes.c_size_t
]

r2RDM = vmcLib.r2RDM
r2RDM.restype = None
r2RDM.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t,
    ctypes.c_char_p
]


class VMC(pyscf.lib.StreamObject):
    r'''VMC program interface and object to hold VMC program input parameters.

    Attributes:
        initialStates: [[int]]
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            VMC program using C1 symmetry.
        useExtraSymm : False
            if the symmetry of the molecule is Dooh or Cooh, then this keyword uses
            complex orbitals to make full use of this symmetry

    Examples:

    '''

    def __init__(self, mol=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.executable = settings.VMCEXE
        self.scratchDirectory = settings.VMCSCRATCHDIR
        self.mpiprefix = settings.MPIPREFIX

        self.wavefunction = "jastrowslater"
        self.slater = "ghf"
        self.maxIter = 100
        self.stochasticIter = 1000
        self.stochasticIterTight = 4*self.stochasticIter

        self.integralFile = "FCIDUMP"
        self.configFile = "vmc.dat"
        self.rdmconfigFile = "rdmvmc.dat"
        self.outputFile = "vmc.out"
        self.rdmoutputFile = "rdmvmc.out"
        if getattr(settings, 'VMCRUNTIMEDIR', None):
            self.runtimeDir = settings.VMCRUNTIMEDIR
        else:
            self.runtimeDir = '.'
        self.extraline = []

        if mol is None:
            self.groupname = None
        else:
            if mol.symmetry:
                self.groupname = mol.groupname
            else:
                self.groupname = None


    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info('')
        log.info('******** VMC flags ********')
        log.info('executable             = %s', self.executable)
        log.info('mpiprefix              = %s', self.mpiprefix)
        log.info('scratchDirectory       = %s', self.scratchDirectory)
        log.info('integralFile           = %s',
                 os.path.join(self.runtimeDir, self.integralFile))
        log.info('configFile             = %s',
                 os.path.join(self.runtimeDir, self.configFile))
        log.info('outputFile             = %s',
                 os.path.join(self.runtimeDir, self.outputFile))
        log.info('maxIter                = %d', self.maxIter)
        log.info('')
        return self

    # ABOUT RDMs AND INDEXES: -----------------------------------------------------------------------
    #   There is two ways to stored an RDM
    #   (the numbers help keep track of creation/annihilation that go together):
    #     E3[i1,j2,k3,l3,m2,n1] is the way DICE outputs text and bin files
    #     E3[i1,j2,k3,l1,m2,n3] is the way the tensors need to be written for SQA and ICPT
    #
    #   --> See various remarks in the pertinent functions below.
    # -----------------------------------------------------------------------------------------------

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        # The 2RDMs written by "VMCrdm::saveRDM" in DICE
        # are written as E2[i1,j2,k1,l2]
        # and stored here as E2[i1,k1,j2,l2] (for PySCF purposes)
        # This is NOT done with SQA in mind.
        twopdm = numpy.zeros((norb, norb, norb, norb))
        file2pdm = "spatialRDM.%d.%d.txt" % (state, state)
        # file2pdm = file2pdm.encode()  # .encode for python3 compatibility
        r2RDM(twopdm, norb,
              os.path.join(file2pdm).encode())

        # (This is coherent with previous statement about indexes)
        onepdm = numpy.einsum('ikjj->ki', twopdm)
        onepdm /= (nelectrons - 1)
        return onepdm, twopdm


    def make_rdm12_forSQA(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        # The 2RDMs written by "VMCrdm::saveRDM" in DICE
        # are written as E2[i1,j2,k1,l2]
        # and stored here as E2[i1,k1,j2,l2] (for PySCF purposes)
        # This is NOT done with SQA in mind.
        twopdm = numpy.zeros((norb, norb, norb, norb))
        file2pdm = "spatialRDM.%d.%d.txt" % (state,state)
        r2RDM(twopdm, norb,
              os.path.join(self.scratchDirectory, file2pdm).endcode())
        twopdm=twopdm.transpose(0,2,1,3)

        # (This is coherent with previous statement about indexes)
        onepdm = numpy.einsum('ijkj->ki', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm


    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0,
               **kwargs):
        """
        Approximately solve CI problem for the specified active space.
        """
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        self.writeConfig()
        executeVMC(self)

        #onerdm, twordm = make_rdm12(self, 0, norb, nelec)

        outFile = os.path.join(self.runtimeDir, self.outputFile)
        f = open(outFile, 'r')
        l = f.readlines()
        calc_e = float(l[-1].split()[1])
        roots = 0
        return calc_e, roots


    def writeConfig(self, restart=True, readBestDeterminant=False):
        confFile = os.path.join(self.runtimeDir, self.configFile)

        f = open(confFile, 'w')
        f.write("%s\n" %self.wavefunction)
        f.write("complex\n")
        f.write("%s\n" %self.slater)
        f.write("maxiter %d\n" %self.maxIter)
        f.write("stochasticIter %d\n" % self.stochasticIter)
        #if (restart) :
        #f.write("fullrestart\n")
        if (readBestDeterminant) :
            f.write("determinants bestDet")
        f.close()


        confFile = os.path.join(self.runtimeDir, self.rdmconfigFile)

        f = open(confFile, 'w')
        f.write("slatertwordm\n")
        f.write("complex\n")
        f.write("%s\n" %self.slater)
        f.write("maxiter %d\n" %self.maxIter)
        f.write("stochasticIter %d\n" % self.stochasticIterTight)
        f.close()
        

    def spin_square(self, civec, norb, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec // 2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec
        s = (neleca - nelecb) * .5
        ss = s * (s + 1)
        if isinstance(civec, int):
            return ss, s * 2 + 1
        else:
            return [ss] * len(civec), [s * 2 + 1] * len(civec)

    def cleanup_dice_files(self):
        """
        Remove the files used for Dice communication.
        """
        os.remove("input.dat")
        os.remove("output.dat")
        os.remove("FCIDUMP")


def print1Int(h1, name):
    with open('%s.X' % (name), 'w') as fout:
        fout.write('%d\n' % h1[0].shape[0])
        for i in range(h1[0].shape[0]):
            for j in range(h1[0].shape[0]):
                if (abs(h1[0, i, j]) > 1.e-8):
                    fout.write(
                        '%16.10g %4d %4d\n' % (h1[0, i, j], i + 1, j + 1))

    with open('%s.Y' % (name), 'w') as fout:
        fout.write('%d\n' % h1[1].shape[0])
        for i in range(h1[1].shape[0]):
            for j in range(h1[1].shape[0]):
                if (abs(h1[1, i, j]) > 1.e-8):
                    fout.write(
                        '%16.10g %4d %4d\n' % (h1[1, i, j], i + 1, j + 1))

    with open('%s.Z' % (name), 'w') as fout:
        fout.write('%d\n' % h1[2].shape[0])
        for i in range(h1[2].shape[0]):
            for j in range(h1[2].shape[0]):
                if (abs(h1[2, i, j]) > 1.e-8):
                    fout.write(
                        '%16.10g %4d %4d\n' % (h1[2, i, j], i + 1, j + 1))

    with open('%sZ' % (name), 'w') as fout:
        fout.write('%d\n' % h1[2].shape[0])
        for i in range(h1[2].shape[0]):
            for j in range(h1[2].shape[0]):
                if (abs(h1[2, i, j]) > 1.e-8):
                    fout.write(
                        '%16.10g %4d %4d\n' % (h1[2, i, j], i + 1, j + 1))



def writeIntegralFile(VMC, h1eff, eri_cas, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec // 2 + nelec % 2
        nelecb = nelec - neleca
    else:
        neleca, nelecb = nelec

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(VMC.runtimeDir, VMC.integralFile)

    if not os.path.exists(VMC.scratchDirectory):
        os.makedirs(VMC.scratchDirectory)

    from pyscf import symm
    from pyscf.dmrgscf import dmrg_sym

    if (VMC.groupname == 'Dooh'
            or VMC.groupname == 'Coov') and VMC.useExtraSymm:
        coeffs, nRows, rowIndex, rowCoeffs, orbsym = D2htoDinfh(
            VMC, norb, nelec)

        newintt = numpy.tensordot(coeffs.conj(), h1eff, axes=([1], [0]))
        newint1 = numpy.tensordot(newintt, coeffs, axes=([1], [1]))
        newint1r = numpy.zeros(shape=(norb, norb), order='C')
        for i in range(norb):
            for j in range(norb):
                newint1r[i, j] = newint1[i, j].real
        int2 = pyscf.ao2mo.restore(1, eri_cas, norb)
        eri_cas = numpy.zeros_like(int2)

        transformDinfh(norb, numpy.ascontiguousarray(nRows, numpy.int32),
                       numpy.ascontiguousarray(rowIndex, numpy.int32),
                       numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                       numpy.ascontiguousarray(int2, numpy.float64),
                       numpy.ascontiguousarray(eri_cas, numpy.float64))

        writeIntNoSymm(norb, numpy.ascontiguousarray(newint1r, numpy.float64),
                       numpy.ascontiguousarray(eri_cas, numpy.float64),
                       ecore, neleca + nelecb,
                       numpy.asarray(orbsym, dtype=numpy.int32))

    else:
        if VMC.groupname is not None and VMC.orbsym is not []:
            orbsym = dmrg_sym.convert_orbsym(VMC.groupname, VMC.orbsym)
        else:
            orbsym = [1] * norb

        eri_cas = pyscf.ao2mo.restore(8, eri_cas, norb)
        # Writes the FCIDUMP file using functions in VMC_tools.cpp.
        integralFile = integralFile.encode()  # .encode for python3 compatibility
        fcidumpFromIntegral(integralFile, h1eff, eri_cas, norb,
                            neleca + nelecb, ecore,
                            numpy.asarray(orbsym, dtype=numpy.int32),
                            abs(neleca - nelecb))


def executeVMC(VMC):
    inFile = os.path.join(VMC.runtimeDir, VMC.configFile)
    outFile = os.path.join(VMC.runtimeDir, VMC.outputFile)
    try:
        cmd = ' '.join((VMC.mpiprefix, VMC.executable, inFile))
        cmd = "%s > %s 2>&1" % (cmd, outFile)
        check_call(cmd, shell=True)
        #save_output(VMC)
    except CalledProcessError as err:
        logger.error(VMC, cmd)
        raise err

    inFile = os.path.join(VMC.runtimeDir, VMC.rdmconfigFile)
    outFile = os.path.join(VMC.runtimeDir, VMC.rdmoutputFile)
    try:
        cmd = ' '.join((VMC.mpiprefix, VMC.executable, inFile))
        cmd = "%s > %s 2>&1" % (cmd, outFile)
        check_call(cmd, shell=True)
        #save_output(VMC)
    except CalledProcessError as err:
        logger.error(VMC, cmd)
        raise err


#def save_output(VMC):
#  for i in range(50):
#    if os.path.exists(os.path.join(VMC.runtimeDir, "output%02d.dat"%(i))):
#      continue
#    else:
#      import shutil
#      shutil.copy2(os.path.join(VMC.runtimeDir, "output.dat"),os.path.join(VMC.runtimeDir, "output%02d.dat"%(i)))
#      shutil.copy2(os.path.join(VMC.runtimeDir, "%s/vmc.e"%(VMC.scratchDirectory)), os.path.join(VMC.runtimeDir, "vmc%02d.e"%(i)))
#      #print('BM copied into "output%02d.dat"'%(i))
#      #print('BM copied into "vmc%02d.e"'%(i))
#      break


def readEnergy(VMC):
    file1 = open(
        os.path.join(VMC.runtimeDir, "%s/vmc.e" % (VMC.scratchDirectory)),
        "rb")
    format = ['d'] * VMC.nroots
    format = ''.join(format)
    calc_e = struct.unpack(format, file1.read())
    file1.close()
    if VMC.nroots == 1:
        return calc_e[0]
    else:
        return list(calc_e)
    
def localizeValence(mf, mo_coeff, method="iao"):
    if (method == "iao"):
        return iao.iao(mf.mol, mo_coeff)
    elif (method == "ibo"):
        a = iao.iao(mf.mol, mo_coeff)
        a = lo.vec_lowdin(a, mf.get_ovlp())
        return ibo.ibo(mf.mol, mo_coeff, iaos=a)
    elif (method == "boys"):
        return boys.Boys(mf.mol).kernel(mo_coeff)
    elif (method == "er"):
        return edmiston.ER(mf.mol).kernel(mo_coeff)

# can be used for all electron, but not recommended
def bestDetValence(mol, lmo, occ, eri, writeToFile=True):

    # index of the ao contributing the most to an lmo
    maxLMOContributers = [ numpy.argmax(numpy.abs(lmo[::,i])) for i in range(lmo.shape[1]) ]  

    # end AO index for each atom in ascending order
    atomNumAOs = [ i[1][3] - 1 for i in enumerate(mol.aoslice_nr_by_atom()) ]  

    lmoSites = [ [] for i in range(mol.natm) ] #lmo's cetered on each atom
    for i in enumerate(maxLMOContributers):
        lmoSites[numpy.searchsorted(numpy.array(atomNumAOs), i[1])].append(i[0])

    bestDet = ['0' for i in range(lmo.shape[1])]
    def pair(i):
        return i*(i+1)//2+i
    for i in enumerate(occ):
        if eri.ndim == 2:
            onSiteIntegrals = [ (j, eri[pair(j),pair(j)]) for (n,j) in enumerate(lmoSites[i[0]]) ]
        elif eri.ndim == 1:
            onSiteIntegrals = [ (j, eri[pair(pair(j))]) for (n,j) in enumerate(lmoSites[i[0]]) ]
            onSiteIntegrals.sort(key = lambda tup : tup[1], reverse=True)
        for k in range(i[1][0]):
            bestDet[onSiteIntegrals[k][0]] = '2'
        for l in range(i[1][1]):
            bestDet[onSiteIntegrals[i[1][0] + l][0]] = 'a'
        for m in range(i[1][2]):
            bestDet[onSiteIntegrals[i[1][0] + i[1][1] + m][0]] = 'b'

    bestDetStr = '  '.join(bestDet)
    print('bestDet:  ' + bestDetStr)
    if writeToFile:
        fileh = open("bestDet", 'w')
        fileh.write('1.   ' + bestDetStr + '\n')
        fileh.close()

    return bestDetStr

def writeMat(mat, fileName, isComplex):
    fileh = open(fileName, 'w')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (isComplex):
                fileh.write('(%16.10e, %16.10e) '%(mat[i,j].real, mat[i,j].imag))
            else:
                fileh.write('%16.10e '%(mat[i,j]))
        fileh.write('\n')
    fileh.close()

def VMCSCF(mf, ncore, nact, occ=None, frozen = None, loc="iao", proc=None,*args, **kwargs):

    mol = mf.mol
    mo = mf.mo_coeff
    nelec = mol.nelectron - 2 * ncore
    mc = mcscf.CASSCF(mf, nact, nelec)

    if loc is not None:
        lmo = localizeValence(mf, mo[:, ncore:ncore+nact], loc)
        tools.molden.from_mo(mf.mol, 'valenceOrbs.molden', lmo)

        h1cas, energy_core = mcscf.casci.h1e_for_cas(mc, mf.mo_coeff, nact, ncore)
        mo_core = mc.mo_coeff[:,:ncore]
        core_dm = 2 * mo_core.dot(mo_core.T)
        corevhf = mc.get_veff(mol, core_dm)
        h1eff = lmo.T.dot(mc.get_hcore() + corevhf).dot(lmo)
        eri = ao2mo.kernel(mol, lmo)
    else:
        lmo = numpy.eye(nact)
        h1eff = mf.get_hcore()
        eri = mf._eri
        energy_core = 0
    if occ is not None:
        bestDetValence(mol, lmo, occ, eri, True)


    #prepare initial guess for the HF orbitals
    norb = nact
    molA = gto.M()
    molA.nelectron = nelec
    molA.verbose = 4
    molA.incore_anyway = True
    gmf = scf.GHF(molA)
    gmf.get_hcore = lambda *args: scipy.linalg.block_diag(h1eff, h1eff)
    gmf.get_ovlp = lambda *args: numpy.identity(2*norb)
    gmf.energy_nuc = lambda *args: energy_core
    gmf._eri = eri

    dm = gmf.get_init_guess()
    dm = dm + 2 * numpy.random.rand(2*norb, 2*norb)
    gmf.level_shift = 0.1
    gmf.max_cycle = 500
    print(gmf.kernel(dm0 = dm))
    mocoeff = numpy.zeros((2*norb, 2*norb), dtype=complex)
    mocoeff = 1.*gmf.mo_coeff
    writeMat(gmf.mo_coeff, "hf.txt", True)

    #run an initial very short turn
    mc.fcisolver = VMC(mf.mol)
    mc.fcisolver.maxIter = 5
    writeIntegralFile(mc.fcisolver, h1eff, eri, nact, nelec, energy_core)
    mc.fcisolver.writeConfig(restart=False, readBestDeterminant= (occ is not None))
    mc.fcisolver.mpiprefix = "mpirun"
    if (proc is not None):
        mc.fcisolver.mpiprefix = ("mpirun -np %d" %(proc))
    executeVMC(mc.fcisolver)


    mc.fcisolver.maxIter = 100
    mf.mo_coeff[:,ncore:ncore+nact] = lmo
    mc.mo_coeff = 1.*mf.mo_coeff
    mc.internal_rotation = True

    if frozen is not None:
        mc.frozen = frozen

    if mc.chkfile == mc._scf._chkfile.name:
        # Do not delete chkfile after mcscf
        mc.chkfile = tempfile.mktemp(dir=settings.VMCSCRATCHDIR)
        if not os.path.exists(settings.VMCSCRATCHDIR):
            os.makedirs(settings.VMCSCRATCHDIR)
    return mc


if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, dmrgscf
    from pyscf.vmcscf import vmc

    # Initialize benzene molecule
    atomstring = '''
    C  0.000517 0.000000  0.000299
    C  0.000517 0.000000  1.394692
    C  1.208097 0.000000  2.091889
    C  2.415677 0.000000  1.394692
    C  2.415677 0.000000  0.000299
    C  1.208097 0.000000 -0.696898
    H -0.939430 0.000000 -0.542380
    H -0.939430 0.000000  1.937371
    H  1.208097 0.000000  3.177246
    H  3.355625 0.000000  1.937371
    H  3.355625 0.000000 -0.542380
    H  1.208097 0.000000 -1.782255
    '''
    mol = gto.M(
        atom = atomstring,
        unit = 'angstrom',
        basis = 'sto-6g',
        verbose = 4,
        symmetry= 0,
        spin = 0)

    # Create HF molecule
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.scf()


    ncore, nact = 6, 30
    occ = []
    configC = [[0,4,0], [0,0,4]] # no double occ, 4 up or 4 dn
    configH = [[0,1,0], [0,0,1]] # no double occ, 1 up or 1 dn
    for i in range(6):
        occ.append(configC[i%2])
    for i in range(6):
        occ.append(configH[(i+1)%2])


    frozen = list(range(0,ncore)) + list (range(36, mf.mo_coeff.shape[0]))
    print (frozen)
    # Create VMC molecule for just variational opt.
    # Active spaces chosen to reflect valence active space.
    mch = vmc.VMCSCF(mf, ncore, nact, occ=occ, loc="ibo",\
                     frozen = frozen)
    mch.fcisolver.stochasticIter = 400
    mch.fcisolver.maxIter = 50
    e_noPT = mch.mc2step()[0]

