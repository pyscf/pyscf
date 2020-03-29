#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#         James Smith <james.smith9113@gmail.com>
#
'''
SHCI solver for CASCI and CASSCF.
'''

from functools import reduce
import ctypes
import os
import sys
import re
import struct
import time
import tempfile
import warnings
from subprocess import check_call
from subprocess import CalledProcessError

import numpy
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf import mcscf
ndpointer = numpy.ctypeslib.ndpointer

# Settings
try:
    from pyscf.shciscf import settings
except ImportError:
    from pyscf import __config__
    settings = lambda: None
    settings.SHCIEXE = getattr(__config__, 'shci_SHCIEXE', None)
    settings.ZSHCIEXE = getattr(__config__, 'shci_ZSHCIEXE', None)
    settings.SHCISCRATCHDIR = getattr(__config__, 'shci_SHCISCRATCHDIR', None)
    settings.SHCIRUNTIMEDIR = getattr(__config__, 'shci_SHCIRUNTIMEDIR', None)
    settings.MPIPREFIX = getattr(__config__, 'shci_MPIPREFIX', None)
    if (settings.SHCIEXE is None or settings.SHCISCRATCHDIR is None):
        import sys
        sys.stderr.write('settings.py not found for module shciscf.  Please create %s\n'
                         % os.path.join(os.path.dirname(__file__), 'settings.py'))
        raise ImportError('settings.py not found')

# Libraries
from pyscf.lib import load_library
libE3unpack = load_library('libicmpspt')
# TODO: Organize this better.
shciLib = load_library('libshciscf')

transformDinfh = shciLib.transformDinfh
transformDinfh.restyp = None
transformDinfh.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double)
]

transformRDMDinfh = shciLib.transformRDMDinfh
transformRDMDinfh.restyp = None
transformRDMDinfh.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_int32),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double)
]

writeIntNoSymm = shciLib.writeIntNoSymm
writeIntNoSymm.argtypes = [
    ctypes.c_int,
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double), ctypes.c_double, ctypes.c_int,
    ndpointer(ctypes.c_int)
]

fcidumpFromIntegral = shciLib.fcidumpFromIntegral
fcidumpFromIntegral.restype = None
fcidumpFromIntegral.argtypes = [
    ctypes.c_char_p,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_double,
    ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), ctypes.c_size_t
]

r2RDM = shciLib.r2RDM
r2RDM.restype = None
r2RDM.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), ctypes.c_size_t,
    ctypes.c_char_p
]


class SHCI(pyscf.lib.StreamObject):
    r'''SHCI program interface and object to hold SHCI program input parameters.

    Attributes:
        davidsonTol: double
        epsilon2: double
        epsilon2Large: double
        targetError: double
        sampleN: int
        epsilon1: vector<double>
        onlyperturbative: bool
        restart: bool
        fullrestart: bool
        dE: double
        eps: double
        prefix: str
        stochastic: bool
        nblocks: int
        excitation: int
        nvirt: int
        singleList: bool
        io: bool
        nroots: int
        nPTiter: int
        DoRDM: bool
        DoSOC: bool
        sweep_iter: [int]
        sweep_epsilon: [float]
        initialStates: [[int]]
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            SHCI program using C1 symmetry.
        useExtraSymm : False
            if the symmetry of the molecule is Dooh or Cooh, then this keyword uses
            complex orbitals to make full use of this symmetry

    Examples:

    '''

    def __init__(self, mol=None, maxM=None, tol=None, num_thrds=1,
                 memory=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.executable = settings.SHCIEXE
        self.executableZDice2 = settings.ZSHCIEXE
        self.scratchDirectory = settings.SHCISCRATCHDIR
        self.mpiprefix = settings.MPIPREFIX
        self.memory = memory

        self.integralFile = "FCIDUMP"
        self.configFile = "input.dat"
        self.outputFile = "output.dat"
        if getattr(settings, 'SHCIRUNTIMEDIR', None):
            self.runtimeDir = settings.SHCIRUNTIMEDIR
        else:
            self.runtimeDir = '.'
        self.extraline = []

        # TODO: Organize into pyscf and SHCI parameters
        # Standard SHCI Input parameters
        self.dets = None
        self.davidsonTol = 5.e-5
        self.epsilon2 = 1.e-7
        self.epsilon2Large = 1000.
        self.targetError = 1.e-4
        self.sampleN = 200
        self.epsilon1 = None
        self.onlyperturbative = False
        self.fullrestart = False
        self.dE = 1.e-8
        self.eps = None
        self.stochastic = True
        self.nblocks = 1
        self.excitation = 1000
        self.nvirt = 1e6
        self.singleList = True
        self.io = True
        self.nroots = 1
        self.nPTiter = 0
        self.DoRDM = True
        self.DoTRDM = False
        self.DoSOC = False
        self.sweep_iter = []
        self.sweep_epsilon = []
        self.maxIter = 6
        self.restart = False
        self.num_thrds = num_thrds
        self.orbsym = []
        self.onlywriteIntegral = False
        self.spin = None
        self.orbsym = []
        if mol is None:
            self.groupname = None
        else:
            if mol.symmetry:
                self.groupname = mol.groupname
            else:
                self.groupname = None
        ##################################################
        # don't modify the following attributes, if you do not finish part of calculation, which can be reused.
        #DO NOT CHANGE these parameters, unless you know the code in details
        self.twopdm = True  #By default, 2rdm is calculated after the calculations of wave function.
        self.shci_extra_keyword = []  #For shci advanced user only.
        self.has_fourpdm = False
        self.has_threepdm = False
        self.has_nevpt = False
        # This flag _restart is set by the program internally, to control when to make
        # SHCI restart calculation.
        self._restart = False
        self.generate_schedule()
        self.returnInt = False
        self._keys = set(self.__dict__.keys())
        self.irrep_nelec = None
        self.useExtraSymm = False
        self.initialStates = None

    def generate_schedule(self):
        return self

    @property
    def threads(self):
        return self.num_thrds

    @threads.setter
    def threads(self, x):
        self.num_thrds = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** SHCI flags ********')
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
        log.info(
            'sweep_iter             = %s',
            '[' + ','.join(['{:>5}' for item in self.sweep_iter
                            ]).format(*self.sweep_iter) + ']')
        log.info(
            'sweep_epsilon          = %s',
            '[' + ','.join(['{:>5}' for item in self.sweep_epsilon
                            ]).format(*self.sweep_epsilon) + ']')
        log.info('nPTiter                = %i', self.nPTiter)
        log.info('Stochastic             = %r', self.stochastic)
        log.info('restart                = %s',
                 str(self.restart or self._restart))
        log.info('fullrestart            = %s', str(self.fullrestart))
        log.info('num_thrds              = %d', self.num_thrds)
        log.info('memory                 = %s', self.memory)
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
        
    def trans_rdm1(self, state_i, state_j, norb, nelec, link_index=None, **kwargs):
        if self.DoSOC:
            trdm1 = numpy.zeros((norb, norb), dtype=complex)
        else:
            trdm1 = numpy.zeros((norb, norb))
        # assume Dice prints only i < j transition rdm.
        if state_i > state_j:
            tmp = state_i
            state_i = state_j
            state_j = tmp

        filetrdm1 = os.path.join(self.scratchDirectory, "transition1RDM.%d.%d.txt" % (state_i, state_j))
        with open(filetrdm1) as f:
            line = f.readline()
            file_orb = int(line.split()[0])
            for line in f:
                orb1 = int(line.split()[0])
                orb2 = int(line.split()[1])
                if self.DoSOC:
                    val = re.split("[(,)]", line.split()[2])
                    val = complex(float(val[1]), float(val[2]))
                else:
                    val  = float(line.split()[2])
                trdm1[orb1][orb2] = val
        return trdm1
        
    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
        # Avoid calling self.make_rdm12 because it may be overloaded
        return self.make_rdm12(state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        # The 2RDMs written by "SHCIrdm::saveRDM" in DICE
        # are written as E2[i1,j2,k1,l2]
        # and stored here as E2[i1,k1,j2,l2] (for PySCF purposes)
        # This is NOT done with SQA in mind.
        twopdm = numpy.zeros((norb, norb, norb, norb))
        file2pdm = "spatialRDM.%d.%d.txt" % (state, state)
        # file2pdm = file2pdm.encode()  # .encode for python3 compatibility
        r2RDM(twopdm, norb,
              os.path.join(self.scratchDirectory, file2pdm).encode())

        # Symmetry addon
        if (self.groupname == 'Dooh'
                or self.groupname == 'Coov') and self.useExtraSymm:
            nRows, rowIndex, rowCoeffs = DinfhtoD2h(self, norb, nelec)
            twopdmcopy = 1. * twopdm
            twopdm = 0. * twopdm
            transformRDMDinfh(
                norb, numpy.ascontiguousarray(nRows, numpy.int32),
                numpy.ascontiguousarray(rowIndex, numpy.int32),
                numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                numpy.ascontiguousarray(twopdmcopy, numpy.float64),
                numpy.ascontiguousarray(twopdm, numpy.float64))
            twopdmcopy = None

        # (This is coherent with previous statement about indexes)
        onepdm = numpy.einsum('ikjj->ki', twopdm)
        onepdm /= (nelectrons - 1)
        return onepdm, twopdm

    def trans_rdm12(self,
                    statebra,
                    stateket,
                    norb,
                    nelec,
                    link_index=None,
                    **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        writeSHCIConfFile(self, nelec, True)
        executeSHCI(self)

        # The 2RDMs written by "SHCIrdm::saveRDM" in DICE
        # are written as E2[i1,j2,k1,l2]
        # and stored here as E2[i1,k1,j2,l2] (for PySCF purposes)
        # This is NOT done with SQA in mind.
        twopdm = numpy.zeros((norb, norb, norb, norb))
        file2pdm = "spatialRDM.%d.%d.txt" % (root, root)
        r2RDM(twopdm, norb,
              os.path.join(self.scratchDirectory, file2pdm).endcode())

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

        # The 2RDMs written by "SHCIrdm::saveRDM" in DICE
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

    def make_rdm123(self, state, norb, nelec, link_index=None, **kwargs):
        if self.has_threepdm == False:
            writeSHCIConfFile(self, nelec, True)
            if self.verbose >= logger.DEBUG1:
                inFile = os.path.join(self.runtimeDir, self.configFile)
                logger.debug1(self, 'SHCI Input conf')
                logger.debug1(self, open(inFile, 'r').read())

            start = time.time()
            mpisave = self.mpiprefix
            #self.mpiprefix=""
            executeSHCI(self)
            self.mpiprefix = mpisave
            end = time.time()
            print('......production of RDMs took %10.2f sec' % (end - start))
            sys.stdout.flush()

            if self.verbose >= logger.DEBUG1:
                outFile = os.path.join(self.runtimeDir, self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True

        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0] + nelec[1]

        # The 3RDMs written by "SHCIrdm::save3RDM" in DICE
        # are written as E3[i1,j2,k3,l3,m2,n1]
        # and are also stored here as E3[i1,j2,k3,l3,m2,n1]
        # This is NOT done with SQA in mind.
        start = time.time()
        threepdm = numpy.zeros((norb, norb, norb, norb, norb, norb))
        file3pdm = "spatial3RDM.%d.%d.txt" % (state, state)
        with open(os.path.join(self.scratchDirectory, file3pdm), "r") as f:
            norb_read = int(f.readline().split()[0])
            assert (norb_read == norb)
            for line in f:
                linesp = line.split()
                i,j,k, l,m,n = [int(x) for x in linesp[:6]]
                threepdm[i,j,k,l,m,n] = float(linesp[6])

        # (This is coherent with previous statement about indexes)
        twopdm = numpy.einsum('ijkklm->ijlm', threepdm)
        twopdm /= (nelectrons - 2)
        onepdm = numpy.einsum('ijjk->ki', twopdm)
        onepdm /= (nelectrons - 1)
        end = time.time()
        print('......reading the RDM took    %10.2f sec' % (end - start))
        sys.stdout.flush()
        return onepdm, twopdm, threepdm

    def _make_dm123(self, state, norb, nelec, link_index=None, **kwargs):
        r'''Note this function does NOT compute the standard density matrix.
        The density matrices are reordered to match the the fci.rdm.make_dm123
        function (used by NEVPT code).
        The returned "2pdm" is :math:`\langle p^\dagger q r^\dagger s\rangle`;
        The returned "3pdm" is :math:`\langle p^\dagger q r^\dagger s t^\dagger u\rangle`.
        '''
        onepdm, twopdm, threepdm = self.make_rdm123(state, norb, nelec, None,
                                                    **kwargs)
        threepdm = numpy.einsum('mkijln->ijklmn', threepdm).copy()
        threepdm += numpy.einsum('jk,lm,in->ijklmn', numpy.identity(norb),
                                 numpy.identity(norb), onepdm)
        threepdm += numpy.einsum('jk,miln->ijklmn', numpy.identity(norb),
                                 twopdm)
        threepdm += numpy.einsum('lm,kijn->ijklmn', numpy.identity(norb),
                                 twopdm)
        threepdm += numpy.einsum('jm,kinl->ijklmn', numpy.identity(norb),
                                 twopdm)

        twopdm =(numpy.einsum('iklj->ijkl',twopdm)
               + numpy.einsum('li,jk->ijkl',onepdm,numpy.identity(norb)))

        return onepdm, twopdm, threepdm

    def make_rdm3(self,
                  state,
                  norb,
                  nelec,
                  dt=numpy.float64,
                  filetype="binary",
                  link_index=None,
                  bypass=False,
                  cumulantE4=False, **kwargs):
        import os

        if self.has_threepdm == False:
            self.twopdm = False
            self.extraline.append('DoThreeRDM')
            if cumulantE4:
              self.extraline.append('dospinrdm')

            writeSHCIConfFile(self, nelec, False)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'SHCI Input conf')
                logger.debug1(self, open(inFile, 'r').read())

            start = time.time()
            mpisave = self.mpiprefix
            #self.mpiprefix=""
            executeSHCI(self)
            self.mpiprefix = mpisave
            end = time.time()
            print('......production of RDMs took %10.2f sec' % (end - start))
            sys.stdout.flush()

            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_threepdm = True
            self.extraline.pop()

        if (bypass): return None

        # The 3RDMS binary files written by "SHCIrdm::save3RDM" in DICE
        # are written as E3[i1,j2,k3,l3,m2,n1]
        # and are stored here as E3[i1,j2,k3,n1,m2,l3]
        # This is done with SQA in mind.
        start = time.time()
        if (filetype == "binary"):
            print('Reading binary 3RDM from DICE')
            fname = os.path.join(self.scratchDirectory,
                                 "spatial3RDM.%d.%d.bin" % (state, state))
            E3 = self.unpackE3_DICE(fname, norb)

        # The 3RDMs text files written by "SHCIrdm::save3RDM" in DICE
        # are written as E3[i1,j2,k3,l3,m2,n1]
        # and are stored here as E3[i1,j2,k3,n1,m2,l3]
        # This is done with SQA in mind.
        else:
            print('Reading text-file 3RDM')
            fname = os.path.join(self.scratchDirectory,
                                 "spatial3RDM.%d.%d.txt" % (state, state))
            f = open(fname, 'r')
            lines = f.readlines()
            E3 = numpy.zeros(
                shape=(norb, norb, norb, norb, norb, norb),
                dtype=dt,
                order='F')
            assert (int(lines[0]) == norb)
            for line in lines[1:]:
                linesp = line.split()
                if (len(linesp) != 7):
                    continue
                a,b,c, d,e,f, integral = int(linesp[0]), int(linesp[1]), int(linesp[2]),\
                                         int(linesp[3]), int(linesp[4]), int(linesp[5]), float(linesp[6])
                self.populate(E3, [a,b,c,  f,e,d], integral)
        end = time.time()
        print('......reading the RDM took    %10.2f sec' % (end - start))
        print('')
        sys.stdout.flush()
        return E3

        #if (filetype == "binary") :
        #    fname = os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.bin" %(state, state))
        #    fnameout = os.path.join('%s/%s/'%(self.scratchDirectory,"node0"), "spatial_threepdm.%d.%d.bin.unpack" %(state, state))
        #    libE3unpack.unpackE3(ctypes.c_char_p(fname.encode()),
        #                         ctypes.c_char_p(fnameout.encode()),
        #                         ctypes.c_int(norb))

    def make_rdm4(self,
                  state,
                  norb,
                  nelec,
                  dt=numpy.float64,
                  filetype="binary",
                  link_index=None,
                  bypass=False,
                  **kwargs):
        import os

        if self.has_fourpdm == False:
            self.twopdm = False
            self.threepdm = False
            self.extraline.append('DoThreeRDM')
            self.extraline.append('DoFourRDM')

            writeSHCIConfFile(self, nelec, False)
            if self.verbose >= logger.DEBUG1:
                inFile = self.configFile
                #inFile = os.path.join(self.scratchDirectory,self.configFile)
                logger.debug1(self, 'SHCI Input conf')
                logger.debug1(self, open(inFile, 'r').read())

            start = time.time()
            mpisave = self.mpiprefix
            #self.mpiprefix=""
            executeSHCI(self)
            self.mpiprefix = mpisave
            end = time.time()
            print('......production of RDMs took %10.2f sec' % (end - start))
            sys.stdout.flush()

            if self.verbose >= logger.DEBUG1:
                outFile = self.outputFile
                #outFile = os.path.join(self.scratchDirectory,self.outputFile)
                logger.debug1(self, open(outFile).read())
            self.has_fourpdm = True
            self.has_threepdm = True
            self.extraline.pop()

        if (bypass): return None

        # The 4RDMS binary files written by "SHCIrdm::save3RDM" in DICE
        # are written as E4[i1,j2,k3,l4,m4,n3,o2,p1]
        # and are stored here as E4[i1,j2,k3,l4,p1,o2,n3,m4]
        # This is done with SQA in mind.
        start = time.time()
        if (filetype == "binary"):
            print('Reading binary 4RDM from DICE')
            fname = os.path.join(self.scratchDirectory,
                                 "spatial4RDM.%d.%d.bin" % (state, state))
            E4 = self.unpackE4_DICE(fname, norb)

        # The 4RDMs text files written by "SHCIrdm::save3RDM" in DICE
        # are written as E4[i1,j2,k3,l4,m4,n3,o2,p1]
        # and are stored here as E4[i1,j2,k3,l4,p1,o2,n3,m4]
        # This is done with SQA in mind.
        else:
            print('Reading text-file 4RDM')
            fname = os.path.join(self.scratchDirectory,
                                 "spatial4RDM.%d.%d.txt" % (state, state))
            f = open(fname, 'r')
            lines = f.readlines()
            E4 = numpy.zeros(
                shape=(norb, norb, norb, norb, norb, norb, norb, norb),
                dtype=dt,
                order='F')
            assert (int(lines[0]) == norb)
            for line in lines[1:]:
                linesp = line.split()
                if (len(linesp) != 9):
                    continue
                a,b,c,d, e,f,g,h, integral = int(linesp[0]), int(linesp[1]), int(linesp[2]), int(linesp[3]),\
                                             int(linesp[4]), int(linesp[5]), int(linesp[6]), int(linesp[7]), float(linesp[8])
                self.populate(E4, [a,b,c,d,  h,g,f,e], integral)
        end = time.time()
        print('......reading the RDM took    %10.2f sec' % (end - start))
        print('')
        sys.stdout.flush()
        return E4

    def populate(self, array, list, value):
        dim = len(list) / 2
        up = list[:dim]
        dn = list[dim:]
        import itertools
        for t in itertools.permutations(range(dim), dim):
            updn = [up[i] for i in t] + [dn[i] for i in t]
            array[tuple(updn)] = value

    def unpackE2_DICE(self,fname,norb):
        # The 2RDMs written by "SHCIrdm::saveRDM" in DICE
        # are written as E2[i1,j2,k2,l2]
        # and are stored here as E2[i1,j2,l2,k2]
        # This is done with SQA in mind.
        fil=open(fname,"rb")
        print("     [fil.seek: How dangerous is that??]")
        fil.seek(53)  # HOW DANGEROUS IS THAT ???!?!?!?
        spina=-1
        spinb=-1
        spinc=-1
        spind=-1
        ab='ab'
        E2hom=numpy.zeros((norb,norb,norb,norb), order='F')
        E2het=numpy.zeros((norb,norb,norb,norb), order='F')
        for a in range(2*norb):
            spina=(spina+1)%2
            for b in range(a+1):
                spinb=(spinb+1)%2
                for c in range(2*norb):
                    spinc=(spinc+1)%2
                    for d in range(c+1):
                        spind=(spind+1)%2
                        A,B,C,D=int(a/2.),int(b/2.),int(c/2.),int(d/2.)
                        (value,)=struct.unpack('d',fil.read(8))
                        if spina==spinb and spina==spinc and spina==spind:
                           #print '%3i%3i%3i%3i %3i%3i%3i%3i %1s%1s%1s%1s E2hom %13.5e'%(a,b,c,d,A,B,C,D,ab[spina],ab[spinb],ab[spinc],ab[spind],value)
                            if (a%2==d%2 and b%2==c%2):
                                E2hom[A,B, D,C]-=value
                                E2hom[B,A, C,D]-=value
                            if (a%2==c%2 and b%2==d%2) and (a!=b):
                                E2hom[A,B, C,D]+=value
                                E2hom[B,A, D,C]+=value
                        elif (spina==spinb and spinc==spind)\
                           or(spina==spinc and spinb==spind)\
                           or(spina==spind and spinb==spinc):
                           #print '%3i%3i%3i%3i %3i%3i%3i%3i %1s%1s%1s%1s E2het %13.5e'%(a,b,c,d,A,B,C,D,ab[spina],ab[spinb],ab[spinc],ab[spind],value)
                            if (a%2==d%2 and b%2==c%2):
                                E2het[A,B, D,C]-=value
                                E2het[B,A, C,D]-=value
                            if (a%2==c%2 and b%2==d%2) and (a!=b):
                                E2het[A,B, C,D]+=value
                                E2het[B,A, D,C]+=value
                       #else:
                       #    print '%3i%3i%3i%3i %3i%3i%3i%3i %1s%1s%1s%1s NONE  %13.5e'%(a,b,c,d,A,B,C,D,ab[spina],ab[spinb],ab[spinc],ab[spind],value)
                    spind=-1
                spinc=-1
            spinb=-1
        spina=-1
        try:
            (value,)=struct.unpack('c',fil.read(1))
            print("     [MORE bytes TO READ!]")
        except:
            print("     [at least, no more bytes to read!]")
          #exit(0)
        fil.close()
        return E2hom,E2het

    def unpackE3_DICE(self, fname, norb):
        # The 3RDMs written by "SHCIrdm::save3RDM" in DICE
        # are written as E3[i1,j2,k3,l3,m2,n1]
        # and are stored here as E3[i1,j2,k3,n1,m2,l3]
        # This is done with SQA in mind.
        E3 = numpy.zeros((norb, norb, norb, norb, norb, norb), order='F')
        fil = open(fname, "rb")
        print("     [fil.seek: How dangerous is that??]")
        #fil.seek(93) # HOW DANGEROUS IS THAT ???!?!?!?
        fil.seek(53)  # HOW DANGEROUS IS THAT ???!?!?!?
        for a in range(norb):
            for b in range(norb):
                for c in range(norb):
                    for d in range(norb):
                        for e in range(norb):
                            for f in range(norb):
                                (value, ) = struct.unpack('d', fil.read(8))
                                E3[a,b,c,  f,e,d]=value
        try:
            (value, ) = struct.unpack('c', fil.read(1))
            print("     [MORE bytes TO READ!]")
        except:
            print("     [at least, no more bytes to read!]")
        #exit(0)
        fil.close()
        return E3

    def unpackE4_DICE(self, fname, norb):
        # The 4RDMs written by "SHCIrdm::save4RDM" in DICE
        # are written as E4[i1,j2,k3,l4,m4,n3,o2,p1]
        # and are stored here as E4[i1,j2,k3,l4,p1,o2,n3,m4]
        # This is done with SQA in mind.
        E4 = numpy.zeros((norb, norb, norb, norb, norb, norb, norb, norb),
                         order='F')
        fil = open(fname, "rb")
        print("     [fil.seek: How dangerous is that??]")
        fil.seek(53)  # HOW DANGEROUS IS THAT ???!?!?!?
        for a in range(norb):
            for b in range(norb):
                for c in range(norb):
                    for d in range(norb):
                        for e in range(norb):
                            for f in range(norb):
                                for g in range(norb):
                                    for h in range(norb):
                                        (value, ) = struct.unpack('d', fil.read(8))
                                        E4[a,b,c,d,  h,g,f,e]=value
        try:
            (value, ) = struct.unpack('c', fil.read(1))
            print("     [MORE bytes TO READ!]")
        except:
            print("     [at least, no more bytes to read!]")
        #exit(0)
        fil.close()
        return E4

    def clearSchedule(self):
        """
        TODO Tagged for removal
        """
        self.scheduleSweeps = []
        self.scheduleMaxMs = []
        self.scheduleTols = []
        self.scheduleNoises = []

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0,
               **kwargs):
        """
        Approximately solve CI problem for the specified active space.
        """

        # Warning about behavior of SHCI
        if hasattr(self, 'prefix'):
            warnings.warn(
                "\n\nThe `SHCI.prefix` attribute is no longer supported.\n" +
                "To set the Dice option `prefix` please set the " +
                "`SHCI.scratchDirectory` attribute in PySCF\n",
                FutureWarning,
            )
            self.scratchDirectory = self.prefix
            delattr(self, "prefix")

        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeSHCIConfFile(self, nelec, fciRestart)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = readEnergy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else:
                    calc_e = [0.0] * self.nroots
            return calc_e, roots
        if self.returnInt:
            return h1e, eri
        executeSHCI(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        return calc_e, roots

    def approx_kernel(self,
                      h1e,
                      eri,
                      norb,
                      nelec,
                      fciRestart=None,
                      ecore=0,
                      **kwargs):
        fciRestart = True

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeSHCIConfFile(self, nelec, fciRestart)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeSHCI(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        return calc_e, roots

    def restart_scheduler_(self):
        def callback(envs):
            if (envs['norm_gorb'] < self.shci_switch_tol
                    or ('norm_ddm' in envs
                        and envs['norm_ddm'] < self.shci_switch_tol * 10)):
                self._restart = True
            else:
                self._restart = False

        return callback

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
        
        os.remove(os.path.join(self.runtimeDir, self.configFile))
        os.remove(os.path.join(self.runtimeDir, self.outputFile))
        os.remove(os.path.join(self.runtimeDir, self.integralFile))


def transition_dipole(mc, state_i, state_j):
    t_dm1 = mc.fcisolver.trans_rdm1(state_i, state_j, mc.ncas, mc.nelecas)
    ncore = mc.ncore
    ncasorb = mc.ncas
    mol = mc.mol
    mo_cas = mc.mo_coeff[:, ncore:ncore+ncasorb]
    t_dm1 = pyscf.lib.einsum('pi, ij, qj->pq', mo_cas, t_dm1, mo_cas)
    #print(t_dm1)
    charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords()) / mol.atom_charges().sum())
    with mol.with_common_origin(charge_center):
        t_dip = numpy.einsum('xij,ji->x', mol.intor('int1e_r'), t_dm1)
    return t_dip

def oscillator_strength(mc, state_i, state_j):
    t_dip = transition_dipole(mc, state_i, state_j)
    calc_e = readEnergy(mc.fcisolver)
    delta_e = abs(calc_e[state_i] - calc_e[state_j])
    return 2./3.*delta_e*sum(abs(t_dip)**2), 2./3.*delta_e*abs(t_dip)**2

def phospherescence_lifetime(mc, s0=0, triplet_list=[1,2,3]):
    au2wavenumber = 219470.
    tau = numpy.zeros(len(triplet_list))
    calc_e = readEnergy(mc.fcisolver)
    for state in triplet_list:
        delta_e = calc_e[state] - calc_e[s0]
        fosc = oscillator_strength(mc, s0, state)
        tau[triplet_list.index(state)] = 1.5/(fosc*(delta_e*au2wavenumber)**2)
    tau_av = 3./(sum(1/tau))
    return tau_av, tau
    

def make_sched(SHCI):

    nIter = len(SHCI.sweep_iter)
    # Check that the number of different epsilons match the number of iter steps.
    assert (nIter == len(SHCI.sweep_epsilon))

    if (nIter == 0):
        SHCI.sweep_iter = [0]
        SHCI.sweep_epsilon = [1.e-3]

    schedStr = 'schedule '
    for it, eps in zip(SHCI.sweep_iter, SHCI.sweep_epsilon):
        schedStr += '\n' + str(it) + '\t' + str(eps)

    schedStr += '\nend\n'

    return schedStr


def writeSHCIConfFile(SHCI, nelec, Restart):
    confFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)

    f = open(confFile, 'w')

    # Reference determinant section
    f.write('#system\n')
    f.write('nocc %i\n' % (nelec[0] + nelec[1]))
    if SHCI.__class__.__name__ == 'FakeCISolver':
        if SHCI.initialStates is not None:
            print("write determinants")
            for i in range(len(SHCI.initialStates)):
                for j in SHCI.initialStates[i]:
                    f.write('%i ' % (j))
                if (i != len(SHCI.initialStates) - 1):
                    f.write('\n')
        else:
            for i in range(nelec[0]):
                f.write('%i ' % (2 * i))
            for i in range(nelec[1]):
                f.write('%i ' % (2 * i + 1))
    else:
        if SHCI.initialStates is not None:
            print("write determinants")
            for i in range(len(SHCI.initialStates)):
                for j in SHCI.initialStates[i]:
                    f.write('%i ' % (j))
                if (i != len(SHCI.initialStates) - 1):
                    f.write('\n')
        elif SHCI.irrep_nelec is None:
            for i in range(int(nelec[0])):
                f.write('%i ' % (2 * i))
            for i in range(int(nelec[1])):
                f.write('%i ' % (2 * i + 1))
        else:
            from pyscf import symm
            from pyscf.dmrgscf import dmrg_sym
            from pyscf.symm.basis import DOOH_IRREP_ID_TABLE
            if SHCI.groupname is not None and SHCI.orbsym is not []:
                orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
            else:
                orbsym = [1] * norb
            done = []
            for k, v in SHCI.irrep_nelec.items():

                irrep, nalpha, nbeta = [dmrg_sym.irrep_name2id(SHCI.groupname, k)],\
                                       v[0], v[1]

                for i in range(len(orbsym)):  #loop over alpha electrons
                    if (orbsym[i] == irrep[0] and nalpha != 0
                            and i * 2 not in done):
                        done.append(i * 2)
                        f.write('%i ' % (i * 2))
                        nalpha -= 1
                    if (orbsym[i] == irrep[0] and nbeta != 0
                            and i * 2 + 1 not in done):
                        done.append(i * 2 + 1)
                        f.write('%i ' % (i * 2 + 1))
                        nbeta -= 1
                if (nalpha != 0):
                    print("number of irreps %s in active space = %d" %
                          (k, v[0] - nalpha))
                    print(
                        "number of irreps %s alpha electrons = %d" % (k, v[0]))
                    exit(1)
                if (nbeta != 0):
                    print("number of irreps %s in active space = %d" %
                          (k, v[1] - nbeta))
                    print(
                        "number of irreps %s beta  electrons = %d" % (k, v[1]))
                    exit(1)
    f.write('\nend\n')

    # Handle different cases for FCIDUMP file names/paths
    f.write("orbitals {}\n".format(os.path.join(SHCI.runtimeDir, SHCI.integralFile)))

    f.write('nroots %r\n' % SHCI.nroots)

    # Variational Keyword Section
    f.write('\n#variational\n')
    if (not Restart):
        schedStr = make_sched(SHCI)
        f.write(schedStr)
    else:
        f.write('schedule\n')
        f.write('%d  %g\n' % (0, SHCI.sweep_epsilon[-1]))
        f.write('end\n')

    f.write('davidsonTol %g\n' % SHCI.davidsonTol)
    f.write('dE %g\n' % SHCI.dE)

    # Sets maxiter to 6 more than the last iter in sweep_iter[] if restarted.
    if (not Restart):
        f.write('maxiter %i\n' % (SHCI.sweep_iter[-1] + 6))
    else:
        f.write('maxiter 10\n')
        f.write('fullrestart\n')

    # Perturbative Keyword Section
    f.write('\n#pt\n')
    if (SHCI.stochastic == False):
        f.write('deterministic \n')
    else:
        f.write('nPTiter %d\n' % SHCI.nPTiter)
    f.write('epsilon2 %g\n' % SHCI.epsilon2)
    f.write('epsilon2Large %g\n' % SHCI.epsilon2Large)
    f.write('targetError %g\n' % SHCI.targetError)
    f.write('sampleN %i\n' % SHCI.sampleN)

    # Miscellaneous Keywords
    f.write('\n#misc\n')
    if Restart is True:
        f.write('noio \n')
    if (SHCI.scratchDirectory != ""):
        if not os.path.exists(SHCI.scratchDirectory):
            os.makedirs(SHCI.scratchDirectory)
        f.write('prefix %s\n' % (SHCI.scratchDirectory))
    if (SHCI.DoSOC):
        f.write('DoSOC\n')
        SHCI.DoRDM = False
    if (SHCI.DoRDM):
        f.write('DoRDM\n')
    if (SHCI.nroots > 1 and (SHCI.DoRDM or SHCI.DoTRDM)):
        f.write('DoTRDM\n')
    for line in SHCI.extraline:
        f.write('%s\n' % line)

    f.write('\n')  # SHCI requires that there is an extra line.
    f.close()


def D2htoDinfh(SHCI, norb, nelec):
    coeffs = numpy.zeros(shape=(norb, norb)).astype(complex)
    nRows = numpy.zeros(shape=(norb, ), dtype=int)
    rowIndex = numpy.zeros(shape=(2 * norb, ), dtype=int)
    rowCoeffs = numpy.zeros(shape=(2 * norb, ), dtype=float)
    orbsym1 = numpy.zeros(shape=(norb, ), dtype=int)

    orbsym = numpy.asarray(SHCI.orbsym)
    A_irrep_ids = set([0, 1, 4, 5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)

    # A1g/A2g/A1u/A2u for Dooh or A1/A2 for Coov
    for ir in A_irrep_ids:
        is_gerade = ir in (0, 1)
        for i in numpy.where(orbsym == ir)[0]:
            coeffs[i, i] = 1.0
            nRows[i] = 1
            rowIndex[2 * i] = i
            rowCoeffs[2 * i] = 1.
            if is_gerade:  # A1g/A2g for Dooh or A1/A2 for Coov
                orbsym1[i] = 1
            else:  # A1u/A2u for Dooh
                orbsym1[i] = 2

    # See L146 of pyscf/symm/basis.py
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
    for ir in Ex_irrep_ids:
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir + 1)[0]
        else:
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir - 1)[0]

        if ir % 10 in (0, 5):
            l = (ir // 10) * 2
        else:
            l = (ir // 10) * 2 + 1

        for ix, iy in zip(Ex, Ey):
            nRows[ix] = nRows[iy] = 2
            if is_gerade:
                orbsym1[ix], orbsym1[iy] = 2 * l + 3, -(2 * l + 3)
            else:
                orbsym1[ix], orbsym1[iy] = 2 * l + 4, -(2 * l + 4)

            rowIndex[2 * ix], rowIndex[2 * ix + 1] = ix, iy
            rowIndex[2 * iy], rowIndex[2 * iy + 1] = ix, iy

            coeffs[ix, ix], coeffs[ix, iy] = ((-1)**l) * 1.0 / (2.0**0.5), (
                (-1)**l) * 1.0j / (2.0**0.5)
            coeffs[iy, ix], coeffs[iy, iy] = 1.0 / (2.0**0.5), -1.0j / (2.0**
                                                                        0.5)
            rowCoeffs[2 * ix], rowCoeffs[2 * ix + 1] = (
                (-1)**l) * 1.0 / (2.0**0.5), ((-1)**l) * 1.0 / (2.0**0.5)
            rowCoeffs[2 * iy], rowCoeffs[
                2 * iy + 1] = 1.0 / (2.0**0.5), -1.0 / (2.0**0.5)

    return coeffs, nRows, rowIndex, rowCoeffs, orbsym1


def DinfhtoD2h(SHCI, norb, nelec):
    nRows = numpy.zeros(shape=(norb, ), dtype=int)
    rowIndex = numpy.zeros(shape=(2 * norb, ), dtype=int)
    rowCoeffs = numpy.zeros(shape=(4 * norb, ), dtype=float)

    orbsym = numpy.asarray(SHCI.orbsym)
    A_irrep_ids = set([0, 1, 4, 5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)

    for ir in A_irrep_ids:
        for i in numpy.where(orbsym == ir)[0]:
            nRows[i] = 1
            rowIndex[2 * i] = i
            rowCoeffs[4 * i] = 1.

    # See L146 of pyscf/symm/basis.py
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir % 10) in (0, 2, 5, 7)]
    for ir in Ex_irrep_ids:
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir + 1)[0]
        else:
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir - 1)[0]

        if ir % 10 in (0, 5):
            l = (ir // 10) * 2
        else:
            l = (ir // 10) * 2 + 1

        for ix, iy in zip(Ex, Ey):
            nRows[ix] = nRows[iy] = 2

            rowIndex[2 * ix], rowIndex[2 * ix + 1] = ix, iy
            rowIndex[2 * iy], rowIndex[2 * iy + 1] = ix, iy

            rowCoeffs[4 * ix], rowCoeffs[4 * ix + 2] = (
                (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)
            rowCoeffs[4 * iy + 1], rowCoeffs[4 * iy + 3] = -(
                (-1)**l) * 1.0 / (2.0**0.5), 1.0 / (2.0**0.5)

    return nRows, rowIndex, rowCoeffs


def writeIntegralFile(SHCI, h1eff, eri_cas, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec // 2 + nelec % 2
        nelecb = nelec - neleca
    else:
        neleca, nelecb = nelec

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(SHCI.runtimeDir, SHCI.integralFile)

    if not os.path.exists(SHCI.scratchDirectory):
        os.makedirs(SHCI.scratchDirectory)

    from pyscf import symm
    from pyscf.dmrgscf import dmrg_sym

    if (SHCI.groupname == 'Dooh'
            or SHCI.groupname == 'Coov') and SHCI.useExtraSymm:
        coeffs, nRows, rowIndex, rowCoeffs, orbsym = D2htoDinfh(
            SHCI, norb, nelec)

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
        if SHCI.groupname is not None and SHCI.orbsym is not []:
            orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
        else:
            orbsym = [1] * norb

        eri_cas = pyscf.ao2mo.restore(8, eri_cas, norb)
        # Writes the FCIDUMP file using functions in SHCI_tools.cpp.
        integralFile = integralFile.encode()  # .encode for python3 compatibility
        fcidumpFromIntegral(integralFile, h1eff, eri_cas, norb,
                            neleca + nelecb, ecore,
                            numpy.asarray(orbsym, dtype=numpy.int32),
                            abs(neleca - nelecb))


def executeSHCI(SHCI):
    file1 = os.path.join(SHCI.runtimeDir, "%s/shci.e" % (SHCI.scratchDirectory))#what?
    if os.path.exists(file1):                                                   #what?
        os.remove(file1)                                                        #what?
    inFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)
    outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
    try:
        if SHCI.DoSOC:
            cmd = ' '.join((SHCI.mpiprefix, SHCI.zdiceExecutable, inFile))
            cmd = "%s > %s 2>&1" % (cmd, outFile)
            check_call(cmd, shell=True)
        else:
            cmd = ' '.join((SHCI.mpiprefix, SHCI.executable, inFile))
            cmd = "%s > %s 2>&1" % (cmd, outFile)
            check_call(cmd, shell=True)
            #save_output(SHCI)
    except CalledProcessError as err:
        logger.error(SHCI, cmd)
        raise err


#def save_output(SHCI):
#  for i in range(50):
#    if os.path.exists(os.path.join(SHCI.runtimeDir, "output%02d.dat"%(i))):
#      continue
#    else:
#      import shutil
#      shutil.copy2(os.path.join(SHCI.runtimeDir, "output.dat"),os.path.join(SHCI.runtimeDir, "output%02d.dat"%(i)))
#      shutil.copy2(os.path.join(SHCI.runtimeDir, "%s/shci.e"%(SHCI.scratchDirectory)), os.path.join(SHCI.runtimeDir, "shci%02d.e"%(i)))
#      #print('BM copied into "output%02d.dat"'%(i))
#      #print('BM copied into "shci%02d.e"'%(i))
#      break


def readEnergy(SHCI):
    file1 = open(
        os.path.join(SHCI.runtimeDir, "%s/shci.e" % (SHCI.scratchDirectory)),
        "rb")
    format = ['d'] * SHCI.nroots
    format = ''.join(format)
    calc_e = struct.unpack(format, file1.read())
    file1.close()
    if SHCI.nroots == 1:
        return calc_e[0]
    else:
        return list(calc_e)


def SHCISCF(mf, norb, nelec, maxM=1000, tol=1.e-8, *args, **kwargs):
    '''Shortcut function to setup CASSCF using the SHCI solver.  The SHCI
    solver is properly initialized in this function so that the 1-step
    algorithm can applied with SHCI-CASSCF.

    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = SHCISCF(mf, 4, 4)
    >>> mc.kernel()
    -74.414908818611522
    '''

    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = SHCI(mf.mol, maxM, tol=tol)
    #mc.callback = mc.fcisolver.restart_scheduler_() #TODO
    if mc.chkfile == mc._scf._chkfile.name:
        # Do not delete chkfile after mcscf
        mc.chkfile = tempfile.mktemp(dir=settings.SHCISCRATCHDIR)
        if not os.path.exists(settings.SHCISCRATCHDIR):
            os.makedirs(settings.SHCISCRATCHDIR)
    return mc


def dryrun(mc, mo_coeff=None):
    '''
    Generate FCIDUMP and SHCI input.dat file for a give multiconfigurational 
    object. This method DOES NOT run a single iteration using Dice it just 
    generates all the inputs you'd need to do so.

    Args:
        mc: a CASSCF/CASCI object (or RHF object)

    Kwargs: 
        mo_coeff: `np.ndarray` of the molecular orbital coefficients. 
            Default is None.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.shciscf import shci
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 2; O 0 1 1;", symmetry=True)
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASCI(mf, 3, 4)
    >>> mc.fcisolver = shci.SHCI(mol)
    >>> shci.dryrun(mc)

    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    
    # Set orbsym b/c it's needed in `writeIntegralFile()`
    if mc.fcisolver.orbsym is not []:
        mc.fcisolver.orbsym = getattr(mo_coeff, "orbsym", [])

    #mc.kernel(mo_coeff) # Works, but runs full CASCI/CASSCF
    h1e, ecore = mc.get_h1eff(mo_coeff)
    h2e = mc.get_h2eff(mo_coeff)
    writeIntegralFile(mc.fcisolver, h1e, h2e, mc.ncas, mc.nelecas, ecore)
    writeSHCIConfFile(mc.fcisolver, mc.nelecas, False)




if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, dmrgscf
    from pyscf.shciscf import shci

    # Initialize N2 molecule
    b = 1.098
    mol = gto.Mole()
    mol.build(
        verbose=5,
        output=None,
        atom=[
            ['N', (0.000000, 0.000000, -b / 2)],
            ['N', (0.000000, 0.000000, b / 2)],
        ],
        basis={
            'N': 'ccpvdz',
        },
    )

    # Create HF molecule
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-9
    mf.scf()

    # Number of orbital and electrons
    norb = 8
    nelec = 10
    dimer_atom = 'N'

    mc = mcscf.CASSCF(mf, norb, nelec)
    e_CASSCF = mc.mc2step()[0]

    # Create SHCI molecule for just variational opt.
    # Active spaces chosen to reflect valence active space.
    mch = shci.SHCISCF(mf, norb, nelec)
    mch.fcisolver.nPTiter = 0  # Turn off perturbative calc.
    mch.fcisolver.outputFile = 'no_PT.dat'
    mch.fcisolver.sweep_iter = [0, 3]
    mch.fcisolver.DoRDM = True
    # Setting large epsilon1 thresholds highlights improvement from perturbation.
    mch.fcisolver.sweep_epsilon = [1e-2, 1e-2]
    e_noPT = mch.mc1step()[0]

    # Run a single SHCI iteration with perturbative correction.
    mch.fcisolver.stochastic = False  # Turns on deterministic PT calc.
    mch.fcisolver.outputFile = 'PT.dat'  # Save output under different name.
    shci.writeSHCIConfFile(mch.fcisolver, [nelec / 2, nelec / 2], True)
    shci.executeSHCI(mch.fcisolver)

    # Open and get the energy from the binary energy file hci.e.
    # Open and get the energy from the
    with open(mch.fcisolver.outputFile, 'r') as f:
        lines = f.readlines()

    e_PT = float(lines[len(lines) - 1].split()[2])

    #e_PT = shci.readEnergy( mch.fcisolver )

    # Comparison Calculations
    del_PT = e_PT - e_noPT
    del_shci = e_CASSCF - e_PT

    print('\n\nEnergies for %s2 give in E_h.' % dimer_atom)
    print('=====================================')
    print('SHCI Variational: %6.12f' % e_noPT)
    # Prints the total energy including the perturbative component.
    print('SHCI Perturbative: %6.12f' % e_PT)
    print('Perturbative Change: %6.12f' % del_PT)
    print('CASSCF Total Energy: %6.12f' % e_CASSCF)
    print('E(CASSCF) - E(SHCI): %6.12f' % del_shci)
