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
# Author:          Sheng Guo <shengg91@gmail.com>
#

'''
Perturbation for DMRG with small bond dimensions.
'''

import os
import sys
import struct
import time
import tempfile
from subprocess import check_call, CalledProcessError
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf.dmrgscf import dmrgci
from dmrgci import DMRGCI
from pyscf.dmrgscf import dmrg_sym
from subprocess import *


class PDMRG(DMRGCI):

  def __init__(self, mol, maxM=None, tol=None, num_thrds=1, memory=None):
      super(PDMRG,self).__init__(mol, maxM, tol, num_thrds, memory)
      self.H0factor=0.5
      self.H0_type = 'Hd'
      self.H0_file = 'H0'
      self.H1_file = 'H1'
      self.state=0

  def generateH0(self):
      with open(self.integralFile,'r') as fin:
          with open(self.H0_file,'w') as fout:
              while True:
                s = fin.readline()
                fout.write(s)
                if s.endswith('END\n'):
                    break
              for line in fin:
                  h, i, j, k, l = line.split()
                  if (i==j and k==l ):
                      fout.write(line)
                  elif ( i==k and j == l and (not k==l)):
                      fout.write(str('\t\t'+h+' '+i+' '+j+' '+j+' '+i+'\n'))

  def generateH1(self):
      with open(self.integralFile,'r') as fin:
          with open(self.H1_file,'w') as fout:
              while True:
                  s = fin.readline()
                  fout.write(s)
                  if s.endswith('END\n'):
                      break
                  for line in fin:
                      h, i, j, k, l = line.split()
                      if (int(k)==0 and int(l) == 0):
                          fout.write(line)
                      else :
                          fout.write(str('\t\t'+h+' '+i+' '+j+' '+k+' '+l+'\n'))
                          fout.write(str('\t\t'+h+' '+j+' '+i+' '+k+' '+l+'\n'))
                          fout.write(str('\t\t'+h+' '+i+' '+j+' '+l+' '+k+'\n'))
                          fout.write(str('\t\t'+h+' '+j+' '+i+' '+l+' '+k+'\n'))

  def readEnergy(self):
      file1 = open(os.path.join(self.scratchDirectory, "node0", "dmrg.e"), "rb")
      format = ['d']*self.nroots
      format = ''.join(format)
      calc_e = struct.unpack(format, file1.read())
      file1.close()
      if self.nroots == 1:
          return calc_e[0]
      else:
          return numpy.asarray(calc_e)[self.state]

  def adjust_H_files(self):
      self.E_dmrg= self.readEnergy()
      with open(self.integralFile,'r') as f:
          self.core_energy = float(f.readlines()[-1].split()[0])
      self.E_dmrg -= self.core_energy
      check_call(["echo 0 > wavenum"],shell=True)
      with open('dmrg.conf','r') as fin:
          with open('oh.conf','w') as fout:
              for line in fin.readlines():
                  if len(line.split()) and line.split()[0] == 'orbitals':
                      fout.write('orbitals %s\n'%self.H0_file)
                  else:
                      fout.write(line)
              fout.write('fullrestart')

    import os
    block_path = os.path.dirname(self.executable)
    check_call(["%s %s/OH oh.conf wavenum > oh.out"%(self.mpiprefix,block_path)], shell=True)
    with open('oh.out','r') as f:
        H0 = float(f.readlines()[-4])
    H0 -=self.core_energy
    E0  = self.E_dmrg*self.H0factor + H0*(1.0-self.H0factor)

    check_call(["head -n -1 %s > tmpfile"%self.H0_file], shell=True)
    check_call(["cp tmpfile %s"%self.H0_file], shell=True)
    with open('%s'%self.H0_file,'a') as f:
        f.write('%s 0 0 0 0\n'%(-E0))

    check_call(["head -n -1 %s > tmpfile"%self.H1_file], shell=True)
    check_call(["cp tmpfile %s"%self.H1_file], shell=True)
    with open('%s'%self.H1_file,'a') as f:
        f.write('%s 0 0 0 0\n'%(-self.E_dmrg))

  def compress(self):
      newsolver = self
      newsolver.scheduleSweeps
      newsolver.scheduleSweeps = [0, 1, 2]
      newsolver.scheduleMaxMs  = [self.maxM, self.maxM, self.maxM]
      newsolver.scheduleTols   = [0.0001, 1e-7, 1e-7]
      newsolver.scheduleNoises = [0.0001, 0.0001, 0.0]
      newsolver.twodot_to_onedot = 2
      newsolver.maxIter = 6
      newsolver.twopdm = False
      newsolver.block_extra_keyword.append('compress %d'%self.state)
      newsolver.block_extra_keyword.append('targetState %d'%(self.state+1000))

      newsolver.configFile = "compress.conf"
      newsolver.outputFile = "compress.out"
      newsolver.integralFile = self.H1_file
      dmrgci.writeDMRGConfFile(newsolver,newsolver.nelec,Restart=False,with_2pdm=False)
      dmrgci.executeBLOCK(newsolver)

  def pt(self):
      newsolver = self
      newsolver.twopdm = False
      newsolver.scheduleSweeps = [0, 2, 6]
      newsolver.scheduleMaxMs  = [self.maxM, self.maxM, self.maxM]
      newsolver.scheduleTols   = [1e-5, 1e-5, 1e-7]
      newsolver.scheduleNoises = [1e-4, 5e-5, 0.0]
      newsolver.twodot_to_onedot = 0
      newsolver.maxIter = 12
      newsolver.block_extra_keyword.append('response')
      newsolver.block_extra_keyword.append('baseStates %d'%self.state)
      newsolver.block_extra_keyword.append('projectorStates %d'%self.state)
      newsolver.block_extra_keyword.append('GuessState %d'%(self.state+1000))
      newsolver.block_extra_keyword.append('targetState %d'%(self.state+2000))
      newsolver.block_extra_keyword.append('occ 9999')
      newsolver.block_extra_keyword.append('twodot')
      newsolver.configFile = "pdmrg.conf"
      newsolver.outputFile = "pdmrg.out"
      newsolver.integralFile = self.H0_file+' '+self.H1_file
      dmrgci.writeDMRGConfFile(newsolver,newsolver.nelec,Restart=False,with_2pdm=False)
      dmrgci.executeBLOCK(newsolver)

  def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
      self.nelec = nelec
      super(PDMRG,self).kernel(h1e, eri, norb, nelec, fciRestart, ecore)
      self.maxM = min(5*self.maxM,5000)
      self.generateH0()
      self.generateH1()
      self.adjust_H_files()
      self.compress()
      self.pt()
      lines = Popen(["grep", "Sweep Energy", "pdmrg.out"], stdout=PIPE).communicate()[0]
      lines = lines.strip().split('\n')
      energy_line = lines[-1]
      pt_e = float(energy_line.split()[-1])
      log = logger.Logger(self.stdout,self.verbose)
      log.info('PT energy is %s'%pt_e)
      log.info('Total energy is %s'%(pt_e+self.E_dmrg+self.core_energy))
      return pt_e+self.E_dmrg+self.core_energy, 0

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    from pyscf import dmrgscf

    import os
    from pyscf.dmrgscf import settings
    settings.MPIPREFIX =''

    b = 1.4
    mol = gto.Mole()
    mol.build(
        verbose = 7,
        output = 'out-pdmrg',
        atom = [['H', (0.,0.,i-3.5)] for i in range(8)],
        basis = {'H': 'sto-3g'},
        symmetry = True
    )
    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASCI(m, 8, 8)
    mc.fcisolver = DMRGCI(mol,maxM=1000)
    emc_0 = mc.casci()[0]

    mc = mcscf.CASCI(m, 8, 8)
    mc.fcisolver = PDMRG(mol,maxM=20)
    emc_1 = mc.casci()[0]


    print('DMRG  = %.15g PDMRG  = %.15g' % (emc_0, emc_1))

