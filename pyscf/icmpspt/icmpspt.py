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
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
Internal-contracted MPS perturbation method.  You can contact Sandeep
Sharma for the "icpt" program required by this module.  If this method
is used in your work, please cite
S. Sharma and G. Chan,  J. Chem. Phys., 136 (2012), 124121
S. Sharma, G. Jeanmairet, and A. Alavi,  J. Chem. Phys., 144 (2016), 034103
'''

import pyscf
import os
import time
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import mcscf
from pyscf import ao2mo
from pyscf import scf
from pyscf.ao2mo import _ao2mo
from pyscf.dmrgscf import dmrgci
from pyscf.dmrgscf import dmrg_sym
from pyscf import tools
import sys

libmc = lib.load_library('libmcscf')

float_precision = numpy.dtype('Float64')
mpiprefix=""
executable="/home/mussard/softwares/icpt/icpt"

if not os.path.isfile(executable):
    msg = ('MPSLCC executable %s not found.  Please specify "executable" in %s'
           % (executable, __file__))
    raise ImportError(msg)

NUMERICAL_ZERO = 1e-14


#in state average calculationg dm1eff will be different than dm1
#this means that the h1eff in the fock operator which is stored in eris_sp['h1eff'] will be
#calculated using the dm1eff and will in general not result in diagonal matrices
def writeNEVPTIntegrals(mc, E1, E2, E1eff, aaavsplit, nfro, fully_ic=False, third_order=False):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff
    intfolder=mc.fcisolver.scratchDirectory+'/int/'
    intfolder='int/'



    # (Note: Integrals are in chemistry notation)
    eris = _ERIS(mc, mo)
    eris_sp={}
    if (third_order):
      eris['pcpc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,nfro:ncor], mo, mo[:,nfro:ncor]), compact=False)
      eris['ppcc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nfro:ncor], mo[:,nfro:ncor]), compact=False)
      eris['ppee'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nocc:], mo[:,nocc:]), compact=False)
      eris['pcpc'].shape=(norb, ncor-nfro, norb, ncor-nfro)
      eris['ppcc'].shape=(norb, norb, ncor-nfro, ncor-nfro)
      eris['ppee'].shape=(norb, norb, nvir, nvir)

    # h1eff
    eris_sp['h1eff']= eris['h1eff']
    eris_sp['h1eff'][:ncor,:ncor] += numpy.einsum('abcd,cd', eris['ppaa'][:ncor,:ncor,:,:], E1eff)
    eris_sp['h1eff'][:ncor,:ncor] -= numpy.einsum('abcd,bd', eris['papa'][:ncor,:,:ncor,:], E1eff)*0.5
    eris_sp['h1eff'][nocc:,nocc:] += numpy.einsum('abcd,cd', eris['ppaa'][nocc:,nocc:,:,:], E1eff)
    eris_sp['h1eff'][nocc:,nocc:] -= numpy.einsum('abcd,bd', eris['papa'][nocc:,:,nocc:,:], E1eff)*0.5
    numpy.save(intfolder+"int1eff",numpy.asfortranarray(eris_sp['h1eff'][nfro:,nfro:]))

    # CVCV
    eriscvcv = eris['cvcv']
    if (not isinstance(eris['cvcv'], type(eris_sp['h1eff']))):
      eriscvcv = lib.chkfile.load(eris['cvcv'].name, "eri_mo")#h5py.File(eris['cvcv'].name,'r')["eri_mo"]
    eris_sp['cvcv'] = eriscvcv.reshape(ncor, nvir, ncor, nvir)

    # energy_core
    hcore  = mc.get_hcore()
    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    energy_core = numpy.einsum('ij,ji', dmcore, hcore) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*numpy.einsum('ij,ij',     E1, eris_sp['h1eff'][ncor:nocc,ncor:nocc])\
             + 0.5*numpy.einsum('ijkl,ijkl', E2, eris['ppaa'][ncor:nocc,ncor:nocc,:,:].transpose(0,2,1,3))
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print("Energy_core = ",energy_core)
    print("Energy      = ", energyE0)
    print("")

    # offdiagonal warning
    offdiagonal = 0.0
    for k in range(ncor):
      for l in range(ncor):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    for k in range(nocc, norb):
      for l in range(nocc,norb):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    if (abs(offdiagonal) > 1e-6):
      print("WARNING: Have to use natural orbitals from CAASCF")
      print("         offdiagonal elements:", offdiagonal)
      print("")

    # Write out ingredients to intfolder
    # 2 "C"
    numpy.save(intfolder+"W:ccae", numpy.asfortranarray(eris['pacv'][nfro:ncor,     :    , nfro:    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:eecc", numpy.asfortranarray(eris_sp['cvcv'][nfro: ,     :    , nfro:    ,     :    ].transpose(1,3,0,2)))
    if (third_order):
      numpy.save(intfolder+"W:ceec", numpy.asfortranarray(eris['pcpc'][nocc:    ,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
      numpy.save(intfolder+"W:cccc", numpy.asfortranarray(eris['ppcc'][nfro:ncor, nfro:ncor,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:cece", numpy.asfortranarray(eris['ppcc'][nocc:    , nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    if (third_order):
      numpy.save(intfolder+"W:ccca", numpy.asfortranarray(eris['ppcc'][nfro:ncor, ncor:nocc,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:ccce", numpy.asfortranarray(eris['ppcc'][nfro:ncor, nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:cace", numpy.asfortranarray(eris['ppcc'][ncor:nocc, nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    # 2 "A"
    numpy.save(intfolder+"W:caac", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,3,1,2)))
    numpy.save(intfolder+"W:ccaa", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:aeca", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
    numpy.save(intfolder+"W:eeaa", numpy.asfortranarray(eris['papa'][nocc:    ,     :    , nocc:    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:aaaa", numpy.asfortranarray(eris['ppaa'][ncor:nocc, ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:eaca", numpy.asfortranarray(eris['ppaa'][nocc:    , nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:caca", numpy.asfortranarray(eris['ppaa'][nfro:ncor, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
    if (third_order):
      numpy.save(intfolder+"W:aeea", numpy.asfortranarray(eris['papa'][nocc:    ,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
      numpy.save(intfolder+"W:aeae", numpy.asfortranarray(eris['ppaa'][nocc:    , nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    if (fully_ic):
      numpy.save(intfolder+"W:eaaa", numpy.asfortranarray(eris['ppaa'][nocc:    , ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
      numpy.save(intfolder+"W:caaa", numpy.asfortranarray(eris['ppaa'][nfro:ncor, ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
    # 2 "E"
    numpy.save(intfolder+"W:eeca", numpy.asfortranarray(eris['pacv'][nocc:    ,     :    , nfro:    ,     :    ].transpose(3,0,2,1)))
    if (third_order):
      numpy.save(intfolder+"W:aece", numpy.asfortranarray(eris['ppee'][ncor:nocc, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
      numpy.save(intfolder+"W:eeee", numpy.asfortranarray(eris['ppee'][nocc:    ,nocc :    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:eeec", numpy.asfortranarray(eris['ppee'][nocc:    ,     :    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:eeea", numpy.asfortranarray(eris['ppee'][nocc:    , ncor:nocc,     :    ,     :    ].transpose(2,0,3,1)))

    # OUTPUT EVERYTHING (for debug of PT3)
    #int2=ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo, mo), compact=False)
    #int2.shape=(norb,norb,norb,norb)
    #dom=['c','a','e']
    #inout={}
    #inout['c']=[nfro,ncor]
    #inout['a']=[ncor,nocc]
    #inout['e']=[nocc,norb]
    #for p in range(3):
    #  for q in range(3):
    #    for r in range(3):
    #      for s in range(3):
    #        name="W:"+dom[p]+dom[q]+dom[r]+dom[s]
    #        test=int2[inout[dom[p]][0]:inout[dom[p]][1],\
    #                  inout[dom[r]][0]:inout[dom[r]][1],\
    #                  inout[dom[q]][0]:inout[dom[q]][1],\
    #                  inout[dom[s]][0]:inout[dom[s]][1]].transpose(0,2,1,3)
    #        print('Output: '+name+' Shape:',test.shape)
    #        numpy.save(intfolder+name, numpy.asfortranarray(test))

    print("Basic ingredients wrote to "+intfolder)
    print("")


    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (not fully_ic):
      # About symmetry...
      from pyscf import symm
      mol = mc.mol
      orbsymout=[]
      orbsym = []
      if (mol.symmetry):
          orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                       mol.symm_orb, mo, s=mc._scf.get_ovlp())
      if mol.symmetry and orbsym:
          if mol.groupname.lower() == 'dooh':
              orbsymout = [dmrg_sym.IRREP_MAP['D2h'][i % 10] for i in orbsym]
          elif mol.groupname.lower() == 'coov':
              orbsymout = [dmrg_sym.IRREP_MAP['C2v'][i % 10] for i in orbsym]
          else:
              orbsymout = [dmrg_sym.IRREP_MAP[mol.groupname][i] for i in orbsym]
      else:
          orbsymout = []

      virtOrbs = range(nocc, eris_sp['h1eff'].shape[0])
      chunks = len(virtOrbs)/aaavsplit
      virtRange = [virtOrbs[i:i+chunks] for i in xrange(0, len(virtOrbs), chunks)]

      for K in range(aaavsplit):
          currentOrbs = range(ncor, nocc)+virtRange[K]
          fout = open('FCIDUMP_aaav%d'%(K),'w')
          #tools.fcidump.write_head(fout, eris_sp['h1eff'].shape[0]-ncor, mol.nelectron-2*ncor, orbsym= orbsymout[ncor:])

          tools.fcidump.write_head(fout, nact+len(virtRange[K]), mol.nelectron-2*ncor, orbsym= (orbsymout[ncor:nocc]+orbsymout[virtRange[K][0]:virtRange[K][-1]+1]) )
          for i in range(len(currentOrbs)):
              for j in range(ncor, nocc):
                  for k in range(nact):
                      for l in range(k+1):
                          I = currentOrbs[i]
                          if abs(eris['ppaa'][I,j,k,l]) > 1.e-8 :
                              fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                             % (eris['ppaa'][I,j,k,l], i+1, j+1-ncor, k+1, l+1))

          h1eff = numpy.zeros(shape=(nact+len(virtRange[K]), nact+len(virtRange[K])))
          h1eff[:nact, :nact] = eris_sp['h1eff'][ncor:nocc,ncor:nocc]
          h1eff[nact:, nact:] = eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, virtRange[K][0]:virtRange[K][-1]+1]
          h1eff[:nact, nact:] = eris_sp['h1eff'][ncor:nocc, virtRange[K][0]:virtRange[K][-1]+1]
          h1eff[nact:, :nact] = eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, ncor:nocc]

          tools.fcidump.write_hcore(fout, h1eff, nact+len(virtRange[K]), tol=1e-8)
          #tools.fcidump.write_hcore(fout, eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, virtRange[K][0]:virtRange[K][-1]+1], len(virtRange[K]), tol=1e-8)
          fout.write(' %17.9e  0  0  0  0\n' %( mol.energy_nuc()+energy_core-energyE0))
          fout.close()
          print("Wrote FCIDUMP_aaav%d file"%(K))

      nocc = ncor+nact
      fout = open('FCIDUMP_aaac','w')
      tools.fcidump.write_head(fout, nocc-nfro, mol.nelectron-2*nfro, orbsym= orbsymout[nfro:nocc])
      for i in range(nfro,nocc):
          for j in range(ncor, nocc):
              for k in range(ncor, nocc):
                  for l in range(ncor,k+1):
                      if abs(eris['ppaa'][i,j,k-ncor,l-ncor]) > 1.e-8 :
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                     % (eris['ppaa'][i,j,k-ncor,l-ncor], i+1-nfro, j+1-nfro, k+1-nfro, l+1-nfro))

      dmrge = energyE0-mol.energy_nuc()-energy_core

      ecore_aaac = 0.0;
      for i in range(nfro,ncor):
          ecore_aaac += 2.0*eris_sp['h1eff'][i,i]
      tools.fcidump.write_hcore(fout, eris_sp['h1eff'][nfro:nocc,nfro:nocc], nocc-nfro, tol=1e-8)
      fout.write(' %17.9e  0  0  0  0\n' %( -dmrge-ecore_aaac))
      fout.close()
      print("Wrote FCIDUMP_aaac  file")
      print("")

    return norb, energyE0


def writeMRLCCIntegrals(mc, E1, E2, nfro, fully_ic=False, third_order=False):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff
    intfolder=mc.fcisolver.scratchDirectory+'/int/'
    intfolder='int/'



    # (Note: Integrals are in chemistry notation)
    eris={}
    eris_sp={}
    eris['pcpc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,nfro:ncor], mo, mo[:,nfro:ncor]), compact=False)
    eris['ppcc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nfro:ncor], mo[:,nfro:ncor]), compact=False)
    eris['papa'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,ncor:nocc], mo, mo[:,ncor:nocc]), compact=False)
    eris['ppaa'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,ncor:nocc], mo[:,ncor:nocc]), compact=False)
    eris['pepe'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,nocc:], mo, mo[:,nocc:]), compact=False)
    eris['ppee'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nocc:], mo[:,nocc:]), compact=False)
    eris['pcpc'].shape=(norb, ncor-nfro, norb, ncor-nfro)
    eris['ppcc'].shape=(norb, norb, ncor-nfro, ncor-nfro)
    eris['papa'].shape=(norb, nact, norb, nact)
    eris['ppaa'].shape=(norb, norb, nact, nact)
    eris['pepe'].shape=(norb, nvir, norb, nvir)
    eris['ppee'].shape=(norb, norb, nvir, nvir)

    # h1eff
    hcore    = mc.get_hcore()
    dmcore   = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    eris_sp['h1eff'] = reduce(numpy.dot, (mo.T, hcore    , mo))+vhfcore
    numpy.save(intfolder+"int1eff",numpy.asfortranarray(eris_sp['h1eff'][nfro:,nfro:]))

    # int1
    hcore    = mc.get_hcore()
    dmcore   = numpy.dot(mo[:,:nfro], mo[:,:nfro].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1     = reduce(numpy.dot, (mo.T, hcore    , mo))+vhfcore
    numpy.save(intfolder+"int1",   numpy.asfortranarray(int1[nfro:,nfro:]))

    # energy_core
    hcore  = mc.get_hcore()
    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    energy_core = numpy.einsum('ij,ji', dmcore, hcore) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*numpy.einsum('ij,ij',     E1, eris_sp['h1eff'][ncor:nocc,ncor:nocc])\
             + 0.5*numpy.einsum('ijkl,ijkl', E2, eris['ppaa'][ncor:nocc,ncor:nocc,:,:].transpose(0,2,1,3))
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print("Energy_core = ",energy_core)
    print("Energy      = ", energyE0)
    print("")

    # offdiagonal warning
    offdiagonal = 0.0
    for k in range(ncor):
      for l in range(ncor):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    for k in range(nocc, norb):
      for l in range(nocc,norb):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    if (abs(offdiagonal) > 1e-6):
      print("WARNING: Have to use natural orbitals from CAASCF")
      print("         offdiagonal elements:", offdiagonal)
      print("")

    # Write out ingredients to intfolder
    # 2 "C"
    numpy.save(intfolder+"W:ccae", numpy.asfortranarray(eris['pcpc'][ncor:nocc,     :    , nocc:    ,     :    ].transpose(1,3,0,2)))
    numpy.save(intfolder+"W:eecc", numpy.asfortranarray(eris['pcpc'][nocc:    ,     :    , nocc:    ,     :    ].transpose(0,2,1,3)))
    if (True):
      numpy.save(intfolder+"W:ceec", numpy.asfortranarray(eris['pcpc'][nocc:    ,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
      numpy.save(intfolder+"W:cccc", numpy.asfortranarray(eris['ppcc'][nfro:ncor, nfro:ncor,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:cece", numpy.asfortranarray(eris['ppcc'][nocc:    , nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    if (third_order):
      numpy.save(intfolder+"W:ccca", numpy.asfortranarray(eris['ppcc'][nfro:ncor, ncor:nocc,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:ccce", numpy.asfortranarray(eris['ppcc'][nfro:ncor, nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:cace", numpy.asfortranarray(eris['ppcc'][ncor:nocc, nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    # 2 "A"
    numpy.save(intfolder+"W:caac", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,3,1,2)))
    numpy.save(intfolder+"W:ccaa", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:aeca", numpy.asfortranarray(eris['papa'][nfro:ncor,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
    numpy.save(intfolder+"W:eeaa", numpy.asfortranarray(eris['papa'][nocc:    ,     :    , nocc:    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:aaaa", numpy.asfortranarray(eris['ppaa'][ncor:nocc, ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:eaca", numpy.asfortranarray(eris['ppaa'][nocc:    , nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
    numpy.save(intfolder+"W:caca", numpy.asfortranarray(eris['ppaa'][nfro:ncor, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
    if (True):
      numpy.save(intfolder+"W:aeea", numpy.asfortranarray(eris['papa'][nocc:    ,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
      numpy.save(intfolder+"W:aeae", numpy.asfortranarray(eris['ppaa'][nocc:    , nocc:    ,     :    ,     :    ].transpose(2,0,3,1)))
    if (fully_ic):
      numpy.save(intfolder+"W:eaaa", numpy.asfortranarray(eris['ppaa'][nocc:    , ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
      numpy.save(intfolder+"W:caaa", numpy.asfortranarray(eris['ppaa'][nfro:ncor, ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
    # 2 "E"
    numpy.save(intfolder+"W:eeca", numpy.asfortranarray(eris['pepe'][nfro:ncor,     :    , ncor:nocc,     :    ].transpose(1,3,0,2)))
    if (third_order):
      numpy.save(intfolder+"W:aece", numpy.asfortranarray(eris['ppee'][ncor:nocc, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
      numpy.save(intfolder+"W:eeee", numpy.asfortranarray(eris['ppee'][nocc:    ,nocc :    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:eeec", numpy.asfortranarray(eris['ppee'][nocc:    ,     :    ,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:eeea", numpy.asfortranarray(eris['ppee'][nocc:    , ncor:nocc,     :    ,     :    ].transpose(2,0,3,1)))

    feri = h5py.File(intfolder+"int2eeee.hdf5", 'w')
    ao2mo.full(mc.mol, mo[:,nocc:], feri, compact=False)
    for o in range(nvir):
      int2eee = feri['eri_mo'][o*(norb-nocc):(o+1)*(norb-nocc),:]
      numpy.asfortranarray(int2eee).tofile(intfolder+"W:eeee%04d"%(o))

    # OUTPUT EVERYTHING (for debug of PT3)
    #int2=ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo, mo), compact=False)
    #int2.shape=(norb,norb,norb,norb)
    #dom=['c','a','e']
    #inout={}
    #inout['c']=[nfro,ncor]
    #inout['a']=[ncor,nocc]
    #inout['e']=[nocc,norb]
    #for p in range(3):
    #  for q in range(3):
    #    for r in range(3):
    #      for s in range(3):
    #        name="W:"+dom[p]+dom[q]+dom[r]+dom[s]
    #        test=int2[inout[dom[p]][0]:inout[dom[p]][1],\
    #                  inout[dom[r]][0]:inout[dom[r]][1],\
    #                  inout[dom[q]][0]:inout[dom[q]][1],\
    #                  inout[dom[s]][0]:inout[dom[s]][1]].transpose(0,2,1,3)
    #        print('Output: '+name+' Shape:',test.shape)
    #        numpy.save(intfolder+name, numpy.asfortranarray(test))

    print("Basic ingredients wrote to "+intfolder)
    print("")


    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (not fully_ic):
      # About symmetry...
      from pyscf import symm
      mol = mc.mol
      orbsymout=[]
      orbsym = []
      if (mol.symmetry):
          orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                       mol.symm_orb, mo, s=mc._scf.get_ovlp())
      if mol.symmetry and orbsym:
          if mol.groupname.lower() == 'dooh':
              orbsymout = [dmrg_sym.IRREP_MAP['D2h'][i % 10] for i in orbsym]
          elif mol.groupname.lower() == 'coov':
              orbsymout = [dmrg_sym.IRREP_MAP['C2v'][i % 10] for i in orbsym]
          else:
              orbsymout = [dmrg_sym.IRREP_MAP[mol.groupname][i] for i in orbsym]
      else:
          orbsymout = []


      fout = open('FCIDUMP_aaav0','w')
      tools.fcidump.write_head(fout, int1.shape[0]-ncor, mc.mol.nelectron-2*ncor, orbsym= orbsymout[ncor:])
      for i in range(ncor,int1.shape[0]):
          for j in range(ncor, i+1):
              for k in range(mc.ncas):
                  for l in range(k+1):
                      if abs(eris['ppaa'][i,j, k,l]) > 1.e-8 :
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                     % (eris['ppaa'][i,j, k,l], i+1-ncor, j+1-ncor, k+1, l+1))
                      if (j >= nocc and abs(eris['papa'][i, k, j, l]) > 1.e-8):
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                         % (eris['papa'][i,k,j, l], i+1-ncor, k+1, l+1, j+1-ncor))
                      if (j >= nocc and abs(eris['papa'][i, l, j, k]) > 1.e-8):
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                         % (eris['papa'][i,l, j, k], i+1-ncor, l+1, k+1, j+1-ncor))

      tools.fcidump.write_hcore(fout, eris_sp['h1eff'][ncor:,ncor:], int1.shape[0]-ncor, tol=1e-8)
      fout.write(' %17.9e  0  0  0  0\n' %( mc.mol.energy_nuc()+energy_core-energyE0))
      fout.close()
      print("Wrote FCIDUMP_aaav0 file")

      eri1cas = ao2mo.outcore.general_iofree(mc.mol, (mo[:,nfro:nocc], mo[:,nfro:nocc], mo[:,nfro:nocc], mo[:,nfro:nocc]), compact=True)
      tools.fcidump.from_integrals("FCIDUMP_aaac", int1[nfro:nocc,nfro:nocc], eri1cas, nocc-nfro, mc.mol.nelectron-2*nfro, nuc=mc.mol.energy_nuc()-energyE0, orbsym = orbsymout[nfro:nocc], tol=1e-8)
      print("Wrote FCIDUMP_aaac  file")
      print("")

    return energyE0, norb


def writeNEVPTIntegralsDF(mc, dm1, dm2, dm1eff, nfro, fully_ic=False):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nocc = ncor+nact
    mo   = mc.mo_coeff


    # Lpq
    Lpq = None
    #eri_test=numpy.zeros((norb*norb,norb*norb))
    for eris in mc.with_df.loop():
      Lpq = _ao2mo.nr_e2(eris, mo,\
                        (0,norb,0,norb), aosym='s2', out=Lpq)
    #  lib.dot(Lpq.T, Lpq, 1, eri_test, 1)
    #eri_test=eri_test.reshape(norb,norb,norb,norb)
    #for m in range(norb):
    #  for n in range(norb):
    #    for p in range(nact):
    #      for q in range(nact):
    #        print('{:5}{:5}{:5}{:5}{:13.6f}'.format(m,n,p,q,eri_test[m,n,ncor+p,ncor+q]))
    Lpq=Lpq.reshape(-1,norb,norb)
    naux  = Lpq.shape[0]

    #from pyscf.ao2mo.incore import _conc_mos, iden_coeffs
    #mo_here=mo
    #if isinstance(mo_here, numpy.ndarray) and mo_here.ndim == 2:
    #  mo_here = (mo_here,) * 4
    #ijmosym, nij_pair, moij, ijslice = _conc_mos(mo_here[0], mo_here[1], False)
    #klmosym, nkl_pair, mokl, klslice = _conc_mos(mo_here[2], mo_here[3], False)
    #mo_eri = numpy.zeros((nij_pair,nkl_pair))
    #sym = (iden_coeffs(mo_here[0], mo_here[2]) and
    #       iden_coeffs(mo_here[1], mo_here[3]))
    #Lij = Lkl = None
    #print('info:',ijmosym,klmosym,sym)
    #print('info:',ijslice,klslice,moij.shape)
    #for eri1 in mc.with_df.loop():
    #  print('loop?')
    #  Lij = _ao2mo.nr_e2(eri1, moij, ijslice, aosym='s2', mosym=ijmosym, out=Lij)
    #  if sym:
    #    Lkl = Lij
    #  else:
    #    Lkl = _ao2mo.nr_e2(eri1, mokl, klslice, aosym='s2', mosym=klmosym, out=Lkl)
    #  lib.dot(Lij.T, Lkl, 1, mo_eri, 1)
    #mo_eri=mo_eri.reshape(norb,norb,norb,norb)
    #print(numpy.allclose(mo_eri,eri_test))
    ##for m in range(norb):
    ##  for n in range(norb):
    ##    for p in range(nact):
    ##      for q in range(nact):
    ##        print('{:5}{:5}{:5}{:5}{:13.6f}'.format(m,n,p,q,mo_eri[m,n,ncor+p,ncor+q]))



    # int1_eff and energy_core
    hcore    = mc.get_hcore()
    dmcore   = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1_eff = reduce(numpy.dot, (mo.T, hcore,     mo))+vhfcore
    energy_core = numpy.einsum('ij,ji', dmcore, hcore) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*numpy.einsum('ij,ij',        dm1, int1_eff[ncor:nocc, ncor:nocc])\
             + 0.5*numpy.einsum('ikjl,Pij,Pkl', dm2, Lpq[:,   ncor:nocc, ncor:nocc],\
                                                     Lpq[:,   ncor:nocc, ncor:nocc])
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print("Energy_core = ",energy_core)
    print("Energy      = ", energyE0)
    print("")


    # Write out ingredients to intfolder
    intfolder=mc.fcisolver.scratchDirectory+'/int/'
    intfolder='int/'
    numpy.save(intfolder+"W:Laa", numpy.asfortranarray(Lpq[:,ncor:nocc, ncor:nocc]))
    numpy.save(intfolder+"W:Lcc", numpy.asfortranarray(Lpq[:,    :ncor,     :ncor]))
    numpy.save(intfolder+"W:Lee", numpy.asfortranarray(Lpq[:,nocc:    , nocc:    ]))
    numpy.save(intfolder+"W:Lca", numpy.asfortranarray(Lpq[:,    :ncor, ncor:nocc]))
    numpy.save(intfolder+"W:Lac", numpy.asfortranarray(Lpq[:,ncor:nocc,     :ncor]))
    numpy.save(intfolder+"W:Lce", numpy.asfortranarray(Lpq[:,    :ncor, nocc:    ]))
    numpy.save(intfolder+"W:Lec", numpy.asfortranarray(Lpq[:,nocc:    ,     :ncor]))
    numpy.save(intfolder+"W:Lea", numpy.asfortranarray(Lpq[:,nocc:    , ncor:nocc]))
    numpy.save(intfolder+"W:Lae", numpy.asfortranarray(Lpq[:,ncor:nocc, nocc:    ]))
    numpy.save(intfolder+"int1eff",numpy.asfortranarray(int1_eff[nfro:,nfro:]))

    print("Basic ingredients wrote to "+intfolder)
    print("")


    # "fully_ic" isn't ready
    if (not fully_ic):
        print("Did not think about Density Fitting for uncontracted AAAC and AAAV: do 'fully_ic'")
        exit(0)

    return norb, naux, energyE0


def writeMRLCCIntegralsDF(mc, E1, E2, nfro, fully_ic=False):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nocc = ncor+nact
    mo   = mc.mo_coeff


    # Lpq
    Lpq = None
    for eris in mc.with_df.loop():
      Lpq = _ao2mo.nr_e2(eris, mo,\
                        (0,norb,0,norb), aosym='s2', out=Lpq)
    Lpq=Lpq.reshape(-1,norb,norb)
    naux  = Lpq.shape[0]

    # int1
    hcore    = mc.get_hcore()
    dmcore   = numpy.dot(mo[:,:nfro], mo[:,:nfro].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1     = reduce(numpy.dot, (mo.T, hcore    , mo))+vhfcore

    # int1_eff and energy_core
    dmcore   = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1_eff = reduce(numpy.dot, (mo.T, hcore,     mo))+vhfcore
    energy_core = numpy.einsum('ij,ji', dmcore, hcore) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*numpy.einsum('ij,ij',        E1, int1_eff[ncor:nocc, ncor:nocc])\
             + 0.5*numpy.einsum('ikjl,Pij,Pkl', E2, Lpq[:,   ncor:nocc, ncor:nocc],\
                                                    Lpq[:,   ncor:nocc, ncor:nocc])
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print("Energy_core = ",energy_core)
    print("Energy      = ", energyE0)
    print("")


    # Write out ingredients to intfolder
    intfolder=mc.fcisolver.scratchDirectory+'/int/'
    intfolder='int/'
    numpy.save(intfolder+"W:Laa", numpy.asfortranarray(Lpq[:,ncor:nocc, ncor:nocc]))
    numpy.save(intfolder+"W:Lcc", numpy.asfortranarray(Lpq[:,    :ncor,     :ncor]))
    numpy.save(intfolder+"W:Lee", numpy.asfortranarray(Lpq[:,nocc:    , nocc:    ]))
    numpy.save(intfolder+"W:Lca", numpy.asfortranarray(Lpq[:,    :ncor, ncor:nocc]))
    numpy.save(intfolder+"W:Lac", numpy.asfortranarray(Lpq[:,ncor:nocc,     :ncor]))
    numpy.save(intfolder+"W:Lce", numpy.asfortranarray(Lpq[:,    :ncor, nocc:    ]))
    numpy.save(intfolder+"W:Lec", numpy.asfortranarray(Lpq[:,nocc:    ,     :ncor]))
    numpy.save(intfolder+"W:Lea", numpy.asfortranarray(Lpq[:,nocc:    , ncor:nocc]))
    numpy.save(intfolder+"W:Lae", numpy.asfortranarray(Lpq[:,ncor:nocc, nocc:    ]))
    numpy.save(intfolder+"W:Lee", numpy.asfortranarray(Lpq[:,nocc:    , nocc:    ]))
    numpy.save(intfolder+"int1",   numpy.asfortranarray(int1[nfro:,nfro:]))
    numpy.save(intfolder+"int1eff",numpy.asfortranarray(int1_eff[nfro:,nfro:]))

    print("Basic ingredients wrote to "+intfolder)
    print("")


    # "fully_ic" isn't ready
    if (not fully_ic):
        print("Did not think about Density Fitting for uncontracted AAAC and AAAV: do 'fully_ic'")
        exit(0)

    return energyE0, norb, naux


def executeMRLCC(nelec, ncor, ncas, nfro, ms2, naux=0, memory=10, fully_ic=False, third_order=False, cumulantE4=False, df=False, no_handcoded_E3=False):
    methods = ['MRLCC_CCVV', 'MRLCC_CCAV', 'MRLCC_ACVV', 'MRLCC_CCAA', 'MRLCC_AAVV', 'MRLCC_CAAV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']
    if (fully_ic):
        methods+=['MRLCC_AAAV', 'MRLCC_AAAC']
        domains+=['eaaa','caaa']
    if (third_order):
        methods+=['MRLCC3']
    if (ncor - nfro) == 0:
        methods = ['MRLCC_AAVV']
        domains = ['eeaa']
    totalE = 0.0
    print("Second-order:")
    for method in methods:
        f = open("%s.inp"%(method), 'w')
        if (memory is not None):
          f.write('work-space-mb %d\n'%(memory*1000))
        f.write('method %s\n'%(method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncor-nfro)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
        f.write('ms2 %d\n'%(ms2))
        f.write('int1e/fock int/int1eff.npy\n')
        f.write('int1e/coreh int/int1.npy\n')
        #f.write('E3  int/E3.npy\n')
        #f.write('E2  int/E2.npy\n')
        #f.write('E1  int/E1.npy\n')
        f.write('thr-den 1.000000e-05\n')
        f.write('thr-var 1.000000e-05\n')
        f.write('thr-trunc 1.000000e-04\n')
        if (third_order and method!='MRLCC3'):
          f.write('save int/D:'+domains[methods.index(method)]+'.npy\n')
        if (cumulantE4):
          f.write('cumulantE4 1\n')
        if (df):
          f.write('naux %d\n'%(naux))
          f.write('handcodedWeeee 0\n')
          f.write('handcodedE3 0\n')
        if (no_handcoded_E3 and not df):
          f.write('handcodedE3 0\n')
        f.close();

        from subprocess import check_call
        infile="%s.inp"%(method)
        outfile="%s.out"%(method)
        output = check_call("%s  %s  %s > %s"%(mpiprefix, executable, infile, outfile), shell=True)
        energy=ReadWriteEnergy(outfile,method)
        if (method!='MRLCC3'):
          try:
            totalE+=energy
          except ValueError:
            continue
        sys.stdout.flush()
    if (fully_ic):
        print("Total:             %18.9e"%(totalE))
        print("")
    if (third_order):
      try:
        print("Third-order:       %18.9e"%(energy))
        print("")
        totalE+=energy
      except ValueError:
        print("Third-order    --  NA")
        print("")
    return totalE


def executeNEVPT(nelec, ncor, ncas, nfro, ms2, naux=0, memory=10, fully_ic=False, third_order=False, cumulantE4=False, df=False, no_handcoded_E3=False):
    methods = ['NEVPT2_CCVV', 'NEVPT2_CCAV', 'NEVPT2_ACVV', 'NEVPT2_CCAA', 'NEVPT2_AAVV', 'NEVPT2_CAAV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']
    if (fully_ic):
        methods+=['NEVPT2_AAAV', 'NEVPT2_AAAC']
        domains+=['eaaa','caaa']
    if (third_order):
        methods+=['NEVPT3']
    if (ncor - nfro) == 0:
        methods = ['NEVPT2_AAVV']
        domains = ['eeaa']
    totalE = 0.0
    print("Second-order:")
    for method in methods:
        f = open("%s.inp"%(method), 'w')
        if (memory is not None):
            f.write('work-space-mb %d\n'%(memory*1000))
        f.write('method %s\n'%(method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncor-nfro)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
        f.write('ms2 %d\n'%(ms2))
        f.write('int1e/fock int/int1eff.npy\n')
        #f.write('E3  int/E3.npy\n')
        #f.write('E2  int/E2.npy\n')
        #f.write('E1  int/E1.npy\n')
        f.write('thr-den 1.000000e-05\n')
        f.write('thr-var 1.000000e-05\n')
        f.write('thr-trunc 1.000000e-03\n')
        if (third_order and method!='NEVPT3'):
          f.write('save int/D:'+domains[methods.index(method)]+'.npy\n')
        if (cumulantE4):
          f.write('cumulantE4 1\n')
        if (df):
          f.write('naux %d\n'%(naux))
          f.write('handcodedWeeee 0\n')
          f.write('handcodedE3 0\n')
        if (no_handcoded_E3 and not df):
          f.write('handcodedE3 0\n')
        f.close();

        from subprocess import check_call
        infile="%s.inp"%(method)
        outfile="%s.out"%(method)
        output = check_call("%s  %s  %s > %s"%(mpiprefix, executable, infile, outfile), shell=True)
        energy=ReadWriteEnergy(outfile,method)
        if (method!='NEVTP3'):
          try:
            totalE+=energy
          except ValueError:
            continue
        sys.stdout.flush()
    if (fully_ic):
        print("Total:             %18.9e"%(totalE))
        print("")
    if (third_order):
      try:
        print("Third-order:       %18.9e"%(energy))
        print("")
        totalE+=energy
      except ValueError:
        print("Third-order:       NA")
        print("")
    return totalE


def ReadWriteEnergy(outfile,pattern):
  import re
  energy='not_found'
  with open(outfile,'r') as origin:
    for line in origin:
      if re.search(pattern+'.*ENERGY',line):
        energy=line.split()[-1]
        break
  try:
    energy=float(energy)
    if((pattern!='MRLCC3')and(pattern!='NEVPT3')):
      print("perturber %s --  %18.9e"%(pattern[-4:], energy))
  except ValueError:
    if((pattern!='MRLCC3')and(pattern!='NEVPT3')):
      print("perturber %s --  NA"%(pattern[-4:]))
  return energy


def kernel(mc, *args, **kwargs):
    return icmpspt(mc, *args, **kwargs)


def _ERIS(mc, mo, method='incore'):
    nmo = mo.shape[1]
    ncor = mc.ncore
    ncas = mc.ncas

    if ((method == 'outcore') or
        (mcscf.mc_ao2mo._mem_usage(ncor, ncas, nmo)[0] +
         nmo**4*2/1e6 > mc.max_memory*.9) or
        (mc._scf._eri is None)):
        ppaa, papa, pacv, cvcv = \
                trans_e1_outcore(mc, mo, max_memory=mc.max_memory,
                                 verbose=mc.verbose)
    else:
        ppaa, papa, pacv, cvcv = trans_e1_incore(mc, mo)

    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj*2-vk, mo))


    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore


    return eris


# see mcscf.mc_ao2mo
def trans_e1_incore(mc, mo):
    eri_ao = mc._scf._eri
    ncor = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nocc = ncor + ncas
    nav = nmo - ncor
    eri1 = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo[:,ncor:]),
                                      compact=False)
    load_buf = lambda r0,r1: eri1[r0*nav:r1*nav]
    ppaa, papa, pacv, cvcv = _trans(mo, ncor, ncas, load_buf)
    return ppaa, papa, pacv, cvcv


def trans_e1_outcore(mc, mo, max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=0):
    time0 = (time.clock(), time.time())
    mol = mc.mol
    log = logger.Logger(mc.stdout, verbose)
    ncor = mc.ncore
    ncas = mc.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncor + ncas
    nvir = nmo - nocc
    nav = nmo - ncor

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo[:,ncor:]), swapfile.name,
                                max_memory=max_memory, ioblk_size=ioblk_size,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(r0,r1):
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'between load_buf',
                                              *tuple(time1))
        buf = numpy.empty(((r1-r0)*nav,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:,col0:col1] = dat[r0*nav:r1*nav]
            col0 = col1
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'load_buf', *tuple(time1))
        return buf
    time0 = logger.timer(mol, 'halfe1', *time0)
    time1 = [time.clock(), time.time()]
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    cvcvfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    with h5py.File(cvcvfile.name) as f5:
        cvcv = f5.create_dataset('eri_mo', (ncor*nvir,ncor*nvir), 'f8')
        ppaa, papa, pacv = _trans(mo, ncor, ncas, load_buf, cvcv, ao_loc)[:3]
    time0 = logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return ppaa, papa, pacv, cvcvfile


def _trans(mo, ncor, ncas, fload, cvcv=None, ao_loc=None):
    nao, nmo = mo.shape
    nocc = ncor + ncas
    nvir = nmo - nocc
    nav = nmo - ncor

    if cvcv is None:
        cvcv = numpy.zeros((ncor*nvir,ncor*nvir))
    pacv = numpy.empty((nmo,ncas,ncor*nvir))
    aapp = numpy.empty((ncas,ncas,nmo*nmo))
    papa = numpy.empty((nmo,ncas,nmo*ncas))
    vcv = numpy.empty((nav,ncor*nvir))
    apa = numpy.empty((ncas,nmo*ncas))
    vpa = numpy.empty((nav,nmo*ncas))
    app = numpy.empty((ncas,nmo*nmo))
    for i in range(ncor):
        buf = fload(i, i+1)
        klshape = (0, ncor, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        cvcv[i*nvir:(i+1)*nvir] = vcv[ncas:]
        pacv[i] = vcv[:ncas]

        klshape = (0, nmo, ncor, nocc)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=apa, ao_loc=ao_loc)
        papa[i] = apa
    for i in range(ncas):
        buf = fload(ncor+i, ncor+i+1)
        klshape = (0, ncor, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        pacv[ncor:,i] = vcv

        klshape = (0, nmo, ncor, nocc)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vpa, ao_loc=ao_loc)
        papa[ncor:,i] = vpa

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=app, ao_loc=ao_loc)
        aapp[i] = app
    #lib.transpose(aapp.reshape(ncas**2, -1), inplace=True)
    ppaa = lib.transpose(aapp.reshape(ncas**2,-1))
    return (ppaa.reshape(nmo,nmo,ncas,ncas), papa.reshape(nmo,ncas,nmo,ncas),
            pacv.reshape(nmo,ncas,ncor,nvir), cvcv)


def writeAAAConfFile(neleca, nelecb, ncor, ncas, norb, DMRGCI, maxM, perturber, memory, numthrds, reorder, extraline, root=0, approx= False, aaavsplit=1, aaavIter=0, name= "MRLCC"):
    virtOrbs = range(norb-ncor-ncas)
    chunks = len(virtOrbs)/aaavsplit
    virtRange = [virtOrbs[i:i+chunks] for i in xrange(0, len(virtOrbs), chunks)]

    if (perturber == "AAAC"):
        configFile = "response%s_aaac.conf"%(name)
        r = open("reorder_aaac.txt", 'w')
        for i in range(ncas):
            r.write("%d "%(reorder[i]+1+ncor))
        for i in range(ncor):
            r.write("%d "%(i+1))
        r.close()
    else:
        configFile = "response%s_aaav%d.conf"%(name, aaavIter)
        r = open("reorder_aaav%d.txt"%(aaavIter), 'w')
        for i in range(ncas):
            r.write("%d "%(reorder[i]+1))
        for i in range(len(virtRange[aaavIter])):
            r.write("%d "%(i+1+ncas))
        r.close()
    f = open(configFile, 'w')

    if (memory is not None):
        f.write('memory, %i, g\n'%(memory))
    ncorE = 0
    if (perturber == "AAAC"):
        ncorE = 2*ncor
    f.write('nelec %i\n'%(neleca+nelecb+ncorE))
    f.write('spin %i\n' %(neleca-nelecb))
    if isinstance(DMRGCI.wfnsym, str):
        wfnsym = dmrg_sym.irrep_name2id(DMRGCI.mol.groupname, DMRGCI.wfnsym)
    else:
        wfnsym = DMRGCI.wfnsym
    f.write('irrep %i\n' % wfnsym)

    f.write('schedule\n')
    if (maxM <= DMRGCI.maxM):
        maxM = DMRGCI.maxM+1

    iter = 0
    for M in range(DMRGCI.maxM, maxM, 1000):
        f.write('%6i  %6i  %8.4e  %8.4e \n' %(iter*4, M, 1e-6, 1.0e-5))
        iter += 1

    f.write('%6i  %6i  %8.4e  %8.4e \n' %(iter*4, maxM, 1e-6, 1.0e-5))
    f.write('end\n')
    f.write('twodot \n')

    if DMRGCI.mol.symmetry:
        if DMRGCI.groupname.lower() == 'dooh':
            f.write('sym d2h\n' )
        elif DMRGCI.groupname.lower() == 'coov':
            f.write('sym c2v\n' )
        else:
            f.write('sym %s\n' % DMRGCI.groupname.lower())

    integralFile = "FCIDUMP_aaav%d"%(aaavIter)
    if (perturber == "AAAC"):
        integralFile = "FCIDUMP_aaac"
    f.write('orbitals %s\n' % integralFile)

    f.write('maxiter %i\n'%(4*iter+4))
    f.write('sweep_tol %8.4e\n'%DMRGCI.tol)

    f.write('outputlevel %s\n'%DMRGCI.outputlevel)
    f.write('hf_occ integral\n')

    if(DMRGCI.scratchDirectory):
        f.write('prefix  %s\n'%DMRGCI.scratchDirectory)

    f.write('num_thrds %d\n'%DMRGCI.num_thrds)

    for line in extraline:
        f.write('%s\n'%line)

    f.write('occ %d\n'%(10000))
    if (perturber == "AAAC") :
        f.write("reorder reorder_aaac.txt\n")
        f.write("responseaaac\n")
        f.write("baseStates %d\n"%(root))
        f.write("projectorStates %d\n"%(root))
        f.write("targetState 12\n")
        f.write("partialsweep %d\n"%(ncas))
        f.write("open \n")
        f.write("closed ")
        for i in range(1,ncor+1):
            f.write("%d "%(i))
        f.write("\n")
    else:
        f.write("reorder reorder_aaav%d.txt\n"%(aaavIter))
        f.write("responseaaav\n")
        f.write("baseStates %d\n"%(root))
        f.write("projectorStates %d\n"%(root))
        f.write("targetState 12\n")
        f.write("partialsweep %d\n"%(ncas))
        f.write("open ")
        for i in range( len(virtRange[aaavIter]) ):
            f.write("%d "%(i+1+ncas))
        f.write("\nclosed \n")
    f.close()


#def readIntegrals(infile, dt=numpy.dtype('Float64')):
#    startScaling = False
#
#    norb = -1 #need to read norb from infile
#    nelec = -1
#
#    lines = []
#    f = open(infile, 'r')
#    lines = f.readlines()
#    f.close()
#
#    index = 0
#    for line in lines:
#        linesplit = line.replace("="," ")
#        linesplit = linesplit.replace(","," ")
#        linesp = linesplit.split()
#        if (startScaling == False and len(linesp) == 1 and (linesp[0] == "&END" or linesp[0] == "/")):
#            startScaling = True
#            index += 1
#        elif(startScaling == False):
#            if (len(linesp) > 4):
#                if (linesp[1] == "NORB"):
#                    norb = int(linesp[2])
#                if (linesp[3] == "NELEC"):
#                    nelec = int(linesp[4])
#            index += 1
#
#    if (norb == -1 or nelec == -1):
#        print("could not read the norbs or nelec")
#        exit(0)
#
#    int2 = numpy.zeros(shape=(norb, norb, norb, norb), dtype=dt, order='F')
#    int1 = numpy.zeros(shape=(norb, norb), dtype=dt, order='F')
#    coreE = 0.0
#
#    totalIntegralLines = len(lines) - index
#    for i in range(totalIntegralLines):
#        linesp = lines[i+index].split()
#        if (len(linesp) != 5) :
#            continue
#        integral, a, b, c, d = float(linesp[0]), int(linesp[1]), int(linesp[2]), int(linesp[3]), int(linesp[4])
#
#        if(a==b==c==d==0):
#            coreE = integral
#        elif (c==d==0):
#            int1[a-1,b-1] = integral
#            int1[b-1,a-1] = integral
#            A,B = max(a,b), min(a,b)
#        else:
#            int2[a-1, c-1, b-1, d-1] = integral
#            int2[b-1, c-1, a-1, d-1] = integral
#            int2[a-1, d-1, b-1, c-1] = integral
#            int2[b-1, d-1, a-1, c-1] = integral
#
#            int2[c-1, a-1, d-1, b-1] = integral
#            int2[d-1, a-1, c-1, b-1] = integral
#            int2[c-1, b-1, d-1, a-1] = integral
#            int2[d-1, b-1, c-1, a-1] = integral
#
#    return norb, nelec, int2, int1, coreE


#def makeheff(int1, int2popo, int2ppoo, E1, ncor, nvir, nfro):
#        nocc = int1.shape[0]-nvir
#
#        int1_eff = 1.*int1 + 2.0*numpy.einsum('mnii->mn', int2ppoo[:, :, nfro:ncor, nfro:ncor])\
#                                -numpy.einsum('mini->mn', int2popo[:, nfro:ncor, :, nfro:ncor])
#
#        int1_eff[:ncor, :ncor] += numpy.einsum('lmjk,jk->lm',int2ppoo[:ncor,:ncor,ncor:nocc,ncor:nocc], E1)
#                            - 0.5*numpy.einsum('ljmk,jk->lm',int2popo[:ncor,ncor:nocc,:ncor,ncor:nocc], E1)
#        int1_eff[nocc:, nocc:] += numpy.einsum('lmjk,jk->lm',int2ppoo[nocc:,nocc:,ncor:nocc,ncor:nocc], E1)
#                            - 0.5*numpy.einsum('ljmk,jk->lm',int2popo[nocc:,ncor:nocc,nocc:,ncor:nocc], E1)
#       #int1_eff[nocc:, nocc:] += numpy.einsum('ljmk,jk->lm',int2[nocc:,ncor:nocc,nocc:,ncor:nocc], E1)
#                            - 0.5*numpy.einsum('ljkm,jk->lm',int2[nocc:,ncor:nocc,ncor:nocc,nocc:], E1)
#        return int1_eff


#def makeheff(int1, int2, E1, ncor, nvir):
#        nocc = int1.shape[0]-nvir
#
#        int1_eff = 1.*int1 + 2.0*numpy.einsum('mini->mn', int2[:,:ncor, :, :ncor])
#                                -numpy.einsum('miin->mn', int2[:,:ncor, :ncor, :])
#
#        int1_eff[:ncor, :ncor] += numpy.einsum('ljmk,jk->lm',int2[:ncor,ncor:nocc,:ncor,ncor:nocc], E1)
#                            - 0.5*numpy.einsum('ljkm,jk->lm',int2[:ncor,ncor:nocc,ncor:nocc,:ncor], E1)
#        int1_eff[nocc:, nocc:] += numpy.einsum('ljmk,jk->lm',int2[nocc:,ncor:nocc,nocc:,ncor:nocc], E1)
#                            - 0.5*numpy.einsum('ljkm,jk->lm',int2[nocc:,ncor:nocc,ncor:nocc,nocc:], E1)
#        return int1_eff
#'''




def icmpspt(mc, pttype="NEVPT2", energyE0=0.0, rdmM=0, frozen=0, PTM=1000, PTincore=False, fciExtraLine=[],\
            have3RDM=False, root=0, nroots=1, verbose=None, AAAVsplit=1,\
            do_dm3=True, do_dm4=False, fully_ic=False, third_order=False, cumulantE4=False, no_handcoded_E3=False):
    sys.stdout.flush()
    print("")
    print("")
    print("--------------------------------------------------")
    print("                 ICMPSPT CALCULATION              ")
    print("--------------------------------------------------")
    print("")

    # Check-up consistency of keywords
    if (do_dm4):
      do_dm3=False
    elif (do_dm3):
      do_dm4=False
    else:
      print("WARNING:  Neither do_dm3 nor do_dm4! Turning do_dm3 on.")
      print("")
      do_dm3=True
    #if (fully_ic and not (do_dm4 or cumulantE4)):
    #  print("WARNING: Fully IC needs do_dm4 or cumulantE4!")
    #  print("")
    #  do_dm4=True
    #  do_dm3=False
    if ((third_order)and(not fully_ic)):
      print("WARNING: Third-order needs Fully IC mode! Turning fully_ic on.")
      print("")
      fully_ic=True
    if (pttype != "NEVPT2" and AAAVsplit != 1):
      print("AAAVsplit only works with CASSCF natural orbitals and NEVPT2")
      print("")
      exit(0)
    #if type(mc.fcisolver) is not dmrgci.DMRGCI:
    #  if (mc.fcisolver.fcibase_class is not dmrgci.DMRGCI):
    #    print("this works with dmrgscf and not regular mcscf")
    #    print("")
    #    exit(0)
    if (hasattr(mc,'with_df')):
      df=True
    else:
      df=False

    # Message
    print("Perturbation type: %s"%(pttype))
    if (fully_ic):
      print("With fully internally contracted scheme")
    if (third_order):
      print("With third order correction")
    if (df):
      print("Recognized a Density Fitting run")
    print("")

    # Remove the -1 state
    import os
    os.system("rm -f %s/node0/Rotation*.state-1.tmp"%(mc.fcisolver.scratchDirectory))
    os.system("rm -f %s/node0/wave*.-1.tmp"         %(mc.fcisolver.scratchDirectory))
    os.system("rm -f %s/node0/RestartReorder.dat_1" %(mc.fcisolver.scratchDirectory))

    # FCIsolver initiation
    mc.fcisolver.startM = 100
    mc.fcisolver.maxM = max(rdmM,501)
    mc.fcisolver.clearSchedule()
    mc.fcisolver.restart = False
    mc.fcisolver.generate_schedule()
    mc.fcisolver.extraline = []
    if (PTincore):
        mc.fcisolver.extraline.append('do_npdm_in_core')
    mc.fcisolver.extraline += fciExtraLine
    if (not have3RDM):
      mc.fcisolver.has_threepdm = False
    else:
      mc.fcisolver.has_threepdm = True


    # Prepare directory
    import os
    intfolder=mc.fcisolver.scratchDirectory+'/int/'
    intfolder='int/'
    os.system("mkdir -p "+intfolder)
    if intfolder!='int/':
      os.system("ln -s "+intfolder+" int")

    # RDMs
    print('Preparing necessary RDMs')
    nelec = mc.nelecas[0]+mc.nelecas[1]
    dm1eff = numpy.zeros(shape=(mc.ncas, mc.ncas)) #this is the state average density which is needed in NEVPT2
    # loop over all states besides the current root
    if (pttype == "NEVPT2" and nroots>1):
        stateIter = range(nroots)
        stateIter.remove(root)
        for istate in stateIter:
            dm3 = mc.fcisolver.make_rdm3(state=istate, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision)
            # This is coherent with statement about indexes made in "make_rdm3"
            # This is done with SQA in mind.
            dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
            dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
            dm1eff += dm1
    # now add the contributaion due to the current root
    if (do_dm3):
      dm3 = mc.fcisolver.make_rdm3(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype="notbinary")
      #print(numpy.einsum('ijklmn',dm3))
    elif (do_dm4):
      dm4 = mc.fcisolver.make_rdm4(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype="notbinary")
      #print(numpy.einsum('ijklmnop',dm4))
      trace=numpy.einsum('ijklijkl->',dm4)
      if abs(trace-nelec*(nelec-1)*(nelec-2)*(nelec-3))<0.000001:
          print('(GOOD) Trace 4RDM: {:5} ={:5}*{:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2,nelec-3))
      else:
          print('(BAD)  Trace 4RDM: {:5}!={:5}*{:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2,nelec-3))
      # This is coherent with statement about indexes made in "make_rdm4"
      # This is done with SQA in mind
      dm3 = numpy.einsum('ijklmnol', dm4)/(nelec-3)
      numpy.save(intfolder+"E4",dm4)
      del dm4
    # This is coherent with statement about indexes made in "make_rdm4" and "make_rdm3"
    # This is done with SQA in mind
    dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
    dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
    dm1eff += dm1
    dm1eff = dm1eff/(1.0*nroots)
    numpy.save(intfolder+"E3",dm3)
    numpy.save(intfolder+"E3B.npy", dm3.transpose(0,3,1,4,2,5))
    numpy.save(intfolder+"E3C.npy", dm3.transpose(5,0,2,4,1,3))
    numpy.save(intfolder+"E2.npy", numpy.asfortranarray(dm2))
    numpy.save(intfolder+"E1.npy", numpy.asfortranarray(dm1))
    trace=numpy.einsum('ijkijk->',dm3)
    if abs(trace-nelec*(nelec-1)*(nelec-2))<0.000001:
        print('(GOOD) Trace 3RDM: {:5} ={:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2))
    else:
        print('(BAD)  Trace 3RDM: {:5}!={:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2))
    trace=numpy.einsum('ijij->',dm2)
    if abs(trace-nelec*(nelec-1))<0.000001:
        print('(GOOD) Trace 2RDM: {:5} ={:5}*{:5}'.format(trace,nelec,nelec-1))
    else:
        print('(BAD)  Trace 2RDM: {:5}!={:5}*{:5}'.format(trace,nelec,nelec-1))
    trace=numpy.einsum('ii->',dm1)
    if abs(trace-nelec)<0.000001:
        print('(GOOD) Trace 1RDM: {:5} ={:5}'.format(trace,nelec))
    else:
        print('(BAD)  Trace 1RDM: {:5}!={:5}'.format(trace,nelec))
    del dm3
    print('')
    sys.stdout.flush()

    #backup the restartreorder file to -1. this is because responseaaav and responseaaac both overwrite this file
    #this means that when we want to restart a calculation after lets say responseaaav didnt finish, the new calculaitons
    #will use the restartreorder file that was written by the incomplete responseaaav run instead of the original dmrg run.
    reorderf1 = "%s/node0/RestartReorder.dat_1"%(mc.fcisolver.scratchDirectory)
    reorderf = "%s/node0/RestartReorder.dat"%(mc.fcisolver.scratchDirectory)
    import os.path
    reorder1present = os.path.isfile(reorderf1)
    if (reorder1present):
        from subprocess import check_call
        output = check_call("cp %s %s"%(reorderf1, reorderf), shell=True)
    else :
        from subprocess import check_call
        check_call("cp %s %s"%(reorderf, reorderf1), shell=True)
    reorder = numpy.loadtxt("%s/node0/RestartReorder.dat"%(mc.fcisolver.scratchDirectory))


    if (pttype == "NEVPT2") :
        naux=0
        if (df):
          norb, naux, energyE0 = writeNEVPTIntegralsDF(mc, dm1, dm2, dm1eff, frozen, fully_ic=fully_ic)
        else:
          norb, energyE0 = writeNEVPTIntegrals(mc, dm1, dm2, dm1eff, AAAVsplit, frozen, fully_ic=fully_ic, third_order=third_order)
        sys.stdout.flush()

        totalE = 0.0;
        totalE += executeNEVPT(nelec, mc.ncore, mc.ncas, frozen, mc.mol.spin,\
                               naux=naux, memory=mc.fcisolver.memory,\
                               fully_ic=fully_ic, third_order=third_order,\
                               cumulantE4=cumulantE4, df=df, no_handcoded_E3=no_handcoded_E3)

        if (not fully_ic):
          for k in range(AAAVsplit):
              writeAAAConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore, mc.ncas,  norb,
                               mc.fcisolver, PTM, "AAAV", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine, aaavsplit=AAAVsplit, aaavIter=k, root=root, name = "NEVPT2")
          writeAAAConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore-frozen, mc.ncas,  norb-frozen,
                           mc.fcisolver, PTM, "AAAC", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine,root=root, name = "NEVPT2")
          sys.stdout.flush()

          from subprocess import check_call
          try:
              for k in range(AAAVsplit):
                  outfile, infile = "responseNEVPT2_aaav%d.out"%(k), "responseNEVPT2_aaav%d.conf"%(k)
                  output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
                  file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
                  import struct
                  energy = struct.unpack('d', file1.read(8))[0]
                  file1.close()
                  totalE += energy
                  print("perturber AAAV%i --  %18.9e"%(k, energy))
                  sys.stdout.flush()

              if (mc.ncore-frozen != 0):
                  outfile, infile = "responseNEVPT2_aaac.out", "responseNEVPT2_aaac.conf"
                  output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
                  file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
                  energy = struct.unpack('d', file1.read(8))[0]
                  file1.close()
                  totalE += energy
                  print("perturber AAAC --  %18.9e"%(energy))
              print("")

          except ValueError:
              print(output)
          print("Total:             %18.9e"%(totalE))
          print("")

        print("Total PT       --  %18.9e"%(totalE))
        print("")
        return totalE

    else :
        #this is a bad way to do it, the problem is
        #that pyscf works with double precision and
        #energyE0 = writeMRLCCIntegrals(mc, dm1, dm2)
        #sys.stdout.flush()
        naux=0
        if (df):
          energyE0, norb, naux = writeMRLCCIntegralsDF(mc, dm1, dm2, frozen, fully_ic=fully_ic)
        else:
          energyE0, norb = writeMRLCCIntegrals(mc, dm1, dm2, frozen, fully_ic=fully_ic, third_order=third_order)
        sys.stdout.flush()

        totalE = 0.0
        totalE +=  executeMRLCC(nelec, mc.ncore, mc.ncas, frozen, mc.mol.spin,\
                                naux=naux, memory=mc.fcisolver.memory,\
                                fully_ic=fully_ic, third_order=third_order,\
                                cumulantE4=cumulantE4, df=df, no_handcoded_E3=no_handcoded_E3)

        if (not fully_ic):
          writeAAAConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore, mc.ncas,  norb,
                           mc.fcisolver, PTM, "AAAV", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine, root=root, name="MRLCC")
          writeAAAConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore-frozen, mc.ncas,  norb-frozen,
                           mc.fcisolver, PTM, "AAAC", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine, root=root, name="MRLCC")
          sys.stdout.flush()

          from subprocess import check_call
          try:
              outfile, infile = "responseMRLCC_aaav0.out", "responseMRLCC_aaav0.conf"
              output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
              file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
              import struct
              energy = struct.unpack('d', file1.read(8))[0]
              file1.close()
              totalE += energy
              print("perturber AAAV --  %18.9e"%(energy))
          except ValueError:
              print("perturber AAAV -- NA")

          try:
              if (mc.ncore-frozen != 0):
                  outfile, infile = "responseMRLCC_aaac.out", "responseMRLCC_aaac.conf"
                  output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
                  file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
                  energy = struct.unpack('d', file1.read(8))[0]
                  file1.close()
                  totalE += energy
                  print("perturber AAAC --  %18.9e"%(energy))
          except ValueError:
              print("perturber AAAC -- NA")
          print("Total:             %18.9e"%(totalE))
          print("")

        print("Total PT       --  %18.9e"%(totalE))
        print("")
        return totalE

if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None
    mol.atom = [
        ['N', ( 0., 0.    , 0.    )],
        ['N', ( 0., 0.    , 1.207 )],
    ]
    mol.basis = '6-31g'
    mol.spin = 0
    mol.build()

    m = scf.RHF(mol)
    m.conv_tol = 1e-20
    ehf = m.scf()
    mc = dmrgci.DMRGSCF(m, 6, 6)
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.mpiprefix=""
    mc.fcisolver.num_thrds=20
    ci_e = mc.kernel()[0]
    mc.verbose = 4
    print(ci_e)

    print(icmpspt(mc, pttype="MRLCC", rdmM=500, PTM=1000), -0.16978546152699392)

