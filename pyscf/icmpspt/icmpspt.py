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

float_precision = numpy.dtype('Float64')
mpiprefix=""
executable="/home/mussard/softwares/icpt/icpt"
#executable="/home/mussard/softwares/icpt_before_me/icpt"

if not os.path.isfile(executable):
    msg = ('MPSLCC executable %s not found.  Please specify "executable" in %s'
           % (executable, __file__))
    raise ImportError(msg)

#in state average calculationg E1eff will be different than E1
#this means that the h1eff in the fock operator which is stored in eris_sp['h1eff'] will be
#calculated using the E1eff and will in general not result in diagonal matrices
def writeNEVPTIntegrals(mc, E1, E2, E1eff, aaavsplit, nfro, fully_ic=False, third_order=False):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff
    intfolder=mc.fcisolver.scratchDirectory+'/int/'



    # (Note: Integrals are in chemistry notation)
    start = time.time()
    print('Producing the integrals')
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
    if (not isinstance(eris_sp['h1eff'], type(eris['cvcv']))):
      eriscvcv = lib.chkfile.load(eris['cvcv'].name, "eri_mo")#h5py.File(eris['cvcv'].name,'r')["eri_mo"]
    eris_sp['cvcv'] = eriscvcv.reshape(ncor, nvir, ncor, nvir)
    end = time.time()
    print('......production of INT took %10.2f sec' %(end-start))
    print('')

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

    print("Energy_nuc  = %13.8f"%(mc.mol.energy_nuc()))
    print("Energy_core = %13.8f"%(energy_core))
    print("Energy      = %13.8f"%(energyE0))
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
      print("         offdiagonal elements: {:13.6f}".format(offdiagonal))
      print("")

    # Write out ingredients to intfolder
    # 2 "C"
    start = time.time()
    print("Basic ingredients written to "+intfolder,nfro,ncor,nocc,norb)
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
    ##dom=['c','a','e']
    #inout={}
    #inout['c']=[nfro,ncor]
    #inout['a']=[ncor,nocc]
    #inout['e']=[nocc,norb]
    ##for p in range(3):
    ##  for q in range(3):
    ##    for r in range(3):
    ##      for s in range(3):
    ##        name="W:"+dom[p]+dom[q]+dom[r]+dom[s]
    ##        test=int2[inout[dom[p]][0]:inout[dom[p]][1],\
    ##                  inout[dom[r]][0]:inout[dom[r]][1],\
    ##                  inout[dom[q]][0]:inout[dom[q]][1],\
    ##                  inout[dom[s]][0]:inout[dom[s]][1]].transpose(0,2,1,3)
    ##        print("Output: {:} Shape: {:}".format(name,test.shape))
    ##        numpy.save(intfolder+name, numpy.asfortranarray(test))
    #list=['aaaa','aeae','aeca','aece','aeea',\
    #      'caaa','caac','caca','cace','ccaa','ccae','ccca','cccc','ccce','cece','ceec',\
    #      'eaaa','eaca','eeaa','eeca','eecc','eeea','eeec']
    #for i in range(len(list)):
    #  p=list[i][0]
    #  q=list[i][1]
    #  r=list[i][2]
    #  s=list[i][3]
    #  name="W:"+p+q+r+s
    #  test=int2[inout[p][0]:inout[p][1],\
    #            inout[r][0]:inout[r][1],\
    #            inout[q][0]:inout[q][1],\
    #            inout[s][0]:inout[s][1]].transpose(0,2,1,3)
    #  print("Output: {:} Shape: {:}".format(name,test.shape))
    #  numpy.save(intfolder+name, numpy.asfortranarray(test))

    end = time.time()
    print('......savings of INGREDIENTS took %10.2f sec' %(end-start))
    print("")

    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (not fully_ic):
      writeFCIDUMPs_NEVPT(mc,eris,eris_sp,aaavsplit,energy_core,energyE0,nfro)

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



    # (Note: Integrals are in chemistry notation)
    start = time.time()
    print('Producing the integrals')
    eris={}
    eris_sp={}
    eris['pcpc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,nfro:ncor], mo, mo[:,nfro:ncor]), compact=False)
    eris['ppcc'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nfro:ncor], mo[:,nfro:ncor]), compact=False)
    eris['ppee'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,nocc:], mo[:,nocc:]), compact=False)
    eris['pcpc'].shape=(norb, ncor-nfro, norb, ncor-nfro)
    eris['ppcc'].shape=(norb, norb, ncor-nfro, ncor-nfro)
    eris['ppee'].shape=(norb, norb, nvir, nvir)
    eris['papa'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,ncor:nocc], mo, mo[:,ncor:nocc]), compact=False)
    eris['ppaa'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,ncor:nocc], mo[:,ncor:nocc]), compact=False)
    eris['pepe'] = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,nocc:], mo, mo[:,nocc:]), compact=False)
    eris['papa'].shape=(norb, nact, norb, nact)
    eris['ppaa'].shape=(norb, norb, nact, nact)
    eris['pepe'].shape=(norb, nvir, norb, nvir)
    end = time.time()
    print('......production of INT took %10.2f sec' %(end-start))
    print('')

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

    print("Energy_nuc  = %13.8f"%(mc.mol.energy_nuc()))
    print("Energy_core = %13.8f"%(energy_core))
    print("Energy      = %13.8f"%(energyE0))
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
      print("         offdiagonal elements: %13.8f"%(offdiagonal))
      print("")

    # Write out ingredients to intfolder
    # 2 "C"
    start = time.time()
    print("Basic ingredients written to "+intfolder)
    numpy.save(intfolder+"W:ccae", numpy.asfortranarray(eris['pcpc'][ncor:nocc,     :    , nocc:    ,     :    ].transpose(1,3,0,2)))
    numpy.save(intfolder+"W:eecc", numpy.asfortranarray(eris['pcpc'][nocc:    ,     :    , nocc:    ,     :    ].transpose(0,2,1,3)))
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
    numpy.save(intfolder+"W:eeca", numpy.asfortranarray(eris['pepe'][nfro:ncor,     :    , ncor:nocc,     :    ].transpose(1,3,0,2)))
    if (third_order):
      numpy.save(intfolder+"W:aece", numpy.asfortranarray(eris['ppee'][ncor:nocc, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
      numpy.save(intfolder+"W:eeec", numpy.asfortranarray(eris['ppee'][nocc:    , nfro:ncor,     :    ,     :    ].transpose(2,0,3,1)))
      numpy.save(intfolder+"W:eeea", numpy.asfortranarray(eris['ppee'][nocc:    , ncor:nocc,     :    ,     :    ].transpose(2,0,3,1)))

    feri = h5py.File(intfolder+"int2eeee.hdf5", 'w')
    ao2mo.full(mc.mol, mo[:,nocc:], feri, compact=False)
    for o in range(nvir):
      int2eee = feri['eri_mo'][o*(norb-nocc):(o+1)*(norb-nocc),:]
      numpy.asfortranarray(int2eee).tofile(intfolder+"W:eeee%04d"%(o))

    # OUTPUT EVERYTHING (for debug of PT3)
    #int2=ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo, mo), compact=False)
    #int2.shape=(norb,norb,norb,norb)
    #inout={}
    #inout['c']=[nfro,ncor]
    #inout['a']=[ncor,nocc]
    #inout['e']=[nocc,norb]
    ##dom=['c','a','e']
    ##for p in range(3):
    ##  for q in range(3):
    ##    for r in range(3):
    ##      for s in range(3):
    ##        name="W:"+dom[p]+dom[q]+dom[r]+dom[s]
    ##        test=int2[inout[dom[p]][0]:inout[dom[p]][1],\
    ##                  inout[dom[r]][0]:inout[dom[r]][1],\
    ##                  inout[dom[q]][0]:inout[dom[q]][1],\
    ##                  inout[dom[s]][0]:inout[dom[s]][1]].transpose(0,2,1,3)
    ##        print("Output: {:} Shape: {:}".format(name,test.shape))
    ##        numpy.save(intfolder+name, numpy.asfortranarray(test))
    #list=['aaaa','aeae','aeca','aece','aeea',\
    #      'caaa','caac','caca','cace','ccaa',\
    #      'ccae','ccca','cccc','ccce','cece',\
    #      'ceec','eaaa','eaca','eeaa','eeca',\
    #      'eecc','eeea','eeec','eeee']
    #for i in range(len(list)):
    #  p=list[i][0]
    #  q=list[i][1]
    #  r=list[i][2]
    #  s=list[i][3]
    #  name="W:"+p+q+r+s
    #  test=int2[inout[p][0]:inout[p][1],\
    #            inout[r][0]:inout[r][1],\
    #            inout[q][0]:inout[q][1],\
    #            inout[s][0]:inout[s][1]].transpose(0,2,1,3)
    #  print("Output: {:} Shape: {:}".format(name,test.shape))
    #  numpy.save(intfolder+name, numpy.asfortranarray(test))

    end = time.time()
    print('......savings of INGREDIENTS took %10.2f sec' %(end-start))
    print("")

    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (not fully_ic):
      writeFCIDUMPs_MRLCC(mc,eris,eris_sp,int1,energy_core,energyE0,nfro)

    return energyE0, norb


def writeFCIDUMPs_NEVPT(mc,eris,eris_sp,aaavsplit,energy_core,energyE0,nfro):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    nocc = ncor+nact
    mo   = mc.mo_coeff

    # About symmetry...
    from pyscf import symm
    mol = mc.mol
    orbsymout=[]
    orbsym = []
    if (mol.symmetry):
        orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                     mol.symm_orb, mo, s=mc._scf.get_ovlp())
    if mol.symmetry and orbsym.any():
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

        tools.fcidump.write_head(fout, nact+len(virtRange[K]), mol.nelectron-2*ncor,\
                orbsym= (orbsymout[ncor:nocc]+orbsymout[virtRange[K][0]:virtRange[K][-1]+1]) )
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
        #tools.fcidump.write_hcore(fout,\
        #        eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, virtRange[K][0]:virtRange[K][-1]+1], len(virtRange[K]), tol=1e-8)
        fout.write(' %17.9e  0  0  0  0\n' %( mol.energy_nuc()+energy_core-energyE0))
        fout.close()
        print("Wrote FCIDUMP_aaav%d file"%(K))

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


def writeFCIDUMPs_MRLCC(mc,eris,eris_sp,int1,energy_core,energyE0,nfro):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    nocc = ncor+nact
    mo   = mc.mo_coeff

    # About symmetry...
    from pyscf import symm
    mol = mc.mol
    orbsymout=[]
    orbsym = []
    if (mol.symmetry):
        orbsym = symm.label_orb_symm(mol, mol.irrep_id,
                                     mol.symm_orb, mo, s=mc._scf.get_ovlp())
    if mol.symmetry and orbsym.any():
        if mol.groupname.lower() == 'dooh':
            orbsymout = [dmrg_sym.IRREP_MAP['D2h'][i % 10] for i in orbsym]
        elif mol.groupname.lower() == 'coov':
            orbsymout = [dmrg_sym.IRREP_MAP['C2v'][i % 10] for i in orbsym]
        else:
            orbsymout = [dmrg_sym.IRREP_MAP[mol.groupname][i] for i in orbsym]
    else:
        orbsymout = []


    fout = open('FCIDUMP_aaav0','w')
    tools.fcidump.write_head(fout, int1.shape[0]-ncor, mol.nelectron-2*ncor, orbsym= orbsymout[ncor:])
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
    fout.write(' %17.9e  0  0  0  0\n' %( mol.energy_nuc()+energy_core-energyE0))
    fout.close()
    print("Wrote FCIDUMP_aaav0 file")

    eri1cas = ao2mo.outcore.general_iofree(mol, (mo[:,nfro:nocc], mo[:,nfro:nocc], mo[:,nfro:nocc], mo[:,nfro:nocc]), compact=True)
    tools.fcidump.from_integrals("FCIDUMP_aaac", int1[nfro:nocc,nfro:nocc], eri1cas,\
            nocc-nfro, mol.nelectron-2*nfro, nuc=mol.energy_nuc()-energyE0, orbsym = orbsymout[nfro:nocc], tol=1e-8)
    print("Wrote FCIDUMP_aaac  file")
    print("")


def writeNEVPTIntegralsLEGACY(mc, E1, E2, E1eff, aaavsplit, nfro):
    print("LEGACY code from INTEGRALS ::begin::")
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff

    eris = _ERIS(mc, mo)
    eris_sp={}
    eris_sp['h1eff']= 1.*eris['h1eff'] #numpy.zeros(shape=(norb, norb))
    eris_sp['h1eff'][:ncor,:ncor] += numpy.einsum('abcd,cd', eris['ppaa'][:ncor, :ncor,:    ,:], E1eff)
    eris_sp['h1eff'][:ncor,:ncor] -= numpy.einsum('abcd,bd', eris['papa'][:ncor, :    ,:ncor,:], E1eff)*0.5
    eris_sp['h1eff'][nocc:,nocc:] += numpy.einsum('abcd,cd', eris['ppaa'][nocc:, nocc:,:    ,:], E1eff)
    eris_sp['h1eff'][nocc:,nocc:] -= numpy.einsum('abcd,bd', eris['papa'][nocc:, :    ,nocc:,:], E1eff)*0.5

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
      print("         offdiagonal elements: {:13.6f}".format(offdiagonal))
      print("")

    eriscvcv = eris['cvcv']
    if (not isinstance(eris_sp['h1eff'], type(eris['cvcv']))):
      eriscvcv = lib.chkfile.load(eris['cvcv'].name, "eri_mo")
    eris_sp['cvcv'] = eriscvcv.reshape(ncor, nvir, ncor, nvir)


    import os
    os.system("mkdir int")
    numpy.save("int/W:caac", numpy.asfortranarray(eris['papa'][nfro:ncor, :, nfro:ncor, :].transpose(0,3,1,2)))
    numpy.save("int/W:aeca", numpy.asfortranarray(eris['papa'][nfro:ncor, :, nocc:, :].transpose(1,2,0,3)))
    numpy.save("int/W:ccaa", numpy.asfortranarray(eris['papa'][nfro:ncor, :, nfro:ncor, :].transpose(0,2,1,3)))
    numpy.save("int/W:eeaa", numpy.asfortranarray(eris['papa'][nocc:, :, nocc:, :].transpose(0,2,1,3)))
    numpy.save("int/W:caca", numpy.asfortranarray(eris['ppaa'][nfro:ncor, nfro:ncor, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eaca", numpy.asfortranarray(eris['ppaa'][nocc:, nfro:ncor, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eecc", numpy.asfortranarray(eris_sp['cvcv'][nfro:,:,nfro:,:].transpose(1,3,0,2)))
    numpy.save("int/W:ccae", numpy.asfortranarray(eris['pacv'][nfro:ncor,:,nfro:,:].transpose(0,2,1,3)))
    numpy.save("int/W:aaaa", numpy.asfortranarray(eris['ppaa'][ncor:nocc, ncor:nocc, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eeca", numpy.asfortranarray(eris['pacv'][nocc:, :, nfro:, :].transpose(3,0,2,1)))
    numpy.save("int/int1eff", numpy.asfortranarray(eris_sp['h1eff'][nfro:,nfro:]))
    numpy.save("int/E1.npy", numpy.asfortranarray(E1))
    numpy.save("int/E2.npy", numpy.asfortranarray(E2))

    energyE0 = 1.0*numpy.einsum('ij,ij', eris_sp['h1eff'][ncor:nocc, ncor:nocc], E1)
    energyE0 += 0.5*numpy.einsum('ijkl,ijkl', eris['ppaa'][ncor:nocc, ncor:nocc, :, :].transpose(0,2,1,3), E2)
    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    energyE0 += numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
                  + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5
    energyE0 += mc.mol.energy_nuc()
    print("Energy = ", energyE0)

    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    energy_core = numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
                  + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5
    print(energy_core)

    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (False):
      writeFCIDUMPs_NEVPT(mc,eris,eris_sp,aaavsplit,energy_core,energyE0,nfro)
    print("LEGACY code from INTEGRALS ::end::")

    return norb, energyE0


def writeMRLCCIntegralsLEGACY(mc, E1, E2, nfro):
    print("LEGACY code from INTEGRALS ::begin::")
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff

    #this is chemistry notation
    int2popo = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,:nocc], mo, mo[:,:nocc]), compact=False)
    int2ppoo = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,:nocc], mo[:,:nocc]), compact=False)
    int2popo.shape=(norb, nocc, norb, nocc)
    int2ppoo.shape=(norb, norb, nocc, nocc)

    dmcore = numpy.dot(mo[:,:nfro], mo[:,:nfro].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1 = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) +vhfcore

    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1_eff = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) +vhfcore

    numpy.save("int/E2",numpy.asfortranarray(E2))
    numpy.save("int/E1",numpy.asfortranarray(E1))
    numpy.save("int/int1",numpy.asfortranarray(int1[nfro:,nfro:]))
    numpy.save("int/int1eff",numpy.asfortranarray(int1_eff[nfro:, nfro:]))
    numpy.save("int/W:caca", numpy.asfortranarray(int2ppoo[nfro:ncor, nfro:ncor, ncor:nocc, ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:caac", numpy.asfortranarray(int2popo[nfro:ncor,ncor:nocc, ncor:nocc, nfro:ncor].transpose(0,2,1,3)))
    numpy.save("int/W:cece", numpy.asfortranarray(int2ppoo[nocc:, nocc:, nfro:ncor, nfro:ncor].transpose(2,0,3,1)))
    numpy.save("int/W:ceec", numpy.asfortranarray(int2popo[nocc:, nfro:ncor, nocc:, nfro:ncor].transpose(1,2,0,3)))
    numpy.save("int/W:aeae", numpy.asfortranarray(int2ppoo[nocc:, nocc:, ncor:nocc,ncor:nocc].transpose(2,0,3,1)))
    numpy.save("int/W:aeea", numpy.asfortranarray(int2popo[nocc:, ncor:nocc,nocc:, ncor:nocc].transpose(1,2,0,3)))
    numpy.save("int/W:cccc", numpy.asfortranarray(int2ppoo[nfro:ncor,nfro:ncor, nfro:ncor, nfro:ncor].transpose(0,2,1,3)))
    numpy.save("int/W:aaaa", numpy.asfortranarray(int2ppoo[ncor:nocc,ncor:nocc, ncor:nocc, ncor:nocc].transpose(0,2,1,3)))
    feri = h5py.File("int/int2eeee.hdf5", 'w')
    ao2mo.full(mc.mol, mo[:,nocc:], feri, compact=False)
    for o in range(nvir):
        int2eee = feri['eri_mo'][o*(norb-nocc):(o+1)*(norb-nocc),:]
        numpy.asfortranarray(int2eee).tofile("int/W:eeee%04d"%(o))
    numpy.save("int/W:eecc", numpy.asfortranarray(int2popo[nocc:,nfro:ncor,nocc:,nfro:ncor].transpose(0,2,1,3)))
    numpy.save("int/W:eeca", numpy.asfortranarray(int2popo[nocc:,nfro:ncor, nocc:, ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:ccaa", numpy.asfortranarray(int2popo[nfro:ncor,ncor:nocc, nfro:ncor, ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:eeaa", numpy.asfortranarray(int2popo[nocc:,ncor:nocc, nocc:, ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:eaca", numpy.asfortranarray(int2popo[nocc:,nfro:ncor, ncor:nocc, ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:aeca", numpy.asfortranarray(int2popo[ncor:nocc,nfro:ncor, nocc:,ncor:nocc].transpose(0,2,1,3)))
    numpy.save("int/W:ccae", numpy.asfortranarray(int2popo[nfro:ncor,ncor:nocc, nocc:, nfro:ncor].transpose(0,3,1,2)))

    dmcore = numpy.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    h1eff = int1_eff
    energy_core = numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
        + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    #print energy_core2+mc.mol.energy_nuc(), energy_core+mc.mol.energy_nuc(), energy_nfro_core+mc.mol.energy_nuc()
    energyE0 = 1.0*numpy.einsum('ij,ij', h1eff[ncor:nocc, ncor:nocc], E1)
        #+ 0.5*numpy.einsum('ikjl,ijkl', E2, int2ppoo[ncor:,ncor:,ncor:,ncor:])
    for i in range(mc.ncas):
        for j in range(mc.ncas):
            for k in range(mc.ncas):
                for l in range(mc.ncas):
                    energyE0 += 0.5*E2[i,k,j,l] * int2ppoo[i+ncor, j+ncor, k+ncor, l+ncor]

    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()
    print("Energy = ", energyE0)

    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (False):
      writeFCIDUMPs_MRLCC(mc,eris,eris_sp,int1,energy_core,energyE0,nfro)
    print("LEGACY code from INTEGRALS ::end::")

    return energyE0, norb


def execute(nelec, ncor, ncas, nfro, ms2, type,\
                naux=0, memory=10,\
                fully_ic=False, third_order=False, cumulantE4=False, df=False, no_handcoded_E3=False):
    methods = ['_CCVV', '_CCAV', '_ACVV', '_CCAA', '_AAVV', '_CAAV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']
    if (fully_ic):
        methods+=['_AAAV', '_AAAC']
        domains+=['eaaa','caaa']
    if (ncor - nfro) == 0:
        methods = ['_AAVV']
        domains = ['eeaa']
    if (third_order):
        methods+=['3']

    # For each method
    totalE = 0.0
    print("--ICPT executable:%s\n"%(executable))
    print("Second-order:")
    for method in methods:
        # Prepare Input
        f = open("%s.inp"%(type+method), 'w')
        if (memory is not None):
            f.write('work-space-mb %d\n'%(memory*1000))
        f.write('method %s\n'%(type+method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncor-nfro)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
        f.write('ms2 %d\n'%(ms2))
        f.write('int1e/fock int/int1eff.npy\n')
        if (type=='MRLCC'):
          f.write('int1e/coreh int/int1.npy\n')
        #f.write('E3  int/E3.npy\n')
        #f.write('E2  int/E2.npy\n')
        #f.write('E1  int/E1.npy\n')
        f.write('thr-den 1.000000e-05\n')
        f.write('thr-var 1.000000e-05\n')
        f.write('thr-trunc 1.000000e-04\n')
        if (third_order and method!='3'):
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

        # Run icpt
        from subprocess import check_call,CalledProcessError
        infile="%s.inp"%(type+method)
        outfile="%s.out"%(type+method)
        try:
          check_call("%s  %s  %s > %s"%(mpiprefix, executable, infile, outfile), shell=True)
          energy=ReadWriteEnergy(outfile,type+method)
        except CalledProcessError:
          energy='NA'
          if(method!='3'):
            print("perturber %s --  NA"%(method[-4:]))
        if (method!='3'):
          try:
            totalE+=energy
          except:
            totalE='nan'
        sys.stdout.flush()

    # Print out
    if (fully_ic):
      try:
        print("Total:             %18.9f"%(totalE))
        print("")
      except:
        print("Total:             NA")
        print("")
    if (third_order):
      try:
        print("Third-order:       %18.9f"%(energy))
        print("")
        totalE+=energy
      except:
        print("Third-order:       NA")
        print("")
    if isinstance(totalE,float):
      return totalE
    else:
      return 0.0


def executeUC(mc, reorder,fciExtraLine,root,norb,\
                  nfro=0,\
                  totalE=0,\
                  type="MRLCC",\
                  AAAVsplit=0,\
                  PTM=1000):
    for k in range(AAAVsplit):
      writeAAAConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore,        mc.ncas,  norb,\
                       mc.fcisolver, PTM, "AAAV", mc.fcisolver.memory,\
                       mc.fcisolver.num_thrds, reorder, fciExtraLine,\
                       root=root, name = type,\
                       aaavsplit=AAAVsplit, aaavIter=k)
    writeAAAConfFile(  mc.nelecas[0], mc.nelecas[1], mc.ncore-nfro, mc.ncas,  norb-nfro,\
                       mc.fcisolver, PTM, "AAAC", mc.fcisolver.memory,\
                       mc.fcisolver.num_thrds, reorder, fciExtraLine,\
                       root=root, name=type)
    sys.stdout.flush()

    from subprocess import check_call
    import struct
    try:
      for k in range(AAAVsplit):
        outfile, infile = "response"+type+"_aaav%d.out"%(k), "response"+type+"_aaav%d.conf"%(k)
        check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
        file1  = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
        energy = struct.unpack('d', file1.read(8))[0]
        file1.close()
        totalE += energy
        print("perturber AAAV%i --  %18.9f"%(k,energy))
    except ValueError:
        print("perturber AAAV -- NA")

    try:
      if (mc.ncore-nfro != 0):
        outfile, infile = "response"+type+"_aaac.out", "response"+type+"_aaac.conf"
        check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
        file1  = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
        energy = struct.unpack('d', file1.read(8))[0]
        file1.close()
        totalE += energy
        print("perturber AAAC --  %18.9f"%(energy))
    except ValueError:
        print("perturber AAAC -- NA")
    print("")
    print("Total:             %18.9f"%(totalE))
    print("")
    return totalE


def ReadWriteEnergy(outfile,method):
  import re
  energy='not_found'
  with open(outfile,'r') as origin:
    for line in origin:
      if re.search(method+'.*ENERGY',line):
        energy=line.split()[-1]
        break
  try:
    energy=float(energy)
    if((method!='MRLCC3')and(method!='NEVPT3')):
      print("perturber %s --  %18.9f"%(method[-4:], energy))
  except ValueError:
    if((method!='MRLCC3')and(method!='NEVPT3')):
      print("perturber %s --  NA"%(method[-4:]))
  return energy


def writeAAAConfFile(neleca, nelecb, ncor, ncas, norb,\
        DMRGCI, maxM, perturber, memory, numthrds,\
        reorder, extraline, root=0, approx= False, aaavsplit=1, aaavIter=0, name= "MRLCC"):
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


def mrlcc(mc, pttype="MRLCC", energyE0=0.0, rdmM=0, nfro=0, PTM=1000, PTincore=False, fciExtraLine=[],\
            have3RDM=False, root=0, nroots=1, verbose=None, AAAVsplit=0,\
            do_dm3=False, do_dm4=True, fully_ic=True, third_order=True, cumulantE4=False, no_handcoded_E3=False, filetype="binary"):
  return icmpspt(mc, pttype, energyE0, rdmM, nfro, PTM, PTincore, fciExtraLine,\
              have3RDM, root, nroots, verbose, AAAVsplit,\
              do_dm3, do_dm4, fully_ic, third_order, cumulantE4, no_handcoded_E3, filetype)

def nevpt(mc, pttype="NEVPT", energyE0=0.0, rdmM=0, nfro=0, PTM=1000, PTincore=False, fciExtraLine=[],\
            have3RDM=False, root=0, nroots=1, verbose=None, AAAVsplit=1,\
            do_dm3=False, do_dm4=True, fully_ic=True, third_order=True, cumulantE4=False, no_handcoded_E3=False, filetype="binary"):
  return icmpspt(mc, pttype, energyE0, rdmM, nfro, PTM, PTincore, fciExtraLine,\
              have3RDM, root, nroots, verbose, AAAVsplit,\
              do_dm3, do_dm4, fully_ic, third_order, cumulantE4, no_handcoded_E3, filetype)


def icmpspt(mc, pttype="NEVPT", energyE0=0.0, rdmM=0, nfro=0, PTM=1000, PTincore=False, fciExtraLine=[],\
            have3RDM=False, root=0, nroots=1, verbose=None, AAAVsplit=1,\
            do_dm3=True, do_dm4=False, fully_ic=False, third_order=False, cumulantE4=False, no_handcoded_E3=False, filetype="binary", legacy=False):
    sys.stdout.flush()
    print("")
    print("")
    print("--------------------------------------------------")
    print("                   MRPT CALCULATION               ")
    print("--------------------------------------------------")
    print("")

    # =========================================
    # SANITY CHECKS OF KEYWORDS
    # =========================================
    if (pttype == "NEVPT2"):
      print("pttype = NEVPT2 is deprecated, use pttype = NEVPT")
      print("")
      pttype = "NEVPT"
    if (pttype != "NEVPT" and pttype != "MRLCC"):
      print("pttype = ", pttype, " is not supported, use NEVPT or MRLCC")
      exit()
    if (do_dm4):
      do_dm3=False
    elif (do_dm3):
      do_dm4=False
    else:
      print("WARNING:  Neither do_dm3 nor do_dm4! Turning do_dm3 on.")
      print("")
      do_dm3=True
    if ((third_order)and(not fully_ic)):
      print("WARNING: Third-order needs Fully IC mode! Turning fully_ic on.")
      print("")
      fully_ic=True
    if (pttype != "NEVPT" and AAAVsplit != 1):
      print("AAAVsplit only works with CASSCF natural orbitals and NEVPT")
      print("")
      exit(0)
    if (getattr(mc, 'with_df', None)):
      df=True
    else:
      df=False
    #if (fully_ic and not (do_dm4 or cumulantE4)):
    #  print("WARNING: Fully IC needs do_dm4 or cumulantE4!")
    #  print("")
    #  do_dm4=True
    #  do_dm3=False
    #if type(mc.fcisolver) is not dmrgci.DMRGCI:
    #  if (mc.fcisolver.fcibase_class is not dmrgci.DMRGCI):
    #    print("this works with dmrgscf and not regular mcscf")
    #    print("")
    #    exit(0)

    # =========================================
    # MESSAGES
    # =========================================
    print("Perturbation type: %s"%(pttype))
    if (fully_ic):
      print("With fully internally contracted scheme")
    if (third_order):
      print("With third order correction")
    if (df):
      print("Recognized a Density Fitting run")
      print("NOT READY YET, NOT TESTED!!")
      exit()
    print("")

    # =========================================
    # INITIATIONS
    # =========================================
    # Remove the -1 state
    import os
    os.system("rm -f %s/node0/Rotation*.state-1.tmp"%(mc.fcisolver.scratchDirectory))
    os.system("rm -f %s/node0/wave*.-1.tmp"         %(mc.fcisolver.scratchDirectory))
    # os.system("rm -f %s/node0/RestartReorder.dat_1" %(mc.fcisolver.scratchDirectory))

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
    os.system("mkdir -p "+intfolder)
    if intfolder!='int/':
      os.system("mv int int_old_$$")
      os.system("ln -s "+intfolder+" int")

    # =========================================
    # RDMS
    # =========================================
    print("Preparing necessary RDMs")
    sys.stdout.flush()
    nelec = mc.nelecas[0]+mc.nelecas[1]

    # State-average density (needed in NEVPT)
    dm1eff = numpy.zeros(shape=(mc.ncas, mc.ncas))
    # Loop over all states besides the current root
    if (pttype == "NEVPT" and nroots>1):
        stateIter = range(nroots)
        stateIter.remove(root)
        for istate in stateIter:
            dm3 = mc.fcisolver.make_rdm3(state=istate, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision)
            # This is coherent with statement about indexes made in "make_rdm3"
            # This is done with SQA in mind.
            dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
            dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
            dm1eff += dm1

    # RDM4 and RDM3
    if (do_dm4):
      dm4 = mc.fcisolver.make_rdm4(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype=filetype)
      # This is coherent with statement about indexes made in "make_rdm4"
      # This is done with SQA in mind
      if (nelec>3):
        dm3 = numpy.einsum('ijklmnol', dm4)/(nelec-3)
      else:
        dm3 = mc.fcisolver.make_rdm3(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype=filetype)
      numpy.save(intfolder+"E4",dm4)
    elif (do_dm3):
      dm3 = mc.fcisolver.make_rdm3(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype=filetype)

    # RDM2 and RDM1
    # This is coherent with statement about indexes made in "make_rdm4" and "make_rdm3"
    # This is done with SQA in mind
    if (nelec>2):
      dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
    else:
      print("UNNECESSARY FOR NOW, right..?")
      exit(0)
    if (nelec>1):
      dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
    else:
      print("UNNECESSARY FOR NOW, right..?")
      exit(0)
    dm1eff += dm1
    dm1eff = dm1eff/(1.0*nroots)

    # NUMPY
    if (do_dm4):
      numpy.save(intfolder+"E4",dm4)
    numpy.save(intfolder+"E3",dm3)
    numpy.save(intfolder+"E3B.npy", dm3.transpose(0,3,1,4,2,5))
    numpy.save(intfolder+"E3C.npy", dm3.transpose(5,0,2,4,1,3))
    numpy.save(intfolder+"E2.npy", numpy.asfortranarray(dm2))
    numpy.save(intfolder+"E1.npy", numpy.asfortranarray(dm1))

    # TRACES
    if (do_dm4):
      trace=numpy.einsum('ijklijkl->',dm4)
      if abs(trace-nelec*(nelec-1)*(nelec-2)*(nelec-3))<0.000001:
          print('(GOOD) Trace 4RDM: {:8} ={:5}*{:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2,nelec-3))
      else:
          print('(BAD)  Trace 4RDM: {:8}!={:5}*{:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2,nelec-3))
      del dm4
    trace=numpy.einsum('ijkijk->',dm3)
    if abs(trace-nelec*(nelec-1)*(nelec-2))<0.000001:
        print('(GOOD) Trace 3RDM: {:8} ={:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2))
    else:
        print('(BAD)  Trace 3RDM: {:8}!={:5}*{:5}*{:5}'.format(trace,nelec,nelec-1,nelec-2))
    trace=numpy.einsum('ijij->',dm2)
    if abs(trace-nelec*(nelec-1))<0.000001:
        print('(GOOD) Trace 2RDM: {:8} ={:5}*{:5}'.format(trace,nelec,nelec-1))
    else:
        print('(BAD)  Trace 2RDM: {:8}!={:5}*{:5}'.format(trace,nelec,nelec-1))
    trace=numpy.einsum('ii->',dm1)
    if abs(trace-nelec)<0.000001:
        print('(GOOD) Trace 1RDM: {:8} ={:5}'.format(trace,nelec))
    else:
        print('(BAD)  Trace 1RDM: {:8}!={:5}'.format(trace,nelec))
    del dm3
    print("")
    sys.stdout.flush()

    # =========================================
    # BACKUP RestartReorder.dat FILE TO _1
    # =========================================
    #backup the restartreorder file to _1. this is because responseaaav and responseaaac both overwrite this file
    #this means that when we want to restart a calculation after lets say responseaaav didnt finish, the new calculaitons
    #will use the restartreorder file that was written by the incomplete responseaaav run instead of the original dmrg run.
    reorder=[]
    reorder_bak_f = "%s/node0/RestartReorder.dat_1"%(mc.fcisolver.scratchDirectory)
    reorderf = "%s/node0/RestartReorder.dat"%(mc.fcisolver.scratchDirectory)
    import os.path
    reorder_bak_present = os.path.isfile(reorder_bak_f)
    if (reorder_bak_present):
        from subprocess import check_call
        if os.path.isfile(reorder_bak_f):
          check_call("cp -p %s %s"%(reorder_bak_f, reorderf), shell=True)
    else :
        from subprocess import check_call
        if os.path.isfile(reorderf):
          check_call("cp -p %s %s"%(reorderf, reorder_bak_f), shell=True)
    if os.path.isfile(reorderf):
      reorder = numpy.loadtxt(reorderf)

    # =========================================
    # NEVPT
    # =========================================
    if (pttype == "NEVPT") :
        naux=0
        #if (df):
        #  norb, naux, energyE0 = writeNEVPTIntegralsDF(mc, dm1, dm2, dm1eff, nfro, fully_ic=fully_ic)
        #else:
        #  norb, energyE0 = writeNEVPTIntegrals(mc, dm1, dm2, dm1eff, AAAVsplit, nfro, fully_ic=fully_ic, third_order=third_order)
        if (legacy):
          norb, energyE0 = writeNEVPTIntegralsLEGACY(mc, dm1, dm2, dm1eff, AAAVsplit, nfro)
        else:
          norb, energyE0 = writeNEVPTIntegrals(mc, dm1, dm2, dm1eff, AAAVsplit, nfro, fully_ic=fully_ic, third_order=third_order)
        sys.stdout.flush()

        totalE = 0.0;
        totalE += execute(nelec, mc.ncore, mc.ncas, nfro, mc.mol.spin, 'NEVPT2',\
                          naux=naux, memory=mc.fcisolver.memory,\
                          fully_ic=fully_ic, third_order=third_order,\
                          cumulantE4=cumulantE4, df=df, no_handcoded_E3=no_handcoded_E3)
        if (not fully_ic):
          totalE +=  executeUC(mc,reorder,fciExtraLine,root,norb,\
                               type="NEVPT2",\
                               AAAVsplit=1)
        print("Total PT       --  %18.9f"%(totalE))
        print("Total energy   --  %18.9f"%(totalE+energyE0))
        print("")
        return totalE

    # =========================================
    # MRLCC
    # =========================================
    else :
        #this is a bad way to do it, the problem is
        #that pyscf works with double precision and
        #
        #energyE0 = writeMRLCCIntegrals(mc, dm1, dm2)
        #sys.stdout.flush()
        naux=0
        #if (df):
        #  energyE0, norb, naux = writeMRLCCIntegralsDF(mc, dm1, dm2, nfro, fully_ic=fully_ic)
        #else:
        #  energyE0, norb = writeMRLCCIntegrals(mc, dm1, dm2, nfro, fully_ic=fully_ic, third_order=third_order)
        if (legacy):
          energyE0, norb = writeMRLCCIntegralsLEGACY(mc, dm1, dm2, nfro)
        else:
          energyE0, norb = writeMRLCCIntegrals(mc, dm1, dm2, nfro, fully_ic=fully_ic, third_order=third_order)
        sys.stdout.flush()

        totalE = 0.0
        totalE +=  execute(nelec, mc.ncore, mc.ncas, nfro, mc.mol.spin,'MRLCC',\
                           naux=naux, memory=mc.fcisolver.memory,\
                           fully_ic=fully_ic, third_order=third_order,\
                           cumulantE4=cumulantE4, df=df, no_handcoded_E3=no_handcoded_E3)
        if (not fully_ic):
          totalE +=  executeUC(mc,reorder,fciExtraLine,root,norb,\
                               type="MRLCC",\
                               AAAVsplit=0)

        print("Total PT       --  %18.9f"%(totalE))
        print("Total energy   --  %18.9f"%(totalE+energyE0))
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

