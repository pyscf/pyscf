#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#
# Descritpion:
#       Internal-contracted MPS perturbation method.  You can contact Sandeep
#       Sharma for the "icpt" program required by this module.  If this method
#       is used in your work, please cite
#       S. Sharma and G. Chan,  J. Chem. Phys., 136 (2012), 124121
#       S. Sharma, G. Jeanmairet, and A. Alavi,  J. Chem. Phys., 144 (2016), 034103
#

import pyscf
import os
import ctypes
import time
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import fci
from pyscf import mcscf
from pyscf import ao2mo
from pyscf import scf
from pyscf.ao2mo import _ao2mo
from pyscf.dmrgscf import dmrgci
from pyscf.dmrgscf import dmrg_sym
from pyscf import tools
from pyscf import lib
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
#        print "could not read the norbs or nelec"
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


#def makeheff(int1, int2popo, int2ppoo, E1, ncore, nvirt, frozen):
#        nc = int1.shape[0]-nvirt
#
#        int1_eff = 1.*int1 + 2.0*numpy.einsum('mnii->mn', int2ppoo[:, :, frozen:ncore, frozen:ncore])-numpy.einsum('mini->mn', int2popo[:,frozen:ncore,:, frozen:ncore])
#
#        int1_eff[:ncore, :ncore] += numpy.einsum('lmjk,jk->lm',int2ppoo[:ncore,:ncore,ncore:nc,ncore:nc], E1) - 0.5*numpy.einsum('ljmk,jk->lm',int2popo[:ncore,ncore:nc,:ncore,ncore:nc], E1)
#        int1_eff[nc:, nc:] += numpy.einsum('lmjk,jk->lm',int2ppoo[nc:,nc:,ncore:nc,ncore:nc], E1) - 0.5*numpy.einsum('ljmk,jk->lm',int2popo[nc:,ncore:nc,nc:,ncore:nc], E1)
#        #int1_eff[nc:, nc:] += numpy.einsum('ljmk,jk->lm',int2[nc:,ncore:nc,nc:,ncore:nc], E1) - 0.5*numpy.einsum('ljkm,jk->lm',int2[nc:,ncore:nc,ncore:nc,nc:], E1)
#        return int1_eff
#'''
#def makeheff(int1, int2, E1, ncore, nvirt):
#        nc = int1.shape[0]-nvirt
#
#        int1_eff = 1.*int1 + 2.0*numpy.einsum('mini->mn', int2[:,:ncore, :, :ncore])-numpy.einsum('miin->mn', int2[:,:ncore,:ncore,:])
#
#        int1_eff[:ncore, :ncore] += numpy.einsum('ljmk,jk->lm',int2[:ncore,ncore:nc,:ncore,ncore:nc], E1) - 0.5*numpy.einsum('ljkm,jk->lm',int2[:ncore,ncore:nc,ncore:nc,:ncore], E1)
#        int1_eff[nc:, nc:] += numpy.einsum('ljmk,jk->lm',int2[nc:,ncore:nc,nc:,ncore:nc], E1) - 0.5*numpy.einsum('ljkm,jk->lm',int2[nc:,ncore:nc,ncore:nc,nc:], E1)
#        return int1_eff
#'''


def writeNumpyforMRLCC(mc, E1, E2, frozen, fully_ic=False) :
    # Initializations
    ncore = mc.ncore
    nact  = mc.ncas
    norbs = mc.mo_coeff.shape[1]
    nvirt = norbs-ncore-nact
    nc    = ncore+nact
    mo = mc.mo_coeff


    # int2popo
    # (Note: Integrals are in chemistry notation)
    int2popo = ao2mo.outcore.general_iofree(mc.mol, (mo, mo[:,:nc], mo, mo[:,:nc]), compact=False)
    int2ppoo = ao2mo.outcore.general_iofree(mc.mol, (mo, mo, mo[:,:nc], mo[:,:nc]), compact=False)
    int2popo.shape=(norbs, nc, norbs, nc)
    int2ppoo.shape=(norbs, norbs, nc, nc)

    # int1 and int1_eff
    dmcore   = numpy.dot(mo[:,:frozen], mo[:,:frozen].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1     = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) +vhfcore
    dmcore   = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)*2
    vj, vk   = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore  = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    int1_eff = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo)) +vhfcore
    #int1_eff = makeheff(int1, int2popo, int2ppoo, E1, ncore, nvirt, frozen)


    # Write out ingredients to "int/"
    import os
    os.system("mkdir -p int")

    numpy.save("int/W:caca", numpy.asfortranarray(int2ppoo[frozen:ncore, frozen:ncore, ncore:nc, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:caac", numpy.asfortranarray(int2popo[frozen:ncore,ncore:nc, ncore:nc, frozen:ncore].transpose(0,2,1,3)))
    numpy.save("int/W:cece", numpy.asfortranarray(int2ppoo[nc:, nc:, frozen:ncore, frozen:ncore].transpose(2,0,3,1)))
    numpy.save("int/W:ceec", numpy.asfortranarray(int2popo[nc:, frozen:ncore, nc:, frozen:ncore].transpose(1,2,0,3)))
    numpy.save("int/W:aeae", numpy.asfortranarray(int2ppoo[nc:, nc:, ncore:nc,ncore:nc].transpose(2,0,3,1)))
    numpy.save("int/W:aeea", numpy.asfortranarray(int2popo[nc:, ncore:nc,nc:, ncore:nc].transpose(1,2,0,3)))
    numpy.save("int/W:cccc", numpy.asfortranarray(int2ppoo[frozen:ncore,frozen:ncore, frozen:ncore, frozen:ncore].transpose(0,2,1,3)))
    numpy.save("int/W:aaaa", numpy.asfortranarray(int2ppoo[ncore:nc,ncore:nc, ncore:nc, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:eecc", numpy.asfortranarray(int2popo[nc:,frozen:ncore,nc:,frozen:ncore].transpose(0,2,1,3)))
    numpy.save("int/W:eeca", numpy.asfortranarray(int2popo[nc:,frozen:ncore, nc:, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:ccaa", numpy.asfortranarray(int2popo[frozen:ncore,ncore:nc, frozen:ncore, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:eeaa", numpy.asfortranarray(int2popo[nc:,ncore:nc, nc:, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:eaca", numpy.asfortranarray(int2popo[nc:,frozen:ncore, ncore:nc, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:aeca", numpy.asfortranarray(int2popo[ncore:nc,frozen:ncore, nc:,ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/W:ccae", numpy.asfortranarray(int2popo[frozen:ncore,ncore:nc, nc:, frozen:ncore].transpose(0,3,1,2)))
    numpy.save("int/int1",   numpy.asfortranarray(int1[frozen:,frozen:]))
    numpy.save("int/int1eff",numpy.asfortranarray(int1_eff[frozen:, frozen:]))
    if (fully_ic):
      numpy.save("int/W:eaaa", numpy.asfortranarray(int2popo[nc:,          ncore:nc, ncore:nc, ncore:nc].transpose(0,2,1,3)))
      numpy.save("int/W:caaa", numpy.asfortranarray(int2popo[frozen:ncore, ncore:nc, ncore:nc, ncore:nc].transpose(0,2,1,3)))
    numpy.save("int/E1",numpy.asfortranarray(E1))
    numpy.save("int/E2",numpy.asfortranarray(E2))
    #numpy.save("int/E3",E3)
    #numpy.save("int/E3B.npy", E3.transpose(0,3,1,4,2,5))
    #numpy.save("int/E3C.npy", E3.transpose(5,0,2,4,1,3))

    feri = h5py.File("int/int2eeee.hdf5", 'w')
    ao2mo.full(mc.mol, mo[:,nc:], feri, compact=False)
    for o in range(nvirt):
        int2eee = feri['eri_mo'][o*(norbs-nc):(o+1)*(norbs-nc),:]
        numpy.asfortranarray(int2eee).tofile("int/W:eeee%04d"%(o))

    print "Basic ingredients wrote to int/"
    print ""


    # energy_frozen_core
    energy_frozen_core = numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
                       + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energy_core
    dmcore  = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)*2
    vj, vk  = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    h1eff   = int1_eff
    energy_core = numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    h1eff = int1_eff
    energyE0 = 1.0*numpy.einsum('ij,ij', h1eff[ncore:nc, ncore:nc], E1)
            #+ 0.5*numpy.einsum('ikjl,ijkl', E2, int2ppoo[ncore:,ncore:,ncore:,ncore:])
    for i in range(mc.ncas):
        for j in range(mc.ncas):
            for k in range(mc.ncas):
                for l in range(mc.ncas):
                    I,J = max(i,j)+ncore, min(i,j)+ncore
                    K,L = max(k,l)+ncore, min(k,l)+ncore
                    energyE0 += 0.5*E2[i,k,j,l] * int2ppoo[i+ncore, j+ncore, k+ncore, l+ncore]
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print "Energy_core = ",energy_core
    print "Energy      = ", energyE0
    print ""


    # Write "FCIDUMP_aaav0" and "FCIDUMP_aaac"
    if (not fully_ic):
      # About symmetry...
      from pyscf import symm
      mol = mc.mol
      orbsymout=[]
      orbsym = []
      if (mol.symmetry):
          orbsym = symm.label_orb_symm(mc.mol, mc.mol.irrep_id,
                                       mc.mol.symm_orb, mo, s=mc._scf.get_ovlp())
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
      tools.fcidump.write_head(fout, int1.shape[0]-ncore, mc.mol.nelectron-2*ncore, orbsym= orbsymout[ncore:])
      for i in range(ncore,int1.shape[0]):
          for j in range(ncore, i+1):
              for k in range(mc.ncas):
                  for l in range(k+1):
                      if abs(int2ppoo[i,j, k+ncore,l+ncore]) > 1.e-8 :
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                     % (int2ppoo[i,j, k+ncore,l+ncore], i+1-ncore, j+1-ncore, k+1, l+1))
                      if (j >= nc and abs(int2popo[i, k+ncore, j, l+ncore]) > 1.e-8):
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                         % (int2popo[i,k+ncore,j, l+ncore], i+1-ncore, k+1, l+1, j+1-ncore))
                      if (j >= nc and abs(int2popo[i, l+ncore, j, k+ncore]) > 1.e-8):
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                         % (int2popo[i,l+ncore, j, k+ncore], i+1-ncore, l+1, k+1, j+1-ncore))

      tools.fcidump.write_hcore(fout, h1eff[ncore:,ncore:], int1.shape[0]-ncore, tol=1e-8)
      fout.write(' %17.9e  0  0  0  0\n' %( mc.mol.energy_nuc()+energy_core-energyE0))
      fout.close()
      print "Wrote FCIDUMP_aaav0 file"

      eri1cas = ao2mo.outcore.general_iofree(mc.mol, (mo[:,frozen:nc], mo[:,frozen:nc], mo[:,frozen:nc], mo[:,frozen:nc]), compact=True)
      tools.fcidump.from_integrals("FCIDUMP_aaac", int1[frozen:nc,frozen:nc], eri1cas, nc-frozen, mc.mol.nelectron-2*frozen, nuc=mc.mol.energy_nuc()-energyE0, orbsym = orbsymout[frozen:nc], tol=1e-8)
      print "Wrote FCIDUMP_aaac  file"
      print ""

    return energyE0, norbs


#in state average calculationg dm1eff will be different than dm1
#this means that the h1eff in the fock operator which is stored in eris_sp['h1eff'] will be
#calculated using the dm1eff and will in general not result in diagonal matrices
def writeNevpt2Integrals(mc, dm1, dm2, dm1eff, aaavsplit, frozen, fully_ic=False):
    # Initializations
    ncore = mc.ncore
    nact  = mc.ncas
    norbs = mc.mo_coeff.shape[1]
    nvirt = norbs-ncore-nact
    nc    = ncore+nact
    mo = mc.mo_coeff


    # eris_sp
    # (Note: Integrals are in chemistry notation)
    eris = _ERIS(mc, mo)
    eris_sp={}
    eris_sp['h1eff']= 1.*eris['h1eff'] #numpy.zeros(shape=(norbs, norbs))
    eris_sp['h1eff'][:mc.ncore,:mc.ncore] += numpy.einsum('abcd,cd', eris['ppaa'][:mc.ncore, :mc.ncore,:,:], dm1eff)
    eris_sp['h1eff'][:mc.ncore,:mc.ncore] -= numpy.einsum('abcd,bd', eris['papa'][:mc.ncore, :, :mc.ncore,:], dm1eff)*0.5
    eris_sp['h1eff'][mc.ncas+mc.ncore:,mc.ncas+mc.ncore:] += numpy.einsum('abcd,cd', eris['ppaa'][mc.ncas+mc.ncore:,mc.ncas+mc.ncore:,:,:], dm1eff)
    eris_sp['h1eff'][mc.ncas+mc.ncore:,mc.ncas+mc.ncore:] -= numpy.einsum('abcd,bd', eris['papa'][mc.ncas+mc.ncore:,:,mc.ncas+mc.ncore:,:], dm1eff)*0.5
    eriscvcv = eris['cvcv']
    if (not isinstance(eris['cvcv'], type(eris_sp['h1eff']))):
        eriscvcv = lib.chkfile.load(eris['cvcv'].name, "eri_mo")#h5py.File(eris['cvcv'].name,'r')["eri_mo"]
    eris_sp['cvcv'] = eriscvcv.reshape(mc.ncore, norbs-mc.ncore-mc.ncas, mc.ncore, norbs-mc.ncore-mc.ncas)

    # int1
    int1 = reduce(numpy.dot, (mo.T, mc.get_hcore(), mo))

    # offdiagonal
    offdiagonal = 0.0
    #zero out off diagonal core
    for k in range(mc.ncore):
        for l in range(mc.ncore):
            if(k != l):
                offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    #zero out off diagonal virtuals
    for k in range(mc.ncore+mc.ncas, norbs):
        for l in range(mc.ncore+mc.ncas,norbs):
            if(k != l):
                offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    # warning
    if (abs(offdiagonal) > 1e-6):
        print "WARNING: Have to use natural orbitals from CAASCF"
        print "         offdiagonal elements:", offdiagonal
        print ""


    # Write out ingredients to "int/"
    import os
    os.system("mkdir -p int")

    numpy.save("int/W:caac", numpy.asfortranarray(eris['papa'][frozen:mc.ncore, :, frozen:mc.ncore, :].transpose(0,3,1,2)))
    numpy.save("int/W:aeca", numpy.asfortranarray(eris['papa'][frozen:mc.ncore, :, mc.ncore+mc.ncas:, :].transpose(1,2,0,3)))
    numpy.save("int/W:ccaa", numpy.asfortranarray(eris['papa'][frozen:mc.ncore, :, frozen:mc.ncore, :].transpose(0,2,1,3)))
    numpy.save("int/W:eeaa", numpy.asfortranarray(eris['papa'][mc.ncore+mc.ncas:, :, mc.ncore+mc.ncas:, :].transpose(0,2,1,3)))
    numpy.save("int/W:caca", numpy.asfortranarray(eris['ppaa'][frozen:mc.ncore, frozen:mc.ncore, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eaca", numpy.asfortranarray(eris['ppaa'][mc.ncore+mc.ncas:, frozen:mc.ncore, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eecc", numpy.asfortranarray(eris_sp['cvcv'][frozen:,:,frozen:,:].transpose(1,3,0,2)))
    numpy.save("int/W:ccae", numpy.asfortranarray(eris['pacv'][frozen:mc.ncore,:,frozen:,:].transpose(0,2,1,3)))
    numpy.save("int/W:aaaa", numpy.asfortranarray(eris['ppaa'][mc.ncore:mc.ncore+mc.ncas, mc.ncore:mc.ncore+mc.ncas, :, :].transpose(0,2,1,3)))
    numpy.save("int/W:eeca", numpy.asfortranarray(eris['pacv'][mc.ncore+mc.ncas:, :, frozen:, :].transpose(3,0,2,1)))
    if (fully_ic):
      numpy.save("int/W:eaaa", numpy.asfortranarray(eris['ppaa'][mc.ncore+mc.ncas:, mc.ncore:mc.ncore+mc.ncas, :, :].transpose(0,2,1,3)))
      numpy.save("int/W:caaa", numpy.asfortranarray(eris['ppaa'][frozen:mc.ncore, mc.ncore:mc.ncore+mc.ncas, :, :].transpose(0,2,1,3)))
    numpy.save("int/int1eff",numpy.asfortranarray(eris_sp['h1eff'][frozen:,frozen:]))
    numpy.save("int/E1.npy", numpy.asfortranarray(dm1))
    numpy.save("int/E2.npy", numpy.asfortranarray(dm2))
    #numpy.save("int/E3",dm3)
    #numpy.save("int/E3B.npy", dm3.transpose(0,3,1,4,2,5))
    #numpy.save("int/E3C.npy", dm3.transpose(5,0,2,4,1,3))

    print "Basic ingredients wrote to int/"
    print ""


    # energy_core
    dmcore = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mo.T, vj-vk*0.5, mo))
    energy_core = numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
                + numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*numpy.einsum('ij,ij', eris_sp['h1eff'][ncore:nc, ncore:nc], dm1)\
             + 0.5*numpy.einsum('ijkl,ijkl', eris['ppaa'][mc.ncore:mc.ncore+mc.ncas, mc.ncore:mc.ncore+mc.ncas, :, :].transpose(0,2,1,3), dm2)
    energyE0 += numpy.einsum('ij,ji', dmcore, mc.get_hcore()) \
             +  numpy.einsum('ij,ji', dmcore, vj-0.5*vk) * .5
    energyE0 += mc.mol.energy_nuc()

    print "Energy_core = ",energy_core
    print "Energy      = ", energyE0
    print ""


    #write FCIDUMP_aaav* and FCIDUMP_aaac
    if (not fully_ic):
      # About symmetry...
      from pyscf import symm
      mol = mc.mol
      orbsymout=[]
      orbsym = []
      if (mol.symmetry):
          orbsym = symm.label_orb_symm(mc.mol, mc.mol.irrep_id,
                                       mc.mol.symm_orb, mo, s=mc._scf.get_ovlp())
      if mol.symmetry and orbsym:
          if mol.groupname.lower() == 'dooh':
              orbsymout = [dmrg_sym.IRREP_MAP['D2h'][i % 10] for i in orbsym]
          elif mol.groupname.lower() == 'coov':
              orbsymout = [dmrg_sym.IRREP_MAP['C2v'][i % 10] for i in orbsym]
          else:
              orbsymout = [dmrg_sym.IRREP_MAP[mol.groupname][i] for i in orbsym]
      else:
          orbsymout = []

      virtOrbs = range(ncore+mc.ncas, eris_sp['h1eff'].shape[0])
      chunks = len(virtOrbs)/aaavsplit
      virtRange = [virtOrbs[i:i+chunks] for i in xrange(0, len(virtOrbs), chunks)]

      for K in range(aaavsplit):
          currentOrbs = range(ncore, mc.ncas+ncore)+virtRange[K]
          fout = open('FCIDUMP_aaav%d'%(K),'w')
          #tools.fcidump.write_head(fout, eris_sp['h1eff'].shape[0]-ncore, mol.nelectron-2*ncore, orbsym= orbsymout[ncore:])

          tools.fcidump.write_head(fout, mc.ncas+len(virtRange[K]), mol.nelectron-2*ncore, orbsym= (orbsymout[ncore:ncore+mc.ncas]+orbsymout[virtRange[K][0]:virtRange[K][-1]+1]) )
          ij = ncore*(ncore+1)/2
          for i in range(len(currentOrbs)):
              for j in range(ncore, mc.ncas+ncore):
                  for k in range(mc.ncas):
                      for l in range(k+1):
                          I = currentOrbs[i]
                          if abs(eris['ppaa'][I,j,k,l]) > 1.e-8 :
                              fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                             % (eris['ppaa'][I,j,k,l], i+1, j+1-ncore, k+1, l+1))

          h1eff = numpy.zeros(shape=(mc.ncas+len(virtRange[K]), mc.ncas+len(virtRange[K])))
          h1eff[:mc.ncas, :mc.ncas] = eris_sp['h1eff'][ncore:ncore+mc.ncas,ncore:ncore+mc.ncas]
          h1eff[mc.ncas:, mc.ncas:] = eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, virtRange[K][0]:virtRange[K][-1]+1]
          h1eff[:mc.ncas, mc.ncas:] = eris_sp['h1eff'][ncore:ncore+mc.ncas, virtRange[K][0]:virtRange[K][-1]+1]
          h1eff[mc.ncas:, :mc.ncas] = eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, ncore:ncore+mc.ncas]

          tools.fcidump.write_hcore(fout, h1eff, mc.ncas+len(virtRange[K]), tol=1e-8)
          #tools.fcidump.write_hcore(fout, eris_sp['h1eff'][virtRange[K][0]:virtRange[K][-1]+1, virtRange[K][0]:virtRange[K][-1]+1], len(virtRange[K]), tol=1e-8)
          fout.write(' %17.9e  0  0  0  0\n' %( mol.energy_nuc()+energy_core-energyE0))
          fout.close()
          print "Wrote FCIDUMP_aaav%d file"%(K)

      nc = ncore+mc.ncas
      fout = open('FCIDUMP_aaac','w')
      tools.fcidump.write_head(fout, nc-frozen, mol.nelectron-2*frozen, orbsym= orbsymout[frozen:nc])
      for i in range(frozen,nc):
          for j in range(ncore, nc):
              for k in range(ncore, nc):
                  for l in range(ncore,k+1):
                      if abs(eris['ppaa'][i,j,k-ncore,l-ncore]) > 1.e-8 :
                          fout.write(' %17.9e %4d %4d %4d %4d\n' \
                                     % (eris['ppaa'][i,j,k-ncore,l-ncore], i+1-frozen, j+1-frozen, k+1-frozen, l+1-frozen))

      dmrge = energyE0-mol.energy_nuc()-energy_core

      ecore_aaac = 0.0;
      for i in range(frozen,ncore):
          ecore_aaac += 2.0*eris_sp['h1eff'][i,i]
      tools.fcidump.write_hcore(fout, eris_sp['h1eff'][frozen:nc,frozen:nc], nc-frozen, tol=1e-8)
      fout.write(' %17.9e  0  0  0  0\n' %( -dmrge-ecore_aaac))
      fout.close()
      print "Wrote FCIDUMP_aaac  file"
      print ""

    return norbs, energyE0


def executeMRLCC(nelec, ncore, ncas, frozen, memory=10, fully_ic=False, third_order=False):
    methods = ['MRLCC_CCVV', 'MRLCC_CCAV', 'MRLCC_ACVV', 'MRLCC_CCAA', 'MRLCC_AAVV', 'MRLCC_CAAV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']
    if (fully_ic):
        methods+=['MRLCC_AAAV', 'MRLCC_AAAC']
        domains+=['eaaa','caaa']
    if (third_order):
        methods+=['MRLCC3']
    if (ncore - frozen) == 0:
        methods = ['MRLCC_AAVV']
        domains = ['eeaa']
    totalE = 0.0
    print "Second-order:"
    for method in methods:
        f = open("%s.inp"%(method), 'w')
        if (memory is not None):
            f.write('work-space-mb %d\n'%(memory*1000))
        f.write('method %s\n'%(method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncore-frozen)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
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
        print "Total:             %18.9e"%(totalE)
        print ""
    if (third_order):
      try:
        print "Third-order:       %18.9e"%(energy)
        print ""
        totalE+=energy
      except ValueError:
        print "Third-order    --  NA"
        print ""
    return totalE


def executeNEVPT(nelec, ncore, ncas, frozen, memory=10, fully_ic=False, third_order=False):
    methods = ['NEVPT2_CCVV', 'NEVPT2_CCAV', 'NEVPT2_ACVV', 'NEVPT2_CCAA', 'NEVPT2_AAVV', 'NEVPT2_CAAV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']
    if (fully_ic):
        methods+=['NEVPT2_AAAV', 'NEVPT2_AAAC']
        domains+=['eaaa','caaa']
    if (third_order):
        methods+=['NEVPT3']
    if (ncore - frozen) == 0:
        methods = ['NEVPT2_AAVV']
        domains = ['eeaa']
    totalE = 0.0
    print "Second-order:"
    for method in methods:
        f = open("%s.inp"%(method), 'w')
        if (memory is not None):
            f.write('work-space-mb %d\n'%(memory*1000))
        f.write('method %s\n'%(method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncore-frozen)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
        f.write('int1e/fock int/int1eff.npy\n')
        #f.write('E3  int/E3.npy\n')
        #f.write('E2  int/E2.npy\n')
        #f.write('E1  int/E1.npy\n')
        f.write('thr-den 1.000000e-05\n')
        f.write('thr-var 1.000000e-05\n')
        f.write('thr-trunc 1.000000e-03\n')
        if (third_order and method!='NEVPT3'):
          f.write('save int/D:'+domains[methods.index(method)]+'.npy\n')
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
        print "Total:             %18.9e"%(totalE)
        print ""
    if (third_order):
      try:
        print "Third-order:       %18.9e"%(energy)
        print ""
        totalE+=energy
      except ValueError:
        print "Third-order:       NA"
        print ""
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
      print "perturber %s --  %18.9e"%(pattern[-4:], energy)
  except ValueError:
    if((pattern!='MRLCC3')and(pattern!='NEVPT3')):
      print "perturber %s --  NA"%(pattern[-4:])
  return energy


def kernel(mc, *args, **kwargs):
    return icmpspt(mc, *args, **kwargs)


def _ERIS(mc, mo, method='incore'):
    nmo = mo.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas

    if ((method == 'outcore') or
        (mcscf.mc_ao2mo._mem_usage(ncore, ncas, nmo)[0] +
         nmo**4*2/1e6 > mc.max_memory*.9) or
        (mc._scf._eri is None)):
        ppaa, papa, pacv, cvcv = \
                trans_e1_outcore(mc, mo, max_memory=mc.max_memory,
                                 verbose=mc.verbose)
    else:
        ppaa, papa, pacv, cvcv = trans_e1_incore(mc, mo)

    dmcore = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)
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
    ncore = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nocc = ncore + ncas
    nav = nmo - ncore
    eri1 = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo[:,ncore:]),
                                      compact=False)
    load_buf = lambda r0,r1: eri1[r0*nav:r1*nav]
    ppaa, papa, pacv, cvcv = _trans(mo, ncore, ncas, load_buf)
    return ppaa, papa, pacv, cvcv


def trans_e1_outcore(mc, mo, max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=0):
    time0 = (time.clock(), time.time())
    mol = mc.mol
    log = logger.Logger(mc.stdout, verbose)
    ncore = mc.ncore
    ncas = mc.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas
    nvir = nmo - nocc
    nav = nmo - ncore

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo[:,ncore:]), swapfile.name,
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
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        ppaa, papa, pacv = _trans(mo, ncore, ncas, load_buf, cvcv, ao_loc)[:3]
    time0 = logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return ppaa, papa, pacv, cvcvfile


def _trans(mo, ncore, ncas, fload, cvcv=None, ao_loc=None):
    nao, nmo = mo.shape
    nocc = ncore + ncas
    nvir = nmo - nocc
    nav = nmo - ncore

    if cvcv is None:
        cvcv = numpy.zeros((ncore*nvir,ncore*nvir))
    pacv = numpy.empty((nmo,ncas,ncore*nvir))
    aapp = numpy.empty((ncas,ncas,nmo*nmo))
    papa = numpy.empty((nmo,ncas,nmo*ncas))
    vcv = numpy.empty((nav,ncore*nvir))
    apa = numpy.empty((ncas,nmo*ncas))
    vpa = numpy.empty((nav,nmo*ncas))
    app = numpy.empty((ncas,nmo*nmo))
    for i in range(ncore):
        buf = fload(i, i+1)
        klshape = (0, ncore, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        cvcv[i*nvir:(i+1)*nvir] = vcv[ncas:]
        pacv[i] = vcv[:ncas]

        klshape = (0, nmo, ncore, nocc)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=apa, ao_loc=ao_loc)
        papa[i] = apa
    for i in range(ncas):
        buf = fload(ncore+i, ncore+i+1)
        klshape = (0, ncore, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        pacv[ncore:,i] = vcv

        klshape = (0, nmo, ncore, nocc)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vpa, ao_loc=ao_loc)
        papa[ncore:,i] = vpa

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=app, ao_loc=ao_loc)
        aapp[i] = app
    #lib.transpose(aapp.reshape(ncas**2, -1), inplace=True)
    ppaa = lib.transpose(aapp.reshape(ncas**2,-1))
    return (ppaa.reshape(nmo,nmo,ncas,ncas), papa.reshape(nmo,ncas,nmo,ncas),
            pacv.reshape(nmo,ncas,ncore,nvir), cvcv)


def writeDMRGConfFile(neleca, nelecb, ncore, ncas, norbs, DMRGCI, maxM, perturber, memory, numthrds, reorder, extraline, root=0, approx= False, aaavsplit=1, aaavIter=0, name= "MRLCC"):
    virtOrbs = range(norbs-ncore-ncas)
    chunks = len(virtOrbs)/aaavsplit
    virtRange = [virtOrbs[i:i+chunks] for i in xrange(0, len(virtOrbs), chunks)]

    if (perturber == "AAAC"):
        configFile = "response%s_aaac.conf"%(name)
        r = open("reorder_aaac.txt", 'w')
        for i in range(ncas):
            r.write("%d "%(reorder[i]+1+ncore))
        for i in range(ncore):
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
    ncoreE = 0
    if (perturber == "AAAC"):
        ncoreE = 2*ncore
    f.write('nelec %i\n'%(neleca+nelecb+ncoreE))
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
        for i in range(1,ncore+1):
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




def icmpspt(mc, pttype="NEVPT2", energyE0=0.0, rdmM=0, frozen=0, PTM=1000, PTincore=False, fciExtraLine=[],\
            have3RDM=False, root=0, nroots=1, verbose=None, AAAVsplit=1, do_dm3=True, do_dm4=False, fully_ic=False, third_order=False):
    print ""
    print ""
    print "--------------------------------------------------"
    print "                 ICMPSPT CALCULATION              "
    print "--------------------------------------------------"
    print ""

    # Check-up consistency of keywords
    if (do_dm4):
      do_dm3=False
    elif (do_dm3):
      do_dm4=False
    else:
      print "WARNING:  Neither do_dm3 nor do_dm4!"
      print ""
      do_dm3=True
    if (fully_ic and not do_dm4):
      print "WARNING: Fully IC needs 4RDM (for now!)"
      print ""
      do_dm4=True
      do_dm3=False
    if ((third_order)and(not fully_ic)):
      print "WARNING: Third-order needs Fully IC mode"
      print ""
      fully_ic=True
    if (pttype != "NEVPT2" and AAAVsplit != 1):
      print "AAAVsplit only works with CASSCF natural orbitals and NEVPT2"
      print ""
      exit(0)
    #if type(mc.fcisolver) is not dmrgci.DMRGCI:
    #  if (mc.fcisolver.fcibase_class is not dmrgci.DMRGCI):
    #    print "this works with dmrgscf and not regular mcscf"
    #    print ""
    #    exit(0)

    # Message
    print "Perturbation type: %s"%(pttype)
    if (fully_ic):
        print "With fully internally contracted scheme"
    if (third_order):
        print "With third order correction"
    print ""

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



    # Construct basic objects
    if (len(mc.fcisolver.orbsym) == 0 and mc.fcisolver.mol.symmetry):
        mcscf.casci_symm.label_symmetry_(mc, mc.mo_coeff)
    ericas = mc.get_h2cas()
    h1e = reduce(numpy.dot, (mc.mo_coeff.T, mc.get_hcore(), mc.mo_coeff))
    dmcore = numpy.dot(mc.mo_coeff[:,:mc.ncore], mc.mo_coeff[:,:mc.ncore].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(numpy.dot, (mc.mo_coeff.T, vj-vk*0.5, mc.mo_coeff))
    h1effcas = h1e+vhfcore
    dmrgci.writeIntegralFile(mc.fcisolver, h1effcas[mc.ncore:mc.ncore+mc.ncas, mc.ncore:mc.ncore+mc.ncas], ericas, mc.ncas, mc.nelecas)
    dm1eff = numpy.zeros(shape=(mc.ncas, mc.ncas)) #this is the state average density which is needed in NEVPT2



    # RDMs
    import os
    os.system("mkdir -p int")
    nelec = mc.nelecas[0]+mc.nelecas[1]
    # loop over all states besides the current root
    if (pttype == "NEVPT2" and nroots>1):
        stateIter = range(nroots)
        stateIter.remove(root)
        for istate in stateIter:
            dm3 = mc.fcisolver.make_rdm3(state=istate, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision)
            dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
            dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
            dm1eff += dm1
    # now add the contributaion due to the current root
    if (do_dm3):
      dm3 = mc.fcisolver.make_rdm3(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype="notbinary")
    elif (do_dm4):
      dm4 = mc.fcisolver.make_rdm4(state=root, norb=mc.ncas, nelec=mc.nelecas, dt=float_precision, filetype="notbinary")
      dm3 = numpy.einsum('ijklmnol', dm4)/(nelec-3)
      numpy.save("int/E4",dm4)
      del dm4
    dm2 = numpy.einsum('ijklmk', dm3)/(nelec-2)
    dm1 = numpy.einsum('ijkj', dm2)/(nelec-1)
    dm1eff += dm1
    dm1eff = dm1eff/(1.0*nroots)
    numpy.save("int/E3",dm3)
    numpy.save("int/E3B.npy", dm3.transpose(0,3,1,4,2,5))
    numpy.save("int/E3C.npy", dm3.transpose(5,0,2,4,1,3))
    del dm3


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
        norbs, energyE0 = writeNevpt2Integrals(mc, dm1, dm2, dm1eff, AAAVsplit, frozen, fully_ic)
        sys.stdout.flush()

        totalE = 0.0;
        totalE += executeNEVPT(nelec, mc.ncore, mc.ncas, frozen, mc.fcisolver.memory, fully_ic, third_order)

        if (not fully_ic):
          for k in range(AAAVsplit):
              writeDMRGConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore, mc.ncas,  norbs,
                                mc.fcisolver, PTM, "AAAV", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine, aaavsplit=AAAVsplit, aaavIter=k, root=root, name = "NEVPT2")
          writeDMRGConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore-frozen, mc.ncas,  norbs-frozen,
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
                  print "perturber AAAV%i --  %18.9e"%(k, energy)
                  sys.stdout.flush()

              if (mc.ncore-frozen != 0):
                  outfile, infile = "responseNEVPT2_aaac.out", "responseNEVPT2_aaac.conf"
                  output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
                  file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
                  energy = struct.unpack('d', file1.read(8))[0]
                  file1.close()
                  totalE += energy
                  print "perturber AAAC --  %18.9e"%(energy)
              print ""

          except ValueError:
              print(output)
          print "Total:             %18.9e"%(totalE)
          print ""

        print "Total PT       --  %18.9e"%(totalE)
        print ""
        return totalE

    else :
        #this is a bad way to do it, the problem is
        #that pyscf works with double precision and
        #energyE0 = writeMRLCCIntegrals(mc, dm1, dm2)
        #sys.stdout.flush()
        energyE0, norbs = writeNumpyforMRLCC(mc, dm1, dm2, frozen, fully_ic)
        sys.stdout.flush()

        totalE = 0.0
        totalE +=  executeMRLCC(nelec, mc.ncore, mc.ncas, frozen, mc.fcisolver.memory, fully_ic, third_order)

        if (not fully_ic):
          writeDMRGConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore, mc.ncas,  norbs,
                            mc.fcisolver, PTM, "AAAV", mc.fcisolver.memory, mc.fcisolver.num_thrds, reorder, fciExtraLine, root=root, name="MRLCC")
          writeDMRGConfFile(mc.nelecas[0], mc.nelecas[1], mc.ncore-frozen, mc.ncas,  norbs-frozen,
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
              print "perturber AAAV --  %18.9e"%(energy)
          except ValueError:
              print "perturber AAAV -- NA"

          try:
              if (mc.ncore-frozen != 0):
                  outfile, infile = "responseMRLCC_aaac.out", "responseMRLCC_aaac.conf"
                  output = check_call("%s  %s  %s > %s"%(mc.fcisolver.mpiprefix, mc.fcisolver.executable, infile, outfile), shell=True)
                  file1 = open("%s/node0/dmrg.e"%(mc.fcisolver.scratchDirectory),"rb")
                  energy = struct.unpack('d', file1.read(8))[0]
                  file1.close()
                  totalE += energy
                  print "perturber AAAC --  %18.9e"%(energy)
          except ValueError:
              print "perturber AAAC -- NA"
          print "Total:             %18.9e"%(totalE)
          print ""

        print "Total PT       --  %18.9e"%(totalE)
        print ""
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
    from pyscf.dmrgscf import dmrgci
    mc = dmrgci.DMRGSCF(m, 6, 6)
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.mpiprefix=""
    mc.fcisolver.num_thrds=20
    ci_e = mc.kernel()[0]
    mc.verbose = 4
    print(ci_e)

    print(icmpspt(mc, pttype="MRLCC", rdmM=500, PTM=1000), -0.16978546152699392)

