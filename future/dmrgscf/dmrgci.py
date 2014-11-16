#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#

import numpy

class DMRGCI(object):
    scratchDirectory = "scratch"
    integralFile = "FCIDUMP"
    configFile="dmrg.conf"
    outputFile="dmrg.out"
    maxIter=20
    twodot_to_onedot=15
    tol=1e-12
    maxM=1000
    executable="/home/sharma/apps/Block/block.spin_adapted"

def make_rdm12(fcivec, norb, nelec, link_index=None):

    nelectrons = 0
    if isinstance(nelec, int):
        nelectrons = nelec
    else:
        nelectrons = nelec[0]+nelec[1]

    import os
    f = open("%s%s%s"%(DMRGCI.scratchDirectory, os.sep, "spatial_twopdm.0.0.txt"), 'r')

    twopdm = numpy.zeros( (norb, norb, norb, norb) )
    norb_read = int(f.readline().split()[0])
    assert norb_read == norb

    for line in f.readlines():
        linesp = line.split()
        twopdm[int(linesp[0]),int(linesp[3]),int(linesp[1]),int(linesp[2])] = 2.0*float(linesp[4])
    
    onepdm = numpy.einsum('ikjj->ik', twopdm)
    onepdm /= (nelectrons-1)


    return onepdm, twopdm

def writeDMRGConfFile(neleca, nelecb, Restart):
    import os
    confFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.configFile)

    f = open(confFile, 'w')
    f.write('nelec %i\n'%(neleca+nelecb))
    f.write('spin %i\n' %(neleca-nelecb))

    if (not Restart):
        f.write('schedule\n')
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 10.0))
        f.write('1 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-5, 1e-4))
        f.write('10 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, 1e-6, 1e-5))
        f.write('16 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10.0, 0e-6))
        f.write('end\n')
        f.write('twodot_to_onedot %i\n'%DMRGCI.twodot_to_onedot)
    else :
        f.write('schedule\n')
        f.write('0 %6i  %8.4e  %8.4e \n' %(DMRGCI.maxM, DMRGCI.tol/10.0, 0e-6))
        f.write('end\n')
        f.write('fullrestart\n')
        f.write('onedot \n')
        
    f.write('orbitals %s%s%s\n'%(DMRGCI.scratchDirectory, os.sep, DMRGCI.integralFile))
    f.write('maxiter %i\n'%DMRGCI.maxIter)
    f.write('sweep_tol %8.4e\n'%DMRGCI.tol)
    f.write('outputlevel 2\n')
    f.write('hf_occ integral\n')
    f.write('twopdm\n')
    f.write('prefix  %s\n'%DMRGCI.scratchDirectory)
    f.close()
    #no reorder
    #f.write('noreorder\n')

def writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb):
    import os
    integralFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.integralFile)

    f = open(integralFile, 'w')
    f.write(' &FCI NORB= %i,NELEC= %i,MS2= %i,\n' %(ncas, neleca+nelecb, neleca-nelecb))
    f.write(' ORBSYM=')
    for i in range(ncas):
        f.write('1 ')

    f.write('\nISYM=1\n')
    f.write('&END\n')
    index1 = 0
    for i in range(ncas):
        for j in range(i+1):
            index2=0
            for k in range(ncas):
                for l in range(k+1):
                    f.write('%18.10e %3i  %3i  %3i  %3i\n' %(eri_cas[index1,index2], i+1, j+1, k+1, l+1))
                    index2=index2+1
            index1=index1+1
    for i in range(ncas):
        for j in range(i+1):
            f.write('%18.10e %3i  %3i  %3i  %3i\n' %(h1eff[i,j], i+1, j+1, 0, 0))

    f.close()


def executeBLOCK():
    import os
    inFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.configFile)
    outFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.outputFile)
    from subprocess import call
    call("%s  %s > %s"%(DMRGCI.executable, inFile, outFile), shell=True)

def readEnergy():
    import struct, os
    file1 = open("%s%s%s"%(DMRGCI.scratchDirectory, os.sep, "dmrg.e"),"rb")
    calc_e = struct.unpack('d', file1.read(8))[0]

    return calc_e

def kernel(h1eff, eri_cas, ncas, neleccas, ci0=False, fciRestart=False):
    if isinstance(neleccas, int):
        neleca=neleccas/2 + neleccas%2
        nelecb=neleccas - neleca
    else :
        neleca, nelecb = neleccas

    writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb)
    writeDMRGConfFile(neleca, nelecb, fciRestart)
    executeBLOCK()
    calc_e = readEnergy()

    return calc_e, None

