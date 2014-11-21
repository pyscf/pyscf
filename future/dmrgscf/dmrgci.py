#!/usr/bin/env python
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import numpy
import pyscf.tools

try:
    import settings
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
             'C1' : 1}

class DMRGCI(object):
    scratchDirectory = settings.BLOCKSCRATCHDIR
    integralFile = "FCIDUMP"
    configFile="dmrg.conf"
    outputFile="dmrg.out"
    maxIter=20
    twodot_to_onedot=15
    tol=1e-12
    maxM=1000
    executable = settings.BLOCKEXE
    orbsym = []

    def __init__(self, mol):
        self.mol = mol
        if mol.symmetry:
            self.groupname = mol.groupname
        else:
            self.groupname = None

    def make_rdm12(self, fcivec, norb, nelec, link_index=None, **kwargs):
        return make_rdm12(fcivec, norb, nelec, link_index, DMRGCI=self)

    def kernel(self, fcivec, norb, nelec, link_index=None,
               fciRestart=False, **kwargs):
        if self.mol.symmetry and self.orbsym:
            orbsym = [IRREP_MAP[self.groupname][i] for i in self.orbsym]
        return kernel(fcivec, norb, nelec, link_index, fciRestart, DMRGCI=self)

def make_rdm12(fcivec, norb, nelec, link_index=None, DMRGCI=DMRGCI):

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

def writeDMRGConfFile(neleca, nelecb, Restart, DMRGCI=DMRGCI):
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

def writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb, DMRGCI=DMRGCI):
    import os
    integralFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.integralFile)
# ensure 4-fold symmetry
    eri_cas = pyscf.ao2mo.restore(4, eri_cas)
    pyscf.tools.fcidump.from_integrals(integralFile, h1eff, eri, ncas,
                                       neleca+nelecb, ms=abs(neleca-nelecb),
                                       orbsym=DMRGCI.orbsym)

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


def executeBLOCK(DMRGCI=DMRGCI):
    import os
    inFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.configFile)
    outFile = "%s%s%s"%(DMRGCI.scratchDirectory,os.sep,DMRGCI.outputFile)
    from subprocess import call
    call("%s  %s > %s"%(DMRGCI.executable, inFile, outFile), shell=True)

def readEnergy(DMRGCI=DMRGCI):
    import struct, os
    file1 = open("%s%s%s"%(DMRGCI.scratchDirectory, os.sep, "dmrg.e"),"rb")
    calc_e = struct.unpack('d', file1.read(8))[0]

    return calc_e

def kernel(h1eff, eri_cas, ncas, neleccas, ci0=False, fciRestart=False,
           DMRGCI=DMRGCI):
    if isinstance(neleccas, int):
        neleca=neleccas/2 + neleccas%2
        nelecb=neleccas - neleca
    else :
        neleca, nelecb = neleccas

    writeIntegralFile(h1eff, eri_cas, ncas, neleca, nelecb, DMRGCI)
    writeDMRGConfFile(neleca, nelecb, fciRestart, DMRGCI)
    executeBLOCK(DMRGCI)
    calc_e = readEnergy(DMRGCI)

    return calc_e, None


if __name__ == '__main__':
    import numpy
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

    mc = mcscf.CASSCF(mol, m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    mc.fcisolver.tol = 1e-9
    emc_1 = mc.mc2step()[0] + mol.nuclear_repulsion()

    mc = mcscf.CASCI(mol, m, 4, 4)
    mc.fcisolver = DMRGCI(mol)
    emc_0 = mc.casci()[0] + mol.nuclear_repulsion()

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

    mc = mcscf.CASSCF(mol, m, 4, 4)
    emc_1ref = mc.mc2step()[0] + mol.nuclear_repulsion()

    mc = mcscf.CASCI(mol, m, 4, 4)
    emc_0ref = mc.casci()[0] + mol.nuclear_repulsion()

    print('DMRGCI  = %.15g CASCI  = %.15g' % (emc_0, emc_0ref))
    print('DMRGSCF = %.15g CASSCF = %.15g' % (emc_1, emc_1ref))
