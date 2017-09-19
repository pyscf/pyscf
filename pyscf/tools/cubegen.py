#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com> 
#         Paul J. Robinson <pjrobinson@ucla.edu>
#

import numpy
import time
import pyscf
from pyscf import lib
from pyscf.dft.gen_grid import prange


'''
Gaussian cube file format
'''

def density(mol, outfile, dm, nx=60, ny=60, nz=60, moN = -1, fileFormat = "cube"):
    ## PJ added a MO generating functionlity , if moN (mo num) is set to a positive value that MO wil be generated else the whole density will be generated
    ## when plotting MOs the density matrix should actually just be the moCoeffs
    '''
    INPUT: mol--molecule to be visulized
            dm--density matrix in the case of densities and the MO coefficents for Mos
            outfile--the name of the file where a gaussian cube is written
            n(x,y,z) =number of grid points along these vectors
            moN = the MO which is to be visualized (if negative [default] the density is produced)
            fileFormat  = cube or vasp...cube is the default for non-periodic and vasp is the default for periodic (MANUALLY CHANGING IS NOT RECCOMENDED)
    Output: a gaussian cube file or a VASP chgcarlike file (with phase if desired)...both can be opened in VESTA or VMD or many other softwares
    
    Example for generating a single mo:
    >>>cubegen.density(mol, 'h2_mo1.cube', mf.mo_coeff,moN=1)
        generates the first MO from the list of mo_coefficents 
        
    Future additions: specify alpha or beta orbitals, spin density
    --------------------------------------------------------------------------------
    In this section we set the parameters for two cases:
    1-molecule: large, sensible, box generated with the molecule at the center
    2-crystal:  lattice vectors define the positioning
    '''
    #cube is the default if the program doesn't do anything else
    writeFormat = "cube"
    #the format writes a cube by default including in cases where the input is not recognized
    if fileFormat.lower() == "vasp" or fileFormat.lower() =="chgcar":
        writeFormat = "vasp"
    coord = mol.atom_coords()

    isPeriodic = True
    #for PBC, we must use the pbc code for evaluating the integrals lest the pbc conditions be ignored
    try:
        dummyVar = mol.a
    except AttributeError:
        isPeriodic = False
    
    if isPeriodic:
        writeFormat = "vasp"
        from pyscf.pbc.dft import numint, gen_grid
        box = mol.lattice_vectors()
        boxorig = numpy.zeros(3)
    
    if not isPeriodic:
        writeFormat = "cube"
        from pyscf.dft import numint, gen_grid
        box = numpy.diag((numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 14))
        boxorig = numpy.min(coord,axis=0) + 7

    normVecs = [numpy.divide(bA,numpy.linalg.norm(bA)) for bA in box]
    mags = [numpy.linalg.norm(bA) for bA in box]
    #xs = numpy.outer(numpy.arange(nx),(numpy.divide(box[:,0],nx)))
    #ys = numpy.outer(numpy.arange(ny),(numpy.divide(box[:,1],ny)))
    #zs = numpy.outer(numpy.arange(nz),(numpy.divide(box[:,2],nz)))
    #the mistake i was making is forgetting to normalize before projecting
    xs = numpy.outer(numpy.arange(nx),mags[0]*(numpy.divide(normVecs[0],nx)))
    ys = numpy.outer(numpy.arange(ny),mags[1]*(numpy.divide(normVecs[1],ny)))
    zs = numpy.outer(numpy.arange(nz),mags[2]*(numpy.divide(normVecs[2],nz)))
    coords =  lib.cartesian_prod((numpy.arange(nx),numpy.arange(ny),numpy.arange(nz)))
    coords = numpy.asarray(map(lambda coordX: sum((xs[coordX[0]], ys[coordX[1]], zs[coordX[2]])), coords))
    #coords = [numpy.add(numpy.add(xs[coordX[0]], ys[coordX[1]]), zs[coordX[2]]) for coordX in coords]
    #try with list comprehension if faster
    #coords = map(lambda coordX: numpy.add(numpy.add(xs[coordX[0]], ys[coordX[1]]), zs[coordX[2]]), coords)
    #this coord function is an extension of the 1D version which now allows for arbitrary vectors
    coords = numpy.subtract(numpy.asarray(coords, order='C'),boxorig)
    '''
    In this section we either generate mos (moN > 0) or the density (moN < 0)
    '''
    
    if moN > 0:
        nao = mol.nao_nr()
        ngrids = nx * ny * nz
        blksize = min(200, ngrids)
        rho = numpy.empty(ngrids)
        numAOs = dm.shape[0]
        rhoPlacehold =  numpy.empty(ngrids)
        for ip0, ip1 in prange(0, ngrids, blksize):
            ao = numint.eval_ao(mol, coords[ip0:ip1])
            '''this loop prints a single MO by num...should loop over the density matrix '''
            for i in range(numAOs):
                rho[ip0:ip1] += numpy.multiply(dm[i,moN-1],ao[:,i])
        rho = rho.reshape(nx,ny,nz)

    else:
        nao = mol.nao_nr()
        ngrids = nx * ny * nz
        blksize = min(200, ngrids)
        rho = numpy.empty(ngrids)
        for ip0, ip1 in prange(0, ngrids, blksize):
            ao = numint.eval_ao(mol, coords[ip0:ip1])
            rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
        rho = rho.reshape(nx,ny,nz)

    '''
    This part just writes the cube file
    '''
    if writeFormat == "cube":
        coord = [numpy.add(coordX,2*boxorig) for coordX in coord ]
        writeCube(mol,rho,nx,ny,nz,xs,ys,zs,boxorig,box,outfile,coord)
    if writeFormat == "vasp":
        writeVasp(mol,rho*numpy.linalg.det(box),nx,ny,nz,xs,ys,zs,boxorig,box,outfile,coord)
    
def writeCube(mol,rho,nx,ny,nz,xs,ys,zs,boxorig,box,outfile,coord):
    with open(outfile, 'w') as f:
        f.write('Cube File: Electron density in real space (e/Bohr^3)\n')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('%5d' % mol.natm)
        f.write(' %14.8f %14.8f %14.8f\n' % tuple(boxorig.tolist()))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (nx, xs[1,0], xs[1,1], xs[1,2]))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (ny, ys[1,0], ys[1,1], ys[1,2]))
        f.write('%5d %14.8f %14.8f %14.8f\n' % (nz, zs[1,0], zs[1,1], zs[1,2]))
        for ia in range(mol.natm):
            chg = mol.atom_charge(ia)
            f.write('%5d %f' % (chg, chg))
            f.write(' %14.8f %14.8f %14.8f\n' % tuple(coord[ia]))
        fmt = ' %14.8e' * nz + '\n'
        for ix in range(nx):
            for iy in range(ny):
                f.write(fmt % tuple(rho[ix,iy].tolist()))

def writeVasp(mol,rho,nx,ny,nz,xs,ys,zs,boxorig,box,outfile,coord):
    bhToAng =  0.529177249
    boxA = box*bhToAng
    atomList= [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    Axyz = zip(atomList, coord.tolist())
    Axyz.sort(key = lambda x: x[0])
    swappedCoords = [numpy.add(vec[1],boxorig)*bhToAng for vec in Axyz]
    #DirectCoords = map( lambda vecIn: [numpy.dot(numpy.multiply(vecIn,bhToAng),latVec)/(numpy.linalg.norm(latVec)) for latVec in boxA ], swappedCoords)
    #print DirectCoords
    vaspAtomicInfo = vaspCoordSort([xyz[0] for xyz in Axyz ])
    with open(outfile, 'w') as f:
        f.write('VASP file: Electron density in real space (e/Bohr^3)  ')
        f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
        f.write('1.0000000000\n')
        f.write('%14.8f %14.8f %14.8f \n' % (boxA[0,0],boxA[0,1],boxA[0,2]))
        f.write('%14.8f %14.8f %14.8f \n' % (boxA[1,0],boxA[1,1],boxA[1,2]))
        f.write('%14.8f %14.8f %14.8f \n' % (boxA[2,0],boxA[2,1],boxA[2,2]))
        [f.write('%5.3s' % (atomN[0])) for atomN in vaspAtomicInfo ]
        f.write('\n')
        [f.write('%5d' % (atomN[1])) for atomN in vaspAtomicInfo ]
        f.write('\n')
        f.write('Cartesian \n')
        for ia in range(mol.natm):
            f.write(' %14.8f %14.8f %14.8f\n' % tuple(swappedCoords[ia]))
        f.write('\n')
        f.write('%6.5s %6.5s %6.5s \n' % (nx,ny,nz))
        fmt = ' %14.8e '
        for iz in range(nx):
            for iy in range(ny):
                f.write('\n')
                for ix in range(nz):
                    f.write(fmt % rho[ix,iy,iz])

#WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
#Vasp format fortran loop ^^
def vaspCoordSort(atomName):
    uniqueAtoms = []
    numOfUnique = []
    for anAtom in atomName:
        if anAtom not in uniqueAtoms:
            uniqueAtoms.append(anAtom)
            numOfUnique.append(1)
        else:
            for k, elem in enumerate(uniqueAtoms):
                if elem == anAtom:
                    numOfUnique[k] += 1
                    break
    return [ [a,numOfUnique[i]] for i,a in enumerate(uniqueAtoms)]

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.tools import cubegen
    mol = gto.M(atom='H 0 0 0; H 0 0 1')
    mol.basis = 'ccPVTZ'
    mf = scf.RHF(mol)
    mf.scf()
    cubegen.density(mol, 'h2.cube', mf.make_rdm1()) #makes total density
    cubegen.density(mol, 'h2_mo1.cube', mf.mo_coeff,moN=1) # makes mo#1 (sigma)
    cubegen.density(mol, 'h2_mo2.cube', mf.mo_coeff,moN=2) # makes mo#2 (sigma*)

