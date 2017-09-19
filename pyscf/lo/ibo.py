
#!/usr/bin/env python
#

'''
Intrinsic Bonding Orbitals
ref. JCTC, 9, 4834
'''

from functools import reduce
import numpy
import scipy.linalg
from time import time

'''
Below here is work done by Paul Robinson.
much of the code below is adapted from code published freely on the website of Gerald Knizia
Ref: JCTC, 2013, 9, 4834-4843
'''
def ibo(mol, orbocc, iaos, mf, p = True):
    '''
input:  mol:    the molecule
        orbocc: the array of OCCUPIED molecular orbitals
        iaos:   the array of orthonormalized IAOs
        mf:     the mean field object
        p:      print or no

returns:
        IBOs in the basis of orbocc
    '''
    #static variables
    StartTime = time()
    L  = 0 # initialize a value of the localization function for safety
    Exponent = 4    #see ref
    maxIter = 20000 #for some reason the convergence of solid is slower
    fGradConv = 1e-10 #this ought to be pumped up to about 1e-8 but for testing purposes it's fine
    swapGradTolerance = 1e-12

    #dynamic variables
    isPeriodic = True 
    Converged = False
    mol.copy()
    

    #We don't really need this, but we might as well check in case we at some point want to know
    try:
        dummyVar = mol.a
    except AttributeError:
        isPeriodic = False
    Atoms  = [mol.atom_symbol(i) for i in range(mol.natm)]
    if not isPeriodic:
        from pyscf import gto
    if isPeriodic:
        from pyscf.pbc import gto


###this is a poor way to do this...there's a builtinFunction    
#    #works for both array and string geometry formats
#    if isinstance(mol.atom,str):
#        geometry = [ a.split() for  a in (mol.atom).split('\n') ]    
#        Atoms =  [ at[0] for at in geometry]
#    else:
#        Atoms =  [ at[0] for at in mol.atom]

    #generates the parameters we need about the atomic structure
    nAtoms = len(Atoms)
    AtomOffsets = MakeAtomIbOffsets(Atoms)[0]
    iAtSl = [slice(AtomOffsets[A],AtomOffsets[A+1]) for A in range(nAtoms)]
    ovlpS = mf.get_ovlp()
    #converts the occupied MOs to the IAO basis
    CIb = reduce(numpy.dot, (iaos.T, ovlpS , orbocc))
    CIb = CIb.copy()
    numOccOrbitals = CIb.shape[1]


    if p: print("   {0:^5s} {1:^14s} {2:^11s} {3:^8s}".format("ITER.","LOC(Orbital)","GRADIENT", "TIME"))


    for it in range(maxIter):
        fGrad = 0.00

        #calculate L for convergence checking
        L = 0.
        for A in range(nAtoms):
            for i in range(numOccOrbitals):
                CAi = CIb[iAtSl[A],i]
                L += numpy.dot(CAi,CAi)**Exponent

        # loop over the occupied orbitals pairs i,j
        for i in range(numOccOrbitals):
            for j in range(i):
                # I eperimented with exponentially falling off random noise
                Aij  = 0.0 #numpy.random.random() * numpy.exp(-1*it)
                Bij  = 0.0 #numpy.random.random() * numpy.exp(-1*it)
                for k in range(nAtoms):
                    CIbA = CIb[iAtSl[k],:]
                    Cii  = numpy.dot(CIbA[:,i], CIbA[:,i])
                    Cij  = numpy.dot(CIbA[:,i], CIbA[:,j])
                    Cjj  = numpy.dot(CIbA[:,j], CIbA[:,j])    
                    #now I calculate Aij and Bij for the gradient search
                    Bij += 4.*Cij*(Cii**3-Cjj**3)
                    Aij += -Cii**4 - Cjj**4 + 6*(Cii**2 + Cjj**2)*Cij**2 + Cii**3 * Cjj + Cii*Cjj**3 
                if (Aij**2 + Bij**2 < swapGradTolerance) and False:
                    continue
                    #this saves us from replacing already fine orbitals
                else:
                    #THE BELOW IS TAKEN DIRECLTY FROMG KNIZIA's FREE CODE
                    # Calculate 2x2 rotation angle phi.
                    # This correspond to [2] (12)-(15), re-arranged and simplified.
                    phi = .25*numpy.arctan2(Bij,-Aij)
                    fGrad += Bij**2
                    # ^- Bij is the actual gradient. Aij is effectively
                    #    the second derivative at phi=0.
        
                    # 2x2 rotation form; that's what PM suggest. it works
                    # fine, but I don't like the asymmetry.
                    cs = numpy.cos(phi)
                    ss = numpy.sin(phi)
                    Ci = 1. * CIb[:,i]
                    Cj = 1. * CIb[:,j]
                    CIb[:,i] =  cs * Ci + ss * Cj
                    CIb[:,j] = -ss * Ci + cs * Cj
        fGrad = fGrad**.5
        
        if p: print(" {0:5d} {1:12.8f} {2:11.2e} {3:8.2f}".format(it+1, L**(1./Exponent), fGrad, time()-StartTime))
        if fGrad < fGradConv:
            Converged = True
            break
    Note = "IB/P%i/2x2, %i iter; Final gradient %.2e" % (Exponent, it+1, fGrad)
    if not Converged:
        print("\nWARNING: Iterative localization failed to converge!"\
             "\n         %s" % Note)
    else:
        if p: #print()
            print(" Iterative localization: %s" % Note)
    print(" Localized orbitals deviation from orthogonality: %8.2e" % numpy.linalg.norm(numpy.dot(CIb.T, CIb) - numpy.eye(numOccOrbitals)))
    return numpy.dot(iaos,CIb)
#    inBigBasis =  reduce(numpy.dot, (iaos, CIb))
#    #normalize the MOs to be returned
#    inBigBasis = numpy.asarray(map(lambda x: x/numpy.linalg.norm(x),inBigBasis))
#    return reduce(numpy.dot,( numpy.linalg.inv(numpy.dot(iaos.T, ovlpS)), CIb))
#    return inBigBasis
               



'''
These are parameters for selecting the valence space correctly
'''         
def MakeAtomInfos():
   nCoreX = {"H": 0, "He": 0}
   for At in "Li Be B C O N F Ne".split(): nCoreX[At] = 1
   for At in "Na Mg Al Si P S Cl Ar".split(): nCoreX[At] = 5
   for At in "Na Mg Al Si P S Cl Ar".split(): nCoreX[At] = 5
   for At in "K Ca".split(): nCoreX[At] = 18/2
   for At in "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(): nCoreX[At] = 18/2
   for At in "Ga Ge As Se Br Kr".split(): nCoreX[At] = 18/2+5 # [Ar] and the 5 d orbitals.
   nAoX = {"H": 1, "He": 1}
   for At in "Li Be".split(): nAoX[At] = 2
   for At in "B C O N F Ne".split(): nAoX[At] = 5
   for At in "Na Mg".split(): nAoX[At] = 3*1 + 1*3
   for At in "Al Si P S Cl Ar".split(): nAoX[At] = 3*1 + 2*3
   for At in "K Ca".split(): nAoX[At] = 18/2+1
   for At in "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(): nAoX[At] = 18/2+1+5   # 4s, 3d
   for At in "Ga Ge As Se Br Kr".split(): nAoX[At] = 18/2+1+5+3

   AoLabels = {}
   def SetAo(At, AoDecl):
      Labels = AoDecl.split()
      AoLabels[At] = Labels
      assert(len(Labels) == nAoX[At])
      nCore = len([o for o in Labels if o.startswith('[')])
      assert(nCore == nCoreX[At])

   # atomic orbitals in the MINAO basis: [xx] denotes core orbitals.
   for At in "H He".split(): SetAo(At, "1s")
   for At in "Li Be".split(): SetAo(At, "[1s] 2s")
   for At in "B C O N F Ne".split(): SetAo(At, "[1s] 2s 2px 2py 2pz")
   for At in "Na Mg".split(): SetAo(At, "[1s] [2s] 3s [2px] [2py] [2pz]")
   for At in "Al Si P S Cl Ar".split(): SetAo(At, "[1s] [2s] 3s [2px] [2py] [2pz] 3px 3py 3pz")
   for At in "K Ca".split(): SetAo(At, "[1s] [2s] [3s] 4s [2px] [2py] [2pz] [3px] [3py] [3pz]")
   for At in "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split(): SetAo(At, "[1s] [2s] [3s] 4s [2px] [2py] [2pz] [3px] [3py] [3pz] 3d0 3d2- 3d1+ 3d2+ 3d1-")
   for At in "Ga Ge As Se Br Kr".split(): SetAo(At, "[1s] [2s] [3s] 4s [2px] [2py] [2pz] [3px] [3py] [3pz] 4px 4py 4pz [3d0] [3d2-] [3d1+] [3d2+] [3d1-]")
   # note: f order is '4f1+','4f1-','4f0','4f3+','4f2-','4f3-','4f2+',

   return nCoreX, nAoX, AoLabels


def MakeAtomIbOffsets(Atoms):
   """calcualte offset of first orbital of individual atoms
   in the valence minimal basis (IB)"""
   nCoreX, nAoX, AoLabels = MakeAtomInfos()
   iBfAt = [0]
   for Atom in Atoms:
      iBfAt.append(iBfAt[-1] + nAoX[Atom])
   return iBfAt, nCoreX, nAoX, AoLabels

