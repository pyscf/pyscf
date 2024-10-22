#!/usr/bin/env python
# Copyright 2019-2021 The PySCF Developers. All Rights Reserved.
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
# Author: Elvira R. Sayfutyarova
#


''' When using results of this code for publications, please cite the following paper:
    "Constructing molecular pi-orbital active spaces for multireference calculations of conjugated systems"
     E. R. Sayfutyarova and S. Hammes-Schiffer, J. Chem. Theory Comput., 15, 1679 (2019).
'''


import numpy as np
import numpy.linalg as la
from pyscf import gto

def mdot(*args):
    """chained matrix product: mdot(A,B,C,..) = A*B*C*...
    No attempt is made to optimize the contraction order."""
    r = args[0]
    for a in args[1:]:
        r = np.dot(r,a)
    return r

def MakeShellsForElement(mol, Element):
    """make a list with MINAO basis set data for a given element in the mol object"""
    PyScfShellInfo = gto.basis.load(mol.basis, Element)
    AtShells = []
    for ShellData in PyScfShellInfo:
        # see: gto/basis/minao.py on format of data.
        # data comes as a list of:
        #   [l, (...), (...), (...)]
        # where each (...) specifies both exponents and contractions and has the same length.

        l, ExponentsAndContractions = ShellData[0], np.asarray(ShellData[1:])
        # primitive exponents
        # Exp = ExponentsAndContractions[:,0]
        # contraction coefficient matrix
        CGTO = ExponentsAndContractions[:,1:]
        nCGTO = CGTO.shape[1]
        nLk = nCGTO*(2*l+1)
        AtShells.append((l, nCGTO, nLk))
    return AtShells

def MakeShells(mol, Elements):
    """collect MINAO basis set data for all elements"""
    Shells = []
    for iAt in range(mol.natm):
        ElementShells = MakeShellsForElement(mol, Elements[iAt])
        Shells.append(ElementShells)
    return Shells

def Atoms_w_Coords(mol):
    """collect info about atoms' positions"""
    AtomCoords = mol.atom_coords()

    assert (AtomCoords.shape == (mol.natm, 3))
    Elements = [mol.atom_pure_symbol(iAt) for iAt in range(mol.natm)]
    Elements =np.asarray(Elements)

    return Elements,AtomCoords

def MakePiOS(mol,mf,PiAtomsList, nPiOcc=None,nPiVirt=None):
    np.set_printoptions(precision=4,linewidth=10000,edgeitems=3,suppress=False)

    print("================================CONSTRUCTING PI-ORBITALS===========================================")
    # PiAtoms list contains the set of main-group atoms involved in the pi-system of the chromophores(?)
    # indices are 1-based.

    mol2=mol.copy()
    mol2.basis = 'MINAO'
    mol2.build()

    # make a minimal AO basis for our atoms. We load the basis from a library
    # just to have access to its shell-composition. Need that to find the indices
    # of all atoms, and the AO indices of the px/py/pz functions.

    Elements,Coords = Atoms_w_Coords(mol2)
    Shells = MakeShells(mol2,Elements)
    def AssignTag(iAt, Element, Tag):
        assert (Elements[iAt-1] == Element)
        Elements[iAt-1] = Element + Tag


    # fix type of atom for the donor-acceptor which are exchanging the hydrogens.
    # During the process, the hydrogens are moving and the Huckel-Theory
    # formal number of contributed pi-electrons may change.
    # These settings override the auto-detection of number of pi electrons
    # based on atomic connectivity in GetNumPiElec() below.
#    AssignTag(50, "N", "1e")



    # OrbBasis = mol.basis
    C = mf.mo_coeff
    S1 = mol.intor_symmetric("int1e_ovlp")
    S2 = mol2.intor_symmetric("int1e_ovlp")
    S12 = gto.intor_cross('int1e_ovlp', mol, mol2)
    SMo = mdot(C.T, S1, C)
    print("    MO deviation from orthogonality  {:8.2e} \n".format(rmsd(SMo - np.eye(SMo.shape[0]))))

    # make arrays of occupation numbers and orbital eigenvalues.
    Occ = mf.mo_occ
    Eps = mf.mo_energy

    nOrb = C.shape[1]
    if (mol.spin==0):
        nOcc = np.sum(Occ == 2)
    else:
        nOcc = np.sum(Occ == 2)+np.sum(Occ == 1)
        n1=np.sum(Occ == 1)
        print("    Number of singly occupied orbitals      {} ".format(n1))

    nVir = np.sum(Occ == 0)
    assert (nOcc + nVir == nOrb)
    print("    Number of occupied orbitals      {} ".format(nOcc))
    print("    Number of unoccupied orbitals    {} ".format(nVir))

    # Compute Fock matrix from orbital eigenvalues (SCF has to be fully converged)
    Fock = mdot(S1, C, np.diag(Eps), C.T, S1.T)
    # Rdm = mdot(C, np.diag(Occ), C.T)

    COcc = C[:,:nOcc]
    CVir = C[:,nOcc:]

    # Smh1 = MakeSmh(S1)
    # Sh1 = np.dot(S1, Smh1)

    # Compute IAO basis (non-orthogonal). Will be used to identify the
    # pi-MO-space of the target atom groups.
    CIb = MakeIaosRaw(COcc, S1, S2, S12)
    # CIbOcc = np.linalg.lstsq(CIb, COcc)[0]
    # Err = np.dot(Sh1, np.dot(CIb, CIbOcc) - COcc)

    # check orthogonality of IAO basis occupied orbitals
    SIb = mdot(CIb.T, S1, CIb)
    # SIbOcc = mdot(CIbOcc.T, SIb, CIbOcc)
    nIb = SIb.shape[0]

    if 0:
        # make the a representation of the virtual valence space.
        # SmhIb = MakeSmh(SIb)

        nIbVir = nIb - nOcc  # number of virtual valence orbitals

        CTargetIb = CIb
        STargetIb = mdot(CTargetIb.T, S1, CTargetIb)
        SmhTargetIb = MakeSmh(STargetIb)
        STargetIbVir = mdot(SmhTargetIb, CTargetIb.T, S1, CVir)
        U,sig,Vt = np.linalg.svd(STargetIbVir, full_matrices=False)
        print("    Number of target MINAO basis fn     {}\n".format(CTargetIb.shape[1]))

        print("SIbVir Singular Values (n={})".format(len(sig)))
        print("    [{}]".format(', '.join('{:.4f}'.format(k) for k in sig)))


        assert (np.abs(sig[nIbVir-1] - 1.0) < 1e-4)
        assert (np.abs(sig[nIbVir] - 0.0) < 1e-4)

        # for pi systems: do it like here -^ for the virtuals, but
        # for COcc and CVir both. What we should do is:
        #   - instead of CIb, we use CTargetIb = CIb * (AO-linear-comb-matrix)
        #   - instead of SmhIb, we use its corresponding target-AO overlap matrix:
        #     SmhTargetIb = MakeSmh(mdot(CTargetIb.T, S1, CTargetIb))
        #   - instead of using nOcc/nVir/nIb to determine the target number
        #     of orbitals, we obtain the target number of pi electrons by counting
        #     their subset from the selected main group atoms (see CHEM 408 u11).
        #     From this determine the number of occupied pi-orbitals this system
        #     is supposed to have, and virtual orbitals as nTargetIb - nPiOcc
        # The AoMix matrix is made from the pi-system's inertial tensor to get the
        # local z-direction, and then linearly-combining this onto the highest-N p-AOs.

    CActOcc = []
    CActVir = []

    # add pi-HOMOs and pi-LUMOs
    CFragOcc, CFragVir,nOccOrbExpected,nVirtOrbExpected = MakePiSystemOrbitals(
        "Pi-System", PiAtomsList, None, Elements,Coords, CIb, Shells, S1, S12, S2, Fock, COcc, CVir)
    if (nPiOcc is None):
        for i in range(1,nOccOrbExpected+1):
            CActOcc.append(CFragOcc[:,-i])
    else:
        for i in range(1,nPiOcc+1):
            CActOcc.append(CFragOcc[:,-i])

    if (nPiVirt is None):
        for j in range(nVirtOrbExpected):
            CActVir.append(CFragVir[:,j])
    else:
        for j in range(nPiVirt):
            CActVir.append(CFragVir[:,j])

    print("\n -- Joining active spaces")
    if (mol.spin==0):
        nElec = 2*len(CActOcc)
    else:
        nElec = 2*len(CActOcc)-n1
    CAct = np.array(CActOcc + CActVir).T
    if 0:
        # orthogonalize and semi-canonicalize
        SAct = mdot(CAct.T, S1, CAct)
        ew, ev = np.linalg.eigh(SAct)
        print("    CAct initial overlap (if all ~approx 1, then the initial "
              "active orbitals are near-orthogonal. That is good.)")
        print("    [{}]".format(', '.join('{:.4f}'.format(k) for k in ew)))

        CAct = np.dot(CAct, MakeSmh(SAct))
        CAct = SemiCanonicalize(CAct, Fock, S1, "active")

    print("    Number of Active Electrons     {} ".format(nElec))
    print("    Number of Active Orbitals      {} ".format( CAct.shape[1]))

    # make new non-active occupied and non-active virtual orbitals,
    # in order to rebuild a full MO matrix.
    def MakeInactiveSpace(Name, CActList, COrb1):
        CAct = np.array(CActList).T
        SAct = mdot(CAct.T, S1, CAct)
        CAct = np.dot(CAct, MakeSmh(SAct))

        SActMo = mdot(COrb1.T, S1.T, CAct, CAct.T, S1, COrb1)
        ew, ev = np.linalg.eigh(SActMo) # small ews first (orbs not overlapping with active orbs).

        nRest = COrb1.shape[1] - CAct.shape[1]
        if 0:
            for i in range(len(ew)):
                print("{:4} {:15.6f}  {}".format(i,ew[i], i == nRest))
        assert (np.abs(ew[nRest-1] - 0.0) < 1e-8)
        assert (np.abs(ew[nRest] - 1.0) < 1e-8)
        CNewOrb1 = np.dot(COrb1, ev[:,:nRest])
        return SemiCanonicalize(CNewOrb1, Fock, S1, Name, Print=False)

    CNewClo = MakeInactiveSpace("NewClo", CActOcc, COcc)
    CNewExt = MakeInactiveSpace("NewVir", CActVir, CVir)

    # re-orthogonalize (should be orthogonal already, but just to be sure).
    COrbNew = np.hstack([CAct])
    if 1:
        COrbNew = np.hstack([CNewClo, CAct, CNewExt])
        SMo = mdot(COrbNew.T, S1, COrbNew)
        COrbNew = np.dot(COrbNew, MakeSmh(SMo))

    print("    Number of Core Orbitals        {} ".format(CNewClo.shape[1]))
    print("    Number of Virtual Orbitals     {} ".format(CNewExt.shape[1]))
    print("    Total Number of Orbitals       {} ".format(COrbNew.shape[1]))

    return CNewClo.shape[1],CAct.shape[1],CNewExt.shape[1],nElec, COrbNew


def rmsd(a, b = None):
    if b is None:
        return np.mean(a.flatten()**2)**.5
    else:
        return rmsd(a - b)


def MakeSmh(S):
    ew,ev = np.linalg.eigh(S)
    assert (np.all(ew > 1e-10))
    v = ev * (ew**-0.25)[np.newaxis,:]
    return np.dot(v, v.T)



def MakeIaosRaw(COcc, S1, S2, S12):
    # calculate the molecule-intrinsic atomic orbital (IAO) basis
    # ref: [1] Knizia, J. Chem. Theory Comput., http://dx.doi.org/10.1021/ct400687b
    # This is the "Simple/2014" version from ibo-ref at sites.psu.edu/knizia/software
    assert (S1.shape[0] == S1.shape[1] and S1.shape[0] == S12.shape[0])
    assert (S2.shape[0] == S2.shape[1] and S2.shape[0] == S12.shape[1])
    assert (COcc.shape[0] == S1.shape[0] and COcc.shape[1] <= S2.shape[0])
    P12 = la.solve(S1, S12)   # P12 = S1^{-1} S12
    COcc2 = mdot(S12.T, COcc)              # O(N m^2)
    CTil = la.solve(S2, COcc2)             # O(m^3)
    STil = mdot(COcc2.T, CTil)             # O(m^3)
    CTil2Bar = la.solve(STil, CTil.T).T    # O(m^3)
    T4 = COcc - mdot(P12, CTil2Bar)        # O(N m^2)
    CIb = P12 + mdot(T4, COcc2.T)          # O(N m^2)
    return CIb





def GetPzOrientation(iTargetAtoms, Coords_, Elements_):
    # make a xyz vector pointing out of the plane containing the target atoms.
    assert (Coords_.shape[1] == 3)
    Coords = Coords_[iTargetAtoms,:].copy()
    Elements = Elements_[iTargetAtoms]

    # get center of mass coordinates
    vCom = np.mean(Coords, axis=0)
    Coords -= vCom

    I = np.zeros((3,3))
    for vCoord in Coords:
        I += np.outer(vCoord, vCoord)
    # diagonalize inertial-like tensor; large eigenvalues first.
    ew,ev = np.linalg.eigh(-I)
    ew *= -1

    print("Target Atom List:")
    print("[{}]".format(', '.join('{}'.format(k) for k in iTargetAtoms)))

    print("Elements        :")
    print("[{}]".format(', '.join('{}'.format(k) for k in Elements)))


    # direction of smallest spatial extend (last one) should be
    # the one indicating the pz-orbital direction.
    vPz = ev[:,2]
    return vPz


def FindValenceAoIndices(iAt, Shells, TargetL):
    ipxyz = None

    iFn0 = 0
    for Atom in range(len(Shells)):
        for AtomL in range(len(Shells[Atom])):
            if Atom == iAt:
                if Shells[Atom][AtomL][0] == TargetL:
                    # this is a generally contracted p-shell on atom iAt.
                    assert (ipxyz is None) # should be only one per atom, I think.
                    nCGTO = Shells[Atom][AtomL][1]
                    nSphComp = 2*TargetL + 1
                    iFnHighestP = iFn0 + nSphComp*(nCGTO-1)
                    ipxyz = np.array([(iFnHighestP + o) for o in range(nSphComp)])
            iFn0 += Shells[Atom][AtomL][2]
    assert (ipxyz is not None)
    return ipxyz



def MakePzMinaoVectors(iTargetAtoms, vPz, Shells):
    # nIb = len(Shells)*len(Shells[Atom])*len( Shells[AtomShell][AtomL])
    nIb=0
    for Atom in range(len(Shells)):
        for AtomL in range(len(Shells[Atom])):
            nIb += Shells[Atom][AtomL][2]
    nVec = len(iTargetAtoms)

    AoVecs = np.zeros((nIb, nVec))
    for (iVec, iAt) in enumerate(iTargetAtoms):
        ipxyz = FindValenceAoIndices(iAt, Shells, TargetL = 1)
        AoVec = np.zeros(nIb)
        AoVec[ipxyz] = vPz

        # each atom brings one pz orbital -> iVec = iiAt
        AoVecs[:,iVec] = AoVec
    return AoVecs


# table of numbers of pi electrons each main group element
# typically brings into a pi-system
_NumPiElec = {
    "C": 1,
    "B": 0,
    "N1e": 1,
    "N2e": 2,
    "O1e": 1,
    "O2e": 2,
    "F": 2,
    "Si":1,
    "P1e": 1,
    "P2e": 2,
    "S1e": 1,
    "S2e": 2,
    "C2e": 2,
}


_CovalentRadii = {1: 38.0, 2: 32.0, 3: 134.0, 4: 90.0, 5: 82.0,
                  6: 77.0, 7: 75.0, 8: 73.0, 9: 71.0, 10: 69.0,
                  11: 154.0, 12: 130.0, 13: 118.0, 14: 111.0, 15: 106.0,
                  16: 102.0, 17: 99.0, 18: 97.0, 19: 196.0, 20: 174.0,
                  21: 144.0, 22: 136.0, 23: 125.0, 24: 127.0, 25: 139.0,
                  26: 125.0, 27: 126.0, 28: 121.0, 29: 138.0, 30: 131.0,
                  31: 126.0, 32: 122.0, 33: 119.0, 34: 116.0, 35: 114.0,
                  36: 110.0, 37: 211.0, 38: 192.0, 39: 162.0, 40: 148.0,
                  41: 137.0, 42: 145.0, 43: 156.0, 44: 126.0, 45: 135.0,
                  46: 131.0, 47: 153.0, 48: 148.0, 49: 144.0, 50: 141.0,
                  51: 138.0, 52: 135.0, 53: 133.0, 54: 130.0, 55: 225.0,
                  56: 198.0, 57: 169.0, 58: None, 59: None, 60: None,
                  61: None, 62: None, 63: None, 64: None, 65: None,
                  66: None, 67: None, 68: None, 69: None, 70: None,
                  71: 160.0, 72: 150.0, 73: 138.0, 74: 146.0, 75: 159.0,
                  76: 128.0, 77: 137.0, 78: 128.0, 79: 144.0, 80: 149.0,
                  81: 148.0, 82: 147.0, 83: 146.0, 84: None, 85: None,
                  86: 145.0, 87: None, 88: None, 89: None, 90: None,
                  91: None, 92: None, 93: None, 94: None, 95: None,
                  96: None, 97: None, 98: None, 99: None, 100: None,
                  101: None, 102: None, 103: None, 104: None, 105: None,
                  106: None, 107: None, 108: None, 109: None, 110: None,
                  111: None, 112: None, 113: None, 114: None, 115: None, 116: None}
# ^- embedded now to remove dependency on finding .txt path.


def GetCovalentRadius(At):
    # get covalent radius in pm
    rcov = _CovalentRadii[At.iElement]
    assert (rcov is not None)
    # convert to bohr radii
    ToAng = 0.52917721092
    return (0.01 * rcov)/ToAng

def GetNumPiElec(iAt, Elements,Coords):
    # deal with C, F, B, Si, and the atoms which have explicit tages assigned.
    At = Elements[iAt]
    nPiTab = _NumPiElec.get(At, None)
    if nPiTab is not None:
        return nPiTab
    print(" ...determining formal number of pi-electrons for {} (not in table).".format(At.Label))

    # find number of bonded partners
    iBonded = []
    for (jAt, AtJ) in enumerate(Elements):
        if iAt == jAt:
            continue
        rij = np.sum((Coords[At] - Coords[AtJ])**2)**.5
        if rij < 1.3 * (GetCovalentRadius(At) + GetCovalentRadius(AtJ)):
            iBonded.append(jAt)

    nBonds = len(iBonded)
    if At in ["N", "P"]:  # 5 valence electrons
        if nBonds == 2:
            # two sigma bonds (1e each) + 1 sigma lone pair (2e) -> 1e left for pi system
            return 1
        elif nBonds == 3:
            # three sigma bonds (1e each) -> 2e left for pi system
            return 2

    if At in ["O","S"]:  # 6 valence electrons
        if nBonds == 1:
            # one sigma bond (1e) + 2 lone pairs in x/y planes (2e each) -> 1e left for pi system
            return 1
        elif nBonds == 3:
            # two sigma bond (1e each) + 1 lone pairs in x/y planes (2e) -> 2e left for pi system
            return 2
    raise Exception("GetNumPiElec({}): {} with {} bonds not tabulated :(".format(iAt, At,  nBonds))


def SemiCanonicalize(COut, Fock, S1, Name, Print=True):
    nOrb = COut.shape[1]
    SOrb = mdot(COut.T, S1, COut)
    fErr = rmsd(SOrb - np.eye(nOrb))
    if fErr > 1e-8:
        print("    SemiCan: Orbital deviation from orthogonality  {:8.2e} \n".format(fErr))
        raise Exception("Orbitals not orthogonal.")
    FockMo = mdot(COut.T, Fock, COut)
    ew,ev = np.linalg.eigh(FockMo)
    if Print:
        print("    Semi-canonicalized orbital energies ({} subspace)".format(Name))
        print("    [{}]".format(', '.join('{:.4f}'.format(k) for k in ew)))
    return np.dot(COut, ev)


def MakeOverlappingOrbSubspace(Space, Name, COrb, nOrbExpected,  CTargetIb, S1, Fock):
    print("\n -- Constructing MO subspace space {}/{}.".format(Space, Name))
    STargetIb = mdot(CTargetIb.T, S1, CTargetIb)
    SmhTargetIb = MakeSmh(STargetIb)

    STargetIbVir = mdot(SmhTargetIb, CTargetIb.T, S1, COrb)
    U,sig,Vt = np.linalg.svd(STargetIbVir, full_matrices=False)

    print("    S[{},{}] singular Values (n={}, nThr={})".format(Space, Name, len(sig), nOrbExpected))
    print("    [{}]".format(', '.join('{:.4f}'.format(k) for k in sig)))
    print("    Sigma[{}]                         {:.4f} (should be 1)".format(nOrbExpected-1, sig[nOrbExpected-1]))
    if nOrbExpected < len(sig):
        print("    Sigma[{}]                         {:.4f} (should be 0)".format(nOrbExpected, sig[nOrbExpected]))
        if sig[nOrbExpected-1] < 0.8 or sig[nOrbExpected] > 0.5:
            raise Exception("{} orbital construction okay?".format(Space))

    V = Vt.T
    COut = np.dot(COrb, V[:,:nOrbExpected])
    return SemiCanonicalize(COut, Fock, S1, Name)



# Use these to add 'stuff' to the proto-active space.
def MakePiSystemOrbitals(TargetName, iTargetAtomsForPlane_,
                         iTargetAtomsForBasis_,Elements,Coords, CIb, Shells,
                         S1, S12, S2, Fock, COcc, CVir):
    print("\n *** TARGET: {}\n".format(TargetName))
    # convert from 1-based atom indices to 0-based indices.
    iTargetAtomsForPlane = np.array(iTargetAtomsForPlane_) - 1
    if iTargetAtomsForBasis_ is None:
        iTargetAtomsForBasis = iTargetAtomsForPlane
    else:
        iTargetAtomsForBasis = np.array(iTargetAtomsForBasis_) - 1

    vPz = GetPzOrientation(iTargetAtomsForPlane, Coords, Elements)
    CAoMix = MakePzMinaoVectors(iTargetAtomsForBasis, vPz, Shells)

    nPiElec = 0
    for iAt in iTargetAtomsForBasis:
        nPiElec += GetNumPiElec(iAt, Elements,Coords)

    print("    Formal number of elec in pi-system      {} ".format(nPiElec))

    if (nPiElec % 2) != 0:
        raise Exception("Got {} pi electrons. Shouldn't this be an even number?".format(nPiElec))

    # idea: make full pi system for target AOs (via IAOs)
    # then we do know how many occupied and virtual pi
    # orbitals there are supposed to be (by counting the contributed pi
    # electrons from the constituent main group atoms)

    # we then isolate those, canonicalize them, and take
    # the frontier MO subset.

    CTargetIb = np.dot(CIb, CAoMix)
    nTargetIb = CAoMix.shape[1]
    print("    Number of target MINAO basis fn      {} ".format(nTargetIb))

    print("    size of CIb1          {} ".format(CIb.shape[0]))
    print("    size of CAomix1       {} ".format(CAoMix.shape[0]))
    print("    size of CIb2          {} ".format(CIb.shape[1]))
    print("    size of CAomix2       {} ".format(CAoMix.shape[1]))


    nOccOrbExpected=nPiElec//2
    nVirtOrbExpected=nTargetIb - nPiElec//2
    CPiOcc = MakeOverlappingOrbSubspace("Pi", "Occ", COcc, nOccOrbExpected,   CTargetIb, S1, Fock)
    CPiVir = MakeOverlappingOrbSubspace("Pi", "Vir", CVir, nVirtOrbExpected,   CTargetIb, S1, Fock)
    return CPiOcc, CPiVir,nOccOrbExpected,nVirtOrbExpected
