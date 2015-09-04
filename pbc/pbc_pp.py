import itertools
import math
import numpy as np
import numpy.fft
import scipy.linalg
import pyscf.gto.mole
import pyscf.dft.numint


pi=math.pi
sqrt=math.sqrt
exp=math.exp

'''PP contributions to PBC module. 
   Notation follows Marx and Hutter (MH), "Ab Initio Molecular Dynamics"
'''

class Cell:
    def __init__(self, **kwargs):
        self.mol = None
        self.h = None
        self.vol = 0.
        # Add a pseudo member (e.g. a list of instances of the PP class)
        self.pseudo = None

def get_vpp(cell, gs):
    '''
    Pseudopotential AO matrix
    '''
    mol=cell.mol

    nuc_coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    coords, weights=setup_ao_grids(cell,gs)
    aoR=get_aoR(cell, coords)

    # The non-divergent part of Vhartree(G=0)+Vloc(G=0)
    nondiv_G0 = 0.
    # Make *some* one-dimensional r vector with the characteristic
    # size and grid-spacing of the unit cell.
    r = STH HERE.

    vlocR=np.zeros(aoR.shape[0])
    #TODO(TCB): I don't like that 6 (# NL terms) is hard-coded
    vnonlocRs=np.zeros(6,aoR.shape[0])
    for ia in range(mol.natm):
        ri = nuc_coords[ia]
        vlocR += cell.pseudo[ia].gth_vloc_r(np.linalg.norm(ri-coords))
        hs, projs_ia = cell.pseudo[ia].gth_vnonloc_r(np.linalg.norm(ri-coords,axis=1))
        vnonlocRs += projs_ia

        # Example way to organize pseudopotentials in Cell
        Zia = cell.pseudo[ia].Zia
        nondiv_G0 += 4*pi*np.vdot( 
            r, (r*gth_vloc_r(np.linalg.norm(r,axis=1)) + Zia) )
    
    # Zero out G=0 from vlocR
    vlocG=fft(vlocR, gs)
    vlocG[0] = 0.
    vlocR=ifft(vlocG, gs)

    nao=aoR.shape[1]
    vpp=np.zeros([nao, nao])
    for i in range(nao):
        for j in range(nao):
            vpp[i,j] = cell.vol/aoR.shape[0]*np.vdot(aoR[:,i],vlocR*aoR[:,j])
            for h, vnonlocR in zip(hs,vnonlocRs):
                vpp[i,j] += ( (cell.vol/aoR.shape[0])**2
                             * h * np.outer(aoR[:,i]*vnonlocR,aoR[:,j]*vnonlocR).sum() )

    return vpp

def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    L=40
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = { 'He': 'cc-pVDZ'}
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=Cell()
    cell.mol=mol
    cell.h=h

    """
    Additional Cell members are here ...
    """
    cell.pseudo = []
    for ia in range(mol.natm):
        atom_pp = PP(mol.atom_charge(ia), mol.atom_coord(ia))
        cell.pseudo.append(atom_pp)
    """
    ... to here.
    """ 

    cell.vol=scipy.linalg.det(cell.h)
    gs=np.array([20,20,20]) # number of G-points in grid. Real-space dim=2*gs+1
    Gv=get_Gv(cell, gs)

    SI=get_SI(cell, Gv)
    ewrc=0.5
    ewcut=(1,1,1)
    ew_rs=ewald_rs(cell, ewrc, ewcut)

    coords, weights=setup_ao_grids(cell, gs)

    aoR=get_aoR(cell, coords)
    print aoR.shape

    dm=m.make_rdm1()

    # print "Kinetic"
    # tao=get_t(cell, gs)
    # print tao

    # tao2 = mol.intor_symmetric('cint1e_kin_sph') 
    # print tao2

    # print "kinetic energies"
    # print np.dot(np.ravel(tao), np.ravel(dm))
    # print np.dot(np.ravel(tao2), np.ravel(dm))
    # sao=get_ovlp(cell,gs)

    # print "Overlap"
    # print sao
    # print m.get_ovlp()
    # print np.dot(np.ravel(sao), np.ravel(dm))
    # print np.dot(np.ravel(m.get_ovlp()), np.ravel(dm))

    print "Coulomb"
    jao=get_j(cell,dm,gs)
    print jao
    print m.get_j(dm)
    print np.dot(np.ravel(dm),np.ravel(m.get_j(dm)))
    print np.dot(np.ravel(dm),np.ravel(jao))

    print "Nuc-el"
    neao=get_nuc(cell,gs)
    print neao
    vne=mol.intor_symmetric('cint1e_nuc_sph') 
    print vne
    print np.dot(np.ravel(dm), np.ravel(vne))
    print np.dot(np.ravel(dm), np.ravel(neao))

    print "Normalization"
    rhoR=get_rhoR(cell, aoR, dm)

    # print cell.vol
    # print rhoR.shape
    # # should be 2.0, gets 1.99004869382. With 27 million pts!!
    print cell.vol/len(rhoR)*np.sum(rhoR) 

    print "total coulomb"
    print ecoul(cell, dm, gs, ewrc, ewcut)
