import math
import numpy as np
import scipy.linalg
import pyscf.gto.mole
import pyscf.dft.numint


pi=math.pi

'''PBC module. Notation follows Marx and Hutter (MH), "Ab Initio Molecular Dynamics"'''

class Cell:
    def __init__(self, **kwargs):
        self.mol = None
        self.h = None
        self.vol = 0.
        self.gs = []
        self.Ns = []

def get_gv(gs):
    '''
    integer cube of indices, -gs...gs along each direction
    gs: [gsx, gsy, gsz]
    
    Returns 
         3 * (gsx*gsy*gsz) matrix
         [-gsx, -gsx,   ..., gsx]
         [-gsy, -gsy,   ..., gsy]
         [-gsz, -gsz+1, ..., gsz]
    '''
    ngs=tuple(2*np.array(gs)+1)
    gv=np.array(list(np.ndindex(ngs)))
    gv-=np.array(gs)

    return gv.T

def get_Gv(cell, gv):
    '''
    cube of G vectors (Eq. (3.8), MH),

    Returns 
        np.array([3, ngs], np.complex128)
    '''
    invhT=scipy.linalg.inv(cell.h.T)
    Gv=2*pi*np.dot(invhT,gv)

    return Gv

def get_SI(cell, Gv):
    '''
    structure factor (Eq. (3.34), MH)

    Returns 
        np.array([natm, ngs], np.complex128)
    '''
    ngs=Gv.shape[1]
    mol=cell.mol
    SI=np.empty([mol.natm, ngs], np.complex128)
    mol=cell.mol

    for ia in range(mol.natm):
        SI[ia,:]=np.exp(-1j*np.dot(Gv.T, mol.atom_coord(ia)))

    return SI

def ewald_rs(cell, ewrc, ewcut):
    '''
    Real-space Ewald sum 

    This implements Eq. (3.46), Eq. (3.47) in MH, but 
    has only a single Ewald length (Rc is same for all ions).
    This is equivalent to the formulation in Martin, App. F.3,
    and should be equivalent to the uniform background formulae of martin, App. F.2

    ewrc : Ewald length (Rc param in MH)
    ewcut : [ewcutx, ewcuty, ewcutz]

    Returns
        float
    '''
    mol=cell.mol
    chargs = [mol.atom_charge(i) for i in range(len(mol._atm))]
    coords = [mol.atom_coord(i) for i in range(len(mol._atm))]

    ewovrl = 0.
    for (ix, iy, iz) in np.ndindex(ewcut):
        L=ix*cell.h[:,0]+iy*cell.h[:,1]+iz*cell.h[:,2]

        # prime in summation to avoid self-interaction in unit cell
        for ia in range(mol.natm):
            qi = chargs[ia]
            ri = coords[ia]

            if ix == 0 and iy == 0 and iz == 0:
                for ja in range(ia):
                    qj = chargs[ja]
                    rj = coords[ja]
                    r = np.linalg.norm(ri-rj)
                    ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
            else:
                for ja in range(mol.natm):
                    qj=chargs[ja]
                    rj=coords[ja]

                    r=np.linalg.norm(ri-rj+L)
                    ewovrl += qi * qj / r * math.erfc(r / math.sqrt(2*ewrc**2))
        
    ewself = 1./(math.sqrt(2*pi) * ewrc) * np.dot(chargs,chargs)
    
    return ewovrl - ewself

def gen_qv(Rs):
    '''
    integer cube of indices, 0...Ns-1 along each direction
    Ns: [Nsx, Nsy, Nsz]

    Returns 
         3 * (Nsx*Nsy*Nsz) matrix
         [0, 0, ... Nsx-1]
         [0, 0, ... Nsy-1]
         [0, 1, ... Nsz-1]
    '''
    return np.array(list(np.ndindex(tuple(Rs)))).T

def setup_ao_grids(cell, qv):
    '''
    Real-space AO uniform grid, following Eq. (3.19) (MH)
    '''
    mol=cell.mol
    invN=np.diag(1./np.array(cell.Ns))
    R=np.dot(np.dot(cell.h, invN), qv)
    coords=R.T.copy() #pyscf notation for grids (also make C-contiguous with copy)
    weights=np.ones(coords.shape)
    return coords, weights

def get_aoR(cell, coords):
    aoR=pyscf.dft.numint.eval_ao(cell.mol, coords)
    return aoR

def get_rhoR(cell, aoR, dm):
    rhoR=pyscf.dft.numint.eval_rho(cell.mol, aoR, dm)
    return rhoR

def get_rhoK(rhoR, coords):
    pass


def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    L=40
    h=np.eye(3.)*L

    mol.atom.extend([['He', (L/2.,L/2.,L/2.)], ])
    mol.basis = { 'He': 'sto-3g'}
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=Cell()
    cell.mol=mol
    cell.h=h
    cell.gs=[2,2,2]
    cell.Ns=[300,300,300]
    cell.vol=scipy.linalg.det(cell.h)
    
    gv=get_gv(cell.gs)
    Gv=get_Gv(cell, gv)

    SI=get_SI(cell, Gv)

    ewrc=0.5
    ewcut=(10,10,10)
    ew_rs=ewald_rs(cell, ewrc, ewcut)

    qv=gen_qv(cell.Ns)
    coords, weights=setup_ao_grids(cell, qv)

    aoR=get_aoR(cell, coords)
    dm=m.make_rdm1()
    rhoR=get_rhoR(cell, aoR, dm)

    print cell.vol
    print rhoR.shape
    # should be 2.0, gets 1.99004869382. With 27 million pts!!
    print cell.vol/len(rhoR)*np.sum(rhoR) 

    
