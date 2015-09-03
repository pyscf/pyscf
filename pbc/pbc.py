import math
import numpy as np
import scipy.linalg
import pyscf.gto.mole

pi=math.pi

class Cell:
    def __init__(self, mol, h):
        self.mol = mol
        self.h=h
        self.vol=scipy.linalg.det(h)

def get_gv(gs):
    '''
    integer cube of indices, -gs...gs along each direction
    gs: [gsx, gsy, gsz]
    
    Returns 
        [[-gsx, -gsx+1, ..., gsx],
         [-gsy  -gsy+1, ..., gsy],
         [-gsz, -gsz+1, ..., gsz]]
    '''
    ngs=tuple(2*np.array(gs)+1)
    gv=np.array(list(np.ndindex(ngs)))
    gv-=np.array(gs)

    return gv.T

def get_Gv(cell, gv):
    '''
    cube of G vectors (Eq. (3.8), HM),

    Returns 
        np.array([3, ngs], np.complex128)
    '''
    invhT=scipy.linalg.inv(cell.h.T)
    Gv=2*pi*np.dot(invhT,gv)

    return Gv

def get_SI(cell, Gv):
    '''
    structure factor (Eq. (3.34), HM)

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

    This implements Eq. (3.46), Eq. (3.47) in HM, but 
    has only a single Ewald length (Rc is same for all ions).
    This is equivalent to the formulation in Martin, App. F.3,
    and should be equivalent to the uniform background formulae of martin, App. F.2

    ewrc : Ewald length (Rc param in HM)
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


def test():
    from pyscf import gto
    from pyscf.dft import rks

    mol = gto.Mole()
    mol.verbose = 7
    mol.output = '/dev/null'#'out_rks'

    mol.atom.extend([['He', (0.,0.,0.)], ])
    mol.basis = { 'He': 'cc-pvdz'}
    mol.build()
    m = rks.RKS(mol)
    m.xc = 'LDA,VWN_RPA'
    m.xc = 'b3lyp'
    print(m.scf()) # -2.90705411168
    
    cell=Cell(mol, np.eye(3))
    gs=np.array([2,2,2])
    gv=get_gv(gs)
    print gv

    Gv=get_Gv(cell, gv)
    print Gv
    print Gv[:,0]
    print Gv[:,-1]

    SI=get_SI(cell, Gv)
    print SI.shape

    ewrc=0.5
    ewcut=(10,10,10)
    ew_rs=ewald_rs(cell, ewrc, ewcut)
    
    print ew_rs
