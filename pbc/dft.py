def _gen_qv(ngs):
    '''
    integer cube of indices, 0...ngs-1 along each direction
    ngs: [ngsx, ngsy, ngsz]

    Returns 
         3 * (ngsx*ngsy*ngsz) matrix
         [0, 0, ... ngsx-1]
         [0, 0, ... ngsy-1]
         [0, 1, ... ngsz-1]
    '''
    return np.array(list(np.ndindex(tuple(ngs)))).T

def setup_uniform_grids(cell):
    '''
    Real-space AO uniform grid, following Eq. (3.19) (MH)
    '''
    gs=cell.gs
    ngs=2*gs+1
    qv=_gen_qv(ngs)
    invN=np.diag(1./np.array(ngs))
    R=np.dot(np.dot(cell.h, invN), qv)
    coords=R.T.copy() # make C-contiguous with copy() for pyscf
    return coords

class Grids(object):
    def __init__(self, cell):
        self.cell = cell
        self.coords = None
        self.weights = None
        
    def setup_grids(self, cell=None):
        return self.setup_grids_(cell)

    def setup_grids_(self, cell=None):
        if cell is None: cell = self.cell

    def dump_flags(self):
        pass

class RKS(pyscf.dft.rks.RKS):
    def __init__(self.cell):
        self.cell=cell
        pyscf.dft.rks.__init__(self,cell.mol)
        
        self.grids=setup_uniform_grids(cell,gs)

    pass
