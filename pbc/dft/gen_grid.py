import numpy as np
from pyscf.lib.numpy_helper import cartesian_prod

def gen_uniform_grids(cell):
    '''Generate a uniform real-space grid consistent w/ samp thm; see MH (3.19).

    Args:
        cell : instance of :class:`Cell`

    Returns:
        coords : (ngx*ngy*ngz, 3) ndarray
            The real-space grid point coordinates.
        
    '''
    ngs = 2*np.asarray(cell.gs)+1
    qv = cartesian_prod([np.arange(x) for x in ngs])
    invN = np.diag(1./ngs)
    coords = np.dot(qv, np.dot(cell.lattice_vectors(), invN).T)
    return coords

class UniformGrids(object):
    '''Uniform Grid class.'''

    def __init__(self, cell):
        self.cell = cell
        self.coords = None
        self.weights = None
        self.stdout = cell.stdout
        self.verbose = cell.verbose

    def setup_grids_(self, cell=None):
        if cell == None: cell = self.cell

        self.coords = gen_uniform_grids(self.cell)
        self.weights = np.ones(self.coords.shape[0]) 
        self.weights *= cell.vol()/self.weights.shape[0]

        return self.coords, self.weights

    def dump_flags(self):
        logger.info(self, 'Uniform grid')

    def kernel(self, cell=None):
        self.dump_flags()
        return self.setup_grids()

