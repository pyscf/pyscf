'''
Self-consistent continuum solvation (SCCS) model
'''
import numpy
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks as pbc_grad
from pyscf.solvent import sccs

boxlen=8.0
cell=pbcgto.Cell()
cell.a=numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom=[["O",          [5.84560,        5.21649,        5.10372]],
           ["H",          [6.30941,        5.30070,        5.92953]],
           ["H",          [4.91429,        5.26674,        5.28886]]]
cell.basis='gth-dzv'
cell.ke_cutoff=420
cell.max_memory=4000
cell.precision=1e-8
cell.pseudo='gth-pade'
cell.verbose=4
cell.rcut_by_shell_radius=True
cell.build()

mf=pbcdft.RKS(cell)
mf.xc = 'pbe,pbe'
mf.with_df = multigrid.MultiGridFFTDF2(cell)
sccs_obj = sccs.SCCS(cell, mf.with_df.mesh, eps=78.3553)
mf.with_df.sccs = sccs_obj
mf.kernel()

grad = pbc_grad.Gradients(mf)
g = grad.kernel()

# reference from finite difference calculation
g0 = numpy.asarray([[-0.01533471,  0.00496358,  0.03351879],
                    [-0.01952710, -0.00288006, -0.02916494],
                    [ 0.03495903, -0.00155499, -0.00415536]])
print(abs(g-g0).max())
