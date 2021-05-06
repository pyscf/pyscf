import numpy as np
import pyscf
import pyscf.pbc

def make_graphite(a=2.461, c=6.708, atoms=["C", "C", "C", "C"]):
    """a = 2.461 A , c = 6.708 A"""
    amat = np.asarray([
            [a/2, -a*np.sqrt(3.0)/2, 0],
            [a/2, +a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [0,     0,      1.0/4],
        [2.0/3, 1.0/3,  1.0/4],
        [0,     0,      3.0/4],
        [1.0/3, 2.0/3,  3.0/4]])
    coords = np.dot(coords_internal, amat)
    atom = [(atoms[i], coords[i]) for i in range(4)]
    return amat, atom

def graphite(a=2.461, c=6.708, basis="gth-dzv"):
    cell = pyscf.pbc.gto.Cell()
    cell.a, cell.atom = make_graphite(a=a, c=c)
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.verbose = 10
    cell.build()
    return cell
