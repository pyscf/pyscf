from abc import ABC, abstractmethod
import numpy as np
from multimethod import multimethod
from pyscf.pbc.symm import geom
from pyscf.pbc.symm import symmetry

class GroupElement(ABC):
    '''
    Group element
    '''
    def __call__(self, other):
        return product(self, other)

    def __matmul__(self, other):
        return product(self, other)

class FiniteGroup():
    '''
    Finite group
    '''
    def __init__(self, elements):
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

class PGElement(GroupElement):
    '''
    Point group element
    '''
    def __init__(self, matrix):
        self.matrix = matrix
        self.dimension = matrix.shape[0]

    def __repr__(self):
        return self.matrix.__repr__()

    def __hash__(self):
        rot = self.matrix.flatten()
        r = 0
        size = self.dimension ** 2
        for i in range(size):
            r += 3**(size-1-i) * (rot[i] + 1)
        return int(r)

    def __lt__(self, other):
        if not isinstance(other, PGElement):
            raise TypeError(f"{other} is not a point group element.")
        return self.__hash__() < other.__hash__()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, a):
        self._matrix = a

    rot = matrix

    def inv(self):
        return PGElement(np.asarray(np.linalg.inv(self.matrix), dtype=np.int32))

class PointGroup(FiniteGroup):
    '''
    Crystallographic point group
    '''
    @property
    def group_name(self):
        return geom.get_crystal_class(None, self.elements)[0]

    def lookup_table(self):
        '''
        Mappings between elements' hash values and their indices.
        '''
        return {hash(g) : i for i, g in enumerate(self.elements)}

    def inverse_table(self):
        '''
        Inverse of the elements. Returns the indices.
        '''
        n = len(self)
        inv_table = np.zeros((n,), dtype=np.int32)
        lookup = self.lookup_table()
        for i, g in enumerate(self.elements):
            inv_table[i] = lookup[hash(g.inv())]
        return inv_table

    def multiplication_table(self):
        '''
        Multiplication table of the group. Returns the indices of elements.
        '''
        n = len(self)
        prod_table = np.zeros((n,n), dtype=np.int32)
        lookup = self.lookup_table()
        for i, g in enumerate(self.elements):
            for j, h in enumerate(self.elements):
                prod = g @ h
                prod_table[i,j] = lookup[hash(prod)]
        return prod_table

    def conjugacy_table(self):
        '''
        conjugacy_table[idx_g, idx_h] gives index of :math:`h^{-1} g h`.
        '''
        inv_table = self.inverse_table()
        prod_table = self.multiplication_table()
        ginv_h = prod_table[inv_table]
        idx = np.arange(len(self))[None, :]
        return ginv_h[ginv_h, idx]

    def conjugacy_classes(self):
        '''
        Conjugacy classes. Unsorted
        '''
        n = len(self)
        idx = np.arange(len(self))[:, None]
        is_conjugate = np.zeros((n,n), dtype=np.int32)
        is_conjugate[idx, self.conjugacy_table()] = 1

        classes, representatives, inverse = np.unique(is_conjugate, axis=0, return_index=True, return_inverse=True)
        return classes, representatives, inverse

    def character_table_by_class(self):
        '''
        Character table of the group.
        '''
        classes, _, _ = self.conjugacy_classes()
        class_sizes = classes.sum(axis=1)

        inv_table = self.inverse_table()
        prod_table = self.multiplication_table()
        ginv_h = prod_table[inv_table]
        M = classes @ np.random.rand(len(self))[ginv_h] @ classes.T
        M /= class_sizes

        _, table = np.linalg.eig(M)
        table = table.T / class_sizes

        norm = np.sum(np.abs(table) ** 2 * class_sizes, axis=1, keepdims=True) ** 0.5
        table /= norm
        table /= (table[:, 0] / np.abs(table[:, 0]))[:, np.newaxis]  # ensure correct sign
        table *= len(self) ** 0.5

        table[np.isclose(table, 0, atol=1e-9)] = 0
        return table

    def character_table(self):
        '''
        Character of each element.
        '''
        _, _, inverse = self.conjugacy_classes()
        CT = self.character_table_by_class()
        return CT[:, inverse]

    '''
    def irrep(self):
        true_product_table = self.multiplication_table()
        inverted_product_table = true_product_table[:, self.inverse_table()]

        def invariant_subspaces(e, seed):
            e = e[inverted_product_table]
            e = e + e.T.conj()
            e, v = np.linalg.eigh(e)
            _, starting_idx = np.unique(e, return_index=True)
            vs = v[:, starting_idx]
            s = np.random.rand(len(self))[inverted_product_table]
            proj = self.character_table().conj() @ s @ vs
            starting_idx = list(starting_idx) + [len(self)]
            return v, starting_idx, proj

        squares = np.diag(true_product_table)
        frob = np.array(
            np.rint(
                np.sum(self.character_table()[:, squares], axis=1).real / len(self)
            ),
            dtype=int,
        )
        eigen = {}
        if np.any(frob == 1):
            e = np.random.rand(len(self))
            eigen["real"] = invariant_subspaces(e, seed=1)
        if np.any(frob != 1):
            raise
            #e = random(len(self), seed=2, cplx=True)
            #eigen["cplx"] = invariant_subspaces(e, seed=3)

        irreps = []
        for i, chi in enumerate(self.character_table()):
            v, idx, proj = eigen["real"] if frob[i] == 1 else eigen["cplx"]
            proj = np.logical_not(np.isclose(proj[i], 0.0))
        for i, chi in enumerate(self.character_table()):
            v, idx, proj = eigen["real"] if frob[i] == 1 else eigen["cplx"]
            proj = np.logical_not(np.isclose(proj[i], 0.0))
            first = np.arange(len(idx) - 1, dtype=int)[proj][0]
            v = v[:, idx[first] : idx[first + 1]]
            irreps.append(np.einsum("gi,ghj ->hij", v.conj(), v[true_product_table, :]))

        return irreps
    '''

@multimethod
def product(g : PGElement, h : PGElement):
    return PGElement(np.dot(g.matrix, h.matrix))

@multimethod # noqa: F811
def product(g : PGElement, h : np.ndarray):
    return np.dot(g.matrix, h)

def symm_adapted_basis(cell):
    sym = symmetry.Symmetry(cell).build(symmorphic=True)
    Dmats = sym.Dmats

    elements = []
    for op in sym.ops:
        assert(op.trans_is_zero)
        elements.append(op.rot)

    elements = np.asarray(elements)
    elements = np.unique(elements, axis=0)
    elements = [PGElement(rot) for rot in elements]

    pg = PointGroup(elements)
    chartab = pg.character_table()
    nirrep = len(chartab)
    nao = cell.nao
    coords = cell.get_scaled_positions()
    atm_maps = []
    for op in sym.ops:
        atm_map, _ = symmetry._get_phase(cell, op, coords, None, ignore_phase=True)
        atm_maps.append(atm_map)
    atm_maps = np.asarray(atm_maps)
    tmp = np.unique(atm_maps, axis=0)
    tmp = np.sort(tmp, axis=0)
    tmp = np.unique(tmp, axis=1)
    eql_atom_ids = []
    for i in range(tmp.shape[-1]):
        eql_atom_ids.append(np.unique(tmp[:,i]))

    aoslice = cell.aoslice_by_atom()
    cbase = np.zeros((nirrep, nao, nao))
    for atom_ids in eql_atom_ids:
        iatm = atom_ids[0]
        op_relate_idx = []
        for iop in range(len(pg)):
            op_relate_idx.append(atm_maps[iop][iatm])
        ao_loc = np.array([aoslice[i,2] for i in op_relate_idx])

        b0, b1 = aoslice[iatm,:2]
        ioff = 0
        icol = aoslice[iatm, 2]
        for ib in range(b0, b1):
            nctr = cell.bas_nctr(ib)
            l = cell.bas_angular(ib)
            if cell.cart:
                degen = (l+1) * (l+2) // 2
            else:
                degen = l * 2 + 1
            for n in range(degen):
                for iop in range(len(pg)):
                    Dmat = Dmats[iop][l]
                    tmp = np.einsum('x,y->xy', chartab[:,iop], Dmat[:,n])
                    idx = ao_loc[iop] + ioff
                    for ictr in range(nctr):
                        cbase[:, idx:idx+degen, icol+n+ictr*degen] += tmp / len(pg)
                        idx += degen
            ioff += degen * nctr
            icol += degen * nctr

    so = []
    for ir in range(nirrep):
        idx = np.where(np.sum(abs(cbase[ir]), axis=0) > 1e-9)[0]
        so.append(cbase[ir][:,idx])

    for ir in range(nirrep):
        norm = np.linalg.norm(so[ir], axis=0)
        so[ir] /= norm[None,:]
    return so

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = [['O' , (1. , 0.    , 0.   ,)],
                 ['H' , (0. , -.757 , 0.587,)],
                 ['H' , (0. , 0.757 , 0.587,)]]
    cell.a = [[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]
    cell.basis = 'ccpvdz'
    cell.verbose = 5
    cell.build()
    so = symm_adapted_basis(cell)

    from pyscf import gto as mol_gto
    from pyscf.symm import geom as mol_geom
    from pyscf.symm.basis import symm_adapted_basis as mol_symm_adapted_basis
    mol = cell.copy()
    gpname, origin, axes = mol_geom.detect_symm(mol._atom)
    atoms = mol_gto.format_atom(cell._atom, origin, axes)
    mol.build(False, False, atom=atoms)
    mol_so = mol_symm_adapted_basis(mol, gpname)[0]

    print(abs(so[0] - mol_so[0]).max())

