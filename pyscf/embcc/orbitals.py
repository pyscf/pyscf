"""Module for easier control of orbital spaces."""


from collections import OrderedDict
import itertools
import copy

import numpy as np

__all__ = [
        "Orbitals",
        ]

def _to_indices(obj, n):
    """Convert slice or boolean mask object to list of indices"""
    indices = np.arange(n)[obj]
    return indices

def _complement_indices(obj, n):
    indices = _to_indices(obj, n)
    indices = np.setdiff1d(np.arange(n), indices)
    return indices

class Orbitals:
    """
    Attributes
    ----------
    coeff : ndarray
        Orbital coefficients
    dm1 : ndarray
        One-particle reduced density matrix
    overlap : ndarray
        Overlap matrix of underlying basis.
    spaces : OrderedDict
        Dictionary of defined spaces

    C : shortcut for coeff
    D : shortcut for dm1
    S : shortcut for overlap
    """

    compl_char = "!"

    def __init__(self, coeff, dm1=None, overlap=None):
        self.coeff = coeff.copy()
        self.dm1 = dm1.copy() if dm1 is not None else None
        self.overlap = overlap.copy() if overlap is not None else None
        self.spaces = OrderedDict()

    def __len__(self):
        """Number of orbitals."""
        return self.coeff.shape[-1]

    def define_space(self, name, indices):
        """Add space to orbitals.spaces"""
        if not isinstance(name, str):
            raise TypeError("Please only use str as space names")
        if name[0] == self.compl_char:
            raise ValueError("Cannot start name with %s" % self.compl_char)
        if name in self.spaces:
            raise KeyError("Space with name %s already defined." % name)
        # Convert slices and boolean masks:
        indices = _to_indices(indices, len(self))
        self.spaces[name] = indices

    def delete_space(self, name):
        del self.spaces[name]

    def sort_spaces(self):
        "Sort according to smallest index"
        spaces = sorted(self.spaces.items(), key=lambda x : min(x[1]))
        print("SORTING SPACES")
        print(self.spaces)
        print(spaces)
        self.spaces = OrderedDict(spaces)

    def get_size(self, space):
        """Get number of orbitals in space."""
        indices = self.get_indices(space)
        return len(indices)

    def get_indices(self, space=None):
        """Get indices of space."""

        # All indices
        if space is None:
            return np.arange(len(self))

        if isinstance(space, str):
            space = [space]

        indices = []
        for s in space:
            if s[0] != self.compl_char:
                indices.append(self.spaces[s])
            else:
                indices.append(_complement_indices(self.spaces[s[1:]], len(self)))

        indices = np.hstack(indices)
        return indices

    def get_maxindex(self, space):
        return max(self.get_indices(space))

    def get_minindex(self, space):
        return min(self.get_indices(space))

    def is_ordered(self, space):
        indices = self.get_indices(space)
        return np.all(np.diff(indices) >= 0)

    def is_contiguous(self, space):
        indices = self.get_indices(space)
        return np.all(np.diff(indices) == 1)

    def get_coeff(self, space=None):
        """Get coefficient of space."""
        indices = self.get_indices(space)
        return self.coeff[:,indices]

    def get_occ(self, space=None, dm1=None, overlap=None):
        """Get occupation of orbitals for a given density matrix."""
        if dm1 is None:
            dm1 = self.dm1
        if overlap is None:
            overlap = self.overlap
        sc = np.dot(overlap, self.get_coeff(space))
        dm1 = np.linalg.multi_dot((sc.T, dm1, sc))
        occ = np.diag(dm1)
        return occ

    def get_nelectron(self, space=None, dm1=None, overlap=None):
        occ = self.get_occ(space, dm1=dm1, overlap=overlap)
        return sum(occ)

    def transform(self, t, space=None):
        """Transform orbitals in target space."""
        indices = self.get_indices(space)
        self.coeff[:,indices] = np.dot(self.coeff[:,indices], t)

        return self.coeff[:,indices]

    def reorder(self, new_order, adjust_spaces=True):
        rank = np.argsort(new_order)
        self.coeff = self.coeff[:,new_order]
        if adjust_spaces:
            for space in list(self.spaces.keys()):
                self.spaces[space] = rank[self.spaces[space]]

    def get_spaces(self):
        """Get an array of of lists, indicating which spaces each orbital is a member of."""
        spaces = np.empty((len(self),), dtype=np.object)
        for i in range(len(self)):
            spaces[i] = []
            for space in self.spaces:
                if i in self.get_indices(space):
                    spaces[i].append(space)

        return spaces

    def add_indices_to_space(self, indices, space):
        if np.any(np.isin(indices, self.spaces[space])):
            raise ValueError("Indices %s already contained in space %s" % (indices, self.space[space]))
        self.spaces[space] = np.append(self.spaces[space], indices)

    def remove_indices_from_space(self, indices, space):
        if np.any(np.isin(indices, self.spaces[space], invert=True)):
            raise ValueError("Indices %s not contained in space %s" % (indices, self.space[space]))
        indices = np.argwhere(np.isin(self.spaces[space], indices)).flatten()
        self.spaces[space] = np.delete(self.spaces[space], indices)

    # For convenience

    @property
    def C(self):
        return self.coeff

    @C.setter
    def C(self, value):
        self.coeff = value

    @property
    def S(self):
        return self.overlap

    @S.setter
    def S(self, value):
        self.overlap = value

    @property
    def D(self):
        return self.dm1

    @D.setter
    def D(self, value):
        self.dm1 = value

    def get_dm1(self, space=None, dm1=None):
        if dm1 is None:
            dm1 = self.dm1
        sc = np.dot(self.S, self.get_coeff(space))
        dm1 = np.linalg.multi_dot((sc.T, dm1, sc))
        return dm1

    def are_orthonormal(self, overlap=None, **kwargs):
        """Check if orbitals are orthonormal wrt. overlap matrix."""
        if overlap is None:
            overlap = self.overlap
        if overlap is None:
            csc = np.dot(self.C.T, self.C)
        else:
            csc = np.linalg.multi_dot((self.C.T, overlap, self.C))
        return np.allclose(csc, np.eye(len(self)), **kwargs)

    def copy(self):
        """Create a copy of orbital object."""
        orbitals = Orbitals(self.C, self.D, self.S)
        orbitals.spaces = copy.deepcopy(self.spaces)
        return orbitals

if __name__ == "__main__":

    from pyscf import gto
    from pyscf import scf

    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="6-31g")
    mol.verbose = 4
    hf = scf.RHF(mol)
    hf.kernel()

    orbs = Orbitals(hf.mo_coeff)

    orbs.define_space("occupied", np.s_[:5])
    orbs.define_space("virtual", np.s_[5:])

    c_occ = hf.mo_coeff[:,hf.mo_occ>0]

    c = orbs.get_coeff("occupied")
    assert np.allclose(c, c_occ)

    c = orbs.get_coeff(("occupied",))
    assert np.allclose(c, c_occ)

    c = orbs.get_coeff(("occupied", "virtual"))
    assert np.allclose(c, hf.mo_coeff)

    print(orbs.get_size("occupied"))
    print(orbs.get_size("virtual"))
    print(orbs.get_size("!virtual"))
    print(orbs.get_size("!occupied"))

    assert np.all(orbs.get_indices("occupied") == orbs.get_indices("!virtual"))

    print(orbs.get_indices("occupied"))
    print(orbs.get_indices(("virtual", "occupied")))
    print(orbs.get_indices(("occupied", "occupied")))


    import scipy
    import scipy.stats

    #nocc = 5
    r = scipy.stats.ortho_group.rvs(len(orbs))
    #print(orbs.get_coeff("occupied"))
    #print(hf.mo_coeff[:,hf.mo_occ>0])

    c2 = orbs.transform(("occupied", "virtual"), r)
    print(c2)

    assert np.allclose(c2, np.dot(hf.mo_coeff, r))
    1/0



    indices = orbs.get_indices("occupied")
    #print(np.dot(hf.mo_coeff[:,hf.mo_occ>0], r))
    print(indices)
    print(np.dot(hf.mo_coeff[:,indices], r))
    1/0

    print(orbs.get_coeff("occupied"))
    print(np.dot(hf.mo_coeff[:,hf.mo_occ>0], r))

    assert np.allclose(c2, np.dot(hf.mo_coeff[:,:nocc], r))



    orbs.delete_space("occupied")

    print(orbs.get_size("occupied"))

