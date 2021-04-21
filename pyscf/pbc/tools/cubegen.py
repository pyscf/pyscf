import numpy as np

import pyscf
from pyscf import __config__
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import numint

DEFAULT_NPOINTS = getattr(__config__, "cubegen_npoints", None)
DEFAULT_RESOLUTION = getattr(__config__, "cubegen_resolution", 15)


class CubeFile:

    def __init__(self, cell, filename, nx=DEFAULT_NPOINTS, ny=DEFAULT_NPOINTS, nz=DEFAULT_NPOINTS,
            resolution=DEFAULT_RESOLUTION, comment1=None, comment2=None, origin=(0.0, 0.0, 0.0), fmt="%13.5E",
            crop=None):
        """Initialize a cube file object. Data can be added using the `add_field` and `add_fields` methods.

        This class can also be used as a context manager:

        >>> with CubeFile(cell, "mycubefile.cube") as f:
        >>>     f.add_field(hf.mo_coeff[:,0])
        >>>     f.add_fields(hf.mo_coeff[:,5:10].T)
        >>>     f.add_field(hf.make_rdm1(), kind="density")

        Arguments
        ---------
            cell : pbc.gto.Cell object
            filename : str
                Filename for cube file. Should include ".cube" extension
            nx, ny, nz : int, optional
                Number of grid points in X, Y, and Z direction. If specified,
                they take precedence over `resolution`. Default: None.
            resolution : float, optional
                Resolution in units of 1/Bohr for automatic choice of `nx`,`ny`, and `nz`. Default: 15.0.
            comment1 : str, optional
                First comment line in cube-file.
            comment2 : str, optional
                Second comment line in cube-file.
            origin : array(3), optional
                Origin in X, Y, Z coordinates
            crop : dict, optional
                By default, the coordinate grid will span the entire unit cell. `crop` can be set
                to crop the unit cell. `crop` should be a dictionary with possible keys
                ["a0", "a1", "b0", "b1", "c0", "c1"], where "a0" crops the first lattice vector at the
                start, "a1" crops the first lattice vector at the end, "b0" crops the second lattice vector
                at the start etc. The corresponding values is the distance which should be cropped in
                units of Bohr.
        """

        self.cell = cell
        self.filename = filename
        if resolution < 1:
            logger.warn(cell, "Warning: resolution is below 1/Bohr. Recommended values are >5/Bohr")
        self.a = self.cell.lattice_vectors().copy()
        self.origin = np.asarray(origin)
        if crop is not None:
            a = self.a.copy()
            norm = np.linalg.norm(self.a, axis=1)
            a[0] -= (crop.get("a0", 0) + crop.get("a1", 0)) * self.a[0]/norm[0]
            a[1] -= (crop.get("b0", 0) + crop.get("b1", 0)) * self.a[1]/norm[1]
            a[2] -= (crop.get("c0", 0) + crop.get("c1", 0)) * self.a[2]/norm[2]
            self.origin += crop.get("a0", 0)*self.a[0]/norm[0]
            self.origin += crop.get("b0", 0)*self.a[1]/norm[1]
            self.origin += crop.get("c0", 0)*self.a[2]/norm[2]
            self.a = a

        self.nx = (nx or min(np.ceil(abs(self.a[0,0]) * resolution).astype(int), 200))
        self.ny = (ny or min(np.ceil(abs(self.a[1,1]) * resolution).astype(int), 200))
        self.nz = (nz or min(np.ceil(abs(self.a[2,2]) * resolution).astype(int), 200))
        self.comment1 = comment1 or "<title>"
        self.comment2 = comment2 or ("Generated with PySCF v%s" % pyscf.__version__)
        self.fmt = fmt
        self.coords = self.get_coords()

        self.fields = []

    def get_coords(self):
        xs = np.arange(self.nx) / (self.nx-1)
        ys = np.arange(self.ny) / (self.ny-1)
        zs = np.arange(self.nz) / (self.nz-1)
        coords = lib.cartesian_prod([xs, ys, zs])
        coords = np.dot(coords, self.a)
        coords = np.asarray(coords, order="C") + self.origin
        return coords

    @property
    def ncoords(self):
        return self.nx*self.ny*self.nz

    @property
    def nfields(self):
        return len(self.fields)

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write()


    def add_field(self, data, kind="orbital", dset_idx=None):
        """Add a field to be included in the cube file.

        Arguments
        ---------
            data : (N) or (N, N), array
                Orbital coefficients or density matrix.
            kind : ["orbital", "density"], optional
                Type of field. Default: "orbital".
            dset_ids : int, optional
                Dataset index.
        """
        if kind not in ("orbital", "density", "mep"):
            raise ValueError("Unknown value for kind= %s" % kind)
        # TODO
        if kind == "mep": raise NotImplementedError()
        self.fields.append((data, kind, dset_idx))

    def add_fields(self, data, kind="orbital", **kwargs):
        """Add multiple fields to be included in the cube file.

        Arguments
        ---------
            data : (N, M) or (M, N, N), array
                Array of Orbital coefficients or density matrices.
                Note that for kind="orbital", the last dimension labels the orbitals,
                however, for kind="density" the first dimension labels different density matrices.
            kind : ["orbital", "density"], optional
                Type of field. Default: "orbital".
            dset_ids : int, optional
                Dataset index.
        """
        if kind == "orbital": data = data.T
        for dat in data:
            self.add_field(dat, kind=kind, **kwargs)

    def write(self, filename=None):
        filename = filename or self.filename
        # Get dataset IDs
        dset_ids = []
        for (field, ftype, fid) in self.fields:
            if fid is None:
                if dset_ids:
                    fid = np.max(dset_ids)+1
                else:
                    fid = 1
            dset_ids.append(fid)

        self.write_header(filename, dset_ids=dset_ids)
        self.write_fields(filename)

    def write_header(self, filename, dset_ids=None):
        if self.nfields > 1 and dset_ids is None:
            dset_ids = range(1, self.nfields+1)
        with open(filename, "w") as f:
            f.write("%s\n" % self.comment1)
            f.write("%s\n" % self.comment2)
            if self.nfields > 1:
                f.write("%5d" % -self.cell.natm)
            else:
                f.write("%5d" % self.cell.natm)
            f.write("%12.6f%12.6f%12.6f" % tuple(self.origin))
            if self.nfields > 1:
                f.write("%5d" % self.nfields)
            f.write("\n")
            # Lattice vectors
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.nx, *(self.a[0]/(self.nx-1))))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.ny, *(self.a[1]/(self.ny-1))))
            f.write("%5d%12.6f%12.6f%12.6f\n" % (self.nz, *(self.a[2]/(self.nz-1))))
            # Atoms
            for atm in range(self.cell.natm):
                sym = self.cell.atom_symbol(atm)
                f.write("%5d%12.6f" % (gto.charge(sym), 0.0))
                f.write("%12.6f%12.6f%12.6f\n" % tuple(self.cell.atom_coords()[atm]))
            # Data set indices
            if self.nfields > 1:
                f.write("%5d" % self.nfields)
                for i in range(self.nfields):
                    f.write("%5d" % dset_ids[i])
                f.write("\n")

    def write_fields(self, filename):
        # Loop over coordinates first, then fields!
        blksize = min(self.ncoords, 8000)
        with open(filename, "a") as f:
            for blk0, blk1 in lib.prange(0, self.ncoords, blksize):
                blksize = blk1-blk0 # Actual blocksize
                data = np.zeros((blksize, self.nfields))
                blk = np.s_[blk0:blk1]
                ao = self.cell.eval_gto("PBCGTOval", self.coords[blk])
                for i, (field, ftype, _) in enumerate(self.fields):
                    if ftype == "orbital":
                        data[:,i] = np.dot(ao, field)
                    elif ftype == "density":
                        data[:,i] = numint.eval_rho(self.cell, ao, field)
                data = data.flatten()
                for d0, d1 in lib.prange(0, len(data), 6):
                    f.write(((d1-d0)*self.fmt + "\n") % tuple(data[d0:d1]))

if __name__ == "__main__":

    def make_graphene(a, c, atoms=["C", "C"], supercell=None):
        amat = np.asarray([
                [a, 0, 0],
                [a/2, a*np.sqrt(3.0)/2, 0],
                [0, 0, c]])
        coords_internal = np.asarray([
            [2.0, 2.0, 3.0],
            [4.0, 4.0, 3.0]])/6
        coords = np.dot(coords_internal, amat)

        if supercell is None:
            atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
        else:
            atom = []
            ncopy = supercell
            nr = 0
            for x in range(ncopy[0]):
                for y in range(ncopy[1]):
                    for z in range(ncopy[2]):
                        shift = x*amat[0] + y*amat[1] + z*amat[2]
                        atom.append((atoms[0]+str(nr), coords[0]+shift))
                        atom.append((atoms[1]+str(nr), coords[1]+shift))
                        nr += 1
            amat = np.einsum("i,ij->ij", ncopy, amat)
        return amat, atom


    from pyscf import pbc
    cell = pbc.gto.Cell(
        #basis = 'gth-szv',
        basis = 'gth-dzv',
        pseudo = 'gth-pade',
        dimension = 2,
        verbose=10)
    cell.a, cell.atom = make_graphene(2.46, 10.0, supercell=(2,2,1))
    hf = pbc.scf.HF(cell)
    hf = hf.density_fit()
    hf.kernel()

    with CubeFile(cell, "graphene.cube", crop={"c0" : 4.0, "c1" : 4.0}) as f:
        f.add_field(hf.mo_coeff[:,0])
        f.add_fields(hf.mo_coeff[:,6:10])
        f.add_field(hf.make_rdm1(), kind="density")
