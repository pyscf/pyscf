# Copyright (C) 2010  CAMd
# Copyright (C) 2010  Argonne National Laboratory
# Please see the accompanying LICENSE file for further information.
# original file from https://gitlab.com/gpaw/gpaw/blob/master/gpaw/blacs.py
# modified the 19-09-2017 to use with pyscf by Marc Barbry

from __future__ import print_function

"""Module for high-level BLACS interface.

Original description from GPAW:
===============================

Usage
-----

A BLACS grid is a logical grid of processors.  To use BLACS, first
create a BLACS grid.  If comm contains 8 or more ranks, this example
will work::

  from gpaw.mpi import world
  from gpaw.blacs import BlacsGrid
  grid = BlacsGrid(world, 4, 2)

Use the processor grid to create various descriptors for distributed
arrays::

  block_desc = grid.new_descriptor(500, 500, 64, 64)
  local_desc = grid.new_descriptor(500, 500, 500, 500)

The first descriptor describes 500 by 500 arrays distributed amongst
the 8 CPUs of the BLACS grid in blocks of 64 by 64 elements (which is
a sensible block size).  That means each CPU has many blocks located
all over the array::

  print(world.rank, block_desc.shape, block_desc.gshape)

Here block_desc.shape is the local array shape while gshape is the
global shape.  The local array shape varies a bit on each CPU as the
block distribution may be slightly uneven.

The second descriptor, local_desc, has a block size equal to the
global size of the array, and will therefore only have one block.
This block will then reside on the first CPU -- local_desc therefore
represents non-distributed arrays.  Let us instantiate some arrays::

  H_MM = local_desc.empty()

  if world.rank == 0:
      assert H_MM.shape == (500, 500)
      H_MM[:, :] = calculate_hamiltonian_or_something()
  else:
      assert H_MM.shape[0] == 0 or H_MM.shape[1] == 0

  H_mm = block_desc.empty()
  print(H_mm.shape)  # many elements on all CPUs

We can then redistribute the local H_MM into H_mm::

  from gpaw.blacs import Redistributor
  redistributor = Redistributor(world, local_desc, block_desc)
  redistributor.redistribute(H_MM, H_mm)

Now we can run parallel linear algebra on H_mm.  This will diagonalize
H_mm, place the eigenvectors in C_mm and the eigenvalues globally in
eps_M::

  eps_M = np.empty(500)
  C_mm = block_desc.empty()
  block_desc.diagonalize_ex(H_mm, C_mm, eps_M)

We can redistribute C_mm back to the master process if we want::

  C_MM = local_desc.empty()
  redistributor2 = Redistributor(world, block_desc, local_desc)
  redistributor2.redistribute(C_mm, C_MM)

If somebody wants to do all this more easily, they will probably write
a function for that.

List of interesting classes
---------------------------

 * BlacsGrid
 * BlacsDescriptor
 * Redistributor

The other classes in this module are coded specifically for GPAW and
are inconvenient to use otherwise.

The module gpaw.utilities.blacs contains several functions like gemm,
gemv and r2k.  These functions may or may not have appropriate
docstings, and may use Fortran-like variable naming.  Also, either
this module or gpaw.utilities.blacs will be renamed at some point.


Notes After port to PYSCF:
==========================
  
  * Added the class MatrixDescriptor that originally was from gpaw.matrix_descriptor
"""

import numpy as np

from pyscf.nao.mpi import SerialCommunicator
from pyscf.nao.mpi.scalapack import scalapack_inverse_cholesky, \
    scalapack_diagonalize_ex, scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_dc, scalapack_general_diagonalize_dc, \
    scalapack_diagonalize_mr3, scalapack_general_diagonalize_mr3

from pyscf.lib import misc
libmpi = misc.load_library("libmpi_wp")


INACTIVE = -1
BLOCK_CYCLIC_2D = 1

class MatrixDescriptor:
  """Class representing a 2D matrix shape.  Base class for parallel
  matrix descriptor with BLACS."""

  def __init__(self, M, N):
    self.shape = (M, N)

  def __bool__(self):
    return self.shape[0] != 0 and self.shape[1] != 0

  __nonzero__ = __bool__  # for Python 2

  def zeros(self, n=(), dtype=float):
    """Return array of zeroes with the correct size on all CPUs.

    The last two dimensions will be equal to the shape of this
    descriptor.  If specified as a tuple, can have any preceding
    dimension."""
    return self._new_array(np.zeros, n, dtype)

  def empty(self, n=(), dtype=float):
    """Return array of zeros with the correct size on all CPUs.

    See zeros()."""
    return self._new_array(np.empty, n, dtype)

  def _new_array(self, func, n, dtype):
    if isinstance(n, int):
    n = n,
    shape = n + self.shape
    return func(shape, dtype)

  def check(self, a_mn):
     """Check that specified array is compatible with this descriptor."""
    return a_mn.shape == self.shape and a_mn.flags.contiguous

  def checkassert(self, a_mn):
    ok = self.check(a_mn)
    if not ok:
      if not a_mn.flags.contiguous:
        msg = 'Matrix with shape %s is not contiguous' % (a_mn.shape,)
      else:
        msg = ('%s-descriptor incompatible with %s-matrix' %
          (self.shape, a_mn.shape))
      raise AssertionError(msg)

  #def general_diagonalize_dc(self, H_mm, S_mm, C_mm, eps_M, UL='L', iu=None):
  #  general_diagonalize(H_mm, eps_M, S_mm, iu=iu)
  #  C_mm[:] = H_mm

  def my_blocks(self, array_mn):
    yield (0, self.shape[0], 0, self.shape[1], array_mn)

  def estimate_memory(self, mem, dtype):
    """Handled by subclass."""
    pass

class BlacsGrid:
    """Class representing a 2D grid of processors sharing a Blacs context.

    A BLACS grid defines a logical M by N ordering of a collection of
    CPUs.  A BLACS grid can be used to create BLACS descriptors.  On
    an npcol by nprow BLACS grid, a matrix is distributed amongst M by
    N CPUs along columns and rows, respectively, while the matrix
    shape and blocking properties are determined by the descriptors.

    Use the method new_descriptor() to create any number of BLACS
    descriptors sharing the same CPU layout.

    Most matrix operations require the involved matrices to all be on
    the same BlacsGrid.  Use a Redistributor to redistribute matrices
    from one BLACS grid to another if necessary.

    Parameters::

      * comm:  MPI communicator for CPUs of the BLACS grid or None.  A BLACS
        grid may use all or some of the CPUs of the communicator.
      * nprow:  Number of CPU rows.
      * npcol: Number of CPU columns.
      * order: 'R' or 'C', meaning rows or columns.  I'm not sure what this
        does, it probably interchanges the meaning of rows and columns. XXX

    Complicated stuff
    -----------------

    It may be useful to know that a BLACS grid is said to be active
    and will evaluate to True on any process where comm is not None
    *and* comm.rank < nprow * npcol.  Otherwise it is considered
    inactive and evaluates to False.  Ranks where a grid is inactive
    never do anything at all.

    BLACS identifies each grid by a unique ID number called the
    context (frequently abbreviated ConTxt).  Grids on inactive ranks
    have context -1."""
    def __init__(self, comm, nprow, npcol, order='R'):
        assert nprow > 0
        assert npcol > 0
        assert len(order) == 1
        assert order in 'CcRr'
        # set a default value for the context leads to fewer
        # if statements below
        self.context = INACTIVE

        # There are three cases to handle:
        # 1. Comm is None is inactive (default).
        # 2. Comm is a legitimate communicator
        # 3. DryRun Communicator is now handled by subclass
        if comm is not None:  # MPI task is part of the communicator
            if nprow * npcol > comm.size:
                raise ValueError('Impossible: %dx%d Blacs grid with %d CPUs'
                                 % (nprow, npcol, comm.size))

            try:
                new = libmpi.new_blacs_context
            except AttributeError as e:
                raise AttributeError(
                    'BLACS is unavailable.  '
                    'PYSCF must be compiled with BLACS/ScaLAPACK, '
                    'Original error: %s' % e)

            self.context = new(comm.get_c_object(), npcol, nprow, order)
            assert (self.context != INACTIVE) == (comm.rank < nprow * npcol)

        self.mycol, self.myrow = libmpi.get_blacs_gridinfo(self.context,
                                                          nprow,
                                                          npcol)

        self.comm = comm
        self.nprow = nprow
        self.npcol = npcol
        self.ncpus = nprow * npcol
        self.order = order

    @property
    def coords(self):
        return self.myrow, self.mycol

    @property
    def shape(self):
        return self.nprow, self.npcol

    def coords2rank(self, row, col):
        return self.nprow * col + row

    def rank2coords(self, rank):
        col, row = divmod(rank, self.nprow)
        return row, col

    def new_descriptor(self, M, N, mb, nb, rsrc=0, csrc=0):
        """Create a new descriptor from this BLACS grid.

        See documentation for BlacsDescriptor.__init__."""
        return BlacsDescriptor(self, M, N, mb, nb, rsrc, csrc)

    def is_active(self):
        """Whether context is active on this rank."""
        return self.context != INACTIVE

    def __bool__(self):
        2 / 0

    __nonzero__ = __bool__  # for Python 2

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[comm:size=%d,rank=%d; context=%d; %dx%d]'
        string = template % (classname, self.comm.size, self.comm.rank,
                             self.context, self.nprow, self.npcol)
        return string

    def __del__(self):
        if self.is_active():
            libmpi.blacs_destroy(self.context)


class DryRunBlacsGrid(BlacsGrid):
    def __init__(self, comm, nprow, npcol, order='R'):
        assert (isinstance(comm, SerialCommunicator) or
                isinstance(comm.comm, SerialCommunicator))
        # DryRunCommunicator is subclass

        if nprow * npcol > comm.size:
            raise ValueError('Impossible: %dx%d Blacs grid with %d CPUs'
                             % (nprow, npcol, comm.size))
        self.context = INACTIVE
        self.comm = comm
        self.nprow = nprow
        self.npcol = npcol
        self.ncpus = nprow * npcol
        self.mycol, self.myrow = INACTIVE, INACTIVE
        self.order = order


# XXX A MAJOR HACK HERE:
#if dry_run:
#    BlacsGrid = DryRunBlacsGrid


class BlacsDescriptor(MatrixDescriptor):
    """Class representing a 2D matrix distribution on a blacs grid.

    A BlacsDescriptor represents a particular shape and distribution
    of matrices.  A BlacsDescriptor has a global matrix shape and a
    rank-dependent local matrix shape.  The local shape is not
    necessarily equal on all ranks.

    A numpy array is said to be compatible with a BlacsDescriptor if,
    on all ranks, the shape of the numpy array is equal to the local
    shape of the BlacsDescriptor.  Compatible arrays can be created
    conveniently with the zeros() and empty() methods.

    An array with a global shape of M by N is distributed such that
    each process gets a number of distinct blocks of size mb by nb.
    The blocks on one process generally reside in very different areas
    of the matrix to improve load balance.

    The following chart describes how different ranks (there are 4
    ranks in this example, 0 through 3) divide the matrix into blocks.
    This is called 2D block cyclic distribution::

        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        ...................
        ...................
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+

    Also refer to:
    http://acts.nersc.gov/scalapack/hands-on/datadist.html

    Parameters:
     * blacsgrid: the BLACS grid of processors to distribute matrices.
     * M: global row count
     * N: global column count
     * mb: number of rows per block
     * nb: number of columns per block
     * rsrc: rank on which the first row is stored
     * csrc: rank on which the first column is stored

    Complicated stuff
    -----------------

    If there is trouble with matrix shapes, the below caveats are
    probably the reason.

    Depending on layout, a descriptor may have a local shape of zero
    by N or something similar.  If the row blocksize is 7, the global
    row count is 10, and the blacs grid contains 3 row processes: The
    first process will have 7 rows, the next will have 3, and the last
    will have 0.  The shapes in this case must still be correctly
    given to BLACS functions, which can be confusing.

    A blacs descriptor must also give the correct local leading
    dimension (lld), which is the local array size along the
    memory-contiguous direction in the matrix, and thus equal to the
    local column number, *except* when local shape is zero, but the
    implementation probably works.

    """
    def __init__(self, blacsgrid, M, N, mb, nb, rsrc=0, csrc=0):
        assert M > 0
        assert N > 0
        assert 1 <= mb
        assert 1 <= nb
        if mb > M:
            mb = M
        if nb > N:
            nb = N
        assert 0 <= rsrc < blacsgrid.nprow
        assert 0 <= csrc < blacsgrid.npcol

        self.blacsgrid = blacsgrid
        self.M = M  # global size 1
        self.N = N  # global size 2
        self.mb = mb  # block cyclic distr dim 1
        self.nb = nb  # and 2.  How many rows or columns are on this processor
        # more info:
        # http://www.netlib.org/scalapack/slug/node75.html
        self.rsrc = rsrc
        self.csrc = csrc

        if blacsgrid.is_active():
            locN, locM = libmpi.get_blacs_local_shape(self.blacsgrid.context,
                                                     self.N, self.M,
                                                     self.nb, self.mb,
                                                     self.csrc, self.rsrc)
            # max 1 is nonsensical, but appears
            # to be required by PBLAS
            self.lld = max(1, locN)
        else:
            # ScaLAPACK has no requirements as to what these values on an
            # inactive blacsgrid should be. This seemed reasonable to me
            # at the time.
            locN, locM = 0, 0
            self.lld = 0

        # locM, locN is not allowed to be negative. This will cause the
        # redistributor to fail. This could happen on active blacsgrid
        # which does not contain any piece of the distribute matrix.
        # This is why there is a final check on the value of locM, locN.
        MatrixDescriptor.__init__(self, max(0, locM), max(0, locN))

        # This is the definition of inactive descriptor; can occur
        # on an active or inactive blacs grid.
        self.active = locM > 0 and locN > 0

        self.bshape = (self.mb, self.nb)  # Shape of one block
        self.gshape = (M, N)  # Global shape of array

    def asarray(self):
        """Return a nine-element array representing this descriptor.

        In the C/Fortran code, a BLACS descriptor is represented by a
        special array of arcane nature.  The value of asarray() must
        generally be passed to BLACS functions in the C code."""
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context,
                        self.N, self.M, self.nb, self.mb, self.csrc, self.rsrc,
                        self.lld], np.intc)
        return arr

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %s, block %s, lld %d, loc %s]'
        string = template % (classname, self.blacsgrid.context,
                             self.gshape,
                             self.bshape, self.lld, self.shape)
        return string

    def index2grid(self, row, col):
        """Get the BLACS grid coordinates storing global index (row, col)."""
        assert row < self.gshape[0], (row, col, self.gshape)
        assert col < self.gshape[1], (row, col, self.gshape)
        gridx = (row // self.bshape[0]) % self.blacsgrid.nprow
        gridy = (col // self.bshape[1]) % self.blacsgrid.npcol
        return gridx, gridy

    def index2rank(self, row, col):
        """Get the rank where global index (row, col) is stored."""
        return self.blacsgrid.coords2rank(*self.index2grid(row, col))

    def diagonalize_dc(self, H_nn, C_nn, eps_N, UL='L'):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_diagonalize_dc(self, H_nn, C_nn, eps_N, UL)

    def diagonalize_ex(self, H_nn, C_nn, eps_N, UL='L', iu=None):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_diagonalize_ex(self, H_nn, C_nn, eps_N, UL, iu=iu)

    def diagonalize_mr3(self, H_nn, C_nn, eps_N, UL='L', iu=None):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_diagonalize_mr3(self, H_nn, C_nn, eps_N, UL, iu=iu)

    def general_diagonalize_dc(self, H_mm, S_mm, C_mm, eps_M,
                               UL='L'):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_general_diagonalize_dc(self, H_mm, S_mm, C_mm, eps_M,
                                         UL)

    def general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                               UL='L', iu=None):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                                         UL, iu=iu)

    def general_diagonalize_mr3(self, H_mm, S_mm, C_mm, eps_M,
                                UL='L', iu=None):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_general_diagonalize_mr3(self, H_mm, S_mm, C_mm, eps_M,
                                          UL, iu=iu)

    def inverse_cholesky(self, S_nn, UL='L'):
        """See documentation in pyscf/mpi/scalapack.py."""
        scalapack_inverse_cholesky(self, S_nn, UL)

    def my_blocks(self, array_mn):
        """Yield the local blocks and their global index limits.

        Yields tuples of the form (Mstart, Mstop, Nstart, Nstop, block),
        for each locally stored block of the array.
        """
        if not self.check(array_mn):
            raise ValueError('Bad array shape (%s vs %s)' % (self,
                                                             array_mn.shape))

        grid = self.blacsgrid
        mb = self.mb
        nb = self.nb
        myrow = grid.myrow
        mycol = grid.mycol
        nprow = grid.nprow
        npcol = grid.npcol
        M, N = self.gshape

        Mmyblocks = -(-self.shape[0] // mb)
        Nmyblocks = -(-self.shape[1] // nb)
        for Mblock in range(Mmyblocks):
            for Nblock in range(Nmyblocks):
                myMstart = Mblock * mb
                myNstart = Nblock * nb
                Mstart = myrow * mb + Mblock * mb * nprow
                Nstart = mycol * mb + Nblock * nb * npcol
                Mstop = min(Mstart + mb, M)
                Nstop = min(Nstart + nb, N)
                block = array_mn[myMstart:myMstart + mb,
                                 myNstart:myNstart + nb]

                yield Mstart, Mstop, Nstart, Nstop, block

    def as_serial(self):
        return self.blacsgrid.new_descriptor(self.M, self.N, self.M, self.N)

    def redistribute(self, otherdesc, src_mn, dst_mn=None,
                     subM=None, subN=None, ia=0, ja=0, ib=0, jb=0, uplo='G'):
        if self.blacsgrid != otherdesc.blacsgrid:
            raise ValueError('Cannot redistribute to other BLACS grid.  '
                             'Requires using Redistributor class explicitly')
        if dst_mn is None:
            dst_mn = otherdesc.empty(dtype=src_mn.dtype)
        r = Redistributor(self.blacsgrid.comm, self, otherdesc)
        r.redistribute(src_mn, dst_mn, subM, subN, ia, ja, ib, jb, uplo)
        return dst_mn

    def collect_on_master(self, src_mn, dst_mn=None, uplo='G'):
        desc = self.as_serial()
        return self.redistribute(desc, src_mn, dst_mn, uplo=uplo)

    def distribute_from_master(self, src_mn, dst_mn=None, uplo='G'):
        desc = self.as_serial()
        return desc.redistribute(self, src_mn, dst_mn, uplo=uplo)


class Redistributor:
    """Class for redistributing BLACS matrices on different contexts."""
    def __init__(self, supercomm, srcdescriptor, dstdescriptor):
        """Create redistributor.

        Source and destination descriptors may reside on different
        BLACS grids, but the descriptors should describe arrays with
        the same number of elements.

        The communicators of the BLACS grid of srcdescriptor as well
        as that of dstdescriptor *must* both be subcommunicators of
        supercomm.

        Allowed values of UPLO are: G for general matrix, U for upper
        triangular and L for lower triangular. The latter two are useful
        for symmetric matrices."""
        self.supercomm = supercomm
        self.supercomm_bg = BlacsGrid(self.supercomm, self.supercomm.size, 1)
        self.srcdescriptor = srcdescriptor
        self.dstdescriptor = dstdescriptor

    def redistribute(self, src_mn, dst_mn=None,
                     subM=None, subN=None,
                     ia=0, ja=0, ib=0, jb=0, uplo='G'):
        """Redistribute src_mn into dst_mn.

        src_mn and dst_mn must be compatible with source and
        destination descriptors of this redistributor.

        If subM and subN are given, distribute only a subM by subN
        submatrix.

        If any ia, ja, ib and jb are given, they denote the global
        index (i, j) of the origin of the submatrix inside the source
        and destination (a, b) matrices."""

        srcdescriptor = self.srcdescriptor
        dstdescriptor = self.dstdescriptor
        dtype = src_mn.dtype

        if dst_mn is None:
            dst_mn = dstdescriptor.empty(dtype=dtype)

        # self.supercomm must be a supercommunicator of the communicators
        # corresponding to the context of srcmatrix as well as dstmatrix.
        # We should verify this somehow.
        srcdescriptor = self.srcdescriptor
        dstdescriptor = self.dstdescriptor

        dtype = src_mn.dtype
        if dst_mn is None:
            dst_mn = dstdescriptor.zeros(dtype=dtype)

        assert dtype == dst_mn.dtype
        assert dtype == float or dtype == complex

        # Check to make sure the submatrix of the source
        # matrix will fit into the destination matrix
        # plus standard BLACS matrix checks.
        srcdescriptor.checkassert(src_mn)
        dstdescriptor.checkassert(dst_mn)

        if subM is None:
            subM = srcdescriptor.gshape[0]
        if subN is None:
            subN = srcdescriptor.gshape[1]

        assert srcdescriptor.gshape[0] >= subM
        assert srcdescriptor.gshape[1] >= subN
        assert dstdescriptor.gshape[0] >= subM
        assert dstdescriptor.gshape[1] >= subN

        # Switch to Fortran conventions
        uplo = {'U': 'L', 'L': 'U', 'G': 'G'}[uplo]
        libmpi.scalapack_redist(srcdescriptor.asarray(),
                               dstdescriptor.asarray(),
                               src_mn, dst_mn,
                               subN, subM,
                               ja + 1, ia + 1, jb + 1, ib + 1,  # 1-indexing
                               self.supercomm_bg.context, uplo)
        return dst_mn


def parallelprint(comm, obj):
    import sys
    for a in range(comm.size):
        if a == comm.rank:
            print('rank=%d' % a)
            print(obj)
            print()
            sys.stdout.flush()
        comm.barrier()
