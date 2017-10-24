# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.
# Original file from https://gitlab.com/gpaw/gpaw/tree/master/gpaw/mpi
# Modified the 19-09-2017 to use with pyscf by Marc Barbry 

import os
import sys
import time
import traceback
import atexit
import pickle

import numpy as np
try:
  from math import gcd
except ImportError:
  from fractions import gcd

from pyscf.nao.mpi.m_utils import is_contiguous
from pyscf.lib import misc
libmpi = misc.load_library("libmpi_wp")

MASTER = 0
debug = True
print("debug: ", debug)


class _Communicator:
    def __init__(self, comm, parent=None):
        """Construct a wrapper of the C-object for any MPI-communicator.

        Parameters:

        comm: MPI-communicator
            Communicator.

        Attributes:

        ============  ======================================================
        ``size``      Number of ranks in the MPI group.
        ``rank``      Number of this CPU in the MPI group.
        ``parent``    Parent MPI-communicator.
        ============  ======================================================
        """
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.parent = parent  # XXX check C-object against comm.parent?

    def new_communicator(self, ranks):
        """Create a new MPI communicator for a subset of ranks in a group.
        Must be called with identical arguments by all relevant processes.

        Note that a valid communicator is only returned to the processes
        which are included in the new group; other ranks get None returned.

        Parameters:

        ranks: ndarray (type int)
            List of integers of the ranks to include in the new group.
            Note that these ranks correspond to indices in the current
            group whereas the rank attribute in the new communicators
            correspond to their respective index in the subset.

        """

        comm = self.comm.new_communicator(ranks)
        if comm is None:
            # This cpu is not in the new communicator:
            return None
        else:
            return _Communicator(comm, parent=self)

    def sum(self, a, root=-1):
        """Perform summation by MPI reduce operations of numerical data.

        Parameters:

        a: ndarray or value (type int, float or complex)
            Numerical data to sum over all ranks in the communicator group.
            If the data is a single value of type int, float or complex,
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the sum of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float, complex)):
            return self.comm.sum(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float or tc == complex
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.sum(a, root)

    def product(self, a, root=-1):
        """Do multiplication by MPI reduce operations of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to multiply across all ranks in the communicator
            group. NB: Find the global product from the local products.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the product
            of the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            return self.comm.product(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.product(a, root)

    def max(self, a, root=-1):
        """Find maximal value by an MPI reduce operation of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to find the maximum value of across all ranks in
            the communicator group. NB: Find global maximum from local max.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the max of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            return self.comm.max(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.max(a, root)

    def min(self, a, root=-1):
        """Find minimal value by an MPI reduce operation of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to find the minimal value of across all ranks in
            the communicator group. NB: Find global minimum from local min.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the min of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            return self.comm.min(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.min(a, root)

    def scatter(self, a, b, root):
        """Distribute data from one rank to all other processes in a group.

        Parameters:

        a: ndarray (ignored on all ranks different from root; use None)
            Source of the data to distribute, i.e. send buffer on root rank.
        b: ndarray
            Destination of the distributed data, i.e. local receive buffer.
            The size of this array multiplied by the number of process in
            the group must match the size of the source array on the root.
        root: int
            Rank of the root process, from which the source data originates.

        The reverse operation is ``gather``.

        Example::

          # The master has all the interesting data. Distribute it.
          if comm.rank == 0:
              data = np.random.normal(size=N*comm.size)
          else:
              data = None
          mydata = np.empty(N, dtype=float)
          comm.scatter(data, mydata, 0)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Extract my part directly
              mydata[:] = data[0:N]
              # Distribute parts to the slaves
              for rank in range(1, comm.size):
                  buf = data[rank*N:(rank+1)*N]
                  comm.send(buf, rank, tag=123)
          else:
              # Receive from the master
              comm.receive(mydata, 0, tag=123)

        """
        if self.rank == root:
            assert a.dtype == b.dtype
            assert a.size == self.size * b.size
            assert a.flags.contiguous
        assert b.flags.contiguous
        assert 0 <= root < self.size
        self.comm.scatter(a, b, root)

    def alltoallv(self, sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls):
        """All-to-all in a group.

        Parameters:

        sbuffer: ndarray
            Source of the data to distribute, i.e., send buffers on all ranks
        scounts: ndarray
            Integer array equal to the group size specifying the number of
            elements to send to each processor
        sdispls: ndarray
            Integer array (of length group size). Entry j specifies the
            displacement (relative to sendbuf from which to take the
            outgoing data destined for process j)
        rbuffer: ndarray
            Destination of the distributed data, i.e., local receive buffer.
        rcounts: ndarray
            Integer array equal to the group size specifying the maximum
            number of elements that can be received from each processor.
        rdispls:
            Integer array (of length group size). Entry i specifies the
            displacement (relative to recvbuf at which to place the incoming
            data from process i
        """
        assert sbuffer.flags.contiguous
        assert scounts.flags.contiguous
        assert sdispls.flags.contiguous
        assert rbuffer.flags.contiguous
        assert rcounts.flags.contiguous
        assert rdispls.flags.contiguous
        assert sbuffer.dtype == rbuffer.dtype
        
        for arr in [scounts, sdispls, rcounts, rdispls]:
            assert arr.dtype == np.int, arr.dtype
            assert len(arr) == self.size

        assert np.all(0 <= sdispls)
        assert np.all(0 <= rdispls)
        assert np.all(sdispls + scounts <= sbuffer.size)
        assert np.all(rdispls + rcounts <= rbuffer.size)
        self.comm.alltoallv(sbuffer, scounts, sdispls,
                            rbuffer, rcounts, rdispls)

    def all_gather(self, a, b):
        """Gather data from all ranks onto all processes in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        b: ndarray
            Destination of the distributed data, i.e. receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        Example::

          # All ranks have parts of interesting data. Gather on all ranks.
          mydata = np.random.normal(size=N)
          data = np.empty(N*comm.size, dtype=float)
          comm.all_gather(mydata, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Insert my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)
          # Broadcast from master to all slaves
          comm.broadcast(data, 0)

        """
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert b.dtype == a.dtype
        assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                a.size * self.size == b.size)
        self.comm.all_gather(a, b)

    def gather(self, a, root, b=None):
        """Gather data from all ranks onto a single process in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        root: int
            Rank of the root process, on which the data is to be gathered.
        b: ndarray (ignored on all ranks different from root; default None)
            Destination of the distributed data, i.e. root's receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        The reverse operation is ``scatter``.

        Example::

          # All ranks have parts of interesting data. Gather it on master.
          mydata = np.random.normal(size=N)
          if comm.rank == 0:
              data = np.empty(N*comm.size, dtype=float)
          else:
              data = None
          comm.gather(mydata, 0, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Extract my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)

        """
        assert a.flags.contiguous
        assert 0 <= root < self.size
        if root == self.rank:
            assert b.flags.contiguous and b.dtype == a.dtype
            assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                    a.size * self.size == b.size)
            self.comm.gather(a, root, b)
        else:
            assert b is None
            self.comm.gather(a, root)

    def broadcast(self, a, root):
        """Share data from a single process to all ranks in a group.

        Parameters:

        a: ndarray
            Data, i.e. send buffer on root rank, receive buffer elsewhere.
            Note that after the broadcast, all ranks have the same data.
        root: int
            Rank of the root process, from which the data is to be shared.

        Example::

          # All ranks have parts of interesting data. Take a given index.
          mydata[:] = np.random.normal(size=N)

          # Who has the element at global index 13? Everybody needs it!
          index = 13
          root, myindex = divmod(index, N)
          element = np.empty(1, dtype=float)
          if comm.rank == root:
              # This process has the requested element so extract it
              element[:] = mydata[myindex]

          # Broadcast from owner to everyone else
          comm.broadcast(element, root)

          # .. which is equivalent to ..

          if comm.rank == root:
              # We are root so send it to the other ranks
              for rank in range(comm.size):
                  if rank != root:
                      comm.send(element, rank, tag=123)
          else:
              # We don't have it so receive from root
              comm.receive(element, root, tag=123)

        """
        assert 0 <= root < self.size
        assert is_contiguous(a)
        self.comm.broadcast(a, root)

    def sendreceive(self, a, dest, b, src, sendtag=123, recvtag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(b)
        return self.comm.sendreceive(a, dest, b, src, sendtag, recvtag)

    def send(self, a, dest, tag=123, block=True):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        if not block:
            pass  # assert sys.getrefcount(a) > 3
        return self.comm.send(a, dest, tag, block)

    def ssend(self, a, dest, tag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        return self.comm.ssend(a, dest, tag)

    def receive(self, a, src, tag=123, block=True):
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(a)
        return self.comm.receive(a, src, tag, block)

    def test(self, request):
        """Test whether a non-blocking MPI operation has completed. A boolean
        is returned immediately and the request is not modified in any way.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.test(request)

    def testall(self, requests):
        """Test whether non-blocking MPI operations have completed. A boolean
        is returned immediately but requests may have been deallocated as a
        result, provided they have completed before or during this invokation.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.testall(requests)  # may deallocate requests!

    def wait(self, request):
        """Wait for a non-blocking MPI operation to complete before returning.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        self.comm.wait(request)

    def waitall(self, requests):
        """Wait for non-blocking MPI operations to complete before returning.

        Parameters:

        requests: list
            List of MPI requests e.g. aggregated from returned requests of
            multiple send/receive calls where block=False was used.

        """
        self.comm.waitall(requests)

    def abort(self, errcode):
        """Terminate MPI execution environment of all tasks in the group.
        This function only returns in the advent of an error occurring.

        Parameters:

        errcode: int
            Error code to return to the invoking environment.

        """
        return self.comm.abort(errcode)

    def name(self):
        """Return the name of the processor as a string."""
        return self.comm.name()

    def barrier(self):
        """Block execution until all process have reached this point."""
        self.comm.barrier()

    def compare(self, othercomm):
        """Compare communicator to other.

        Returns 'ident' if they are identical, 'congruent' if they are
        copies of each other, 'similar' if they are permutations of
        each other, and otherwise 'unequal'.

        This method corresponds to MPI_Comm_compare."""
        if isinstance(self.comm, SerialCommunicator):
            return self.comm.compare(othercomm.comm) # argh!
        result = self.comm.compare(othercomm.get_c_object())
        assert result in ['ident', 'congruent', 'similar', 'unequal']
        return result

    def translate_ranks(self, other, ranks):
        """"Translate ranks from communicator to other.

        ranks must be valid on this communicator.  Returns ranks
        on other communicator corresponding to the same processes.
        Ranks that are not defined on the other communicator are
        assigned values of -1.  (In contrast to MPI which would
        assign MPI_UNDEFINED)."""
        assert hasattr(other, 'translate_ranks'), \
            'Excpected communicator, got %s' % other
        assert all(0 <= rank for rank in ranks)
        assert all(rank < self.size for rank in ranks)
        if isinstance(self.comm, SerialCommunicator):
            return self.comm.translate_ranks(other.comm, ranks) # argh!
        otherranks = self.comm.translate_ranks(other.get_c_object(), ranks)
        assert all(-1 <= rank for rank in otherranks)
        assert ranks.dtype == otherranks.dtype
        return otherranks
        
    def get_members(self):
        """Return the subset of processes which are members of this MPI group
        in terms of the ranks they are assigned on the parent communicator.
        For the world communicator, this is all integers up to ``size``.

        Example::

          >>> world.rank, world.size
          (3, 4)
          >>> world.get_members()
          array([0, 1, 2, 3])
          >>> comm = world.new_communicator(array([2, 3]))
          >>> comm.rank, comm.size
          (1, 2)
          >>> comm.get_members()
          array([2, 3])
          >>> comm.get_members()[comm.rank] == world.rank
          True

        """
        return self.comm.get_members()

    def get_c_object(self):
        """Return the C-object wrapped by this debug interface.

        Whenever a communicator object is passed to C code, that object
        must be a proper C-object - *not* e.g. this debug wrapper.  For
        this reason.  The C-communicator object has a get_c_object()
        implementation which returns itself; thus, always call
        comm.get_c_object() and pass the resulting object to the C code.
        """
        c_obj = self.comm.get_c_object()
        assert isinstance(c_obj, libmpi.Communicator)
        return c_obj


# Serial communicator
class SerialCommunicator:
    size = 1
    rank = 0

    def __init__(self, parent=None):
        self.parent = parent

    def sum(self, array, root=-1):
        if isinstance(array, (int, float, complex)):
            return array

    def scatter(self, s, r, root):
        r[:] = s

    def min(self, value, root=-1):
        return value

    def max(self, value, root=-1):
        return value

    def broadcast(self, buf, root):
        pass

    def send(self, buff, root, tag=123, block=True):
        pass

    def barrier(self):
        pass

    def gather(self, a, root, b):
        b[:] = a

    def all_gather(self, a, b):
        b[:] = a

    def alltoallv(self, sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls):
        assert len(scounts) == 1
        assert len(sdispls) == 1
        assert len(rcounts) == 1
        assert len(rdispls) == 1
        assert len(sbuffer) == len(rbuffer)
        
        rbuffer[rdispls[0]:rdispls[0] + rcounts[0]] = \
            sbuffer[sdispls[0]:sdispls[0] + scounts[0]]

    def new_communicator(self, ranks):
        if self.rank not in ranks:
            return None
        return SerialCommunicator(parent=self)

    def test(self, request):
        return 1

    def testall(self, requests):
        return 1

    def wait(self, request):
        raise NotImplementedError('Calls to mpi wait should not happen in '
                                  'serial mode')

    def waitall(self, requests):
        if not requests:
            return
        raise NotImplementedError('Calls to mpi waitall should not happen in '
                                  'serial mode')

    def get_members(self):
        return np.array([0])
    
    def compare(self, other):
        if self == other:
            return 'ident'
        elif isinstance(other, SerialCommunicator):
            return 'congruent'
        else:
            raise NotImplementedError('Compare serial comm to other')

    def translate_ranks(self, other, ranks):
        if isinstance(other, SerialCommunicator):
            assert all(rank == 0 for rank in ranks)
            return np.zeros(len(ranks), dtype=int)
        raise NotImplementedError('Translate non-trivial ranks with serial comm')

    def get_c_object(self):
        raise NotImplementedError('Should not get C-object for serial comm')


serial_comm = SerialCommunicator()

world = libmpi.Communicator()
if world.size == 1:
    world = serial_comm
else:
    try:
        world = libmpi.Communicator()
    except AttributeError:
        world = serial_comm

    
if debug:
    serial_comm = _Communicator(serial_comm)
    world = _Communicator(world)


size = world.size
rank = world.rank
parallel = (size > 1)
try:
    world.get_c_object()
except NotImplementedError:
    have_mpi = False
else:
    have_mpi = True


# XXXXXXXXXX for easier transition to Parallelization class
def distribute_cpus(parsize_domain, parsize_bands,
                    nspins, nibzkpts, comm=world,
                    idiotproof=True, mode='fd'):
    nsk = nspins * nibzkpts
    if mode in ['fd', 'lcao']:
        if parsize_bands is None:
            parsize_bands = 1
    else:
        # Plane wave mode:
        if parsize_bands is None:
            parsize_bands = comm.size // gcd(nsk, comm.size)

    p = Parallelization(comm, nsk)
    return p.build_communicators(domain=np.prod(parsize_domain),
                                 band=parsize_bands)


def broadcast(obj, root=0, comm=world):
    """Broadcast a Python object across an MPI communicator and return it."""
    if comm.rank == root:
        assert obj is not None
        b = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    else:
        assert obj is None
        b = None
    b = broadcast_bytes(b, root, comm)
    if comm.rank == root:
        return obj
    else:
        return pickle.loads(b)


def synchronize_atoms(atoms, comm, tolerance=1e-8):
    """Synchronize atoms between multiple CPUs removing numerical noise.

    If the atoms differ significantly, raise ValueError on all ranks.
    The error object contains the ranks where the check failed.

    In debug mode, write atoms to files in case of failure."""

    if len(atoms) == 0:
        return

    if comm.rank == 0:
        src = (atoms.positions, atoms.cell, atoms.numbers, atoms.pbc)
    else:
        src = None

    # XXX replace with ase.cell.same_cell in the future
    # (if that functions gets to exist)
    def same_cell(cell1, cell2):
        return ((cell1 is None) == (cell2 is None) and
                (cell1 is None or (cell1 == cell2).all()))

    positions, cell, numbers, pbc = broadcast(src, root=0, comm=comm)
    ok = (len(positions) == len(atoms.positions) and
          (abs(positions - atoms.positions).max() <= tolerance) and
          (numbers == atoms.numbers).all() and
          same_cell(cell, atoms.cell) and
          (pbc == atoms.pbc).all())

    # We need to fail equally on all ranks to avoid trouble.  Thus
    # we use an array to gather check results from everyone.
    my_fail = np.array(not ok, dtype=bool)
    all_fail = np.zeros(comm.size, dtype=bool)
    comm.all_gather(my_fail, all_fail)

    if all_fail.any():
        if debug:
            with open('synchronize_atoms_r%d.pckl' % comm.rank, 'wb') as fd:
                pickle.dump((atoms.positions, atoms.cell,
                             atoms.numbers, atoms.pbc,
                             positions, cell, numbers, pbc), fd)
        err_ranks = np.arange(comm.size)[all_fail]
        raise ValueError('Mismatch of Atoms objects.  In debug '
                         'mode, atoms will be dumped to files.',
                         err_ranks)

    atoms.positions = positions


def broadcast_string(string=None, root=0, comm=world):
    if comm.rank == root:
        string = string.encode()
    return broadcast_bytes(string, root, comm).decode()


def broadcast_bytes(b=None, root=0, comm=world):
    """Broadcast a bytes across an MPI communicator and return it."""
    if comm.rank == root:
        assert isinstance(b, bytes)
        n = np.array(len(b), int)
    else:
        assert b is None
        n = np.zeros(1, int)
    comm.broadcast(n, root)
    if comm.rank == root:
        b = np.fromstring(b, np.int8)
    else:
        b = np.zeros(n, np.int8)
    comm.broadcast(b, root)
    return b.tostring()

    
def send_string(string, rank, comm=world):
    comm.send(np.array(len(string)), rank)
    comm.send(np.fromstring(string, np.int8), rank)

    
def receive_string(rank, comm=world):
    n = np.array(0)
    comm.receive(n, rank)
    string = np.empty(n, np.int8)
    comm.receive(string, rank)
    return string.tostring().decode()

    
def alltoallv_string(send_dict, comm=world):
    scounts = np.zeros(comm.size, dtype=np.int)
    sdispls = np.zeros(comm.size, dtype=np.int)
    stotal = 0
    for proc in range(comm.size):
        if proc in send_dict:
            data = np.fromstring(send_dict[proc], np.int8)
            scounts[proc] = data.size
            sdispls[proc] = stotal
            stotal += scounts[proc]

    rcounts = np.zeros(comm.size, dtype=np.int)
    comm.alltoallv(scounts, np.ones(comm.size, dtype=np.int),
                   np.arange(comm.size, dtype=np.int),
                   rcounts, np.ones(comm.size, dtype=np.int),
                   np.arange(comm.size, dtype=np.int))
    rdispls = np.zeros(comm.size, dtype=np.int)
    rtotal = 0
    for proc in range(comm.size):
        rdispls[proc] = rtotal
        rtotal += rcounts[proc]
        # rtotal += rcounts[proc]  # CHECK: is this correct?

    sbuffer = np.zeros(stotal, dtype=np.int8)
    for proc in range(comm.size):
        sbuffer[sdispls[proc]:(sdispls[proc] + scounts[proc])] = (
            np.fromstring(send_dict[proc], np.int8))

    rbuffer = np.zeros(rtotal, dtype=np.int8)
    comm.alltoallv(sbuffer, scounts, sdispls, rbuffer, rcounts, rdispls)

    rdict = {}
    for proc in range(comm.size):
        i = rdispls[proc]
        rdict[proc] = rbuffer[i:i + rcounts[proc]].tostring().decode()

    return rdict

    
def ibarrier(timeout=None, root=0, tag=123, comm=world):
    """Non-blocking barrier returning a list of requests to wait for.
    An optional time-out may be given, turning the call into a blocking
    barrier with an upper time limit, beyond which an exception is raised."""
    requests = []
    byte = np.ones(1, dtype=np.int8)
    if comm.rank == root:
        # Everybody else:
        for rank in range(comm.size):
            if rank == root:
                continue
            rbuf, sbuf = np.empty_like(byte), byte.copy()
            requests.append(comm.send(sbuf, rank, tag=2 * tag + 0,
                                      block=False))
            requests.append(comm.receive(rbuf, rank, tag=2 * tag + 1,
                                         block=False))
    else:
        rbuf, sbuf = np.empty_like(byte), byte
        requests.append(comm.receive(rbuf, root, tag=2 * tag + 0, block=False))
        requests.append(comm.send(sbuf, root, tag=2 * tag + 1, block=False))

    if comm.size == 1 or timeout is None:
        return requests

    t0 = time.time()
    while not comm.testall(requests):  # automatic clean-up upon success
        if time.time() - t0 > timeout:
            raise RuntimeError('MPI barrier timeout.')
    return []

    
def run(iterators):
    """Run through list of iterators one step at a time."""
    if not isinstance(iterators, list):
        # It's a single iterator - empty it:
        for i in iterators:
            pass
        return

    if len(iterators) == 0:
        return

    while True:
        try:
            results = [next(iter) for iter in iterators]
        except StopIteration:
            return results

            
class Parallelization:
    def __init__(self, comm, nspinkpts):
        self.comm = comm
        self.size = comm.size
        self.nspinkpts = nspinkpts
        
        self.kpt = None
        self.domain = None
        self.band = None
        
        self.nclaimed = 1
        self.navail = comm.size

    def set(self, kpt=None, domain=None, band=None):
        if kpt is not None:
            self.kpt = kpt
        if domain is not None:
            self.domain = domain
        if band is not None:
            self.band = band
        
        nclaimed = 1
        for group, name in zip([self.kpt, self.domain, self.band],
                               ['k-point', 'domain', 'band']):
            if group is not None:
                if self.size % group != 0:
                    msg = ('Cannot parallelize as the '
                           'communicator size %d is not divisible by the '
                           'requested number %d of ranks for %s '
                           'parallelization' % (self.size, group, name))
                    raise ValueError(msg)
                nclaimed *= group
        navail = self.size // nclaimed
        
        assert self.size % nclaimed == 0
        assert self.size % navail == 0

        self.navail = navail
        self.nclaimed = nclaimed

    def get_communicator_sizes(self, kpt=None, domain=None, band=None):
        self.set(kpt=kpt, domain=domain, band=band)
        self.autofinalize()
        return self.kpt, self.domain, self.band

    def build_communicators(self, kpt=None, domain=None, band=None,
                            order='kbd'):
        """Construct communicators.

        Returns a communicator for k-points, domains, bands and
        k-points/bands.  The last one "unites" all ranks that are
        responsible for the same domain.

        The order must be a permutation of the characters 'kbd', each
        corresponding to each a parallelization mode.  The last
        character signifies the communicator that will be assigned
        contiguous ranks, i.e. order='kbd' will yield contiguous
        domain ranks, whereas order='kdb' will yield contiguous band
        ranks."""
        self.set(kpt=kpt, domain=domain, band=band)
        self.autofinalize()
        
        comm = self.comm
        rank = comm.rank
        communicators = {}
        parent_stride = self.size
        offset = 0

        groups = dict(k=self.kpt, b=self.band, d=self.domain)

        # Build communicators in hierarchical manner
        # The ranks in the first group have largest separation while
        # the ranks in the last group are next to each other
        for name in order:
            group = groups[name]
            stride = parent_stride // group
            # First rank in this group
            r0 = rank % stride + offset
            # Last rank in this group
            r1 = r0 + stride * group
            ranks = np.arange(r0, r1, stride)
            communicators[name] = comm.new_communicator(ranks)
            parent_stride = stride
            # Offset for the next communicator
            offset += communicators[name].rank * stride

        # We want a communicator for kpts/bands, i.e. the complement of the
        # grid comm: a communicator uniting all cores with the same domain.
        c1, c2, c3 = [communicators[name] for name in order]
        allranks = [range(c1.size), range(c2.size), range(c3.size)]
        
        def get_communicator_complement(name):
            relevant_ranks = list(allranks)
            relevant_ranks[order.find(name)] = [communicators[name].rank]
            ranks = np.array([r3 + c3.size * (r2 + c2.size * r1)
                              for r1 in relevant_ranks[0]
                              for r2 in relevant_ranks[1]
                              for r3 in relevant_ranks[2]])
            return comm.new_communicator(ranks)
        
        # The communicator of all processes that share a domain, i.e.
        # the combination of k-point and band dommunicators.
        communicators['D'] = get_communicator_complement('d')
        # For each k-point comm rank, a communicator of all
        # band/domain ranks.  This is typically used with ScaLAPACK
        # and LCAO orbital stuff.
        communicators['K'] = get_communicator_complement('k')
        return communicators
    
    def autofinalize(self):
        if self.kpt is None:
            self.set(kpt=self.get_optimal_kpt_parallelization())
        if self.domain is None:
            self.set(domain=self.navail)
        if self.band is None:
            self.set(band=self.navail)

        if self.navail > 1:
            assignments = dict(kpt=self.kpt,
                               domain=self.domain,
                               band=self.band)
            raise RuntimeError('All the CPUs must be used.  Have %s but '
                               '%d times more are available'
                               % (assignments, self.navail))
    
    def get_optimal_kpt_parallelization(self, kptprioritypower=1.4):
        if self.domain and self.band:
            # Try to use all the CPUs for k-point parallelization
            ncpus = min(self.nspinkpts, self.navail)
            return ncpus
        ncpuvalues, wastevalues = self.find_kpt_parallelizations()
        scores = ((self.navail // ncpuvalues)
                  * ncpuvalues**kptprioritypower)**(1.0 - wastevalues)
        arg = np.argmax(scores)
        ncpus = ncpuvalues[arg]
        return ncpus

    def find_kpt_parallelizations(self):
        nspinkpts = self.nspinkpts
        ncpuvalues = []
        wastevalues = []
        
        ncpus = nspinkpts
        while ncpus > 0:
            if self.navail % ncpus == 0:
                nkptsmax = -(-nspinkpts // ncpus)
                effort = nkptsmax * ncpus
                efficiency = nspinkpts / float(effort)
                waste = 1.0 - efficiency
                wastevalues.append(waste)
                ncpuvalues.append(ncpus)
            ncpus -= 1
        return np.array(ncpuvalues), np.array(wastevalues)


def cleanup():
    error = getattr(sys, 'last_type', None)
    if error is not None:  # else: Python script completed or raise SystemExit
        if parallel:
            sys.stdout.flush()
            sys.stderr.write(('PYSCF CLEANUP (node %d): %s occurred.  '
                              'Calling MPI_Abort!\n') % (world.rank, error))
            sys.stderr.flush()
            # Give other nodes a moment to crash by themselves (perhaps
            # producing helpful error messages)
            time.sleep(10)
            world.abort(42)


def print_mpi_stack_trace(type, value, tb):
    """Format exceptions nicely when running in parallel.

    Use this function as an except hook.  Adds rank
    and line number to each line of the exception.  Lines will
    still be printed from different ranks in random order, but
    one can grep for a rank or run 'sort' on the output to obtain
    readable data."""
    
    exception_text = traceback.format_exception(type, value, tb)
    ndigits = len(str(world.size - 1))
    rankstring = ('%%0%dd' % ndigits) % world.rank
    
    lines = []
    # The exception elements may contain newlines themselves
    for element in exception_text:
        lines.extend(element.splitlines())

    line_ndigits = len(str(len(lines) - 1))

    for lineno, line in enumerate(lines):
        lineno = ('%%0%dd' % line_ndigits) % lineno
        sys.stderr.write('rank=%s L%s: %s\n' % (rankstring, lineno, line))

if world.size > 1:  # Triggers for dry-run communicators too, but we care not.
    sys.excepthook = print_mpi_stack_trace

            
def exit(error='Manual exit'):
    # Note that exit must be called on *all* MPI tasks
    atexit._exithandlers = []  # not needed because we are intentially exiting
    if parallel:
        sys.stdout.flush()
        sys.stderr.write(('Pyscf CLEANUP (node %d): %s occurred.  ' +
                          'Calling MPI_Finalize!\n') % (world.rank, error))
        sys.stderr.flush()
    else:
        cleanup(error)
    world.barrier()  # sync up before exiting
    sys.exit()  # quit for serial case, return to libmpi.c for parallel case

atexit.register(cleanup)
