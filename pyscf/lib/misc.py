#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Some helper functions
'''

import os
import sys
import time
import random
import platform
import warnings
import tempfile
import functools
import itertools
import inspect
import collections
import pickle
import weakref
import ctypes
import numpy
import scipy
import h5py
from threading import Thread
from multiprocessing import Queue, Process
try:
    from concurrent.futures import ThreadPoolExecutor
except ImportError:
    ThreadPoolExecutor = None

if sys.platform.startswith('linux'):
    # Avoid too many threads being created in OMP loops.
    # See issue https://github.com/pyscf/pyscf/issues/317
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        pass
    else:
        def _ldd(so_file):
            libs = []
            with open(so_file, 'rb') as f:
                elf = ELFFile(f)
                for seg in elf.iter_segments():
                    if seg.header.p_type != 'PT_DYNAMIC':
                        continue
                    for t in seg.iter_tags():
                        if t.entry.d_tag == 'DT_NEEDED':
                            libs.append(t.needed)
                    break
            return libs

        so_file = os.path.abspath(os.path.join(__file__, '..', 'libnp_helper.so'))
        for p in _ldd(so_file):
            if 'mkl' in p and 'thread' in p:
                warnings.warn(f'PySCF C exteions are incompatible with {p}. '
                              'MKL_NUM_THREADS is set to 1')
                os.environ['MKL_NUM_THREADS'] = '1'
                break
            elif 'openblasp' in p or 'openblaso' in p:
                warnings.warn(f'PySCF C exteions are incompatible with {p}. '
                              'OPENBLAS_NUM_THREADS is set to 1')
                os.environ['OPENBLAS_NUM_THREADS'] = '1'
                break
        del p, so_file, _ldd

from pyscf.lib import parameters as param
from pyscf import __config__

if h5py.version.version[:4] == '2.2.':
    sys.stderr.write('h5py-%s is found in your environment. '
                     'h5py-%s has bug in threading mode.\n'
                     'Async-IO is disabled.\n' % ((h5py.version.version,)*2))

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_null_ptr = ctypes.POINTER(ctypes.c_void_p)

def load_library(libname):
    try:
        _loaderpath = os.path.dirname(__file__)
        return numpy.ctypeslib.load_library(libname, _loaderpath)
    except OSError:
        from pyscf import __path__ as ext_modules
        for path in ext_modules:
            libpath = os.path.join(path, 'lib')
            if os.path.isdir(libpath):
                for files in os.listdir(libpath):
                    if files.startswith(libname):
                        return numpy.ctypeslib.load_library(libname, libpath)
        raise

#Fixme, the standard resource module gives wrong number when objects are released
# http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/#fn:1
#or use slow functions as memory_profiler._get_memory did
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
PAGESIZE = os.sysconf("SC_PAGE_SIZE")
def current_memory():
    '''Return the size of used memory and allocated virtual memory (in MB)'''
    #import resource
    #return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
    if sys.platform.startswith('linux'):
        with open("/proc/%s/statm" % os.getpid()) as f:
            vms, rss = [int(x)*PAGESIZE for x in f.readline().split()[:2]]
            return rss/1e6, vms/1e6
    else:
        return 0, 0

def num_threads(n=None):
    '''Set the number of OMP threads.  If argument is not specified, the
    function will return the total number of available OMP threads.

    It's recommended to call this function to set OMP threads than
    "os.environ['OMP_NUM_THREADS'] = int(n)". This is because environment
    variables like OMP_NUM_THREADS were read when a module was imported. They
    cannot be reset through os.environ after the module was loaded.

    Examples:

    >>> from pyscf import lib
    >>> print(lib.num_threads())
    8
    >>> lib.num_threads(4)
    4
    >>> print(lib.num_threads())
    4
    '''
    from pyscf.lib.numpy_helper import _np_helper
    if n is not None:
        _np_helper.set_omp_threads.restype = ctypes.c_int
        threads = _np_helper.set_omp_threads(ctypes.c_int(int(n)))
        if threads == 0:
            warnings.warn('OpenMP is not available. '
                          'Setting omp_threads to %s has no effects.' % n)
        return threads
    else:
        _np_helper.get_omp_threads.restype = ctypes.c_int
        return _np_helper.get_omp_threads()

class with_omp_threads:
    '''Using this macro to create a temporary context in which the number of
    OpenMP threads are set to the required value. When the program exits the
    context, the number OpenMP threads will be restored.

    Args:
        nthreads : int

    Examples:

    >>> from pyscf import lib
    >>> print(lib.num_threads())
    8
    >>> with lib.with_omp_threads(2):
    ...     print(lib.num_threads())
    2
    >>> print(lib.num_threads())
    8
    '''
    def __init__(self, nthreads=None):
        self.nthreads = nthreads
        self.sys_threads = None
    def __enter__(self):
        if self.nthreads is not None and self.nthreads >= 1:
            self.sys_threads = num_threads()
            num_threads(self.nthreads)
        return self
    def __exit__(self, type, value, traceback):
        if self.sys_threads is not None:
            num_threads(self.sys_threads)

class with_multiproc_nproc:
    '''
    Using this macro to create a temporary context in which the number of
    multi-processing processes are set to the required value.
    '''
    def __init__(self, nproc=1):
        self.nproc = nproc
        self.sys_threads = None
    def __enter__(self):
        if self.nproc is not None and self.nproc >= 1:
            self.sys_threads = num_threads()
            if self.nproc > self.sys_threads:
                warnings.warn('Reset nproc to nthreads: %s' % self.sys_threads)
                self.nproc = self.sys_threads
            nthreads = max(1, self.sys_threads // self.nproc)
            num_threads(nthreads)
        return self
    def __exit__(self, type, value, traceback):
        if self.sys_threads is not None:
            num_threads(self.sys_threads)
            self.nproc = 1

def c_int_arr(m):
    npm = numpy.array(m).flatten('C')
    arr = (ctypes.c_int * npm.size)(*npm)
    # cannot return LP_c_double class,
    #Xreturn npm.ctypes.data_as(c_int_p), which destructs npm before return
    return arr
def f_int_arr(m):
    npm = numpy.array(m).flatten('F')
    arr = (ctypes.c_int * npm.size)(*npm)
    return arr
def c_double_arr(m):
    npm = numpy.array(m).flatten('C')
    arr = (ctypes.c_double * npm.size)(*npm)
    return arr
def f_double_arr(m):
    npm = numpy.array(m).flatten('F')
    arr = (ctypes.c_double * npm.size)(*npm)
    return arr


def member(test, x, lst):
    for l in lst:
        if test(x, l):
            return True
    return False

def remove_dup(test, lst, from_end=False):
    if test is None:
        return set(lst)
    else:
        if from_end:
            lst = list(reversed(lst))
        seen = []
        for l in lst:
            if not member(test, l, seen):
                seen.append(l)
        return seen

def remove_if(test, lst):
    return [x for x in lst if not test(x)]

def find_if(test, lst):
    for l in lst:
        if test(l):
            return l
    raise ValueError('No element of the given list matches the test condition.')

def arg_first_match(test, lst):
    for i,x in enumerate(lst):
        if test(x):
            return i
    raise ValueError('No element of the given list matches the test condition.')

def _balanced_partition(cum, ntasks):
    segsize = float(cum[-1]) / ntasks
    bounds = numpy.arange(ntasks+1) * segsize
    displs = abs(bounds[:,None] - cum).argmin(axis=1)
    return displs

def _blocksize_partition(cum, blocksize):
    n = len(cum) - 1
    displs = [0]
    if n == 0:
        return displs

    p0 = 0
    for i in range(1, n):
        if cum[i+1]-cum[p0] > blocksize:
            displs.append(i)
            p0 = i
    displs.append(n)
    return displs

def flatten(lst):
    '''flatten nested lists
    x[0] + x[1] + x[2] + ...

    Examples:

    >>> flatten([[0, 2], [1], [[9, 8, 7]]])
    [0, 2, 1, [9, 8, 7]]
    '''
    return list(itertools.chain.from_iterable(lst))

def prange(start, end, step):
    '''This function splits the number sequence between "start" and "end"
    using uniform "step" length. It yields the boundary (start, end) for each
    fragment.

    Examples:

    >>> for p0, p1 in lib.prange(0, 8, 2):
    ...    print(p0, p1)
    (0, 2)
    (2, 4)
    (4, 6)
    (6, 8)
    '''
    if start < end:
        for i in range(start, end, step):
            yield i, min(i+step, end)

def prange_tril(start, stop, blocksize):
    '''Similar to :func:`prange`, yields start (p0) and end (p1) with the
    restriction p1*(p1+1)/2-p0*(p0+1)/2 < blocksize

    Examples:

    >>> for p0, p1 in lib.prange_tril(0, 10, 25):
    ...     print(p0, p1)
    (0, 6)
    (6, 9)
    (9, 10)
    '''
    if start >= stop:
        return []
    idx = numpy.arange(start, stop+1)
    cum_costs = idx*(idx+1)//2 - start*(start+1)//2
    displs = [x+start for x in _blocksize_partition(cum_costs, blocksize)]
    return zip(displs[:-1], displs[1:])

def prange_split(n_total, n_sections):
    '''
    Generate prange sequence that splits n_total elements into n sections.
    The splits parallel to the np.array_split convention: the first (l % n)
    sections of size l//n + 1 and the rest of size l//n.

    Examples:

    >>> for p0, p1 in lib.prange_split(10, 3):
    ...     print(p0, p1)
    (0, 4)
    (4, 7)
    (7, 10)
    '''
    n_each_section, extras = divmod(n_total, n_sections)
    section_sizes = ([0] +
                     extras * [n_each_section+1] +
                     (n_sections-extras) * [n_each_section])
    div_points = numpy.array(section_sizes).cumsum()
    return zip(div_points[:-1], div_points[1:])

izip = zip

if sys.version_info > (3, 8):
    from math import comb
else:
    import math
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return math.factorial(n) // math.factorial(n-k) // math.factorial(k)

def map_with_prefetch(func, *iterables):
    '''
    Apply function to an task and prefetch the next task
    '''
    global_import_lock = False
    if sys.version_info < (3, 6):
        import imp
        global_import_lock = imp.lock_held()

    if not ASYNC_IO or global_import_lock:
        for task in zip(*iterables):
            yield func(*task)

    elif ThreadPoolExecutor is not None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            for task in zip(*iterables):
                if future is None:
                    future = executor.submit(func, *task)
                else:
                    result = future.result()
                    future = executor.submit(func, *task)
                    yield result
            if future is not None:
                yield future.result()
    else:
        def func_with_buf(_output_buf, *args):
            _output_buf[0] = func(*args)
        with call_in_background(func_with_buf) as f_prefetch:
            buf0, buf1 = [None], [None]
            for istep, task in enumerate(zip(*iterables)):
                if istep == 0:
                    f_prefetch(buf0, *task)
                else:
                    buf0, buf1 = buf1, buf0
                    f_prefetch(buf0, *task)
                    yield buf1[0]
        if buf0[0] is not None:
            yield buf0[0]

def index_tril_to_pair(ij):
    '''Given tril-index ij, compute the pair indices (i,j) which satisfy
    ij = i * (i+1) / 2 + j
    '''
    i = (numpy.sqrt(2*ij+.25) - .5 + 1e-7).astype(int)
    j = ij - i*(i+1)//2
    return i, j


def tril_product(*iterables, **kwds):
    '''Cartesian product in lower-triangular form for multiple indices

    For a given list of indices (`iterables`), this function yields all
    indices such that the sub-indices given by the kwarg `tril_idx` satisfy a
    lower-triangular form.  The lower-triangular form satisfies:

    .. math:: i[tril_idx[0]] >= i[tril_idx[1]] >= ... >= i[tril_idx[len(tril_idx)-1]]

    Args:
        ``*iterables``: Variable length argument list of indices for the cartesian product
        ``**kwds``: Arbitrary keyword arguments.  Acceptable keywords include:
            repeat (int): Number of times to repeat the iterables
            tril_idx (array_like): Indices to put into lower-triangular form.

    Yields:
        product (tuple): Tuple in lower-triangular form.

    Examples:
        Specifying no `tril_idx` is equivalent to just a cartesian product.

        >>> list(tril_product(range(2), repeat=2))
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        We can specify only sub-indices to satisfy a lower-triangular form:

        >>> list(tril_product(range(2), repeat=3, tril_idx=[1,2]))
        [(0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]

        We specify all indices to satisfy a lower-triangular form, useful for iterating over
        the symmetry unique elements of occupied/virtual orbitals in a 3-particle operator:

        >>> list(tril_product(range(3), repeat=3, tril_idx=[0,1,2]))
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 2, 0), (2, 2, 1), (2, 2, 2)]
    '''
    repeat = kwds.get('repeat', 1)
    tril_idx = kwds.get('tril_idx', [])
    niterables = len(iterables) * repeat
    ntril_idx = len(tril_idx)

    assert ntril_idx <= niterables, 'Cant have a greater number of tril indices than iterables!'
    if ntril_idx > 0:
        assert numpy.max(tril_idx) < niterables, 'Tril index out of bounds for %d iterables! idx = %s' % \
                                                 (niterables, tril_idx)
    for tup in itertools.product(*iterables, repeat=repeat):
        if ntril_idx == 0:
            yield tup
            continue

        if all(tup[tril_idx[i]] >= tup[tril_idx[i+1]] for i in range(ntril_idx-1)):
            yield tup
        else:
            pass

def square_mat_in_trilu_indices(n):
    '''Return a n x n symmetric index matrix, in which the elements are the
    indices of the unique elements of a tril vector
    [0 1 3 ... ]
    [1 2 4 ... ]
    [3 4 5 ... ]
    [...       ]
    '''
    idx = numpy.tril_indices(n)
    tril2sq = numpy.zeros((n,n), dtype=int)
    tril2sq[idx[0],idx[1]] = tril2sq[idx[1],idx[0]] = numpy.arange(n*(n+1)//2)
    return tril2sq

class capture_stdout:
    '''redirect all stdout (c printf & python print) into a string

    Examples:

    >>> import os
    >>> from pyscf import lib
    >>> with lib.capture_stdout() as out:
    ...     os.system('ls')
    >>> print(out.read())
    '''
    #TODO: handle stderr
    def __enter__(self):
        sys.stdout.flush()
        self._contents = None
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.ftmp = tempfile.NamedTemporaryFile(dir=param.TMPDIR)
        os.dup2(self.ftmp.file.fileno(), self.old_stdout_fileno)
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        self.ftmp.file.seek(0)
        self._contents = self.ftmp.file.read()
        self.ftmp.close()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        os.close(self.bak_stdout_fd)
    def read(self):
        if self._contents:
            return self._contents
        else:
            sys.stdout.flush()
            self.ftmp.file.seek(0)
            return self.ftmp.file.read()
ctypes_stdout = capture_stdout

class quite_run:
    '''capture all stdout (c printf & python print) but output nothing

    Examples:

    >>> import os
    >>> from pyscf import lib
    >>> with lib.quite_run():
    ...     os.system('ls')
    '''
    def __enter__(self):
        sys.stdout.flush()
        #TODO: to handle the redirected stdout e.g. StringIO()
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.fnull = open(os.devnull, 'wb')
        os.dup2(self.fnull.fileno(), self.old_stdout_fileno)
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        self.fnull.close()


# from pygeocoder
# this decorator lets me use methods as both static and instance methods
# In contrast to classmethod, when obj.function() is called, the first
# argument is obj in omnimethod rather than obj.__class__ in classmethod
class omnimethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return functools.partial(self.func, instance)

def view(obj, cls):
    '''New view of object with the same attributes.'''
    new_obj = cls.__new__(cls)
    new_obj.__dict__.update(obj.__dict__)
    return new_obj

def generate_pickle_methods(excludes=(), reset_state=False):
    '''Generate methods for pickle, e.g.:

    class A:
        __getstate__, __setstate__ = generate_pickle_methods(excludes=('a', 'b', 'c'))
    '''
    def getstate(obj):
        dic = {**obj.__dict__}
        dic.pop('stdout', None)
        for key in excludes:
            dic.pop(key, None)
        return dic

    def setstate(obj, state):
        obj.stdout = sys.stdout
        obj.__dict__.update(state)
        for key in excludes:
            setattr(obj, key, None)
        if reset_state and hasattr(obj, 'reset'):
            obj.reset()

    return getstate, setstate


SANITY_CHECK = getattr(__config__, 'SANITY_CHECK', True)
class StreamObject:
    '''For most methods, there are three stream functions to pipe computing stream:

    1 ``.set_`` function to update object attributes, eg
    ``mf = scf.RHF(mol).set(conv_tol=1e-5)`` is identical to proceed in two steps
    ``mf = scf.RHF(mol); mf.conv_tol=1e-5``

    2 ``.run`` function to execute the kernel function (the function arguments
    are passed to kernel function).  If keyword arguments is given, it will first
    call ``.set`` function to update object attributes then execute the kernel
    function.  Eg
    ``mf = scf.RHF(mol).run(dm_init, conv_tol=1e-5)`` is identical to three steps
    ``mf = scf.RHF(mol); mf.conv_tol=1e-5; mf.kernel(dm_init)``

    3 ``.apply`` function to apply the given function/class to the current object
    (function arguments and keyword arguments are passed to the given function).
    Eg
    ``mol.apply(scf.RHF).run().apply(mcscf.CASSCF, 6, 4, frozen=4)`` is identical to
    ``mf = scf.RHF(mol); mf.kernel(); mcscf.CASSCF(mf, 6, 4, frozen=4)``
    '''

    verbose = 0
    stdout = sys.stdout
    # Store the keys appeared in the module.  It is used to check misinput attributes
    _keys = {'output', 'verbose', 'stdout', 'max_memory'}

    def kernel(self, *args, **kwargs):
        '''
        Kernel function is the main driver of a method.  Every method should
        define the kernel function as the entry of the calculation.  Note the
        return value of kernel function is not strictly defined.  It can be
        anything related to the method (such as the energy, the wave-function,
        the DFT mesh grids etc.).
        '''
        pass

    def pre_kernel(self, envs):
        '''
        A hook to be run before the main body of kernel function is executed.
        Internal variables are exposed to pre_kernel through the "envs"
        dictionary.  Return value of pre_kernel function is not required.
        '''
        pass

    def post_kernel(self, envs):
        '''
        A hook to be run after the main body of the kernel function.  Internal
        variables are exposed to post_kernel through the "envs" dictionary.
        Return value of post_kernel function is not required.
        '''
        pass

    def run(self, *args, **kwargs):
        '''
        Call the kernel function of current object.  `args` will be passed
        to kernel function.  `kwargs` will be used to update the attributes of
        current object.  The return value of method run is the object itself.
        This allows a series of functions/methods to be executed in pipe.
        '''
        self.set(**kwargs)
        self.kernel(*args)
        return self

    def set(self, *args, **kwargs):
        '''
        Update the attributes of the current object.  The return value of
        method set is the object itself.  This allows a series of
        functions/methods to be executed in pipe.
        '''
        if args:
            warnings.warn('method set() only supports keyword arguments.\n'
                          'Arguments %s are ignored.' % args)
        #keys_ref = set(self._keys)
        #cls_keys = [cls._keys for cls in self.__class__.__mro__[:-1]
        #            if hasattr(cls, '_keys')]
        #keys_ref = keys_ref.union(*cls_keys)
        #unknown_keys = set(kwargs).difference(keys_ref)
        #if unknown_keys:
        #    warnings.warn(f'{self.__class__} does not have attributes {unknown_keys}')

        for k,v in kwargs.items():
            setattr(self, k, v)
        return self

    # An alias to .set method
    __call__ = set

    def apply(self, fn, *args, **kwargs):
        '''
        Apply the fn to rest arguments:  return ``fn(*args, **kwargs)``.  The
        return value of method set is the object itself.  This allows a series
        of functions/methods to be executed in pipe.
        '''
        return fn(self, *args, **kwargs)

#    def _format_args(self, args, kwargs, kernel_kw_lst):
#        args1 = [kwargs.pop(k, v) for k, v in kernel_kw_lst]
#        return args + args1[len(args):], kwargs

    def check_sanity(self):
        '''
        Check input of class/object attributes, check whether a class method is
        overwritten.  It does not check the attributes which are prefixed with
        "_".  The
        return value of method set is the object itself.  This allows a series
        of functions/methods to be executed in pipe.
        '''
        if SANITY_CHECK and self.verbose > 0:
            cls_keys = [cls._keys for cls in self.__class__.__mro__[:-1]
                        if hasattr(cls, '_keys')]
            keys_ref = set(self._keys).union(*cls_keys)
            check_sanity(self, keys_ref, self.stdout)
        return self

    view = view

    def copy(self):
        '''Returns a shallow copy'''
        return self.view(self.__class__)

    __getstate__, __setstate__ = generate_pickle_methods()

    def reset(self):
        '''
        Clean up intermediates
        '''
        raise NotImplementedError


_warn_once_registry = {}
def check_sanity(obj, keysref, stdout=sys.stdout):
    '''Check misinput of class attributes, check whether a class method is
    overwritten.  It does not check the attributes which are prefixed with
    "_".
    '''
    objkeys = [x for x in obj.__dict__ if not x.startswith('_')]
    keysub = set(objkeys) - set(keysref)
    if keysub:
        class_attr = set(dir(obj.__class__))
        keyin = keysub.intersection(class_attr)
        if keyin:
            msg = ('Overwritten attributes  %s  of %s\n' %
                   (' '.join(keyin), obj.__class__))
            if msg not in _warn_once_registry:
                _warn_once_registry[msg] = 1
                sys.stderr.write(msg)
                if stdout is not sys.stdout:
                    stdout.write(msg)
        keydiff = keysub - class_attr
        if keydiff:
            msg = ('%s does not have attributes  %s\n' %
                   (obj.__class__, ' '.join(keydiff)))
            if msg not in _warn_once_registry:
                _warn_once_registry[msg] = 1
                sys.stderr.write(msg)
                if stdout is not sys.stdout:
                    stdout.write(msg)
    return obj

def with_doc(doc):
    '''Use this decorator to add doc string for function

        @with_doc(doc)
        def fn:
            ...

    is equivalent to

        fn.__doc__ = doc
    '''
    def fn_with_doc(fn):
        fn.__doc__ = doc
        return fn
    return fn_with_doc

def alias(fn, alias_name=None):
    '''
    The statement "fn1 = alias(fn)" in a class is equivalent to define the
    following method in the class:

    .. code-block:: python
        def fn1(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    Using alias function instead of fn1 = fn because some methods may be
    overloaded in the child class. Using "alias" can make sure that the
    overloaded methods were called when calling the aliased method.
    '''
    name = fn.__name__
    if alias_name is None:
        alias_name = name

    _locals = {}
    sig = inspect.signature(fn)
    var_args = []
    for k, v in sig.parameters.items():
        if v.kind == v.VAR_POSITIONAL or v.kind == v.VAR_KEYWORD:
            var_args.append(str(v))
        else:
            var_args.append(k)
    txt = f'''def {alias_name}({", ".join(var_args)}):
    return {var_args[0]}.{name}({", ".join(var_args[1:])})'''
    exec(txt, fn.__globals__, _locals)
    new_fn = _locals[alias_name]

    new_fn.__defaults__ = fn.__defaults__
    new_fn.__module__ = fn.__module__
    new_fn.__doc__ = fn.__doc__
    new_fn.__annotations__ = fn.__annotations__
    return new_fn

def module_method(fn, absences=None):
    '''
    The statement "fn1 = module_method(fn, absences=['a'])"
    in a class is equivalent to define the following method in the class:

    .. code-block:: python
        def fn1(self, ..., a=None, b, ...):
            if a is None: a = self.a
            return fn(..., a, b, ...)

    If absences are not specified, all position arguments will be assigned in
    terms of the corresponding attributes of self, i.e.

    .. code-block:: python
        def fn1(self, a=None, b=None):
            if a is None: a = self.a
            if b is None: b = self.b
            return fn(a, b)

    This function can be used to replace "staticmethod" when inserting a module
    method into a class. In a child class, it allows one to call the method of a
    base class with either "self.__class__.method_name(self, args)" or
    "self.super().method_name(args)". For method created with "staticmethod",
    calling "self.super().method_name(args)" is the only option.
    '''
    _locals = {}
    name = fn.__name__
    sig = inspect.signature(fn)
    body = []
    var_args = []
    for k, v in sig.parameters.items():
        if v.kind == v.VAR_POSITIONAL or v.kind == v.VAR_KEYWORD:
            var_args.append(str(v))
        else:
            var_args.append(k)
            if absences is None and v.default == v.empty:  # positional argument
                body.append(f'    if {k} is None: {k} = self.{k}')

    fn_defaults = fn.__defaults__
    nargs = fn.__code__.co_argcount
    if fn_defaults is None:
        fn_defaults = [None] * nargs
    else:
        fn_defaults = [None] * (nargs-len(fn_defaults)) + list(fn_defaults)

    if absences is not None:
        for k in absences:
            try:
                idx = var_args.index(k)
            except ValueError:
                raise ValueError(f'Unknown argument {k}')
            body.append(f'    if {k} is None: {k} = self.{k}')
            fn_defaults[idx] = None

    body = '\n'.join(body)
    txt = f'''def {name}(self, {", ".join(var_args)}):
{body}
    return {name}({", ".join(var_args)})'''
    exec(txt, fn.__globals__, _locals)
    new_fn = _locals[name]
    new_fn.__module__ = fn.__module__
    new_fn.__defaults__ = tuple(fn_defaults)
    new_fn.__doc__ = fn.__doc__
    new_fn.__annotations__ = fn.__annotations__
    return new_fn

def class_as_method(cls):
    '''
    The statement "fn1 = class_as_method(Class)" is equivalent to:

    .. code-block:: python
        def fn1(self, *args, **kwargs):
            return Class(self, *args, **kwargs)
    '''
    def fn(obj, *args, **kwargs):
        return cls(obj, *args, **kwargs)
    fn.__doc__ = cls.__doc__
    fn.__name__ = cls.__name__
    fn.__module__ = cls.__module__
    return fn

def invalid_method(name):
    '''
    The statement "fn1 = invalid_method(name)" can de-register a method
    '''
    def fn(obj, *args, **kwargs):
        raise NotImplementedError(f'Method {name} invalid or not implemented')
    fn.__name__ = name
    return fn

_registered_classes = {}
def make_class(bases, name=None, attrs=None):
    '''
    Construct a class

    .. code-block:: python

        class {name}(*bases):
            __dict__ = attrs
    '''
    _registered_classes
    if name is None:
        name = ''.join(getattr(x, '__name_mixin__', x.__name__) for x in bases)

    cls = _registered_classes.get((name, bases))
    if cls is None:
        if attrs is None:
            attrs = {}
        cls = type(name, bases, attrs)
        cls.__name_mixin__ = name
        _registered_classes[name, bases] = cls
    return cls

def set_class(obj, bases, name=None, attrs=None):
    '''Change the class of an object'''
    cls = make_class(bases, name, attrs)
    cls.__module__ = obj.__class__.__module__
    obj.__class__ = cls
    return obj

def drop_class(cls, base_cls, name_mixin=None):
    '''Recursively remove the first matched base_cls from cls MRO
    '''
    filter_bases = list(cls.__bases__)
    force_rebuild = False
    for i, base in enumerate(cls.__bases__):
        if base == base_cls:
            filter_bases[i] = None
            break
        elif issubclass(base, base_cls):
            filter_bases[i] = cls_undressed = drop_class(base, base_cls, name_mixin)
            force_rebuild = cls_undressed is not None
            break
    else:
        raise RuntimeError(f'class {base_cls} not found in {cls} MRO')

    filter_bases = [x for x in filter_bases if x is not None]
    if len(filter_bases) < 1:
        # cls is the singly inherited sub-class of base_cls
        return None
    elif not force_rebuild and len(filter_bases) == 1:
        return filter_bases[0]

    if name_mixin is None:
        name_mixin = getattr(base_cls, '__name_mixin__', base_cls.__name__)
    cls_name = cls.__name__.replace(name_mixin, '', 1)

    # rebuild the dynamic_mixin class
    attrs = {**cls.__dict__, '__name_mixin__': cls_name}
    cls_undressed = make_class(tuple(filter_bases), cls_name, attrs)
    cls_undressed.__module__ = cls.__module__
    return cls_undressed

def replace_class(cls, old_cls, new_cls):
    '''Replace the first matched class in MRO
    '''
    if cls == old_cls:
        return new_cls

    bases = list(cls.__bases__)
    any_match = False
    for i, base in enumerate(cls.__bases__):
        if issubclass(base, old_cls):
            bases[i] = replace_class(base, old_cls, new_cls)
            any_match = True
            break

    if not any_match:
        return cls

    name = cls.__name__.replace(old_cls.__name__, new_cls.__name__)
    attrs = {**cls.__dict__, '__name_mixin__': name}
    _cls = make_class(tuple(bases), name, attrs)
    _cls.__module__ = cls.__module__
    return _cls

def overwrite_mro(obj, mro):
    '''A hacky function to overwrite the __mro__ attribute'''
    raise DeprecationWarning

class ProcessWithReturnValue(Process):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        self._q = Queue()
        self._e = None
        def qwrap(*args, **kwargs):
            try:
                self._q.put(target(*args, **kwargs))
            except BaseException as e:
                self._e = e
                raise e
        Process.__init__(self, group, qwrap, name, args, kwargs)
    def join(self):
        Process.join(self)
        if self._e is not None:
            raise ProcessRuntimeError('Error on process %s:\n%s' % (self, self._e))
        else:
            return self._q.get()
    get = join

class ProcessRuntimeError(RuntimeError):
    pass

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        self._q = Queue()
        self._e = None
        def qwrap(*args, **kwargs):
            try:
                self._q.put(target(*args, **kwargs))
            except BaseException as e:
                self._e = e
                raise e
        Thread.__init__(self, group, qwrap, name, args, kwargs)
    def join(self):
        Thread.join(self)
        if self._e is not None:
            raise ThreadRuntimeError('Error on thread %s:\n%s' % (self, self._e))
        else:
            # Note: If the return value of target is huge, Queue.get may raise
            # SystemError: NULL result without error in PyObject_Call
            # It is because return value is cached somewhere by pickle but pickle is
            # unable to handle huge amount of data.
            return self._q.get()
    get = join

class ThreadWithTraceBack(Thread):
    def __init__(self, group=None, target=None, name=None, args=(),
                 kwargs=None):
        self._e = None
        def qwrap(*args, **kwargs):
            try:
                target(*args, **kwargs)
            except BaseException as e:
                self._e = e
                raise e
        Thread.__init__(self, group, qwrap, name, args, kwargs)
    def join(self):
        Thread.join(self)
        if self._e is not None:
            raise ThreadRuntimeError('Error on thread %s:\n%s' % (self, self._e))

class ThreadRuntimeError(RuntimeError):
    pass

def background_thread(func, *args, **kwargs):
    '''applying function in background'''
    thread = ThreadWithReturnValue(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread

def background_process(func, *args, **kwargs):
    '''applying function in background'''
    thread = ProcessWithReturnValue(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread

bg = background = bg_thread = background_thread
bp = bg_process = background_process

ASYNC_IO = getattr(__config__, 'ASYNC_IO', True)
class call_in_background:
    '''Within this macro, function(s) can be executed asynchronously (the
    given functions are executed in background).

    Attributes:
        sync (bool): Whether to run in synchronized mode.  The default value
            is False (asynchronized mode).

    Examples:

    >>> with call_in_background(fun) as async_fun:
    ...     async_fun(a, b)  # == fun(a, b)
    ...     do_something_else()

    >>> with call_in_background(fun1, fun2) as (afun1, afun2):
    ...     afun2(a, b)
    ...     do_something_else()
    ...     afun2(a, b)
    ...     do_something_else()
    ...     afun1(a, b)
    ...     do_something_else()
    '''

    def __init__(self, *fns, **kwargs):
        self.fns = fns
        self.executor = None
        self.handlers = [None] * len(self.fns)
        self.sync = kwargs.get('sync', not ASYNC_IO)

    if h5py.version.version[:4] == '2.2.': # h5py-2.2.* has bug in threading mode
        # Disable back-ground mode
        def __enter__(self):
            if len(self.fns) == 1:
                return self.fns[0]
            else:
                return self.fns

    else:
        def __enter__(self):
            fns = self.fns
            handlers = self.handlers
            ntasks = len(self.fns)

            global_import_lock = False
            if sys.version_info < (3, 6):
                import imp
                global_import_lock = imp.lock_held()

            if self.sync or global_import_lock:
                # Some modules like nosetests, coverage etc
                #   python -m unittest test_xxx.py  or  nosetests test_xxx.py
                # hang when Python multi-threading was used in the import stage due to (Python
                # import lock) bug in the threading module.  See also
                # https://github.com/paramiko/paramiko/issues/104
                # https://docs.python.org/2/library/threading.html#importing-in-threaded-code
                # Disable the asynchronous mode for safe importing
                def def_async_fn(i):
                    return fns[i]

            elif ThreadPoolExecutor is None: # async mode, old python
                def def_async_fn(i):
                    def async_fn(*args, **kwargs):
                        if self.handlers[i] is not None:
                            self.handlers[i].join()
                        self.handlers[i] = ThreadWithTraceBack(target=fns[i], args=args,
                                                               kwargs=kwargs)
                        self.handlers[i].start()
                        return self.handlers[i]
                    return async_fn

            else: # multiple executors in async mode, python 2.7.12 or newer
                executor = self.executor = ThreadPoolExecutor(max_workers=ntasks)
                def def_async_fn(i):
                    def async_fn(*args, **kwargs):
                        if handlers[i] is not None:
                            try:
                                handlers[i].result()
                            except Exception as e:
                                raise ThreadRuntimeError('Error on thread %s:\n%s'
                                                         % (self, e))
                        handlers[i] = executor.submit(fns[i], *args, **kwargs)
                        return handlers[i]
                    return async_fn

            if len(self.fns) == 1:
                return def_async_fn(0)
            else:
                return [def_async_fn(i) for i in range(ntasks)]

    def __exit__(self, type, value, traceback):
        for handler in self.handlers:
            if handler is not None:
                try:
                    if ThreadPoolExecutor is None:
                        handler.join()
                    else:
                        handler.result()
                except Exception as e:
                    raise ThreadRuntimeError('Error on thread %s:\n%s' % (self, e))

        if self.executor is not None:
            self.executor.shutdown(wait=True)

class H5FileWrap(h5py.File):
    '''
    A wrapper for h5py.File that allows global options to be set by
    the user via lib.param.H5F_WRITE_KWARGS, which is imported
    upon startup from the user's configuration file.

    These options are, as the name suggests, not used when the
    HDF5 file is opened in read-only mode.

    Example:

    >>> with temporary_env(lib.param, H5F_WRITE_KWARGS={'driver': 'core'}):
    ...     with lib.H5TmpFile() as f:
    ...         print(f.driver)
    core
    '''
    def __init__(self, filename, mode, *args, **kwargs):
        if mode != 'r':
            options = param.H5F_WRITE_KWARGS.copy()
            options.update(kwargs)
        else:
            options = kwargs
        super().__init__(filename, mode, *args, **options)

    def _finished(self):
        '''
        Close the file and flush it if it is open.
        Flushing explicitly should not be necessary:
        this is intended to avoid a bug that unpredictably
        causes outcore DF to hang on an NFS filesystem.
        '''
        try:
            if super().id and super().id.valid:
                super().flush()
            super().close()
        except AttributeError:  # close not defined in old h5py
            pass
        except ValueError:  # if close() is called twice
            pass
        except ImportError:  # exit program before de-referring the object
            pass

    def __del__(self):
        self._finished()

class H5TmpFile(H5FileWrap):
    '''Create and return an HDF5 temporary file.

    Kwargs:
        filename : str or None
            If a string is given, an HDF5 file of the given filename will be
            created. If filename is not specified, the HDF5 temporary
            file will be deleted when the H5TmpFile object is released.

    The return object is an h5py.File object. The file will be automatically
    deleted when it is closed or the object is released (unless filename is
    specified).

    Examples:

    >>> from pyscf import lib
    >>> ftmp = lib.H5TmpFile()
    '''
    def __init__(self, filename=None, mode='a', prefix='', suffix='',
                 dir=param.TMPDIR, *args, **kwargs):
        self.delete_on_close = False
        if filename is None:
            filename = H5TmpFile._gen_unique_name(dir, pre=prefix, suf=suffix)
            self.delete_on_close = True

        def _delete_with_check(fname, should_delete):
            if should_delete and os.path.exists(fname):
                os.remove(fname)

        self._finalizer = weakref.finalize(self, _delete_with_check,
                                           filename, self.delete_on_close)

        super().__init__(filename, mode, *args, **kwargs)

    # Python 3 stdlib does not have a way to just generate
    # temporary file names.
    @staticmethod
    def _gen_unique_name(directory, pre='', suf=''):
        absdir = os.path.abspath(directory)
        random.seed()
        for seq in range(10000):
            name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            filename = os.path.join(absdir, pre + name + suf)
            try:
                f = open(filename, 'x')
            except FileExistsError:
                continue    # try again
            f.close()
            return filename
        raise FileExistsError("No usable temporary file name found")

    def close(self):
        self._finished()
        self._finalizer()

    def __exit__(self, type, value, traceback):
        self.close()


def fingerprint(a):
    '''Fingerprint of numpy array'''
    a = numpy.asarray(a)
    return numpy.dot(numpy.cos(numpy.arange(a.size)), a.ravel())
finger = fp = fingerprint


def ndpointer(*args, **kwargs):
    base = numpy.ctypeslib.ndpointer(*args, **kwargs)

    @classmethod
    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)
    return type(base.__name__, (base,), {'from_param': from_param})


# A tag to label the derived Scanner class
class SinglePointScanner:
    __name_mixin__ = '_Scanner'

    def undo_scanner(self):
        return view(self, drop_class(self.__class__, SinglePointScanner))

class GradScanner:
    __name_mixin__ = '_Scanner'

    def undo_scanner(self):
        return view(self, drop_class(self.__class__, GradScanner))

    def __init__(self, g):
        self.__dict__.update(g.__dict__)
        self.base = g.base.as_scanner()
    @property
    def e_tot(self):
        return self.base.e_tot
    @e_tot.setter
    def e_tot(self, x):
        self.base.e_tot = x

    @property
    def converged(self):
        # Some base methods like MP2 does not have the attribute converged
        conv = getattr(self.base, 'converged', True)
        return conv

class temporary_env:
    '''Within the context of this macro, the attributes of the object are
    temporarily updated. When the program goes out of the scope of the
    context, the original value of each attribute will be restored.

    Examples:

    >>> with temporary_env(lib.param, LIGHT_SPEED=15., BOHR=2.5):
    ...     print(lib.param.LIGHT_SPEED, lib.param.BOHR)
    15. 2.5
    >>> print(lib.param.LIGHT_SPEED, lib.param.BOHR)
    137.03599967994 0.52917721092
    '''
    def __init__(self, obj, **kwargs):
        self.obj = obj

        # Should I skip the keys which are not presented in obj?
        #keys = [key for key in kwargs.keys() if hasattr(obj, key)]
        #self.env_bak = [(key, getattr(obj, key, 'TO_DEL')) for key in keys]
        #self.env_new = [(key, kwargs[key]) for key in keys]

        self.env_bak = [(key, getattr(obj, key, 'TO_DEL')) for key in kwargs]
        self.env_new = [(key, kwargs[key]) for key in kwargs]

    def __enter__(self):
        for k, v in self.env_new:
            setattr(self.obj, k, v)
        return self

    def __exit__(self, type, value, traceback):
        for k, v in self.env_bak:
            if isinstance(v, str) and v == 'TO_DEL':
                delattr(self.obj, k)
            else:
                setattr(self.obj, k, v)

class light_speed(temporary_env):
    '''Within the context of this macro, the environment variable LIGHT_SPEED
    can be customized.

    Examples:

    >>> with light_speed(15.):
    ...     print(lib.param.LIGHT_SPEED)
    15.
    >>> print(lib.param.LIGHT_SPEED)
    137.03599967994
    '''
    def __init__(self, c):
        temporary_env.__init__(self, param, LIGHT_SPEED=c)
        self.c = c
    def __enter__(self):
        temporary_env.__enter__(self)
        return self.c

class h5filewrite_options(temporary_env):
    '''Within the context of this macro, extra keyword arguments are
    passed to h5py.File() whenever an HDF5 file is opened for writing.

    Examples:

    >>> with h5filewrite_options(alignment_interval=4096, alignment_threshold=4096):
    ...     f = lib.H5FileWrap('mydata.h5', 'w')
    >>> print(h5py.h5p.PropFAID.get_alignment(f.id.get_access_plist()))
    (4096, 4096)
    '''
    def __init__(self, **kwargs):
        super().__init__(param, H5F_WRITE_KWARGS=kwargs)

def repo_info(repo_path):
    '''
    Repo location, version, git branch and commit ID
    '''

    def git_version(orig_head, head, branch):
        git_version = []
        if orig_head:
            git_version.append('GIT ORIG_HEAD %s' % orig_head)
        if branch:
            git_version.append('GIT HEAD (branch %s) %s' % (branch, head))
        elif head:
            git_version.append('GIT HEAD      %s' % head)
        return '\n'.join(git_version)

    repo_path = os.path.abspath(repo_path)

    if os.path.isdir(os.path.join(repo_path, '.git')):
        git_str = git_version(*git_info(repo_path))

    elif os.path.isdir(os.path.abspath(os.path.join(repo_path, '..', '.git'))):
        repo_path = os.path.abspath(os.path.join(repo_path, '..'))
        git_str = git_version(*git_info(repo_path))

    else:
        git_str = None

    # TODO: Add info of BLAS, libcint, libxc, libxcfun, tblis if applicable

    info = {'path': repo_path}
    if git_str:
        info['git'] = git_str
    return info

def git_info(repo_path):
    orig_head = None
    head = None
    branch = None
    try:
        with open(os.path.join(repo_path, '.git', 'ORIG_HEAD'), 'r') as f:
            orig_head = f.read().strip()
    except IOError:
        pass

    try:
        head = os.path.join(repo_path, '.git', 'HEAD')
        with open(head, 'r') as f:
            head = f.read().splitlines()[0].strip()

        if head.startswith('ref:'):
            branch = os.path.basename(head)
            with open(os.path.join(repo_path, '.git', head.split(' ')[1]), 'r') as f:
                head = f.read().strip()
    except IOError:
        pass
    return orig_head, head, branch

def format_sys_info():
    '''Format a list of system information for printing.'''
    import pyscf
    info = repo_info(os.path.join(__file__, '..', '..'))
    result = [
        f'System: {platform.uname()}  Threads {num_threads()}',
        f'Python {sys.version}',
        f'numpy {numpy.__version__}  scipy {scipy.__version__}  '
        f'h5py {h5py.__version__}',
        f'Date: {time.ctime()}',
        f'PySCF version {pyscf.__version__}',
        f'PySCF path  {info["path"]}',
    ]
    if 'git' in info:
        result.append(info['git'])
    return result


def isinteger(obj):
    '''
    Check if an object is an integer.
    '''
    # A bool is also an int in python, but we don't want that.
    # On the other hand, numpy.bool_ is probably not a numpy.integer, but just to be sure...
    if isinstance(obj, (bool, numpy.bool_)):
        return False
    # These are actual ints we expect to encounter.
    else:
        return isinstance(obj, (int, numpy.integer))


def issequence(obj):
    '''
    Determine if the object provided is a sequence.
    '''
    # These are the types of sequences that we permit.
    # numpy.ndarray is not a subclass of collections.abc.Sequence as of version 1.19.
    sequence_types = (collections.abc.Sequence, numpy.ndarray)
    return isinstance(obj, sequence_types)


def isintsequence(obj):
    '''
    Determine if the object provided is a sequence of integers.
    '''
    if not issequence(obj):
        return False
    elif isinstance(obj, numpy.ndarray):
        return issubclass(obj.dtype.type, numpy.integer)
    else:
        are_ints = True
        for i in obj:
            are_ints = are_ints and isinteger(i)
        return are_ints

class _OmniObject:
    '''Class with default attributes. When accessing an attribute that is not
    initialized, a default value will be returned than raising an AttributeError.
    '''
    verbose = 0
    max_memory = param.MAX_MEMORY
    stdout = StreamObject.stdout

    def __init__(self, default_factory=None):
        self._default = default_factory

    def __getattr__(self, key):
        return self._default

# Many methods requires a mol or mf object in initialization.
# These objects can be as the default arguments for these methods.
# Then class can be instantiated easily like cls(omniobj) in the following
# to_gpu function.
omniobj = _OmniObject()
omniobj._built = True
omniobj.mol = omniobj
omniobj._scf = omniobj
omniobj.base = omniobj
omniobj.precision = 1e-8 # utilized by several pbc modules

# Attributes that are kept in np.ndarray during the to_gpu conversion
_ATTRIBUTES_IN_NPARRAY = {'kpt', 'kpts', 'kpts_band', 'mesh', 'frozen'}

def to_gpu(method, out=None):
    '''Convert a method to its corresponding GPU variant, and recursively
    converts all attributes of a method to cupy objects or gpu4pyscf objects.
    '''
    import cupy
    from pyscf import gto

    # If a GPU class inherits a CPU code, the "to_gpu" method may be resolved
    # and available in the GPU class. Skip the conversion in this case.
    if method.__module__.startswith('gpu4pyscf'):
        return method

    if out is None:
        try:
            import gpu4pyscf
        except ImportError:
            print('Library gpu4pyscf not found. You can install this package via\n'
                  '    pip install gpu4pyscf-cuda11x\n'
                  'See more installation info at https://github.com/pyscf/gpu4pyscf')
            raise

        # TODO: Is it necessary to implement scanner in gpu4pyscf?
        if isinstance(method, (SinglePointScanner, GradScanner)):
            method = method.undo_scanner()

        from importlib import import_module
        mod = import_module(method.__module__.replace('pyscf', 'gpu4pyscf'))
        cls = getattr(mod, method.__class__.__name__)
        # A temporary GPU instance. This ensures to initialize private
        # attributes that are only available for GPU code.
        if hasattr(method, 'base'):
            out = cls(method.base)
        elif hasattr(method, '_scf'):
            out = cls(method._scf)
        elif hasattr(method, 'cell'):
            out = cls(method.cell)
        elif hasattr(method, 'mol'):
            out = cls(method.mol)
        else:
            raise TypeError('Conversion for class {cls} not supported')

    # Convert only the keys that are defined in the corresponding GPU class
    cls_keys = [getattr(cls, '_keys', ()) for cls in out.__class__.__mro__[:-1]]
    out_keys = set(out.__dict__).union(*cls_keys)
    # Only overwrite the attributes of the same name.
    keys = out_keys.intersection(method.__dict__)

    for key in keys:
        val = getattr(method, key)
        if isinstance(val, numpy.ndarray):
            if key not in _ATTRIBUTES_IN_NPARRAY:
                val = cupy.asarray(val)
        elif hasattr(val, 'to_gpu'):
            val = val.to_gpu()
        setattr(out, key, val)
    if hasattr(out, 'reset'):
        out.reset()
    return out
