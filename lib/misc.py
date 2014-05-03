#
# File: misc.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import tempfile
import ctypes
import numpy
import functools

c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)

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
            lst = reversed(lst)
        seen = []
        for l in lst:
            if not member(test, l, seen):
                seen.append(l)
        return seen

def remove_if(test, lst):
    return filter(lambda x: not test(x), lst)

def find_if(test, lst):
    for l in lst:
        if test(l):
            return l


def trace_ab(a, b):
    return (numpy.array(a).T*numpy.array(b)).sum()

def pack_lowtri(mat, nd):
    mat1d = numpy.empty(nd*(nd+1)/2)
    n = 0
    for i in range(nd):
        for j in range(i+1):
            mat1d[n] = mat[i,j]
            n += 1
    return mat1d
def unpack_lowtri(mat1d, nd):
    mat = numpy.empty((nd,nd))
    n = 0
    for i in range(nd):
        for j in range(i+1):
            mat[i,j] = mat1d[n]
            mat[j,i] = mat1d[n].conj()
            n += 1
    return mat


LINEAR_DEP_THRESHOLD = 1e-10
def solve_lineq_by_SVD(a, b):
    ''' a * x = b '''
    t, w, vH = numpy.linalg.svd(a)
    idx = []
    for i,wi in enumerate(w):
        if wi > LINEAR_DEP_THRESHOLD:
            idx.append(i)
    if idx:
        idx = numpy.array(idx)
        tb = numpy.dot(numpy.array(t[:,idx]).T.conj(), numpy.array(b))
        x = numpy.dot(numpy.array(vH[idx,:]).T.conj(), tb / w[idx])
    else:
        x = numpy.zeros_like(b)
    return x

class ctypes_stdout:
    '''make c-printf output to string, but keep python print in /dev/pts/1.
    Note it cannot correctly handle c-printf with GCC, don't know why.
    Usage:
        with ctypes_stdout() as stdout:
            ...
        print stdout.read()'''
    def __enter__(self):
        sys.stdout.flush()
        self._contents = None
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.bak_stdout = sys.stdout
        self.fd, self.ftmp = tempfile.mkstemp(dir='/dev/shm')
        os.dup2(self.fd, self.old_stdout_fileno)
        sys.stdout = os.fdopen(self.bak_stdout_fd, 'w')
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        os.fsync(self.fd)
        self._contents = open(self.ftmp, 'r').read()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        sys.stdout = self.bak_stdout # self.bak_stdout_fd is closed
        #os.close(self.fd) is closed when os.fdopen is closed
        os.remove(self.ftmp)
    def read(self):
        if self._contents:
            return self._contents
        else:
            sys.stdout.flush()
            #f = os.fdopen(self.fd, 'r') # need to rewind(0) before reading
            #f.seek(0)
            return open(self.ftmp, 'r').read()

class capture_stdout:
    '''redirect all stdout (c printf & python print) into a string
    Usage:
        with capture_stdout() as stdout:
            ...
        print stdout.read()'''
    def __enter__(self):
        sys.stdout.flush()
        self._contents = None
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.fd, self.ftmp = tempfile.mkstemp(dir='/dev/shm')
        os.dup2(self.fd, self.old_stdout_fileno)
        return self
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        self._contents = open(self.ftmp, 'r').read()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        os.close(self.bak_stdout_fd)
        #os.close(self.fd) will be closed when os.fdopen is closed
        os.remove(self.ftmp)
    def read(self):
        if self._contents:
            return self._contents
        else:
            sys.stdout.flush()
            #f = os.fdopen(self.fd, 'r') # need to rewind(0) before reading
            #f.seek(0)
            return open(self.ftmp, 'r').read()

class quite_run:
    '''output nothing

    Examples
    --------
    with quite_run():
        ...
    '''
    def __enter__(self):
        sys.stdout.flush()
        self.dirnow = os.getcwd()
        self.tmpdir = tempfile.mkdtemp(dir='/dev/shm')
        os.chdir(self.tmpdir)
        self.old_stdout_fileno = sys.stdout.fileno()
        self.bak_stdout_fd = os.dup(self.old_stdout_fileno)
        self.fnull = open(os.devnull, 'wb')
        os.dup2(self.fnull.fileno(), self.old_stdout_fileno)
    def __exit__(self, type, value, traceback):
        sys.stdout.flush()
        os.dup2(self.bak_stdout_fd, self.old_stdout_fileno)
        self.fnull.close()
        shutil.rmtree(self.tmpdir)
        os.chdir(self.dirnow)


# from pygeocoder
# this decorator lets me use methods as both static and instance methods
class omnimethod(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return functools.partial(self.func, instance)

