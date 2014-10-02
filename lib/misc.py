#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import tempfile
import shutil
import functools
import ctypes
import numpy
import math

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

# for give n, generate [(m1,m2),...] that
#       m2*(m2+1)/2 - m1*(m1+1)/2 <= base*(base+1)/2
def tril_equal_pace(n, base=0, npace=0, minimal=1):
    if base == 0:
        assert(npace > 0)
        base = int(math.sqrt(n*(n+1)/npace)) + 1
    m1 = 0
    while m1 < n:
        # m1*m1 + base*base < m1*(m1+1) + base*(base+1) - m2
        m2 = int(max(math.sqrt(m1**2+base**2), m1+minimal))
        yield m1, min(m2,n)
        m1 = m2


class ctypes_stdout:
    '''make c-printf output to string, but keep python print in /dev/pts/1.
    Note it cannot correctly handle c-printf with GCC, don't know why.
    Usage:
        with ctypes_stdout() as stdout:
            ...
        print(stdout.read())'''
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
        print(stdout.read())'''
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
# In contrast to classmethod, when obj.function() is called, the first
# argument is obj in omnimethod rather than obj.__class__ in classmethod
class omnimethod(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return functools.partial(self.func, instance)

if __name__ == '__main__':
    for i,j in tril_equal_pace(90, 30):
        print('base=30', i, j, j*(j+1)/2-i*(i+1)/2)
    for i,j in tril_equal_pace(90, npace=5):
        print('npace=5', i, j, j*(j+1)/2-i*(i+1)/2)
