# -*- coding: utf-8 -*-
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Modified from  https://github.com/adrn/mpipool
#

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MPIPool", "MPIPoolException"]

import os
import sys
import imp
import types
import marshal
#import traceback
from mpi4py import MPI

class MPIPool(object):
    """
    A pool that distributes tasks over a set of MPI processes using
    mpi4py. MPI is an API for distributed memory parallelism, used
    by large cluster computers. This class provides a similar interface
    to Python's multiprocessing Pool, but currently only supports the
    :func:`map` method.

    Contributed initially by `Joe Zuntz <https://github.com/joezuntz>`_.

    Parameters
    ----------
    comm : (optional)
        The ``mpi4py`` communicator.

    debug : bool (optional)
        If ``True``, print out a lot of status updates at each step.
    """
    def __init__(self, comm=None, debug=False):
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.debug = debug
        self.function = _error_function

        if self.debug:
            import platform
            node = platform.node()
            if self.rank == 0:
                print('Master: host {0} PID {1}'.format(node, os.getpid()))
            else:
                print('Worker: host {0} PID {1}'.format(node, os.getpid()))

    def is_master(self):
        """
        Is the current process the master?

        """
        return self.rank == 0

    def wait(self):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            return

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.bcast()
            if self.debug:
                print("Worker {0} got task {1}.".format(self.rank, task))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                if self.debug:
                    print("Worker {0} close.".format(self.rank))
# Handle global import lock for multithreading, see
#   http://stackoverflow.com/questions/12389526/import-inside-of-a-python-thread
#   https://docs.python.org/3.4/library/imp.html#imp.lock_held
# Global import lock affects the ctypes module.  It leads to deadlock when
# ctypes function is called in new threads created by threading module.
                if sys.version_info < (3,4):
                    if not imp.lock_held():
                        imp.acquire_lock()
                sys.exit(0)

            # Check if message is special type containing new function
            # to be applied
            elif isinstance(task, _function_wrapper):
                code = marshal.loads(task.func_code)
                if self.debug:
                    print('function {0}'.format(code))
                    print("Worker {0} replaced its task function: {1}."
                          .format(self.rank, self.function))
                self.function = types.FunctionType(code, globals())

            else:  # message are function args
                self.function(task)

    def apply(self, function, master_args, worker_args):
        if not self.is_master():
            self.wait()
            exit(0)

        if function is not self.function:
            if self.debug:
                print("Master replacing pool function with {0}."
                      .format(function))

            # Tell all the workers what function to use.
            self.function = function
            F = _function_wrapper(function)
            self.comm.bcast(F)

        # Send all the tasks off and wait for them to be received.
        self.comm.bcast(worker_args)

        result = function(master_args)
        return result

    def close(self):
        """
        Just send a message off to all the pool members which contains
        the special :class:`_close_pool_message` sentinel.

        """
        if self.is_master():
            self.comm.bcast(_close_pool_message())
            if self.debug:
                print('master close')

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()


class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"


class _function_wrapper(object):
    def __init__(self, function):
        #print(function.__closure__)
        self.func_code = marshal.dumps(function.func_code)

def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")

class MPIPoolException(Exception):
    def __init__(self, tb):
        self.traceback = tb
