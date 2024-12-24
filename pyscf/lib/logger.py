#!/usr/bin/env python
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
Logging system

Log level
---------

======= ======
Level   number
------- ------
DEBUG4  9
DEBUG3  8
DEBUG2  7
DEBUG1  6
DEBUG   5
INFO    4
NOTE    3
WARN    2
ERROR   1
QUIET   0
======= ======

Large value means more noise in the output file.

.. note::
    Error and warning messages are written to stderr.

Each Logger object has its own output destination and verbose level.  So
multiple Logger objects can be created to manage the message system without
affecting each other.
The methods provided by Logger class has the direct connection to the log level.
E.g.  :func:`info` print messages if the verbose level >= 4 (INFO):

>>> import sys
>>> from pyscf import lib
>>> log = lib.logger.Logger(sys.stdout, 4)
>>> log.info('info level')
info level
>>> log.verbose = 3
>>> log.info('info level')
>>> log.note('note level')
note level


timer
-----
Logger object provides timer method for timing.  Set :attr:`TIMER_LEVEL` to
control at which level the timing information should be output.  It is 5
(DEBUG) by default.

>>> import sys, time
>>> from pyscf import lib
>>> log = lib.logger.Logger(sys.stdout, 4)
>>> t0 = logger.process_clock()
>>> log.timer('test', t0)
>>> lib.logger.TIMER_LEVEL = 4
>>> log.timer('test', t0)
    CPU time for test      0.00 sec

'''

import sys
import time

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

from pyscf.lib import parameters as param
import pyscf.__config__

DEBUG4 = param.VERBOSE_DEBUG + 4
DEBUG3 = param.VERBOSE_DEBUG + 3
DEBUG2 = param.VERBOSE_DEBUG + 2
DEBUG1 = param.VERBOSE_DEBUG + 1
DEBUG  = param.VERBOSE_DEBUG
INFO   = param.VERBOSE_INFO
NOTE   = param.VERBOSE_NOTICE
NOTICE = NOTE
WARN   = param.VERBOSE_WARN
WARNING = WARN
ERR    = param.VERBOSE_ERR
ERROR  = ERR
QUIET  = param.VERBOSE_QUIET
CRIT   = param.VERBOSE_CRIT
ALERT  = param.VERBOSE_ALERT
PANIC  = param.VERBOSE_PANIC

TIMER_LEVEL  = getattr(pyscf.__config__, 'TIMER_LEVEL', DEBUG)

sys.verbose = NOTE

def flush(rec, msg, *args):
    rec.stdout.write(msg%args)
    rec.stdout.write('\n')
    rec.stdout.flush()

def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, '\nERROR: '+msg+'\n', *args)
    sys.stderr.write('ERROR: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, '\nWARN: '+msg+'\n', *args)
        if rec.stdout is not sys.stdout:
            sys.stderr.write('WARN: ' + (msg%args) + '\n')

def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)

def note(rec, msg, *args):
    if rec.verbose >= NOTICE:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)

def debug1(rec, msg, *args):
    if rec.verbose >= DEBUG1:
        flush(rec, msg, *args)

def debug2(rec, msg, *args):
    if rec.verbose >= DEBUG2:
        flush(rec, msg, *args)

def debug3(rec, msg, *args):
    if rec.verbose >= DEBUG3:
        flush(rec, msg, *args)

def debug4(rec, msg, *args):
    if rec.verbose >= DEBUG4:
        flush(rec, msg, *args)

def stdout(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)
    sys.stdout.write('>>> %s\n' % msg)

def timer(rec, msg, cpu0=None, wall0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0

def timer_debug1(rec, msg, cpu0=None, wall0=None):
    if rec.verbose >= DEBUG1:
        return timer(rec, msg, cpu0, wall0)
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        return rec._t0

class Logger:
    '''
    Attributes:
        stdout : file object or sys.stdout
            The file to dump output message.
        verbose : int
            Large value means more noise in the output file.
    '''
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        self.stdout = stdout
        self.verbose = verbose
        self._t0 = process_clock()
        self._w0 = perf_counter()

    log = log
    error = error
    warn = warn
    note = note
    info = info
    debug  = debug
    debug1 = debug1
    debug2 = debug2
    debug3 = debug3
    debug4 = debug4
    timer = timer
    timer_debug1 = timer_debug1

def new_logger(rec=None, verbose=None):
    '''Create and return a :class:`Logger` object

    Args:
        rec : An object which carries the attributes stdout and verbose

        verbose : a Logger object, or integer or None
            The verbose level. If verbose is a Logger object, the Logger
            object is returned. If verbose is not specified (None),
            rec.verbose will be used in the new Logger object.
    '''
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        from pyscf.lib.misc import StreamObject
        if getattr(rec, 'stdout', None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(StreamObject.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log
