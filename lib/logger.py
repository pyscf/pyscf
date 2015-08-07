#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
logger
'''

import sys
import time

from pyscf.lib import parameters as param

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

TIMER_LEVEL  = param.TIMER_LEVEL

sys.verbose = NOTE

class Logger(object):
    def __init__(self, stdout, verbose):
        self.stdout = stdout
        self.verbose = verbose
        self._t0 = time.clock()
        self._w0 = time.time()

    def debug(self, msg, *args):
        debug(self, msg, *args)

    def debug1(self, msg, *args):
        debug1(self, msg, *args)

    def debug2(self, msg, *args):
        debug2(self, msg, *args)

    def debug3(self, msg, *args):
        debug3(self, msg, *args)

    def debug4(self, msg, *args):
        debug4(self, msg, *args)

    def info(self, msg, *args):
        info(self, msg, *args)

    def note(self, msg, *args):
        note(self, msg, *args)

    def warn(self, msg, *args):
        warn(self, msg, *args)

    def error(self, msg, *args):
        error(self, msg, *args)

    def log(self, msg, *args):
        log(self, msg, *args)

    def timer(self, msg, cpu0=None, wall0=None):
        if cpu0:
            return timer(self, msg, cpu0, wall0)
        else:
            self._t0, self._w0 = timer(self, msg, self._t0, wall0)
            return self._t0, self._w0

    def timer_debug1(self, msg, cpu0=None, wall0=None):
        if self.verbose >= DEBUG1:
            return self.timer(msg, cpu0, wall0)
        elif wall0:
            return time.clock(), time.time()
        else:
            return time.clock()

def flush(rec, msg, *args):
    rec.stdout.write(msg%args)
    rec.stdout.write('\n')
    rec.stdout.flush()

def log(rec, msg, *args):
    if rec.verbose > QUIET:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, 'Error: '+msg, *args)
    sys.stderr.write('Error: ' + (msg%args) + '\n')

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, 'Warn: '+msg, *args)
    #if rec.stdout is not sys.stdout:
        sys.stderr.write('Warn: ' + (msg%args) + '\n')

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

def timer(rec, msg, cpu0, wall0=None):
    cpu1, wall1 = time.clock(), time.time()
    if wall0:
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, ' '.join(('    CPU time for', msg,
                                 '%9.2f sec, wall time %9.2f sec')),
                  cpu1-cpu0, wall1-wall0)
        return cpu1, wall1
    else:
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, ' '.join(('    CPU time for', msg, '%9.2f sec')),
                  cpu1-cpu0)
        return cpu1

def timer_debug1(rec, msg, cpu0, wall0=None):
    if rec.verbose >= DEBUG1:
        return timer(rec, msg, cpu0, wall0)
    elif wall0:
        return time.clock(), time.time()
    else:
        return time.clock()
