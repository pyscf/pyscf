#!/usr/bin/env python
# -*- coding: utf-8
#
# File: logger.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#


'''
logger
'''

import sys

import parameters as param

NONE   = param.VERBOSE_NONE
DEBUG  = param.VERBOSE_DEBUG
DEBUG1 = param.VERBOSE_DEBUG + 1
DEBUG2 = param.VERBOSE_DEBUG + 2
DEBUG3 = param.VERBOSE_DEBUG + 3
DEBUG4 = param.VERBOSE_DEBUG + 4
INFO   = param.VERBOSE_INFO
NOTICE = param.VERBOSE_NOTICE
WARN   = param.VERBOSE_WARN
WARNING = WARN
ERR    = param.VERBOSE_ERR
ERROR  = ERR
QUITE  = param.VERBOSE_QUITE
CRIT   = param.VERBOSE_CRIT
ALERT  = param.VERBOSE_ALERT
PANIC  = param.VERBOSE_PANIC

class Logger:
    def __init__(self, fout, verbose):
        self.fout = fout
        self.verbose = verbose

    def debug(self, msg, *args):
        debug(self, msg, *args)

    def info(self, msg):
        info(self, msg, *args)

    def warn(self, msg, *args):
        warn(self, msg, *args)

    def error(self, msg, *args):
        error(self, msg, *args)

    def log(self, msg, *args):
        log(self, msg, *args)

    def stdout(self, msg, *args):
        stdout(self, msg, *args)

def flush(rec, msg, *args):
    rec.fout.write(msg%args)
    rec.fout.write('\n')
    rec.fout.flush()

def msg(level, rec, msg, *args):
    if rec.verbose >= level:
        flush(rec, 'LOGLVL-%d: %s\n' % (level, (msg%args)))

def log(rec, msg, *args):
    if rec.verbose > QUITE:
        flush(rec, msg, *args)

def error(rec, msg, *args):
    if rec.verbose >= ERROR:
        flush(rec, msg, *args)

def warn(rec, msg, *args):
    if rec.verbose >= WARN:
        flush(rec, msg, *args)

def info(rec, msg, *args):
    if rec.verbose >= INFO:
        flush(rec, msg, *args)

def debug(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)

def stdout(rec, msg, *args):
    if rec.verbose >= DEBUG:
        flush(rec, msg, *args)
    sys.stdout.write('>>> %s\n' % msg)
