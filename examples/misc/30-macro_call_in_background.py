#!/usr/bin/env python

'''
This example shows how to use the call_in_background macro
'''

from pyscf import lib
import time

def fa():
    print('a')
    time.sleep(0.5)

def fb():
    print('b')
    time.sleep(0.8)

print('type 1')
w0 = time.time()
with lib.call_in_background(fa) as afa, lib.call_in_background(fb) as afb:
    for i in range(3):
        afa()
        afb()
print('total time = %.1f s  = [fb]0.8 * 3 seconds' % (time.time() - w0))

print('type 2')
w0 = time.time()
with lib.call_in_background(fa, fb) as (afa, afb):
    for i in range(3):
        afa()
        afb()
print('total time = %.1f s  = ([fa]0.5 + [fb]0.8) * 3 seconds' % (time.time() - w0))
