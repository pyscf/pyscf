#!/usr/bin/env python
'''
A simple example to run an MBD calculation.
'''

from pyscf.extras import mbd

atoms = [('Ar', (0, 0, 0)), ('Ar', (3.5/mbd.bohr, 0, 0))]
ene = mbd.mbd_rsscs(atoms, volumes=[1, 1], beta=0.83)
print(ene)
