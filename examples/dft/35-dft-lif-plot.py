#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
from __future__ import print_function
import sys, numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec


dd = np.loadtxt('yz2spin_dens_large.txt')
yy = np.loadtxt('yy.txt')
zz = np.loadtxt('zz.txt')

print(yy.shape, zz.shape, dd.shape)

plt.rc('text', usetex=True)    
fig = plt.figure(1, figsize=(16,16))
gs = gridspec.GridSpec(2,1)

ax = fig.add_subplot(gs[0,0])
ax.minorticks_on()
ax.tick_params(bottom=True, top=True, left=True, right=True, which='both', direction='in', length=6.0)
ax.tick_params(which='minor', length=3.0)

dens = (dd[:,2]+dd[:,3])
dens[ np.where(dens>6.4)[0] ] = 6.4
dens = dens.reshape((len(yy), len(zz)))

ext = [min(zz), max(zz), min(yy), max(yy)]
ims = ax.imshow(dens, extent=ext)
plt.colorbar(mappable=ims, orientation='vertical', ax=ax, shrink=0.77)

#####################################################
ax = fig.add_subplot(gs[1,0])
ax.minorticks_on()
ax.tick_params(bottom=True, top=True, left=True, right=True, which='both', direction='in', length=6.0)
ax.tick_params(which='minor', length=3.0)

dens = (dd[:,2]+dd[:,3]).reshape((len(yy), len(zz)))
ims = ax.imshow(dens, extent=ext)
plt.colorbar(mappable=ims, orientation='vertical', ax=ax, shrink=0.77)

plt.tight_layout()
fig.savefig('lif-large.png')


#
