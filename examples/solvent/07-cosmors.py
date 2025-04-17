#!/usr/bin/env python
'''
An example of using COSMO-RS functionality.
'''

#%% Imports

import io, json

import matplotlib.pyplot as plt

from pyscf import gto, dft
from pyscf.solvent import pcm
from pyscf.solvent.cosmors import get_sas_volume
from pyscf.solvent.cosmors import write_cosmo_file
from pyscf.solvent.cosmors import get_pcm_parameters


#%% Set parameters

# get molecule
coords = '''
C        0.000000    0.000000             -0.542500
O        0.000000    0.000000              0.677500
H        0.000000    0.9353074360871938   -1.082500
H        0.000000   -0.9353074360871938   -1.082500
'''
mol = gto.M(atom=coords, basis='6-31G*', verbose=1)

# configure PCM
cm = pcm.PCM(mol)
cm.eps = float('inf') # f_epsilon = 1 is required for COSMO-RS
cm.method = 'C-PCM' # or COSMO, IEF-PCM, SS(V)PE, see https://manual.q-chem.com/5.4/topic_pcm-em.html
cm.lebedev_order = 29 # lebedev grids on the cavity surface, lebedev_order=29  <--> # of grids = 302


#%% COSMO-files

# Please note, that outlying charge correction is not implemented yet

# run DFT SCF (any level of theory is OK, though DFT is optimal)
mf = dft.RKS(mol, xc='b3lyp')
mf = mf.PCM(cm)
mf.kernel()

# generate COSMO-file
with io.StringIO() as outp:
    write_cosmo_file(outp, mf)
    print(outp.getvalue())
    print()

# the corresponding parameters can be extracted as JSON-formatted dictionary
params = get_pcm_parameters(mf)
print(json.dumps(params, indent = '  '))


#%% Volumes

# The molecular volume is computed for the solvent-accessible surface generated
# within pcm.PCM object. Please note that r_sol is assumed to be equal to zero,
# so that SAS is a union of vdW spheres.
# Increasing integration step increases accuracy of integration, but for most COSMO-RS
# modelling and ML step=0.2 should be enough:

# compute volumes with different step values
steps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
Vs = [get_sas_volume(mf.with_solvent.surface, step) for step in steps]
# plot
ax = plt.gca()
ax.plot(Vs)
ax.set_xticks(range(len(steps)))
_ = ax.set_xticklabels(steps)
plt.show()

# If user prefers other algorithm to compute molecular volumes (e.g. RDKit),
# then it can be specified manually via the "volume" argument. Please note,
# that volume values must be given in [a.u.]

with io.StringIO() as outp:
    write_cosmo_file(outp, mf, volume=1234.56)
    print(outp.getvalue()[:219]) # slicing truncates the COSMO-file up to the volume line


#%% F_epsilon

# if PCM DFT were computed with fepsilon < 1 <=> eps != inf, the ValueError will be raised

# run DFT SCF
cm = pcm.PCM(mol)
cm.eps = 32.613 # methanol
mf = dft.RKS(mol, xc='b3lyp')
mf = mf.PCM(cm)
mf.kernel()

# try to get COSMO-file
with io.StringIO() as outp:
    try:
        write_cosmo_file(outp, mf)
        print(outp.getvalue())
    except ValueError as e:
        print(e)


