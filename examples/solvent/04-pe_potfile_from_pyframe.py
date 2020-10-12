#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
This example shows how to call the PyFraME library to generate the potential
file for the CPPE library used by pyscf polarizable embedding method

More details of PyFraME usage can be found in PE tutorial paper arXiv:1804.03598
'''

import pyframe

# Read PDB file
system = pyframe.MolecularSystem(input_file='4NP_in_water.pdb')

# System fragments can be accessed through the object  system.fragments
# There several filters to select specific fragments
# get_fragments_by_number(numbers=[1])
# get_fragments_by_name(names=['4NP'])
# get_fragments_by_chain_id(chain_ids=[])
# get_fragments_by_charge(charges=[])
core = system.get_fragments_by_name(names=['4NP'])

system.set_core_region(fragments=core)
solvent = system.get_fragments_by_distance(reference=core, distance=4.0)
system.add_region(name='solvent', fragments=solvent , use_standard_potentials=True)

# Write pot file for CPPE input. The output potfile is
# 4NP_in_water/4NP_in_water.pot
project = pyframe.Project()
project.create_embedding_potential(system)
project.write_core(system)
project.write_potential(system)
