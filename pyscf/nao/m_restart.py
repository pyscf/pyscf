#An HDF5 file is a container for two kinds of objects: 
#  * datasets (array-like collections of data)
#  * groups (folder-like containers that hold datasets).
# Groups work like dictionaries, and datasets work like NumPy arrays

from __future__ import division
import numpy as np

def read_rst_h5py (filename=None):
    import h5py ,os
    if filename is None: 
        path = os.getcwd()
        filename =find('*.hdf5', path)
    #filename= 'SCREENED_COULOMB.hdf5'
    with h5py.File(filename, 'r') as f:
        #print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    msg = 'RESTART: Full matrix elements of screened interactions (W_c) was read from {}'.format(filename)
    return data, msg


def write_rst_h5py(data, filename = None):
    import h5py
    if filename is None: 
      filename= 'SCREENED_COULOMB.hdf5'
    

    with h5py.File(filename, 'w') as data_file:
        try:
            data_file.create_dataset('W_c', data=data)
        except:
            print("failed writting data to SCREENED_COULOMB.hdf5")
            print(type(data))

        data_file.close
    
    msg = 'Full matrix elements of screened interactions (W_c) stored in {}'.format(filename)
    return msg


def write_rst_yaml (data , filename=None):
    import yaml
    if filename is None: filename= 'SCREENED_COULOMB.yaml'
    with open(filename, 'w+', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)
    msg = 'Full matrix elements of screened interactions stored in {}'.format(filename)
    return msg


def read_rst_yaml (filename=None):
    import yaml, os
    if filename is None: 
        path = os.getcwd()
        filename =find('*.yaml', path)
    with open(filename, 'r') as stream:
        try:
            data = yaml.load(stream)
            msg = 'RESTART: Full matrix elements of screened interactions (W_c) was read from {}'.format(filename)
            return data, msg
        except yaml.YAMLError as exc:
            return exc
