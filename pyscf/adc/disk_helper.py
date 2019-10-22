import os
import h5py
import tempfile


def empty_dataset(shape):
    _, fname = tempfile.mkstemp()
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', shape)


def dataset(data):
    _, fname = tempfile.mkstemp()
    f = h5py.File(fname, mode='w')
    return f.create_dataset('data', data=data)


def remove_dataset(dataset):
    os.remove(dataset.file.filename)
