import os.path
import numpy as np
import h5py

mem_units = { 'B' : 1, 'KB' : 1e3, 'MB' : 1e6, 'GB' : 1e9, 'TB' : 1e12 }
DEFAULT_MEM_UNIT = 'MB'

class NOTSET:
    pass

def valid_label(label):
    # TODO
    return True

class IntegralCollection:

    def __init__(self, h5file=None, default_storage='memory', max_memory=None):
        """Create a new integral collection.

        Parameters
        ----------
        h5file : str, optional
            Set the filename of the HDF5 file.
        default_storage : ['memory', 'disk']
            Set the default storage behavior of the integral collection. Default: 'memory'.
        max_memory : int, optional
            Default: None
        """

        self.max_memory = max_memory
        if default_storage not in ('memory', 'disk'):
            raise ValueError("Invalid value for default_storage: %s. Must be in ['memory', 'disk']." % store)
        self.default_storage = default_storage

        self.h5file = "AFAF.h5"

        # Store integrals:
        self.integrals = {}

    def add(self, label, array=None, store=None, chunks=None):
        """Add integral to collection."""
        if not valid_label(label):
            raise ValueError("Integral label %s not valid." % label)
        if label in self.integrals:
            raise ValueError("Integral label %s already defined." % label)

        store = store or self.default_storage
        if store == 'memory':
            itg = Integral(self, label, array)
        elif store == 'disk':
            h5dset = label
            with h5py.File(self.h5file, 'a') as f:
                if chunks is not None:
                    f.create_dataset(h5dset, data=array, chunks=chunks)
                else:
                    f.create_dataset(h5dset, data=array)
            itg = Integral(self, label, h5dset)
        else:
            raise ValueError("Invalid value for store: %s" % store)

        self.integrals[label] = itg
        return itg

    def add_rule(self, label, rule, shape=None, dtype=None):
        """Add rule for integral construction."""
        if not allowed_label(label):
            raise ValueError("Integral label %s not allowed." % label)
        if label in self.integrals:
            raise ValueError()

        itg = Integral(self, label, rule, shape=shape, dtype=dtype)
        self.integrals[label] = itg
        return itg

    def get(self, label, default=NOTSET, key=slice(None)):
        """Access integral with label `label`.

        Parameters
        ----------
        label : str
            Integral label.
        default :
            If set, `default` is returned if integral label is not present in collection.
        key : slice or tuple
            Use `key` to access only a part of the integrals. For integrals stored as an array,
            `get(..., key)` is equivalent to `get(...)[key]`. For integral stored on disk
            or as an abstract rule, `get(..., key)` is preferred, since it can avoid loading or constructing
            the entire integral array.

        Returns
        -------
        Integral
        """
        if label not in self.integrals:
            if default is not NOTSET:
                return default
            raise ValueError("Integral label %s not found" % label)
        return self.integrals[label][key]

    def __getattr__(self, label):
        """To support direct attribute access of integrals, e.g. `eris.ovov`."""
        if label not in self.integrals:
            raise ValueError("IntegralCollection has not attribute or integral '%s'" % label)
        return self.integrals[label]

    def delete(self, label, delete_from_file=True):
        """Delete integral."""
        itg = self.integrals[label]
        # Delete from HDF5 file
        if itg.h5dset is not None and delete_from_file:
            with h5py.File(self.h5file, 'r+') as f:
                del f[itg.h5dset]
        del self.integrals[label]

    def clear(self):
        """Delete all integrals from collection."""
        for label in list(self.integrals):
            self.delete(label)

    def __len__(self):
        """Number of integrals in collection."""
        return len(self.integrals)

    def used_memory(self, unit=DEFAULT_MEM_UNIT):
        """Total memory used by all integrals in collection."""
        return sum([itg.used_memory(unit=unit) for itg in self.integrals.values()])

    def used_disk_space(self, unit=DEFAULT_MEM_UNIT):
        """Total disk space used by all integrals in collection."""
        return sum([itg.used_disk_space(unit=unit) for itg in self.integrals.values()])

    def print_storage(self, unit=DEFAULT_MEM_UNIT, verbose=True):
        unit = unit.upper()
        fmt = "%10s:  memory= %8.3f %s  disk= %8.3f %s"
        print("IntegralCollection:")
        if verbose:
            for itg in self.integrals.values():
                print(fmt % (itg.label, itg.used_memory(unit=unit), unit, itg.used_disk_space(unit=unit), unit))
        mem = self.used_memory(unit=unit)
        disk = self.used_disk_space(unit=unit)
        print(fmt % ("total", mem, unit, disk, unit))



class Integral:

    def __init__(self, collection, label, data, shape=None, dtype=None):
        self.collection = collection
        self.label = label
        self._shape = shape
        self._dtype = dtype

        self.array = None
        self.h5dset = None
        self.rule = None

        # Data is plain array
        if isinstance(data, np.ndarray):
            self.array = data
        # Data is h5dset
        elif isinstance(data, str):
            self.h5dset = data
        # Data has to be constructed by calling rule
        elif callable(data):
            self.rule = data
        else:
            raise ValueError()

    @property
    def h5file(self):
        return self.collection.h5file

    def used_memory(self, unit=DEFAULT_MEM_UNIT):
        if self.array is None:
            return 0
        return self.array.nbytes/mem_units[unit.upper()]

    def used_disk_space(self, unit=DEFAULT_MEM_UNIT):
        if self.h5dset is None:
            return 0
        with h5py.File(self.h5file, 'r') as f:
            return f[self.h5dset].nbytes/mem_units[unit.upper()]

    def __getitem__(self, key):
        if self.array is not None:
            return self.array[key]
        if self.rule is not None:
            return self.rule(self.collection, key)
        if self.h5dset is not None:
            with h5py.File(self.h5file, 'r') as f:
                return f[self.h5dset][key]

    # For convencience implement some NumPy/h5py.

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        if self._shape is not None:
            return self._shape
        if self.array is not None:
            return self.array.shape
        if self.h5dset is not None:
            with h5py.File(self.h5file, 'r') as f:
                return f[self.h5dset].shape
        raise ValueError("Cannot deduce shape of abstract integrals.")

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        if self.array is not None:
            return self.array.dtype
        if self.h5dset is not None:
            with h5py.File(self.h5file, 'r') as f:
                return f[self.h5dset].dtype
        raise ValueError("Cannot deduce dtype of abstract integrals.")

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)





from timeit import default_timer as timer

n = 1000
naux = 200
nocc = 50
nvir = 120
norb = nocc + nvir
j3c_ov = np.random.rand(naux,nocc,nvir)

eris = IntegralCollection()
eris.add('oooo', np.random.rand(nocc,nocc,nocc,nocc))
lov = np.random.rand(naux,nocc,nvir)
eris.add('lov', lov, store='disk')

j3c = np.random.rand(naux,norb,norb)
eris.add('j3c', j3c, store='disk')

occ, vir = np.s_[:nocc], np.s_[nocc:]
eris.add_rule('j3c_oo', rule=lambda coll, key : coll.j3c[:,occ,occ][key], shape=(nocc, nocc), dtype=j3c.dtype)
eris.add_rule('j3c_ov', rule=lambda coll, key : coll.j3c[:,occ,vir][key], shape=(nocc, nvir), dtype=j3c.dtype)
eris.add_rule('j3c_vv', rule=lambda coll, key : coll.j3c[:,vir,vir][key], shape=(nvir, nvir), dtype=j3c.dtype)

print(eris.j3c_oo.shape)
print(eris.j3c_ov.shape)
print(eris.j3c_vv.shape)
assert np.allclose(eris.j3c_vv[:], eris.j3c[:,vir,vir])
# Indexing/slicing still works!
assert np.allclose(eris.j3c_vv[:2,4:5], eris.j3c[:,vir,vir][:2,4:5])

eris.add_rule('ovvv', rule=lambda coll, key : np.einsum("Lij,Lkl->ijkl", coll.j3c_ov[:,:,key], coll.j3c_vv, shape=(nocc,nvir,nvir,nvir), dtype=j3c.dtype))

ovov = np.einsum("Lij,Lkl->ijkl", lov, lov, optimize=True)


# Support for abstract integrals via `eris.add_rule`
def make_ovov(collection, key):
    # Some logic to transform the (ov|ov) key to key1, key2 for (ov|L) and (L|ov), respectively.
    if isinstance(key, tuple):
        key1 = (slice(None), *(key[:2]))
        key2 = (slice(None), *(key[2:]))
    else:
        key1, key2 = (slice(None), key), slice(None)
    return np.einsum('Lij,Lkl->ijkl', collection.lov[key1], collection.lov[key2], optimize=True)
eris.add_rule('ovov', rule=make_ovov, shape=(nocc,nvir,nocc,nvir), dtype=eris.lov.dtype)

eris.add_rule('ovov-2x', rule=lambda coll, key : 2*coll.ovov[key], shape=(nocc,nvir,nocc,nvir), dtype=eris.lov.dtype)


eris.print_storage()

print(eris.ovov.dtype)

print(eris.ovov.shape)

t0 = timer()
assert(np.allclose(eris.ovov[:3], ovov[:3]))
t1 = timer()-t0

t0 = timer()
assert(np.allclose(eris.ovov[:,:3], ovov[:,:3]))
t2 = timer()-t0

t0 = timer()
assert(np.allclose(eris.ovov[:,:,:3], ovov[:,:,:3]))
t3 = timer()-t0


t0 = timer()
eris.ovov[:][:3]
tf = timer()-t0

print(eris.get('ovov-2x')[0,0,0,0])
print(eris.get('ovov')[0,0,0,0])

print(t1, t2, t3, tf)

eris.clear()



1/0
#print(np.einsum("Lij,Lkl->", eris.Lov[:], eris.Lov[:]))
#eris.add_integral('oooo', data)
#eris.add_rule('ovov', rule=lambda base, key: np.einsum('Lij,Lkl->ijkl', base.Lov[:,key], base.Lov, optimize=True))


print(eris.ovov[key])

eris.print_storage()

#print(eris.Lov.shape)

#print(eris.oooo[:]-eris.ovvv[:])
t0 = timer()
print(eris.ovov[:].shape)
print("Time= %.3f" % (timer()-t0))

t0 = timer()
#np.einsum('Lij,Lkl->ijkl', Lov, Lov, optimize=True)
print("Time= %.3f" % (timer()-t0))

eris.clear()

1/0
eris.delete('oooo')
eris.show_storage()
eris.add_integral('oovv', data, store='disk')
eris.show_storage()
eris.delete_integral('oovv')
eris.show_storage()
1/0
t0 = timer()
print(eris.oooo[:])
t1 = timer()
print(eris.oovv[:])
t2 = timer()
print(t1-t0, t2-t1)
print(eris.oooo[:].shape)
print(np.einsum("ij->", eris.oooo[:]))
