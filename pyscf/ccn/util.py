#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Artem Pulkin
#

from collections import Counter, Mapping
import numpy
import itertools
from numbers import Number

from pyscf import lib

class StrictCounter(Counter):
    def __neg__(self):
        return StrictCounter(dict((k, -v) for k, v in self.items()))

    def __add__(self, other):
        result = self.copy()
        result.update(other)
        return result.clean()

    def __sub__(self, other):
        return self + (-StrictCounter(other))

    def __eq__(self, other):
        return (self-other).is_empty()

    def applied_count_condition(self, condition):
        """
        Applies a given condition on counts and returns a copy of StrictCounter.
        Args:
            condition (callable): a condition to apply;

        Returns:
            A StrictCounter with a condition applied.
        """
        return StrictCounter(dict((k, v) for k, v in self.items() if condition(v)))

    def clean(self):
        """
        Removes zero counts.
        Returns:
            An instance of StrictCounter with all zero counts removed.
        """
        return self.applied_count_condition(lambda c: c != 0)

    def positive_only(self):
        """
        Removes negative and zero counts.
        Returns:
            An instance of StrictCounter with positive counts only.
        """
        return self.applied_count_condition(lambda c: c > 0)

    def is_empty(self):
        """
        Checks if empty.
        Returns:
            True if all counts are zero.
        """
        for v in self.values():
            if v != 0:
                return False
        return True

    def to_list(self):
        """
        Makes a list of this counter with repeating elements.
        Returns:
            A list with elements from this counter.
        """
        return sum(([k] * v for k, v in self.items()), [])

    def readonly(self):
        """
        Returns a read-only mapping to self.
        Returns:
            A read-only mapping.
        """
        return ReadOnlySCWrapper(self)

    def __repr__(self):
        return "{{{}}}".format(",".join("{:d}x{}".format(v, repr(k)) for k, v in sorted(self.items())))


class ReadOnlySCWrapper(Mapping):

    def __init__(self, data):
        self.__data__ = data

    def __getitem__(self, key):
        return self.__data__[key]

    def __len__(self):
        return len(self.__data__)

    def __iter__(self):
        return iter(self.__data__)

    def __neg__(self):
        return -self.__data__

    def __add__(self, other):
        return self.__data__ + other

    def __sub__(self, other):
        return self.__data__ - other

    def to_list(self):
        return self.__data__.to_list()


readonly = ReadOnlySCWrapper


class OneToOne(dict):
    def __init__(self, source=None):
        """
        A one-to-one mapping.
        Args:
            source: source to initialize from;
        """
        dict.__init__(self)
        self.__bw__ = {}
        if source is not None:
            self.update(source)

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError("The key {} is already present".format(repr(key)))
        if value in self.__bw__:
            raise KeyError("The value {} is already present".format(repr(value)))
        dict.__setitem__(self, key, value)
        self.__bw__[value] = key

    def __delitem__(self, key):
        if key not in self:
            raise KeyError("Missing key {}".format(repr(key)))
        val = self[key]
        dict.__delitem__(self, key)
        del self.__bw__[val]

    def __repr__(self):
        return "{{{{{}}}}}".format(",".join(
            "{}=>{}".format(repr(k), repr(v)) for k, v in self.items()
        ))

    def clear(self):
        dict.clear(self)
        self.__bw__.clear()
    clear.__doc__ = dict.clear.__doc__

    def copy(self):
        return OneToOne(self)
    copy.__doc__ = dict.copy.__doc__

    def update(self, other):
        present = set(other.keys()) & set(self.keys())
        if len(present) > 0:
            raise KeyError("Keys {} are already present".format(repr(present)))
        counter = StrictCounter(other.values())
        repeating = list(k for k, v in counter.items() if v > 1)
        if len(repeating) > 0:
            raise KeyError("Some of the values are repeated and cannot be used as keys: {}".format(repr(repeating)))
        present = set(other.values()) & set(self.values())
        if len(present) > 0:
            raise KeyError("Values {} are already present".format(repr(present)))
        dict.update(self, other)
        self.__bw__.update(dict((v, k) for k, v in other.items()))
    update.__doc__ = dict.update.__doc__

    def withdraw(self, other):
        """
        Withdraws items from this one-to-one. Inverse of `self.update`.
        Args:
            other (dict): key-values pairs to withdraw;
        """
        for k, v in other.items():
            if k not in self:
                raise KeyError("Missing key {}".format(repr(k)))
            if self[k] != v:
                raise KeyError("Wrong value {} for key {}: expected {}".format(repr(v), repr(k), self[k]))
        for k in other.keys():
            del self[k]

    def inv(self):
        """
        Inverts the one-to-one correspondence.
        Returns:
            An inverted correspondence.
        """
        return OneToOne(self.__bw__)


class Intervals(object):
    def __init__(self, *args):
        """
        A class representing a set of (closed) intervals in 1D.
        Args:
            *args (Intervals, iterable): a set of intervals to initialize with.
        """
        self.__s__ = []
        self.__e__ = []
        if len(args) == 2 and isinstance(args[0], (int, float)):
            args = (args,)
        for i in args:
            self.add(*i)

    def __iter__(self):
        return iter(zip(self.__s__, self.__e__))

    def add(self, fr, to):
        """
        Adds an interval.
        Args:
            fr (float): from;
            to (float): to;
        """
        fr, to = min(fr, to), max(fr, to)
        new_s = []
        new_e = []
        for s, e in self:
            if e < fr or s > to:
                new_s.append(s)
                new_e.append(e)
            elif s >= fr and e <= to:
                pass
            else:
                fr = min(fr, s)
                to = max(to, e)
        new_s.append(fr)
        new_e.append(to)
        self.__s__ = new_s
        self.__e__ = new_e

    def __and__(self, other):
        if not isinstance(other, Intervals):
            other = Intervals(*other)
        result = []
        for s1, e1 in self:
            for s2, e2 in other:
                s = max(s1, s2)
                e = min(e1, e2)
                if s <= e:
                    result.append((s, e))
        return Intervals(*result)

    def __nonzero__(self):
        return bool(self.__s__)

    def __repr__(self):
        return "Intervals({})".format(", ".join("({}, {})".format(i, j) for i, j in self))


class MetaArray(numpy.ndarray):
    """Array with metadata (StackOverflow copy-paste)."""

    def __new__(cls, array, dtype=None, order=None, **kwargs):
        obj = numpy.asarray(array, dtype=dtype, order=order).view(cls)
        obj.metadata = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', None)


def meta(a, **kwargs):
    """
    Prepares an array with metadata.
    Args:
        a (numpy.ndarray): a numpy array;
        **kwargs: metadata to save;

    Returns:
        An array enhanced with metadata.
    """
    if isinstance(a, numpy.ndarray):
        return MetaArray(a, **kwargs)
    else:
        return a


def d2t(d):
    """Dict into tuple."""
    return tuple(sorted(d.items()))


def e(*args):
    """Numpy optimized einsum."""
    for i in args:
        if isinstance(i, Number) and i == 0:
            return 0
    try:
        return numpy.einsum(*args, optimize=True)
    except TypeError:
        return lib.einsum(*args)


def p_count(permutation, destination=None, debug=False):
    """
    Counts permutations.
    Args:
        permutation (iterable): a list of unique integers from 0 to N-1 or any iterable of unique entries if `normal`
        is provided;
        destination (iterable): ordered elements from `permutation`;
        debug (bool): prints debug information if True;

    Returns:
        The number of permutations needed to achieve this list from a 0..N-1 series.
    """
    if destination is None:
        destination = sorted(permutation)
    if len(permutation) != len(destination):
        raise ValueError("Permutation and destination do not match: {:d} vs {:d}".format(len(permutation), len(destination)))
    destination = dict((element, i) for i, element in enumerate(destination))
    permutation = tuple(destination[i] for i in permutation)
    visited = [False] * len(permutation)
    result = 0
    for i in range(len(permutation)):
        if not visited[i]:
            j = i
            while permutation[j] != i:
                j = permutation[j]
                result += 1
                visited[j] = True
    if debug:
        print("p_count(" + ", ".join("{:d}".format(i) for i in permutation) + ") = {:d}".format(result))
    return result


def p(spec, tensor):
    """
    Antisymmetrizes tensor.
    Args:
        spec (str): a string specifying tensor indexes. Each tensor dimension is represented by the corresponding
        symbol in the string using the following rules:

        1. Tensor dimensions which do not need to be antisymmetrized are represented by same symbols;
        2. Each pair of tensor dimensions with different symbols will be antisymmetrized;
        3. The symbol `.` (dot) is a special symbol: the corresponding dimension marked by this symbol will not be
        touched;

        tensor (numpy.ndarray): a tensor to antisymmetrize;

    Returns:
        An antisymmetrized tensor.

    Examples:

        >>> import numpy
        >>> from numpy import testing
        >>> a = numpy.arange(12).reshape(2, 2, 3)
        >>> s = p("ab.", a)  # permutes first and second dimensions
        >>> testing.assert_allclose(s, a - numpy.swapaxes(a, 0, 1))
        True

        >>> s = p("aa.", a)  # does nothing
        >>> testing.assert_allclose(s, a)
        True
    """
    if isinstance(tensor, Number):
        return tensor
    result = numpy.zeros_like(tensor)

    perm_mask = numpy.array([i for i in spec]) != '.'
    all_indexes = numpy.arange(len(spec))
    dims = all_indexes

    included = set()

    for order in itertools.permutations(all_indexes[perm_mask]):
        this_spec = ''.join(spec[_i] for _i in order)
        if this_spec not in included:
            dims[perm_mask] = order

            if p_count(order) % 2 == 0:
                result += tensor.transpose(dims)
            else:
                result -= tensor.transpose(dims)
            included.add(this_spec)
    return result


def _ltri_ix(n, ndims):
    """
    Generates lower-triangular part indexes in arbitrary dimensions.
    Args:
        n (int): dimension size;
        ndims (int): number of dimensions;

    Returns:
        Indexes in an array.
    """
    if ndims == 1:
        return numpy.arange(n)[:, numpy.newaxis]
    else:
        result = []
        for i in range(ndims-1, n):
            x = _ltri_ix(i, ndims-1)
            x_arr = numpy.empty((x.shape[0],1), dtype=int)
            x_arr[:] = i
            result.append(numpy.concatenate((x_arr, x), axis=1,))
        return numpy.concatenate(result, axis=0)


def ltri_ix(n, ndims):
    """
    Generates lower-triangular part indexes in arbitrary dimensions.
    Args:
        n (int): dimension size;
        ndims (int): number of dimensions;

    Returns:
        Indexes in a tuple of arrays.
    """
    return tuple(i for i in _ltri_ix(n, ndims).T)
