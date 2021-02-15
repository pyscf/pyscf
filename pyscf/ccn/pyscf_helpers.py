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

from .util import meta, MetaArray, ltri_ix, p

from pyscf.lib.diis import DIIS
from pyscf.lib.linalg_helper import davidson_nosym1 as davidson
from pyscf.cc.uccsd_slow import _PhysicistsERIs as ERIS_uccsd_slow
from pyscf.cc.gccsd import _PhysicistsERIs as ERIS_gccsd
import numpy

import inspect
from numbers import Number
from collections import OrderedDict
from warnings import warn
import string


def res2amps(residuals, e_occ, e_vir, constant=None):
    """
    Converts residuals into amplitudes update.
    Args:
        residuals (iterable): a list of residuals;
        e_occ (array): occupied energies;
        e_vir (array): virtual energies;
        virtual spaces;
        constant (float): a constant in the denominator;

    Returns:
        A list of updates to amplitudes.
    """
    result = []
    for res in residuals:

        if isinstance(res, Number) and res == 0:
            result.append(0)

        elif isinstance(res, MetaArray):
            diagonal = numpy.zeros_like(res)
            ix = [numpy.newaxis] * len(diagonal.shape)

            if "labels" not in res.metadata:
                raise ValueError("Missing metadata: axes labels")

            for j, s in enumerate(res.metadata["labels"]):
                ix[j] = slice(None)
                if s == 'o':
                    diagonal += e_occ[tuple(ix)]
                elif s == 'v':
                    diagonal -= e_vir[tuple(ix)]
                else:
                    raise ValueError("Unknown spec '{}' in {}".format(s, residuals.metadata["labels"]))
                ix[j] = numpy.newaxis

            if constant is not None:
                result.append(res / (constant + diagonal))

            else:
                result.append(res / diagonal)

        else:
            raise ValueError("Unsupported type: {}".format(type(res)))

    return result


def a2v(amplitudes):
    """List of amplitudes into a single array."""
    result = []
    for v in amplitudes:
        result.append(numpy.reshape(v, -1))
    return numpy.concatenate(result)


def v2a(vec, like):
    """Array into a list amplitudes."""
    result = []
    offset = 0
    for v in like:
        s = v.size
        result.append(numpy.reshape(vec[offset:offset+s], v.shape))
        if isinstance(v, MetaArray):
            result[-1] = MetaArray(result[-1], **v.metadata)
        offset += s
    return result


def eris_hamiltonian(eris):
    """
    Retrieves Hamiltonian matrix elements from pyscf ERIS.
    Args:
        eris (pyscf.cc.ccsd.ERIS): pyscf ERIS;

    Returns:
        A dict with Hamiltonian matrix elements.
    """
    # TODO: decide on adding '**ov', 'vo**'
    nocc = eris.oooo.shape[0]
    if isinstance(eris, ERIS_uccsd_slow):
        def chess(a):
            ix = []
            for d in a.shape:
                ix.append(numpy.dstack((
                    numpy.arange(d // 2),
                    numpy.arange(d // 2, d),
                )).reshape(-1))
            return a[numpy.ix_(*ix)]
        return {k: chess(v) for k, v in dict(
            ov=eris.fock[:nocc, nocc:],
            vo=eris.fock[nocc:, :nocc],
            oo=eris.fock[:nocc, :nocc],
            vv=eris.fock[nocc:, nocc:],
            oooo=eris.oooo,
            oovo=-numpy.transpose(eris.ooov, (0, 1, 3, 2)),
            oovv=eris.oovv,
            ovoo=eris.ovoo,
            ovvo=-numpy.transpose(eris.ovov, (0, 1, 3, 2)),
            ovvv=eris.ovvv,
            vvoo=numpy.transpose(eris.oovv, (2, 3, 0, 1)),
            vvvo=-numpy.transpose(eris.ovvv, (2, 3, 1, 0)),
            vvvv=eris.vvvv,
        ).items()}

    elif isinstance(eris, ERIS_gccsd):
        return dict(
            ov=eris.fock[:nocc, nocc:], #OK
            vo=eris.fock[nocc:, :nocc], #OK
            oo=eris.fock[:nocc, :nocc], #OK
            vv=eris.fock[nocc:, nocc:], #OK
            oooo=eris.oooo, #OK
            oovo=-numpy.transpose(eris.ooov, (0, 1, 3, 2)), #OK
            oovv=eris.oovv,
            # ovoo=eris.ovoo,
            ovoo=numpy.transpose(eris.ooov, (2, 3, 0, 1)), #OK
            # ovvo=-numpy.transpose(eris.ovov, (0, 1, 3, 2)),
            ovvo=eris.ovvo,
            ovvv=eris.ovvv,
            vvoo=numpy.transpose(eris.oovv, (2, 3, 0, 1)),
            vvvo=-numpy.transpose(eris.ovvv, (2, 3, 1, 0)),
            vvvv=eris.vvvv,
        )

    else:
        raise ValueError("Unknown object: {}".format(eris))


def oneshot(equations, *args):
    """
    A one-shot calculation.
    Args:
        equations (callable): coupled-cluster equations;
        args (iterable): amplitudes and hamiltonian matrix elements as dicts;

    Returns:
        Results of the calculation.
    """
    input_args = inspect.getargspec(equations).args
    fw_args = {}
    for i in args:
        fw_args.update(i)
    # Remove excess arguments from the Hamiltonian
    fw_args = {k: v for k, v in fw_args.items() if k in input_args}
    # Check missing arguments
    missing = set(input_args) - set(fw_args.keys())
    if len(missing) > 0:
        raise ValueError("Following arguments are missing: {}".format(', '.join(missing)))
    return equations(**fw_args)


def kernel_solve(hamiltonian, equations, initial_guess, tolerance=1e-9, debug=False, diis=True, equation_energy=None,
                 dim_spec=None, maxiter=50):
    """
    Coupled-cluster solver (linear systems).
    Args:
        hamiltonian (dict): hamiltonian matrix elements or pyscf ERIS;
        equations (callable): coupled-cluster equations;
        initial_guess (OrderedDict): starting amplitudes;
        tolerance (float): convergence criterion;
        debug (bool): prints iterations if True;
        diis (bool, DIIS): converger for iterations;
        equation_energy (callable): energy equation;
        dim_spec (iterable): if `initial_guess` is a dict, this parameter defines shapes of arrays in 'ov' notation
        (list of strings);
        maxiter (int): maximal number of iterations;

    Returns:
        Resulting coupled-cluster amplitudes and energy if specified.
    """
    # Convert ERIS to hamiltonian dict if needed
    if not isinstance(hamiltonian, dict):
        hamiltonian = eris_hamiltonian(hamiltonian)

    if isinstance(initial_guess, (tuple, list)):
        initial_guess = OrderedDict((k, 0) for k in initial_guess)
        if dim_spec is None:
            raise ValueError("dim_spec is not specified")
    elif isinstance(initial_guess, OrderedDict):
        if dim_spec is None and any(not isinstance(i, MetaArray) for i in initial_guess.values()):
            raise ValueError("One or more of initial_guess values is not a MetaArray. Either specify dim_spec or use "
                             "MetaArrays to provide dimensions' labels in the 'ov' notation")
            dim_spec = tuple(i.metadata["labels"] for i in initial_guess.values())
    else:
        raise ValueError("OrderedDict expected for 'initial_guess'")

    tol = None
    e_occ = numpy.diag(hamiltonian["oo"])
    e_vir = numpy.diag(hamiltonian["vv"])

    if diis is True:
        diis = DIIS()

    while tol is None or tol > tolerance and maxiter > 0:
        output = oneshot(equations, hamiltonian, initial_guess)
        if not isinstance(output, tuple):
            output = (output,)
        output = tuple(MetaArray(i, labels=j) if isinstance(i, numpy.ndarray) else i for i, j in zip(output, dim_spec))
        dt = res2amps(output, e_occ, e_vir)
        tol = max(numpy.linalg.norm(i) for i in dt)
        for k, delta in zip(initial_guess, dt):
            initial_guess[k] = initial_guess[k] + delta

        if diis and not any(isinstance(i, Number) for i in initial_guess.values()):
            v = a2v(initial_guess.values())
            initial_guess = OrderedDict(zip(
                initial_guess.keys(),
                v2a(diis.update(v), initial_guess.values())
            ))

        maxiter -= 1

        if debug:
            if equation_energy is not None:
                e = oneshot(equation_energy, hamiltonian, initial_guess)
                print("E = {:.10f} delta={:.3e}".format(e, tol))
            else:
                print("delta={:.3e}".format(tol))

    if equation_energy is not None:
        return initial_guess, oneshot(equation_energy, hamiltonian, initial_guess)

    else:
        return initial_guess


def koopmans_guess_ip(nocc, nvir, amplitudes, n, **kwargs):
    """
    Koopman's guess for IP-EOM-CC amplitudes.
    Args:
        nocc (int): occupied space size;
        nvir (int): virtual space size;
        amplitudes (OrderedDict): an ordered dict with variable name-variable order pairs;
        n (int): the root number;
        kwargs: keyword arguments to `numpy.zeros`.

    Returns:
        An ordered dict with variable name-initial guess pairs.
    """
    result = OrderedDict()
    valid = False
    for k, v in amplitudes.items():
        result[k] = meta(numpy.zeros((nocc,) * v + (nvir,) * (v-1), **kwargs), labels='o' * v + 'v' * (v-1))
        if v == 1:
            if valid:
                raise ValueError("Several first-order amplitudes encountered: {}".format(amplitudes))
            else:
                result[k][-n-1] = 1
                valid = True
    if not valid:
        raise ValueError("No first-order amplitudes found: {}".format(amplitudes))
    return result


def koopmans_guess_ea(nocc, nvir, amplitudes, n, **kwargs):
    """
    Koopman's guess for EA-EOM-CC amplitudes.
    Args:
        nocc (int): occupied space size;
        nvir (int): virtual space size;
        amplitudes (OrderedDict): an ordered dict with variable name-variable order pairs;
        n (int): the root number;
        kwargs: keyword arguments to `numpy.zeros`.

    Returns:
        An ordered dict with variable name-initial guess pairs.
    """
    result = OrderedDict()
    valid = False
    for k, v in amplitudes.items():
        result[k] = meta(numpy.zeros((nocc,) * (v-1) + (nvir,) * v, **kwargs), labels='o' * (v-1) + 'v' * v)
        if v == 1:
            if valid:
                raise ValueError("Several first-order amplitudes encountered: {}".format(amplitudes))
            else:
                result[k][n] = 1
                valid = True
    if not valid:
        raise ValueError("No first-order amplitudes found: {}".format(amplitudes))
    return result


def ltri_ix_amplitudes(a):
    """
    Collects lower-triangular indexes of antisymetric amplitudes.
    Args:
        a (MetaArray): amplitudes to process;

    Returns:
        Lower-triangular indexes.
    """
    if not isinstance(a, MetaArray) or "labels" not in a.metadata:
        raise ValueError("Labels metadata is missing")
    labels = a.metadata["labels"]

    if len(labels) != len(a.shape):
        raise ValueError("The length of 'labels' spec does not match the tensor rank")

    dim_sizes = OrderedDict()
    for label_i, label in enumerate(labels):
        dim_size = a.shape[label_i]
        if label in dim_sizes:
            if dim_sizes[label] != dim_size:
                raise ValueError("Dimensions of the same type '{}' do not match: {:d} vs {:d} in {}".format(
                    label,
                    dim_sizes[label],
                    dim_size,
                    repr(a.shape),
                ))
        else:
            dim_sizes[label] = dim_size

    ix = OrderedDict()
    ix_size = []
    for label, dim_size in dim_sizes.items():
        indexes = ltri_ix(dim_size, labels.count(label))
        ix[label] = iter(indexes)
        ix_size.append(len(indexes[0]))

    # Label order
    label_order = ''.join(ix.keys())

    result = []
    for label in labels:
        x = next(ix[label])
        pos = label_order.index(label)
        bf = numpy.prod([1] + ix_size[:pos])
        ft = numpy.prod([1] + ix_size[pos+1:])
        x = numpy.tile(numpy.repeat(x, ft), bf)
        result.append(x)
    return tuple(result)


def a2v_sym(amplitudes, ixs):
    """
    Symmetric amplitudes into vector.
    Args:
        amplitudes (iterable): amplitudes to join;
        ixs (iterable): indexes of lower-triangle parts;

    Returns:
        A numpy array with amplitudes joined.
    """
    return a2v(a[i] for a, i in zip(amplitudes, ixs))


def v2a_sym(a, labels, shapes, ixs):
    """
    Decompresses the antisymmetric array.
    Args:
        a (numpy.ndarray): array to decompress;
        labels (iterable): array's axes' labels;
        shapes (iterable): arrays' shapes;
        ixs (iterable): indexes of lower-triangle parts;

    Returns:
        Decompressed amplitude tensors.
    """
    result = []
    pos = 0
    for lbls, shape, ix in zip(labels, shapes, ixs):
        ampl = numpy.zeros(shape, dtype=a.dtype)
        end = pos + len(ix[0])
        ampl[ix] = a[pos:end]
        pos = end
        for l in set(lbls):
            letters = iter(string.ascii_lowercase)
            str_spec = ''.join(next(letters) if i == l else '.' for i in lbls)
            ampl = p(str_spec, ampl)
        result.append(ampl)
    return result


def kernel_eig(hamiltonian, equations, amplitudes, tolerance=1e-9):
    """
    Coupled-cluster solver (eigenvalue problem).
    Args:
        hamiltonian (dict): hamiltonian matrix elements or pyscf ERIS;
        equations (callable): coupled-cluster equations;
        amplitudes (iterable): starting amplitudes (a list of OrderedDicts);
        tolerance (float): convergence criterion;

    Returns:
        Resulting coupled-cluster amplitudes and energy if specified.
    """
    # Convert ERIS to hamiltonian dict if needed
    if not isinstance(hamiltonian, dict):
        hamiltonian = eris_hamiltonian(hamiltonian)

    # Preconditioning
    e_occ = numpy.diag(hamiltonian["oo"])
    e_vir = numpy.diag(hamiltonian["vv"])

    # Antisymmetry data
    sample = amplitudes[0].values()
    labels = list(i.metadata["labels"] for i in sample)
    ixs = list(ltri_ix_amplitudes(i) for i in sample)
    shapes = list(i.shape for i in sample)

    def matvec(vec):
        result = []
        for i in vec:
            a = v2a_sym(i, labels, shapes, ixs)
            a = OrderedDict(zip(amplitudes[0].keys(), a))
            r = oneshot(equations, hamiltonian, a)
            result.append(a2v_sym(r, ixs))
        return result

    def precond(res, e0, x0):
        a = v2a_sym(res, labels, shapes, ixs)
        a = list(MetaArray(i, **j.metadata) for i, j in zip(a, amplitudes[0].values()))
        a = res2amps(a, e_occ, e_vir, constant=e0)
        return a2v_sym(a, ixs)

    amplitudes_plain = tuple(a2v_sym(i.values(), ixs) for i in amplitudes)

    conv, values, vectors = davidson(matvec, amplitudes_plain, precond, tol=tolerance, nroots=len(amplitudes))

    if any(not i for i in conv):
        warn("Following eigenvalues did not converge: {}".format(list(
            i for i, x in enumerate(conv) if not x
        )))

    return values, list(v2a_sym(i, labels, shapes, ixs) for i in vectors)
