from pyscf.pbc.tdscf.krhf_slow_supercell import k_nocc

import numpy
from numpy import testing


def retrieve_m(model, **kwargs):
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(numpy.eye(size)).T


def sign(x):
    return x / abs(x)


def make_mf_phase_well_defined(model):
    if "kpts" in dir(model):
        for i in model.mo_coeff:
            i /= sign(i[0])[numpy.newaxis, :]
    else:
        model.mo_coeff /= sign(model.mo_coeff[0])[numpy.newaxis, :]


def unphase(v1, v2, threshold=1e-5):
    v1, v2 = numpy.asarray(v1), numpy.asarray(v2)
    testing.assert_equal(v1.shape, v2.shape)
    v1, v2 = v1.reshape(len(v1), -1), v2.reshape(len(v2), -1)
    g1 = abs(v1) > threshold
    g2 = abs(v2) > threshold
    g12 = numpy.logical_and(g1, g2)
    if numpy.any(g12.sum(axis=1) == 0):
        desired_threshold = numpy.minimum(abs(v1), abs(v2)).max(axis=1).min()
        raise ValueError("Cannot find an anchor for the rotation, minimal value for the threshold is: {:.3e}".format(
            desired_threshold
        ))
    a = tuple(numpy.where(i)[0][0] for i in g12)
    for v in (v1, v2):
        anc = v[numpy.arange(len(v)), a]
        v /= (anc / abs(anc))[:, numpy.newaxis]
    return v1, v2


def assert_vectors_close(v1, v2, threshold=1e-5, atol=1e-8):
    v1, v2 = unphase(v1, v2, threshold=threshold)
    delta = abs(v1 - v2).max(axis=1)
    wrong = delta > atol
    if any(wrong):
        raise AssertionError("Vectors are not close to tolerance atol={}\n\n({:d} roots mismatch)\ndelta {}".format(
            str(atol),
            sum(wrong),
            ", ".join("#{:d}: {:.3e}".format(i, delta[i]) for i in numpy.argwhere(wrong)[:, 0]),
        ))


def ov_order(model):
    nocc = k_nocc(model)
    e_occ = tuple(e[:o] for e, o in zip(model.mo_energy, nocc))
    e_virt = tuple(e[o:] for e, o in zip(model.mo_energy, nocc))
    sort_o = []
    sort_v = []
    for o in e_occ:
        for v in e_virt:
            _v, _o = numpy.meshgrid(v, o)
            sort_o.append(_o.reshape(-1))
            sort_v.append(_v.reshape(-1))
    sort_o, sort_v = numpy.concatenate(sort_o), numpy.concatenate(sort_v)
    vals = numpy.array(
        list(zip(sort_o, sort_v)),
        dtype=[('o', sort_o[0].dtype), ('v', sort_v[0].dtype)]
    )
    result = numpy.argsort(vals, order=('o', 'v'))
    # Double for other blocks
    return numpy.concatenate([result, result + len(result)])
