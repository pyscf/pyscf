from pyscf.tdscf.common_slow import k_nocc
from pyscf.pbc.tdscf.krhf_slow import get_block_k_ix

import numpy
from numpy import testing


def retrieve_m(model, phase=1, **kwargs):
    """Retrieves TDSCF matrix."""
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(phase * numpy.eye(size)).T


def retrieve_m_hf(eri):
    """Retrieves TDHF matrix directly."""
    d = eri.tdhf_diag()
    m = numpy.array([
        [d + 2 * eri["knmj"] - eri["knjm"], 2 * eri["kjmn"] - eri["kjnm"]],
        [- 2 * eri["mnkj"] + eri["mnjk"], - 2 * eri["mjkn"] + eri["mjnk"] - d],
    ])

    return m.transpose(0, 2, 1, 3).reshape(
        (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
    )


def retrieve_m_khf(eri, k):
    """Retrieves TDHF matrix directly (K-version)."""
    r1, r2, c1, c2 = get_block_k_ix(eri, k)

    d1 = eri.tdhf_diag(r1)
    d2 = eri.tdhf_diag(r2)

    m11 = d1 + 2 * eri["knmj", r1, c1] - eri["knjm", r1, c1]
    m12 = 2 * eri["kjmn", r1, c2] - eri["kjnm", r1, c2]
    m21 = 2 * eri["mnkj", r2, c1] - eri["mnjk", r2, c1]
    m22 = d2 + 2 * eri["mjkn", r2, c2] - eri["mjnk", r2, c2]

    m = numpy.array([[m11, m12], [-m21, -m22]])

    return m.transpose((0, 2, 1, 3)).reshape(
        (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
    )


def _test_m_complex_support(model, phase=None, **kwargs):
    if phase is None:
        phase = numpy.exp(2.j * numpy.pi * 0.2468) * 1.3579
    m1 = retrieve_m(model, **kwargs)
    m2 = retrieve_m(model, phase=phase, **kwargs)
    testing.assert_allclose(m2, phase * m1, atol=1e-12)


def sign(x):
    return x / abs(x)


def pull_dim(a, dim):
    """Pulls the specified dimension forward and reshapes array into a 2D matrix."""
    a = a.transpose(*(
            (dim,) + tuple(range(dim)) + tuple(range(dim + 1, len(a.shape)))
    ))
    a = a.reshape(len(a), -1)
    return a


def phase_difference(a, b, axis=0, threshold=1e-5):
    """The phase difference between vectors."""
    v1, v2 = numpy.asarray(a), numpy.asarray(b)
    testing.assert_equal(v1.shape, v2.shape)
    v1, v2 = pull_dim(v1, axis), pull_dim(v2, axis)
    g1 = abs(v1) > threshold
    g2 = abs(v2) > threshold
    g12 = numpy.logical_and(g1, g2)
    if numpy.any(g12.sum(axis=1) == 0):
        desired_threshold = numpy.minimum(abs(v1), abs(v2)).max(axis=1).min()
        raise ValueError("Cannot find an anchor for the rotation, maximal value for the threshold is: {:.3e}".format(
            desired_threshold
        ))
    anchor_index = tuple(numpy.where(i)[0][0] for i in g12)
    return sign(v2[numpy.arange(len(v2)), anchor_index]) / sign(v1[numpy.arange(len(v1)), anchor_index])


def adjust_mf_phase(model1, model2, threshold=1e-5):
    """Tunes the phase of the 2 mean-field models to a common value."""
    signatures = []
    orders = []

    for m in (model1, model2):
        if "kpts" in dir(m):
            signatures.append(numpy.concatenate(m.mo_coeff, axis=1))
            orders.append(numpy.argsort(numpy.concatenate(m.mo_energy)))
        else:
            signatures.append(m.mo_coeff)
            orders.append(numpy.argsort(m.mo_energy))

    m1, m2 = signatures
    o1, o2 = orders
    mdim = min(m1.shape[0], m2.shape[0])
    m1, m2 = m1[:mdim, :][:, o1], m2[:mdim, :][:, o2]

    p = phase_difference(m1, m2, axis=1, threshold=threshold)

    if "kpts" in dir(model2):
        fr = 0
        for k, i in enumerate(model2.mo_coeff):
            to = fr + i.shape[1]
            slc = numpy.logical_and(fr <= o2, o2 < to)
            i[:, o2[slc] - fr] /= p[slc][numpy.newaxis, :]
            fr = to
    else:
        model2.mo_coeff[:, o2] /= p[numpy.newaxis, :]


def tdhf_frozen_mask(eri, kind="ov"):
    if isinstance(eri.nocc, int):
        nocc = int(eri.model.mo_occ.sum() // 2)
        mask = eri.space
    else:
        nocc = numpy.array(tuple(int(i.sum() // 2) for i in eri.model.mo_occ))
        assert numpy.all(nocc == nocc[0])
        assert numpy.all(eri.space == eri.space[0, numpy.newaxis, :])
        nocc = nocc[0]
        mask = eri.space[0]
    mask_o = mask[:nocc]
    mask_v = mask[nocc:]
    if kind == "ov":
        mask_ov = numpy.outer(mask_o, mask_v).reshape(-1)
        return numpy.tile(mask_ov, 2)
    elif kind == "1ov":
        return numpy.outer(mask_o, mask_v).reshape(-1)
    elif kind == "sov":
        mask_ov = numpy.outer(mask_o, mask_v).reshape(-1)
        nk = len(eri.model.mo_occ)
        return numpy.tile(mask_ov, 2 * nk ** 2)
    elif kind == "o,v":
        return mask_o, mask_v


def convert_k2s(vectors, k, eri):
    """Converts vectors from k-representation to the supercell space by padding with zeros."""
    nv, _, nk, nocc, nvirt = vectors.shape
    # _ = 2
    result = numpy.zeros((nv, _, nk, nk, nocc, nvirt), dtype=vectors.dtype)
    r1, r2, _, _ = get_block_k_ix(eri, k)
    for k1 in range(nk):
        k2_x = r1[k1]
        result[:, 0, k1, k2_x] = vectors[:, 0, k1]
        k2_y = r2[k1]
        result[:, 1, k1, k2_y] = vectors[:, 1, k1]
    return result


def adjust_td_phase(model1, model2, threshold=1e-5):
    """Tunes the phase of the 2 time-dependent models to a common value."""
    signatures = []
    orders = []
    space = []

    for m in (model1, model2):
        # Are there k-points?
        if "kpts" in dir(m._scf):
            # Is it a supercell model, Gamma model or a true k-model?
            if isinstance(m.xy, dict) or len(m.xy.shape) == 6:
                if isinstance(m.xy, dict):
                    # A true k-model
                    xy = []
                    e = []
                    for k in range(len(m.e)):
                        xy.append(convert_k2s(m.xy[k], k, m.eri))
                        e.append(m.e[k])
                    xy = numpy.concatenate(xy)
                    e = numpy.concatenate(e)
                else:
                    # A supercell model
                    xy = m.xy
                    e = m.e
                xy = xy.reshape(len(e), -1)
                order_truncated = ov_order(m._scf, m.eri.space)
                order_orig = ov_order(m._scf)
                xy = xy[:, order_truncated]
                signatures.append(xy)
                orders.append(numpy.argsort(e))
                space.append(tdhf_frozen_mask(m.eri, "sov")[order_orig])
            elif len(m.xy.shape) == 5:
                # Gamma model
                raise NotImplementedError("Implement me")
            else:
                raise ValueError("Unknown vectors: {}".format(repr(m.xy)))
        else:
            signatures.append(m.xy.reshape(len(m.e), -1))
            orders.append(numpy.argsort(m.e))
            space.append(tdhf_frozen_mask(m.eri))

    common_space = None
    for i in space:
        if i is not None:
            if common_space is None:
                common_space = i.copy()
            else:
                common_space = numpy.logical_and(common_space, i)

    m1, m2 = signatures
    o1, o2 = orders
    m1, m2 = m1[o1, :], m2[o2, :]

    if common_space is not None:
        space = [common_space[i] if i is not None else common_space for i in space]
        s1, s2 = space
        m1 = m1[:, s1]
        m2 = m2[:, s2]

    p = phase_difference(m1, m2, axis=0, threshold=threshold)

    if "kpts" in dir(model2._scf):
        # Is it a supercell model, Gamma model or a true k-model?
        if isinstance(m.xy, dict):
            # A true k-model
            nvec_per_kp = len(m.e[0])
            for k in range(len(m.e)):
                o2_kp_mask = o2 // nvec_per_kp == k
                o2_kp = o2[o2_kp_mask] % nvec_per_kp
                p_kp = p[o2_kp_mask]
                model2.xy[k][o2_kp, ...] /= p_kp[(slice(None),) + (numpy.newaxis,) * 4]
        elif len(m.xy.shape) == 6:
            # A supercell model
            model2.xy[o2, ...] /= p[(slice(None),) + (numpy.newaxis,) * 5]
        elif len(m.xy.shape) == 5:
            # Gamma model
            raise NotImplementedError("Implement me")
        else:
            raise ValueError("Unknown vectors: {}".format(repr(m.xy)))
    else:
        model2.xy[o2, ...] /= p[(slice(None),) + (numpy.newaxis,) * 3]


def remove_phase_difference(v1, v2, axis=0, threshold=1e-5):
    """Removes the phase difference between two vectors."""
    dtype = numpy.common_type(numpy.asarray(v1), numpy.asarray(v2))
    v1, v2 = numpy.array(v1, dtype=dtype), numpy.array(v2, dtype=dtype)
    v1, v2 = pull_dim(v1, axis), pull_dim(v2, axis)
    v2 /= phase_difference(v1, v2, threshold=threshold)[:, numpy.newaxis]
    return v1, v2


def assert_vectors_close(v1, v2, axis=0, threshold=1e-5, atol=1e-8):
    """Compares two vectors up to a phase difference."""
    v1, v2 = remove_phase_difference(v1, v2, axis=axis, threshold=threshold)
    delta = abs(v1 - v2).max(axis=1)
    wrong = delta > atol
    if any(wrong):
        raise AssertionError("Vectors are not close to tolerance atol={}\n\n({:d} roots mismatch)\ndelta {}".format(
            str(atol),
            sum(wrong),
            ", ".join("#{:d}: {:.3e}".format(i, delta[i]) for i in numpy.argwhere(wrong)[:, 0]),
        ))


def ov_order(model, slc=None):
    nocc = k_nocc(model)
    if slc is None:
        slc = numpy.ones((len(model.mo_coeff), model.mo_coeff[0].shape[1]), dtype=bool)
    e_occ = tuple(e[:o][s[:o]] for e, o, s in zip(model.mo_energy, nocc, slc))
    e_virt = tuple(e[o:][s[o:]] for e, o, s in zip(model.mo_energy, nocc, slc))
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
