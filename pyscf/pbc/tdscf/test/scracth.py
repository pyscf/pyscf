from pyscf.gto import Mole
from pyscf.scf import RHF
from pyscf.tdscf import TDHF
from pyscf import ao2mo

import numpy
from numpy import testing


class TDDFTMatrixBlocks(object):
    symmetries = [
        ((0, 1, 2, 3), False),
    ]

    def __init__(self):
        self.__eri__ = {}

    def __get_mo_energies__(self, *args, **kwargs):
        raise NotImplementedError

    def get_diag_block(self, *args):
        """
        Retrieves the diagonal block.
        Args:
            *args: args passed to `__get_mo_energies__`;

        Returns:
            The diagonal block.
        """
        e_occ, e_virt = self.__get_mo_energies__(*args)
        diag = (- e_occ[:, numpy.newaxis] + e_virt[numpy.newaxis, :]).reshape(-1)
        return numpy.diag(diag).reshape((len(e_occ), len(e_virt), len(e_occ), len(e_virt)))

    def assemble_diag_block(self):
        """
        Assembles the diagonal part of the TDDFT blocks.
        Returns:
            The diagonal block.
        """
        raise NotImplementedError

    def __calc_block__(self, item, *args):
        raise NotImplementedError

    def get_block_ov_notation(self, item, *args):
        """
        Retrieves ERI block using 'ov' notation.
        Args:
            item (str): a 4-character string of 'o' and 'v' letters;
            *args: other args passed to `__calc_block__`;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or not set(item).issubset('ov'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        for i, c in self.symmetries:
            key = ''.join(tuple(item[j] for j in i))
            if key in self.__eri__:
                if c:
                    return self.__eri__[key, args].transpose(*i).conj()
                else:
                    return self.__eri__[key, args].transpose(*i)

        self.__eri__[item, args] = self.__calc_block__(item, *args)
        return self.__eri__[item, args]

    def __mknj2i__(self, item):
        notation = "mknj"
        notation = dict(zip(notation, range(len(notation))))
        return tuple(notation[i] for i in item)

    def get_block_mknj_notation(self, item, *args):
        """
        Retrieves ERI block using 'mknj' notation.
        Args:
            item (str): a 4-character string of 'mknj' letters;
            *args: other arguments passed to `get_block_ov_notation`;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or set(item) != set('mknj'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        item = self.__mknj2i__(item)
        n_ov = ''.join('o' if i % 2 == 0 else 'v' for i in item)
        return self.get_block_ov_notation(n_ov, *args).transpose(*numpy.argsort(item))

    def assemble_block(self, item):
        """
        Assembles the entire TDDFT block.
        Args:
            item (str): a 4-character string of 'mknj' letters;

        Returns:
            The TDDFT matrix block.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        return self.assemble_block(item)


class PhysERI(TDDFTMatrixBlocks):

    def __init__(self, model):
        """
        4-center integrals. Notation corresponds to RevModPhys.36.844. Integral blocks are obtained via __getitem__.

        Args:
            model (RHF): the base model;
            mode (str): the mode, either 'tensor' or 'matrix': affects whether __getitem__ returns
            tensors or the same tensors reshaped into matrixes;
        """
        super(PhysERI, self).__init__()
        self.model = model
        self.__full_eri__ = self.ao2mo((self.model.mo_coeff,) * 4)

    @property
    def nocc(self):
        return int(self.model.mo_occ.sum()) // 2

    def ao2mo(self, coeff):
        """
        Phys ERI in MO basis.
        Args:
            coeff (Iterable): MO orbitals;

        Returns:
            ERI in MO basis.
        """
        coeff = (coeff[0], coeff[2], coeff[1], coeff[3])

        if "with_df" in dir(self.model):
            result = self.model.with_df.ao2mo(coeff, compact=False)
        else:
            result = ao2mo.general(self.model.mol, coeff, compact=False)

        return result.reshape(
            tuple(i.shape[1] for i in coeff)
        ).swapaxes(1, 2)

    def __get_mo_energies__(self):
        return self.model.mo_energy[:self.nocc], self.model.mo_energy[self.nocc:]

    def assemble_diag_block(self):
        result = self.get_diag_block()
        o1, v1, o2, v2 = result.shape
        return result.reshape(o1 * v1, o2 * v2)

    def __calc_block__(self, item):
        slc = tuple(slice(self.nocc) if i == 'o' else slice(self.nocc, None) for i in item)
        return self.__full_eri__[slc]

    def assemble_block(self, item):
        result = self.get_block_mknj_notation(item)
        o1, v1, o2, v2 = result.shape
        return result.reshape(o1 * v1, o2 * v2)


class PhysERI4(PhysERI):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __init__(self, model):
        super(PhysERI, self).__init__()
        self.model = model

    def __calc_block__(self, item):
        o = self.model.mo_coeff[:, :self.nocc]
        v = self.model.mo_coeff[:, self.nocc:]
        return self.ao2mo(tuple(o if i == "o" else v for i in item))


class PhysERI8(PhysERI4):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), False),
        ((3, 2, 1, 0), False),

        ((2, 1, 0, 3), False),
        ((3, 0, 1, 2), False),
        ((0, 3, 2, 1), False),
        ((1, 2, 3, 0), False),
    ]


from pyscf.pbc.gto import Cell
from pyscf.pbc.scf import RHF as pRHF, KRHF as kRHF
from pyscf.pbc.tdscf import TDHF as pTDHF, KTDHF as kTDHF
from pyscf.pbc.tools import get_kconserv

from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.lib.kpts_helper import loop_kkk

import scipy


def k_nocc(model):
    return tuple(int(i.sum()) // 2 for i in model.mo_occ)


class PhysERIkS(TDDFTMatrixBlocks):

    def __init__(self, model):
        """
        4-center integrals (the supercell version). Notation corresponds to RevModPhys.36.844.
        Integral blocks are obtained via __getitem__.

        Args:
            model (RHF): the base model;
        """
        super(PhysERIkS, self).__init__()
        self.model = model
        # Phys representation
        self.kconserv = get_kconserv(self.model.cell, self.model.kpts).swapaxes(1, 2)
        for k in loop_kkk(len(model.kpts)):
            k = k + (self.kconserv[k],)
            self.__eri__[k] = self.ao2mo_k(tuple(self.model.mo_coeff[j] for j in k), k)

    @property
    def nocc(self):
        return k_nocc(self.model)

    def ao2mo_k(self, coeff, k):
        """
        Phys ERI in MO basis.
        Args:
            coeff (Iterable): MO orbitals;
            k (Iterable): the 4 k-points MOs correspond to;

        Returns:
            ERI in MO basis.
        """
        coeff = (coeff[0], coeff[2], coeff[1], coeff[3])
        k = (k[0], k[2], k[1], k[3])
        result = self.model.with_df.ao2mo(coeff, tuple(self.model.kpts[i] for i in k), compact=False)
        return result.reshape(
            tuple(i.shape[1] for i in coeff)
        ).swapaxes(1, 2)

    def __get_mo_energies__(self, k1, k2):
        return self.model.mo_energy[k1][:self.nocc[k1]], self.model.mo_energy[k2][self.nocc[k2]:]

    def assemble_diag_block(self):
        result = []
        for k1 in range(len(self.model.kpts)):
            for k2 in range(len(self.model.kpts)):
                b = self.get_diag_block(k1, k2)
                o1, v1, o2, v2 = b.shape
                b = b.reshape(o1 * v1, o2 * v2)
                result.append(b)
        return scipy.linalg.block_diag(*result)

    def __calc_block__(self, item, k):
        if k in self.__eri__:
            slc = tuple(slice(self.nocc[_k]) if i == 'o' else slice(self.nocc[_k], None) for i, _k in zip(item, k))
            return self.__eri__[k][slc]
        else:
            return numpy.zeros(tuple(
                self.nocc[_k] if i == 'o' else self.model.mo_coeff[_k].shape[-1] - self.nocc[_k]
                for i, _k in zip(item, k)
            ))

    def get_block_mknj_notation(self, item, k):
        """
        Retrieves ERI block using 'mknj' notation.
        Args:
            item (str): a 4-character string of 'mknj' letters;
            k (Iterable): k indexes;

        Returns:
            The corresponding block of ERI (phys notation).
        """
        if len(item) != 4 or not isinstance(item, str) or set(item) != set('mknj'):
            raise ValueError("Unknown item: {}".format(repr(item)))

        item_i = self.__mknj2i__(item)
        k = tuple(k[i] for i in item_i)
        return super(PhysERIkS, self).get_block_mknj_notation(item, k)

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                result.append([])
                for k3 in range(nkpts):
                    for k4 in range(nkpts):
                        x = self.get_block_mknj_notation(item, (k1, k2, k3, k4))
                        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                        result[-1].append(x)

        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERIkS4(PhysERIkS):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    def __calc_block__(self, item, k):
        if k in self.__eri__:
            return self.ao2mo_k(tuple(
                self.model.mo_coeff[_k][:, :self.nocc[_k]] if i == "o" else self.model.mo_coeff[_k][:, self.nocc[_k]:]
                for i, _k in zip(item, k)
            ), k)
        else:
            return numpy.zeros(tuple(
                self.nocc[_k] if i == 'o' else self.model.mo_coeff[_k].shape[-1] - self.nocc[_k]
                for i, _k in zip(item, k)
            ))


class PhysERIkS8(PhysERIkS4):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), False),
        ((3, 2, 1, 0), False),

        ((2, 1, 0, 3), False),
        ((3, 0, 1, 2), False),
        ((0, 3, 2, 1), False),
        ((1, 2, 3, 0), False),
    ]


class PhysERIkG(PhysERIkS):

    def __init__(self, model):
        """
        4-center integrals (the zero-momentum transfer variant). Notation corresponds to RevModPhys.36.844. Integral
        blocks are obtained via __getitem__.

        Args:
            model (RHF): the base model;
        """
        super(PhysERIkG, self).__init__(model)

    def assemble_diag_block(self):
        result = []
        for k in range(len(self.model.kpts)):
            b = self.get_diag_block(k, k)
            o1, v1, o2, v2 = b.shape
            b = b.reshape(o1 * v1, o2 * v2)
            result.append(b)
        return scipy.linalg.block_diag(*result)

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            result.append([])
            for k2 in range(nkpts):
                x = self.get_block_mknj_notation(item, (k1, k1, k2, k2))
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                result[-1].append(x)

        r = numpy.block(result)
        return r / len(self.model.kpts)


class PhysERIkG4(PhysERIkG):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), True),
        ((3, 2, 1, 0), True),
    ]

    __calc_block__ = PhysERIkS4.__calc_block__.im_func


class PhysERIkG8(PhysERIkG4):
    symmetries = [
        ((0, 1, 2, 3), False),
        ((1, 0, 3, 2), False),
        ((2, 3, 0, 1), False),
        ((3, 2, 1, 0), False),

        ((2, 1, 0, 3), False),
        ((3, 0, 1, 2), False),
        ((0, 3, 2, 1), False),
        ((1, 2, 3, 0), False),
    ]


class PhysERIkT(PhysERIkS):

    def __init__(self, model, k):
        """
        4-center integrals (the final variant with momentum transfer). Notation corresponds to RevModPhys.36.844.
        Integral blocks are obtained via __getitem__.

        Args:
            model (RHF): the base model;
            k (int): the momentum transfer index corresponding to all momnetum pairs `i, j` satisfying
            `kconserv[k, 0, i] == j`.
        """
        super(PhysERIkT, self).__init__(model)
        self.k = k
        # Note that kconserv is in phys notation
        self.kconserv_k0 = self.kconserv[k, :, 0]

    def assemble_diag_block(self):
        result = []
        for k in range(len(self.model.kpts)):
            k2 = self.kconserv_k0[k]
            b = self.get_diag_block(k, k2)
            o1, v1, o2, v2 = b.shape
            b = b.reshape(o1 * v1, o2 * v2)
            result.append(b)
        return scipy.linalg.block_diag(*result)

    def __get_adjusted_k__(self, item, k1, k2):
        k3 = self.kconserv_k0[k1]
        k4 = self.kconserv[k1, k2, k3]
        # For item == mknj return k1, k3, k2, k4
        # item_i = self.__mknj2i__(item)
        # result = tuple((k1, k3, k2, k4)[i] for i in numpy.argsort(item_i))
        result = (k1, k3, k2, k4)
        return result

    def assemble_block(self, item):
        result = []
        nkpts = len(self.model.kpts)
        for k1 in range(nkpts):
            result.append([])
            for k2 in range(nkpts):
                x = self.get_block_mknj_notation(item, self.__get_adjusted_k__(item, k1, k2))
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                result[-1].append(x)
        r = numpy.block(result)
        return r / len(self.model.kpts)


def full_tensor(eri):
    """
    Full tensor of the TDRHF problem.
    Args:
        eri (TDDFTMatrixBlocks): ERI of the problem;

    Returns:
        The matrix.
    """
    d = eri.assemble_diag_block()
    m = numpy.array([
        [d + 2 * eri["knmj"] - eri["knjm"],   2 * eri["kjmn"] - eri["kjnm"]],
        [  - 2 * eri["mnkj"] + eri["mnjk"], - 2 * eri["mjkn"] + eri["mjnk"] - d],
    ])

    return m.transpose(0, 2, 1, 3).reshape(
        (m.shape[0] * m.shape[2], m.shape[1] * m.shape[3])
    )


def retrieve_m(model, **kwargs):
    vind, hdiag = model.gen_vind(model._scf, **kwargs)
    size = model.init_guess(model._scf, 1).shape[1]
    return vind(numpy.eye(size)).T


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


def make_phase_well_defined(model):
    for i in model.mo_coeff:
        i *= numpy.sign(i[0])[numpy.newaxis, :]


# ===================
# Mole
# ===================

mol = Mole()
mol.atom = [
    [8, (0., 0., 0.)],
    [1, (0., -0.757, 0.587)],
    [1, (0., 0.757, 0.587)]]

mol.basis = {'H': 'cc-pvdz',
             'O': 'cc-pvdz', }
mol.verbose = 5
mol.build()

model_rhf = RHF(mol).density_fit()
model_rhf.kernel()

td_model_rhf = TDHF(model_rhf)
td_model_rhf.nroots = 5
td_model_rhf.kernel()

ref_m = retrieve_m(td_model_rhf)

m = full_tensor(PhysERI(model_rhf))
m4 = full_tensor(PhysERI4(model_rhf))
m8 = full_tensor(PhysERI8(model_rhf))

vals = numpy.linalg.eigvals(m)
vals.sort()
vals = vals[len(vals) // 2:][:td_model_rhf.nroots]

testing.assert_allclose(td_model_rhf.e, vals, rtol=1e-5)

testing.assert_allclose(ref_m, m, atol=1e-14)
testing.assert_allclose(ref_m, m4, atol=1e-14)
testing.assert_allclose(ref_m, m8, atol=1e-14)

# ===================
# PBC - Gamma
# ===================

cell = Cell()
# Lift some degeneracies
cell.atom = '''
C 0.000000000000   0.000000000000   0.000000000000
C 1.67   1.68   1.69
'''
cell.basis = {'C': [[0, (0.8, 1.0)],
                    [1, (1.0, 1.0)]]}
# cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

model_prhf = pRHF(cell)
model_prhf.kernel()

td_model_prhf = pTDHF(model_prhf)
td_model_prhf.nroots = 5
td_model_prhf.kernel()

ref_m = retrieve_m(td_model_prhf)

m = full_tensor(PhysERI(model_prhf))
m4 = full_tensor(PhysERI4(model_prhf))
m8 = full_tensor(PhysERI8(model_prhf))

vals = numpy.linalg.eigvals(m)
vals.sort()
vals = vals[len(vals) // 2:][:td_model_prhf.nroots]

testing.assert_allclose(td_model_prhf.e, vals, rtol=1e-5)

testing.assert_allclose(ref_m, m, atol=1e-14)
testing.assert_allclose(ref_m, m4, atol=1e-14)
testing.assert_allclose(ref_m, m8, atol=1e-14)

# ===================
# Test
# ===================
#
# model_krhf = kRHF(cell, kpts=cell.make_kpts([3, 1, 1])).density_fit()
# model_krhf.kernel()
#
# m = full_tensor(PhysERIkS(model_krhf))
# from matplotlib import pyplot
# pyplot.matshow((m[:m.shape[0] // 2, :m.shape[1] // 2] == 0).astype(int))
# pyplot.colorbar()
# pyplot.show()
# pyplot.matshow((m[m.shape[0] // 2:, :m.shape[1] // 2] == 0).astype(int))
# pyplot.colorbar()
# pyplot.show()
# pyplot.matshow((m[:m.shape[0] // 2, m.shape[1] // 2:] == 0).astype(int))
# pyplot.colorbar()
# pyplot.show()
# pyplot.matshow((m[m.shape[0] // 2:, m.shape[1] // 2:] == 0).astype(int))
# pyplot.colorbar()
# pyplot.show()
# exit()

# ===================
# PBC - fake Gamma
# ===================

model_srhf = kRHF(super_cell(cell, [2, 1, 1]))#.density_fit()
model_srhf.kernel()

make_phase_well_defined(model_srhf)

td_model_srhf = kTDHF(model_srhf)
td_model_srhf.nroots = 5
td_model_srhf.kernel()

ref_m = retrieve_m(td_model_srhf)

m = full_tensor(PhysERIkS(model_srhf))
m4 = full_tensor(PhysERIkS4(model_srhf))
m8 = full_tensor(PhysERIkS8(model_srhf))

vals = numpy.linalg.eigvals(m)
vals.sort()
vals = vals[len(vals) // 2:][:td_model_srhf.nroots]

# This fails (guess why)
# ----------------------
# print td_model_srhf.converged
# assert all(td_model_srhf.converged)
# testing.assert_allclose(td_model_srhf.e, vals, rtol=1e-5)

testing.assert_allclose(ref_m, m, atol=1e-13)
testing.assert_allclose(ref_m, m4, atol=1e-13)
testing.assert_allclose(ref_m, m8, atol=1e-13)

# =================================
# PBC - 2 kp (sc-type) vs supercell
# =================================

model_krhf = kRHF(cell, kpts=cell.make_kpts([2, 1, 1]))#.density_fit()
model_krhf.kernel()

make_phase_well_defined(model_krhf)

ke = numpy.concatenate(model_krhf.mo_energy)
ke.sort()
testing.assert_allclose(ke, model_srhf.mo_energy[0], atol=1e-6)
assert abs(model_srhf.mo_coeff[0].imag).max() < 1e-10
assert all(abs(i.imag).max() < 1e-10 for i in model_krhf.mo_coeff)

m_ref = m

order = ov_order(model_krhf)

m = full_tensor(PhysERIkS(model_krhf))
m4 = full_tensor(PhysERIkS4(model_krhf))
m8 = full_tensor(PhysERIkS8(model_krhf))

vals_ref = vals
vals = numpy.linalg.eigvals(m)
vals.sort()
vals = vals[len(vals) // 2:][:td_model_srhf.nroots]

testing.assert_allclose(m_ref, m[numpy.ix_(order, order)], atol=1e-5)
testing.assert_allclose(m_ref, m4[numpy.ix_(order, order)], atol=1e-5)
testing.assert_allclose(m_ref, m8[numpy.ix_(order, order)], atol=1e-5)

# Check against own implementation because pyscf davidson fails at the previous step
testing.assert_allclose(vals_ref, vals, rtol=1e-5)

# ======================
# PBC - Gamma that fails
# ======================

fails_rhf = kRHF(cell, kpts=cell.make_kpts([1, 1, 1], scaled_center=(.1, .2, .3)))
fails_rhf.kernel()

fails_td = kTDHF(fails_rhf)
fails_td.nroots = 5
# fails_td.kernel()

ref_m = retrieve_m(fails_td)

m = full_tensor(PhysERIkS(fails_rhf))
m4 = full_tensor(PhysERIkS4(fails_rhf))

# PhysERIk8 no longer works because orbitals are complex
# m8 = full_tensor(PhysERIk8(fails_rhf))

testing.assert_allclose(m, m4, atol=1e-13)
print abs(ref_m - m).max()
# This also fails since pyscf assumes orbitals to be real
# testing.assert_allclose(ref_m, m, atol=1e-13)

# ============================
# PBC - zero momentum transfer
# ============================

assert abs(numpy.array(model_krhf.mo_coeff).imag).max() < 1e-10

td_model_krhf = kTDHF(model_krhf)
td_model_krhf.kernel()

m_ref = retrieve_m(td_model_krhf)

m = full_tensor(PhysERIkG(model_krhf))
m4 = full_tensor(PhysERIkG4(model_krhf))
m8 = full_tensor(PhysERIkG8(model_krhf))

vals = numpy.linalg.eigvals(m)
vals.sort()
vals = vals[len(vals) // 2:][:td_model_krhf.nroots]
testing.assert_allclose(td_model_krhf.e, vals, rtol=1e-5)

testing.assert_allclose(m_ref, m, atol=1e-5)
testing.assert_allclose(m_ref, m4, atol=1e-5)
testing.assert_allclose(m_ref, m8, atol=1e-5)

# ================================
# PBC - non-zero momentum transfer
# ================================

m_ref = m

m = full_tensor(PhysERIkT(model_krhf, 0))

testing.assert_allclose(m_ref, m, atol=1e-10)

m_ref = retrieve_m(td_model_srhf)
vals_ref = numpy.linalg.eigvals(m_ref)
vals_ref.sort()

vals_ref = vals_ref[len(vals_ref) // 2:]

vals = []
for k in range(len(model_krhf.kpts)):
    m = full_tensor(PhysERIkT(model_krhf, k))
    v = numpy.linalg.eigvals(m)
    v.sort()
    v = v[len(v) // 2:]
    vals.append(v)

vals = numpy.concatenate(vals)
vals.sort()
testing.assert_allclose(vals_ref, vals, atol=1e-6)

# testing.assert_allclose(m_ref, m, atol=1e-5)
# testing.assert_allclose(m_ref, m4, atol=1e-5)
# testing.assert_allclose(m_ref, m8, atol=1e-5)
