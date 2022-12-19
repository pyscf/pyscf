# Based on test_fcidump.py in pyscf.
import ase
from functools import reduce
import unittest
import tempfile
import numpy
import pandas
from pyscf import lib
from pyscf import gto, scf, ao2mo
from pyscf.tools import fcidump
from pyscfdump.helpers.ase_and_cell import build_cell
from pyscfdump.helpers.scf import run_khf
import pyscfdump.pbcfcidump

basis = "STO-3G"
ke_cutoff = 200.0
fcid = 'FCIDUMP'
nmp = numpy.array([3, 1, 1])
ase_atom = ase.Atoms("HH", positions=[[0, 0, 0], [4.0, 0, 0]],
                     cell=[10.0, 10.0, 10.0], pbc=True)
cell = build_cell(ase_atom, ke=ke_cutoff, basis=basis, incore_anyway=True,
                 pseudo=None)
mf, scaled_kpts = run_khf(cell, nmp=nmp, gamma=True, exxdiv="vcut_ws")

def tearDownModule():
    # In line with pyscf testing.
    global ase_atom, cell, mf
    del ase_atom, cell, mf

def read_in(filename, exchange=False):
    skiprows = {True: 0, False: 6}
    integral = numpy.loadtxt(filename, skiprows=skiprows[exchange],
                             unpack=True, usecols=(0), dtype=str)
    i, a, j, b = numpy.loadtxt(filename, skiprows=skiprows[exchange],
                               unpack=True, usecols=(1,2,3,4), dtype=int)
    # https://stackoverflow.com/questions/3559559/how-to-delete-a-character-from-a-string-using-python
    integral = numpy.asarray([numpy.complex(
        *map(float,inte.replace("(", "").replace(")", "").split(","))
        ) for inte in integral])
    return pandas.DataFrame({'int': integral, 'i': i, 'a': a, 'j': j, 'b': b})

def compare_intdumps(filename_new, filename_bench, exchange):
    df_new = read_in(filename_new, exchange)
    df_bench = read_in(filename_bench, exchange)
    df_m = df_new.copy().merge(df_bench.copy(), on=['i', 'a', 'j', 'b'],
                               how="inner")
    df_m['abs_mag_diff'] = abs(abs(df_m['int_x']) - abs(df_m['int_y']))
    df_m['abs_real_diff'] = abs(numpy.array([x.real for x in df_m['int_x']])
                              - numpy.array([y.real for y in df_m['int_y']]))
    df_m['abs_imag_diff'] = abs(numpy.array([x.imag for x in df_m['int_x']])
                              - numpy.array([y.imag for y in df_m['int_y']]))
    print("Difference in magnitude of integrals greater than 1e-6:")
    print(df_m.query('abs_mag_diff > 1e-6'))
    print("Difference in real value of integrals greater than 1e-6:")
    print(df_m.query('abs_real_diff > 1e-6'))
    print("Difference in imag value of integrals greater than 1e-6:")
    print(df_m.query('abs_imag_diff > 1e-6'))
    return df_m

class CalcIntegrals(unittest.TestCase):

    def test_setup(self):
        # SCF energy
        self.assertAlmostEqual(mf.e_tot, -0.7616939508075196)
        # scaled_kpts
        numpy.testing.assert_array_almost_equal(
                scaled_kpts, numpy.array([[-0.3333333, 0., 0.],
                                          [0., 0., 0.],
                                          [0.3333333, 0., 0.]]))

    def test_overall_fcidump(self):
        pyscfdump.pbcfcidump.fcidump(fcid, mf, nmp, scaled_kpts, False,
                                     keep_exxdiv=False)
        print("testing fcidump")
        df_m  = compare_intdumps(fcid, fcid+"_v", False)
        self.assertTrue(df_m.query('abs_mag_diff > 1e-6').empty)
        self.assertTrue(df_m.query('abs_real_diff > 1e-6').empty)
        self.assertTrue(df_m.query('abs_imag_diff > 1e-6').empty)
        print("testing fcidump_x")
        df_m = compare_intdumps(fcid+"_X", fcid+"_v_X", True)
        self.assertTrue(df_m.query('abs_mag_diff > 1e-6').empty)
        self.assertTrue(df_m.query('abs_real_diff > 1e-6').empty)
        self.assertTrue(df_m.query('abs_imag_diff > 1e-6').empty)

if __name__ == "__main__":
    print("Full Tests for pbcfcidump")
    unittest.main()



