from __future__ import print_function, division
import unittest,numpy as np
from pyscf.nao.tddft_iter_2ord import tddft_iter_2ord
from os.path import dirname, abspath

class KnowValues(unittest.TestCase):

  def test_tddft_upkc(self):
    """ This is a comparison of two equivalent ways of computing the polarizability for water molecule """
    td = tddft_iter_2ord(label='water', cd=dirname(abspath(__file__)),jcutoff=7,xc_code='RPA',verbosity=0)
    omegas = np.arange(0.0,1.0,0.005)+1j*0.01

    pxx1 = -td.comp_polariz_nonin_xx(omegas).imag
    data1 = np.array([omegas.real*27.2114, pxx1])
    np.savetxt('water.tddft_iter_nonin.txt', data1.T)
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)


    pxx1 = -td.comp_polariz_inter_xx(omegas).imag
    data1 = np.array([omegas.real*27.2114, pxx1])
    np.savetxt('water.tddft_iter_unit.txt', data1.T)
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)

    pxx2 = td.polariz_upkc(omegas)
    wp = np.zeros((2*pxx2.shape[1]+1,pxx2.shape[0]))
    wp[0,:] = omegas.real*27.2114
    wp[1:pxx2.shape[1]+1,:] = pxx2.real.T
    wp[pxx2.shape[1]+1:,:] = pxx2.imag.T
    np.savetxt('water.tddft_iter_upkc.txt', wp.T)
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)

    pxx3 = -td.polariz_dckcd(omegas).imag
    data1 = np.array([omegas.real*27.2114, pxx3])
    np.savetxt('water.tddft_iter_dckcd.txt', data1.T)
    #print('    td.rf0_ncalls ', td.rf0_ncalls)
    #print(' td.matvec_ncalls ', td.matvec_ncalls)

if __name__ == "__main__": unittest.main()
