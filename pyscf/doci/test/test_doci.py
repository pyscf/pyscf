import unittest
import numpy
from pyscf.fci import cistring


class KnownValues(unittest.TestCase):
    def test_fancy_index(self):
        norb = 9
        nelec = 4
        link_index = cistring.gen_linkstr_index(range(norb), nelec)
        na = link_index.shape[0]

        ci0 = numpy.random.random(na)
        t1ref = numpy.zeros((norb,na))
        t2ref = numpy.zeros((norb,norb,na))
        for str0, tab in enumerate(link_index):
            for a, i, str1, sign in tab:
                if a == i:
                    t1ref[i,str1] += ci0[str0]
                else:
                    t2ref[a,i,str1] += ci0[str0]

        link1 = link_index[link_index[:,:,0] == link_index[:,:,1]].reshape(na,-1,4)
        link2 = link_index[link_index[:,:,0] != link_index[:,:,1]].reshape(na,-1,4)
        t1 = numpy.zeros_like(t1ref)
        t2 = numpy.zeros_like(t2ref)
        t1[link1[:,:,1],link1[:,:,2]] = ci0[:,None]
        t2[link2[:,:,0],link2[:,:,1],link2[:,:,2]] = ci0[:,None]
        self.assertAlmostEqual(abs(t1ref - t1).max(), 0, 12)
        self.assertAlmostEqual(abs(t2ref - t2).max(), 0, 12)


if __name__ == "__main__":
    print("Full Tests for doci")
    unittest.main()
