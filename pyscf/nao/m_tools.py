"""
    modules containing tools and utility functions
"""

from __future__ import division
import numpy as np

def read_xyz(fname):
  """ Reads xyz files """
  a2s  = np.loadtxt(fname, skiprows=2, usecols=[0], dtype=str)
  a2xyz = np.loadtxt(fname, skiprows=2, usecols=[1,2,3])
  assert len(a2s)==len(a2xyz)
  return a2s,a2xyz

  
def write_xyz(fname, s, ccc):
  """ Writes xyz files """
  assert len(s) == len(ccc)
  f = open(fname, "w")
  print(len(s), file=f)
  print(fname, file=f)
  for sym,xyz in zip(s,ccc): print("%2s %18.10f %18.10f %18.10f"%(sym, xyz[0],xyz[1],xyz[2]), file=f)
  f.close()
  return

def xyz2rtp( x,y,z):
     r=np.sqrt( x**2+y**2+z**2)
     t=np.acos( z/r )
     p=np.atan2( y, x )
     return (r,t,p)
     
def rtp2xyz( r, t, p):
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    return (x, y, z)


def transformData2newCoordinate(oldCoordinates, newCoordinates, data, transform=rtp2xyz):
    """
        transform a 3D array from a coodinate system to another.
        For example, transforming from cartesian to spherical coordinates:

        from __future__ import division
        import numpy as np
        from pyscf.nao.m_tools import transformData2newCoordinate

        dims = (10, 5, 6)
        x = np.linspace(-5, 5, dims[0])
        y = np.linspace(-2, 2, dims[1])
        z = np.linspace(-3, 3, dims[2])

        dn = np.random.randn(dims[0], dims[1], dims[2])

        r = np.arange(0.0, 2.0, 0.1)
        phi = np.arange(0.0, 2*np.pi, 0.01)
        theta = np.arange(0.0, np.pi, 0.01)

        dn_new = transformData2newCoordinate((x, y, z), (r, phi, theta), dn)
    """

    import scipy

    assert len(oldCoordinates) == len(data.shape)
    assert len(newCoordinates) == len(data.shape)

    xyzinterpolator = scipy.interpolate.RegularGridInterpolator( oldCoordinates, data )

    newData = np.zeros((newCoordinates[0].size, newCoordinates[1].size, newCoordinates[2].size), dtype=data.dtype)

    max_dim = max(newCoordinates[0].size, newCoordinates[1].size, newCoordinates[2].size)

    if max_dim == newCoordinates[0].size:
        for i, v1 in enumerate(newCoordinates[1]):
            for j, v2 in enumerate(newCoordinates[2]):
                newData[:, i, j] = xyzinterpolator(transform(newCoordinates[0], v1, v2))
    elif max_dim == newCoordinates[1].size:
        for i, v1 in enumerate(newCoordinates[0]):
            for j, v2 in enumerate(newCoordinates[2]):
                newData[i, :, j] = xyzinterpolator(transform(v1, newCoordinates[1], v2))
    elif max_dim == newCoordinates[2].size:
        for i, v1 in enumerate(newCoordinates[0]):
            for j, v2 in enumerate(newCoordinates[1]):
                newData[i, j, :] = xyzinterpolator(transform(v1, v2, newCoordinates[2]))

    else:
        raise ValueError("Wrong max dim")

    return newData

def find_nearrest_index(arr, val):
    """
        return the index of an array which is the
        closest from the entered value

        Input Parameters:
        -----------------
            arr (1D numpy arr)
            val: value to find in the array

        Output Parameters:
        ------------------
            idx: index of arr corresponding to the closest
                from value
    """
    idx = (np.abs(arr-val)).argmin()
    return idx

def is_power2(n):
    """
        Check if n is a power of 2
    """
    assert isinstance(n, int)
    return ((n & (n-1)) == 0) and n != 0
