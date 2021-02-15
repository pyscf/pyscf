"""
    Module to that correct the Discret Fourier Transform 
    in order to get the analytical value.
"""
from __future__ import division
import numpy as np
import scipy

try:
    import numba as nb
    use_numba = True
except:
    use_numba = False


def get_fft_freq(t):
    """
    return the frequency range of the fft variable, it
    is a well define array,
        dw = 2*pi/(N*dt)
    with N the size of the array
    """

    dt = t[1]-t[0]
    dw = 2*np.pi/(t.size*dt)

    w_min = - 2*np.pi*(t.size-1)/(dt*t.size) / 2
    arange = np.arange(t.shape[0])
    w = arange*dw +w_min

    return w

#########################
#                       #
#       1D FFT          #
#                       #
#########################

def FT1(t, f, axis=0):
    """
    Calculate the FFT of a ND dimensionnal array over the axes axes
    corresponding to the FT by using numpy fft
    The Fourier transform is done on the first dimension.

    Input:
        t, ND numpy array containing the time variable
        f, ND numpy array containing the data to Fourier transform (f(t, x, ....))
    
    output:
        F, ND numpy of the FT along axis
    """

    w = get_fft_freq(t)

    tmin = t[0]
    wmin = w[0]
    dt = t[1] - t[0]
    dw = w[1] - w[0]
    N = t.size

    param = dw*dt*N/(2*np.pi)
    if abs(param-1)>1e-10 :
        sys.stdout.write('cannot use fft, param != 1\n')
        sys.stdout.write('param = %s\n' %(param, ))
        sys.stdout.write('dw = %s\n' % (dw, ))
        sys.stdout.write('dt = %s\n' % (dt, ))
        sys.stdout.write('N = %s\n' % (N, ))
        sys.exit()

    f_new = dt*f*np.exp(-1.0j*wmin*(t-tmin))
    F = np.fft.fft(f_new, axis=axis)
    F = F*np.exp(-1.0j*(w*tmin))/np.sqrt(2*np.pi)

    return F

def iFT1(w, F, axis = 0):
    """
    Calculate the FFT of a 1D dimensionnal array
    corresponding to the FT by using numpy fft
    input:
    ------
      w, 1D numpy array containing the frequency variable from fft_freq
      F, 1D numpy array containing the data to inverse Fourier transform
    output:
    -------
      F, 1D numpy array containing the iFT
    """

    t = get_fft_freq(w)

    tmin = t[0]
    wmin = w[0]
    dt = t[1] - t[0]
    dw = w[1] - w[0]
    N = t.size

    param = dw*dt*N/2/np.pi
    if abs(param-1)>1e-10 :
        print('cannot use fft, param != 1')
        print('param = ', param)
        print('dw = ', dw)
        print('dt = ', dt)
        print('N = ', N)
        sys.exit()


    F_new = np.zeros(F.shape, dtype = np.complex128)

    if axis != 0:
      raise ValueError("Not yet implemented, for the moment you must use axis =0")

    F_new = dw*F*np.exp(1.0j*tmin*(w-wmin))
    f = N*np.fft.ifft(F_new, axis = axis)
    f *= np.exp(1.0j*( wmin*t))/np.sqrt(2*np.pi)

    return f

#########################
#                       #
#       2D FFT          #
#                       #
#########################

def calc_prefactor_2D(x, y, f, wmin, tmin, dt, sign=1.0):

    if abs(sign) != 1.0:
        raise ValueError("sign must be 1.0 or -1.0")
    fmod = np.zeros(f.shape, dtype=np.complex128)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            fmod[i, j] = f[i, j]*dt[0]*dt[1]*np.exp(sign*1.0j*(wmin[0]*(x[i]-tmin[0]) +
                                    wmin[1]*(y[j]-tmin[1])))

    return fmod

def calc_postfactor_2D(kx, ky, tmin, F, sign=1.0):

    if abs(sign) != 1.0:
        raise ValueError("sign must be 1.0 or -1.0")

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i, j] *= np.exp(sign*1.0j*(kx[i]*tmin[0] + ky[j]*tmin[1]))


def FT2(x, y, f):
    """
        2D Fourier Transform

        Input Parameters:
            x, y (1D array): axis range
            f (2D numpy array of shape (x.size, y.size)): array to FT

        Output Parameters:
            F (2D numpy array of shape (x.size, y.size)): FT

        Example:
            import numpy as np
            import pyscf.nao.m_fft as fft
            import matplotlib.pyplot as plt

            def gauss2D(x, y, a, b):
                return np.exp(-(a*x**2 + b*y**2))

            x = np.linspace(-5.0, 5.0, 100)
            y = np.linspace(-7.0, 7.0, 200)

            a = 1.0
            b = 2.0

            xv, yv = np.meshgrid(y, x)
            f = gauss2D(xv, yv, a, b)

            f_FT = fft.FT2(x, y, f)

            kx = fft.get_fft_freq(x)
            ky = fft.get_fft_freq(y)
            if_FT = fft.iFT2(kx, ky, f_FT)

            plt.imshow(f)
            plt.colorbar()
            plt.show()

            plt.imshow(if_FT.real)
            plt.colorbar()
            plt.show()
    """

    assert f.shape == (x.size, y.size)

    kx = get_fft_freq(x)
    ky = get_fft_freq(y)

    tmin = np.array([x[0], y[0]])
    wmin = np.array([kx[0], ky[0]])
    dw = np.array([kx[1]-kx[0], ky[1]-ky[0]])
    dt = np.array([x[1]-x[0], y[1]-y[0]])
    N = np.array([x.size, y.size])

    param = dw*dt*N/(2*np.pi)
    if any(abs(param-1) > 1e-10):
        sys.stdout.write('cannot use fft, param != 1\n')
        sys.stdout.write('param = %s\n' %(param, ))
        sys.stdout.write('dw = %s\n' % (dw, ))
        sys.stdout.write('dt = %s\n' % (dt, ))
        sys.stdout.write('N = %s\n' % (N, ))
        sys.exit()

    if use_numba:
        calc_pre = nb.jit(nopython=True)(calc_prefactor_2D)
        calc_post = nb.jit(nopython=True)(calc_postfactor_2D)
    else:
        calc_pre = calc_prefactor_2D
        calc_post = calc_postfactor_2D

    f_new = calc_pre(x, y, f, wmin, tmin, dt, sign=-1.0)
    F = np.fft.fftn(f_new)
    calc_post(kx, ky, tmin, F, sign=-1.0)

    return F/(2*np.pi)


def iFT2(kx, ky, F):
    """
        2D Inverse Fourier Transform

        Input Parameters:
            kx, ky (1D array): axis range
            F (2D numpy array of shape (x.size, y.size)): array to IFT

        Output Parameters:
            f (2D numpy array of shape (x.size, y.size)): IFT
            
    """

    assert F.shape == (kx.size, ky.size)

    x = get_fft_freq(kx)
    y = get_fft_freq(ky)

    tmin = np.array([x[0], y[0]])
    wmin = np.array([kx[0], ky[0]])
    dw = np.array([kx[1]-kx[0], ky[1]-ky[0]])
    dt = np.array([x[1]-x[0], y[1]-y[0]])
    N = np.array([kx.size, ky.size])

    param = dw*dt*N/(2*np.pi)
    if any(abs(param-1) > 1e-10):
        sys.stdout.write('cannot use fft, param != 1\n')
        sys.stdout.write('param = %s\n' %(param, ))
        sys.stdout.write('dw = %s\n' % (dw, ))
        sys.stdout.write('dt = %s\n' % (dt, ))
        sys.stdout.write('N = %s\n' % (N, ))
        sys.exit()

    if use_numba:
        calc_pre = nb.jit(nopython=True)(calc_prefactor_2D)
        calc_post = nb.jit(nopython=True)(calc_postfactor_2D)
    else:
        calc_pre = calc_prefactor_2D
        calc_post = calc_postfactor_2D

    F_new = calc_pre(kx, ky, F, tmin, wmin, dw)
    f = N[0]*N[1]*np.fft.ifftn(F_new)
    calc_post(x, y, wmin, f)

    return f/(2*np.pi)

#########################
#                       #
#       3D FFT          #
#                       #
#########################

def calc_prefactor_3D(x, y, z, f, wmin, tmin, dt, sign=1.0):

    if abs(sign) != 1.0:
        raise ValueError("sign must be 1.0 or -1.0")

    fmod = np.zeros(f.shape, dtype=np.complex128)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
                fmod[i, j, k] = f[i, j, k]*dt[0]*dt[1]*dt[2]*\
                                np.exp(sign*1.0j*(wmin[0]*(x[i]-tmin[0]) +\
                                    wmin[1]*(y[j]-tmin[1]) + wmin[2]*(z[k]-tmin[2])))

    return fmod

def calc_postfactor_3D(kx, ky, kz, tmin, F, sign=1.0):

    if abs(sign) != 1.0:
        raise ValueError("sign must be 1.0 or -1.0")

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            for k in range(F.shape[2]):
                F[i, j, k] *= np.exp(sign*1.0j*(kx[i]*tmin[0] + ky[j]*tmin[1] + kz[k]*tmin[2]))


def FT3(x, y, z, f):
    """
        3D Fourier Transform

        Input Parameters:
            x, y, z (1D array): axis range
            f (3D numpy array of shape (x.size, y.size, z.size)): array to FT

        Output Parameters:
            F (3D numpy array of shape (x.size, y.size, z.size)): FT
    """

    assert f.shape == (x.size, y.size, z.size)

    kx = get_fft_freq(x)
    ky = get_fft_freq(y)
    kz = get_fft_freq(z)

    tmin = np.array([x[0], y[0], z[0]])
    wmin = np.array([kx[0], ky[0], kz[0]])
    dw = np.array([kx[1]-kx[0], ky[1]-ky[0], kz[1]-kz[0]])
    dt = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    N = np.array([x.size, y.size, z.size])

    param = dw*dt*N/(2*np.pi)
    if any(abs(param-1) > 1e-10):
        sys.stdout.write('cannot use fft, param != 1\n')
        sys.stdout.write('param = %s\n' %(param, ))
        sys.stdout.write('dw = %s\n' % (dw, ))
        sys.stdout.write('dt = %s\n' % (dt, ))
        sys.stdout.write('N = %s\n' % (N, ))
        sys.exit()

    if use_numba:
        calc_pre = nb.jit(nopython=True)(calc_prefactor_3D)
        calc_post = nb.jit(nopython=True)(calc_postfactor_3D)
    else:
        calc_pre = calc_prefactor_3D
        calc_post = calc_postfactor_3D

    f_new = calc_pre(x, y, z, f, wmin, tmin, dt, sign=-1.0)
    F = np.fft.fftn(f_new)
    calc_post(kx, ky, kz, tmin, F, sign=-1.0)

    return F/(2*np.pi)**(3/2)


def iFT3(kx, ky, kz, F):
    """
        3D Inverse Fourier Transform

        Input Parameters:
            kx, ky, kz (1D array): axis range
            F (3D numpy array of shape (kx.size, ky.size, kz.size)): array to IFT

        Output Parameters:
            f (3D numpy array of shape (kx.size, ky.size, kz.size)): IFT
            
    """

    assert F.shape == (kx.size, ky.size, kz.size)

    x = get_fft_freq(kx)
    y = get_fft_freq(ky)
    z = get_fft_freq(kz)

    tmin = np.array([x[0], y[0], z[0]])
    wmin = np.array([kx[0], ky[0], kz[0]])
    dw = np.array([kx[1]-kx[0], ky[1]-ky[0], kz[1]-kz[0]])
    dt = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    N = np.array([kx.size, ky.size, kz.size])

    param = dw*dt*N/(2*np.pi)
    if any(abs(param-1) > 1e-10):
        sys.stdout.write('cannot use fft, param != 1\n')
        sys.stdout.write('param = %s\n' %(param, ))
        sys.stdout.write('dw = %s\n' % (dw, ))
        sys.stdout.write('dt = %s\n' % (dt, ))
        sys.stdout.write('N = %s\n' % (N, ))
        sys.exit()

    if use_numba:
        calc_pre = nb.jit(nopython=True)(calc_prefactor_3D)
        calc_post = nb.jit(nopython=True)(calc_postfactor_3D)
    else:
        calc_pre = calc_prefactor_3D
        calc_post = calc_postfactor_3D

    F_new = calc_pre(kx, ky, kz, F, tmin, wmin, dw)
    f = N[0]*N[1]*N[2]*np.fft.ifftn(F_new)
    calc_post(x, y, z, wmin, f)

    return f/(2*np.pi)**(3/2)

#################################
#                               #
#       FFT Covolution          #
#                               #
#################################

def FTconvolve(f, g, *args):
    """
        Perform a corrected version of the fft convolution from scipy

        Input arguments:
            f (1, 2 or 3D array): first term of the product
            g (1, 2 or 3D array): second term of the product (same dim than f)
            args (list of 1D array): contains the real space array variable,
                it must contains at least 1 argument and in maximum 3.
                the sumber of arguments must match the dimemension of f and g.

        Output arguments:
            conv (1, 2, or 3D array): convolution product
    """

    if len(args) == 0:
        raise ValueError("you must provide at least one array for the real space variable")
    elif len(args) == 1:
        assert f.size == args[0].size
        assert g.size == args[0].size

        k = get_fft_freq(args[0])
        f_FT = FT1(args[0], f)
        g_FT = FT1(args[0], g)

        return iFT1(k, f_FT*g_FT)

    elif len(args) == 2:
        assert f.shape == (args[0].size, args[1].size)
        assert g.shape == (args[0].size, args[1].size)

        kx = get_fft_freq(args[0])
        ky = get_fft_freq(args[1])

        f_FT = FT2(args[0], args[1], f)
        g_FT = FT2(args[0], args[1], g)

        return iFT2(kx, ky, f_FT*g_FT)

    elif len(args) == 3:
        assert f.shape == (args[0].size, args[1].size, args[2].size)
        assert g.shape == (args[0].size, args[1].size, args[2].size)

        kx = get_fft_freq(args[0])
        ky = get_fft_freq(args[1])
        kz = get_fft_freq(args[2])

        f_FT = FT3(args[0], args[1], args[2], f)
        g_FT = FT3(args[0], args[1], args[2], g)

        return iFT3(kx, ky, kz, f_FT*g_FT)

    else:
        raise ValueError("Only up to 3 real space array are accepted")




