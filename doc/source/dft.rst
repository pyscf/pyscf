dft --- Density functional theory
*********************************

Customizing XC functional
=========================

XC functional of DFT methods can be customized.  The simplest way to customize
XC functional is to assigned a string expression to :attr:`mf.xc`::

    from pyscf import gto, dft
    mol = gto.M(atom='H  0  0  0; F  0.9  0  0', basis='6-31g')
    mf = dft.RKS(mol)
    mf.xc = 'HF*0.2 + .08*LDA + .72*B88, .81*LYP + .19*VWN'
    mf.kernel()
    mf.xc = 'HF*0.5 + .08*LDA + .42*B88, .81*LYP + .19*VWN'
    mf.kernel()
    mf.xc = 'HF*0.8 + .08*LDA + .12*B88, .81*LYP + .19*VWN'
    mf.kernel()
    mf.xc = 'HF'
    mf.kernel()

The XC functional string is parsed against the rules, as described below.

* The given functional description must be a one-line string.
* The functional description is case-insensitive.
* The functional description string has two parts, separated by ``,``.  The
  first part describes the exchange functional, the second is the correlation
  functional.
  - If ``,`` was not appeared in string, the entire string is considered as
    X functional.
  - To neglect X functional (just apply C functional), leave blank in the
    first part, eg ``mf.xc=',vwn'`` for pure VWN functional
* The functional name can be placed in arbitrary order.  Two names needs to
  be separated by operators ``+`` or ``-``.  Blank spaces are ignored.
  NOTE the parser only reads operators ``+ - *``.  ``/`` is not supported.
* A functional name is associated with one factor.  If the factor is not
  given, it is assumed equaling 1.
* String ``'HF'`` stands for exact exchange (HF K matrix).  It is allowed to
  put ``'HF'`` in C (correlation) functional part.
* Be careful with the libxc convention on GGA functional, in which the LDA
  contribution is included.

Another way to customize XC functional is to redefine the :py:meth:`eval_xc`
method of numerical integral class::

    mol = gto.M(atom='H 0 0 0; F 0.9 0 0', basis = '6-31g')
    mf = dft.RKS(mol)
    def eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        # A fictitious XC functional to demonstrate the usage
        rho0, dx, dy, dz = rho
        gamma = (dx**2 + dy**2 + dz**2)
        exc = .01 * rho0**2 + .02 * (gamma+.001)**.5
        vrho = .01 * 2 * rho0
        vgamma = .02 * .5 * (gamma+.001)**(-.5)
        vlapl = None
        vtau = None
        vxc = (vrho, vgamma, vlapl, vtau)
        fxc = None  # 2nd order functional derivative
        kxc = None  # 3rd order functional derivative
        return exc, vxc, fxc, kxc
    dft.libxc.define_xc_(mf._numint, eval_xc, xctype='GGA')
    mf.kernel()

By calling :func:`dft.libxc.define_xc_` function, the customized :func:`eval_xc`
function is patched to the numerical integration class :attr:`mf._numint`
dynamically.

More examples of customizing DFT XC functional can be found in
:file:`examples/dft/24-custom_xc_functional.py` and 
:file:`examples/dft/24-define_xc_functional.py`.


Program reference
=================

.. automodule:: pyscf.dft.rks
   :members:

.. automodule:: pyscf.dft.uks
   :members:

.. automodule:: pyscf.dft.gen_grid
   :members:

.. automodule:: pyscf.dft.numint
   :members:

.. automodule:: pyscf.dft.libxc
   :members:
