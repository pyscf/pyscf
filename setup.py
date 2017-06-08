import sys
import subprocess
from distutils.core import setup
import subprocess
import os

def compile_C_code():
    """
    Compile automaticaly the C and Fortran code
    when installing the program with python setup.py
    """
    ret = subprocess.call("rm -rf lib/build", shell = True)

    os.mkdir("lib/build")
    os.chdir("lib/build")

    ret = subprocess.call("cmake ..", shell = True)
    if ret != 0:
        raise ValueError("cmake returned error {0}".format(ret))
    ret = subprocess.call("make", shell = True)
    if ret != 0:
        raise ValueError("make returned error {0}".format(ret))

    os.chdir("../..")


if sys.version_info[0] >= 3: # from Cython 0.14
    from distutils.command.build_py import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py


CLASSIFIERS = [
'Development Status :: 1.3 stable',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved',
'Programming Language :: C',
'Programming Language :: Python',
'Programming Language :: Python :: 3',
'Topic :: Software Development',
'Topic :: Scientific/Engineering',
'Operating System :: POSIX',
'Operating System :: Unix',
'Operating System :: MacOS',
]

NAME             = 'pyscf'
MAINTAINER       = 'Qiming Sun'
MAINTAINER_EMAIL = 'osirpt.sun@gmail.com'
DESCRIPTION      = 'PySCF: Python-based Simulations of Chemistry Framework'
#LONG_DESCRIPTION = ''
URL              = 'http://www.pyscf.org'
DOWNLOAD_URL     = 'http://github.com/sunqm/pyscf'
LICENSE          = 'BSD 2-clause "Simplified" License (BSD2)'
AUTHOR           = 'Qiming Sun'
AUTHOR_EMAIL     = 'osirpt.sun@gmail.com'
PLATFORMS        = ['Linux', 'Mac OS-X', 'Unix']
VERSION          = '1.3.0'


compile_C_code()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    package_dir={'pyscf': 'pyscf'},
    package_data={'pyscf': ['gto/basis/*.dat', 'lib/*.so',
                            'pbc/gto/basis/*.dat', 'pbc/gto/pseudo/*.dat']},
    packages=['pyscf',
              'pyscf.gto', 'pyscf.gto.basis',
              'pyscf.ao2mo', 'pyscf.cc', 'pyscf.ci', 'pyscf.data',
              'pyscf.df', 'pyscf.dft', 'pyscf.fci', 'pyscf.grad',
              'pyscf.lib', 'pyscf.mcscf', 'pyscf.mp', 'pyscf.nao',
              'pyscf.nmr', 'pyscf.scf', 'pyscf.symm', 'pyscf.tools',
              'pyscf.pbc.gto.basis', 'pyscf.pbc.gto.pseudo',
              'pyscf.pbc.gto', 'pyscf.pbc.scf', 'pyscf.pbc.df', 'pyscf.pbc.dft',
              'pyscf.pbc.cc', 'pyscf.test', 'pyscf.test.nao.water', 'pyscf.test.nao.na2',
              'pyscf.test.nao.ae_fii'],
    cmdclass={'build_py': build_py},
)

#msg = subprocess.check_output(
#    'mkdir -p lib/build && cd lib/build && cmake .. && make install',
#    shell=True, stderr=subprocess.STDOUT)

