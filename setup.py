import sys
import subprocess
from setuptools import setup, find_packages, Extension

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
VERSION          = '1.4.0'

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
    package_dir={'pyscf': 'pyscf'},  # packages are under directory pyscf
    package_data={'': ['*.so', '*.dat']}, # any package contains *.so *.dat files
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*dmrgscf*', '*fciqmcscf*', '*icmpspt*',
                                    '*shciscf*', '*xianci*', '*tdscf*',
                                    '*future*', '*test*', '*examples*',
                                    '*setup.py']),
    cmdclass={'build_py': build_py},
)

#msg = subprocess.check_output(
#    'mkdir -p pyscf/lib/build && cd pyscf/lib/build && cmake .. && make install',
#    shell=True, stderr=subprocess.STDOUT)

