#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py

CLASSIFIERS = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: Apache Software License',
'Programming Language :: C',
'Programming Language :: Python',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
'Programming Language :: Python :: 3.9',
'Programming Language :: Python :: 3.10',
'Programming Language :: Python :: 3.11',
'Programming Language :: Python :: 3.12',
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
DOWNLOAD_URL     = 'http://github.com/pyscf/pyscf'
LICENSE          = 'Apache License 2.0'
AUTHOR           = 'Qiming Sun'
AUTHOR_EMAIL     = 'osirpt.sun@gmail.com'
PLATFORMS        = ['Linux', 'Mac OS-X', 'Unix']
def get_version():
    topdir = os.path.abspath(os.path.join(__file__, '..'))
    with open(os.path.join(topdir, 'pyscf', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise ValueError("Version string not found")
VERSION = get_version()

EXTRAS = {
    'geomopt': ['pyberny>=0.6.2', 'geometric>=0.9.7.2', 'pyscf-qsdopt'],
    #'dmrgscf': ['pyscf-dmrgscf'],
    'doci': ['pyscf-doci'],
    'icmpspt': ['pyscf-icmpspt'],
    'properties': ['pyscf-properties'],
    'semiempirical': ['pyscf-semiempirical'],
    'shciscf': ['pyscf-shciscf'],
    'cppe': ['cppe'],
    'pyqmc': ['pyqmc'],
    'mcfun': ['mcfun>=0.2.1'],
    'bse': ['basis-set-exchange'],
}
EXTRAS['all'] = [p for extras in EXTRAS.values() for p in extras]
# extras which should not be installed by "all" components
EXTRAS['cornell_shci'] = ['pyscf-cornell-shci']
EXTRAS['nao'] = ['pyscf-nao']
EXTRAS['fciqmcscf'] = ['pyscf-fciqmc']
EXTRAS['tblis'] = ['pyscf-tblis']

def get_platform():
    from distutils.util import get_platform
    platform = get_platform()
    if sys.platform == 'darwin':
        arch = os.getenv('CMAKE_OSX_ARCHITECTURES')
        if arch:
            osname = platform.rsplit('-', 1)[0]
            if ';' in arch:
                platform = f'{osname}-universal2'
            else:
                platform = f'{osname}-{arch}'
        elif os.getenv('_PYTHON_HOST_PLATFORM'):
            # the cibuildwheel environment
            platform = os.getenv('_PYTHON_HOST_PLATFORM')
            if platform.endswith('arm64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64')
            elif platform.endswith('x86_64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'x86_64')
            else:
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64;x86_64')
    return platform

class CMakeBuildPy(build_py):
    def run(self):
        self.plat_name = get_platform()
        self.build_base = 'build'
        self.build_lib = os.path.join(self.build_base, 'lib')
        self.build_temp = os.path.join(self.build_base, f'temp.{self.plat_name}')

        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'pyscf', 'lib'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        # Do not use high level parallel compilation. OOM may be triggered
        # when compiling certain functionals in libxc.
        cmd = ['cmake', '--build', self.build_temp, '-j', '2']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)
        super().run()

# build_py will produce plat_name = 'any'. Patch the bdist_wheel to change the
# platform tag because the C extensions are platform dependent.
from wheel.bdist_wheel import bdist_wheel
initialize_options = bdist_wheel.initialize_options
def initialize_with_default_plat_name(self):
    initialize_options(self)
    self.plat_name = get_platform()
bdist_wheel.initialize_options = initialize_with_default_plat_name

# scipy bugs
# https://github.com/scipy/scipy/issues/12533
_scipy_version = 'scipy!=1.5.0,!=1.5.1'
if sys.platform == 'darwin':
    if sys.version_info < (3, 8):
        _scipy_version = 'scipy<=1.1.0'
    else:
        print('scipy>1.1.0 may crash when calling scipy.linalg.eigh. '
              '(Issues https://github.com/scipy/scipy/issues/15362 '
              'https://github.com/scipy/scipy/issues/16151)')

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    #package_dir={'pyscf': 'pyscf'},  # packages are under directory pyscf
    #include *.so *.dat files. They are now placed in MANIFEST.in
    #package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat']},
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*test*', '*examples*']),
    cmdclass={'build_py': CMakeBuildPy},
    install_requires=['numpy>=1.13,!=1.16,!=1.17',
                      _scipy_version,
                      'h5py>=2.7',
                      'setuptools'],
    extras_require=EXTRAS,
)
