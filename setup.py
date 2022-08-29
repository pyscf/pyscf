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
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

CLASSIFIERS = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: Apache Software License',
'Programming Language :: C',
'Programming Language :: Python',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
'Programming Language :: Python :: 3.9',
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
    'dftd3': ['pyscf-dftd3'],
    'dmrgscf': ['pyscf-dmrgscf'],
    'doci': ['pyscf-doci'],
    'icmpspt': ['pyscf-icmpspt'],
    'properties': ['pyscf-properties'],
    'semiempirical': ['pyscf-semiempirical'],
    'shciscf': ['pyscf-shciscf'],
    'cppe': ['cppe'],
    'pyqmc': ['pyqmc'],
    'mcfun': ['mcfun>=0.2.1'],
}
EXTRAS['all'] = [p for extras in EXTRAS.values() for p in extras]
# extras which should not be installed by "all" components
EXTRAS['cornell_shci'] = ['pyscf-cornell-shci']
EXTRAS['nao'] = ['pyscf-nao']
EXTRAS['fciqmcscf'] = ['pyscf-fciqmc']
EXTRAS['tblis'] = ['pyscf-tblis']

class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == 'pyscf_lib_placeholder'
        self.build_cmake(extension)

    def build_cmake(self, extension):
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
        cmd = ['cmake', '--build', self.build_temp, '-j2']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

    # To remove the infix string like cpython-37m-x86_64-linux-gnu.so
    # Python ABI updates since 3.5
    # https://www.python.org/dev/peps/pep-3149/
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix

# Here to change the order of sub_commands to ['build_py', ..., 'build_ext']
# C extensions by build_ext are installed in source directory.
# build_py then copy all .so files into "build_ext.build_lib" directory.
# We have to ensure build_ext being executed earlier than build_py.
# A temporary workaround is to modifying the order of sub_commands in build class
from distutils.command.build import build
build.sub_commands = ([c for c in build.sub_commands if c[0] == 'build_ext'] +
                      [c for c in build.sub_commands if c[0] != 'build_ext'])

# scipy bugs
# https://github.com/scipy/scipy/issues/12533
_scipy_version = 'scipy!=1.5.0,!=1.5.1'
import sys
if sys.platform == 'darwin':
    # https://github.com/scipy/scipy/issues/15362
    if sys.version_info < (3, 8):
        _scipy_version = 'scipy<=1.1.0'
    else:
        # https://github.com/scipy/scipy/issues/16151
        print('scipy>1.1.0 may crash with segmentation fault when calling scipy.linalg.eigh')

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
    #package_dir={'pyscf': 'pyscf'},  # packages are under directory pyscf
    #include *.so *.dat files. They are now placed in MANIFEST.in
    #package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat']},
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*test*', '*examples*']),
    # The ext_modules placeholder is to ensure build_ext getting initialized
    ext_modules=[Extension('pyscf_lib_placeholder', [])],
    cmdclass={'build_ext': CMakeBuildExt},
    install_requires=['numpy>=1.13,!=1.16,!=1.17',
                      'scipy<=1.1.0' if sys.platform == "darwin" else 'scipy!=1.5.0,!=1.5.1',
                      'h5py>=2.7'],
    extras_require=EXTRAS,
)
