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
import sysconfig
from setuptools import setup, find_packages, Extension

#if sys.version_info[0] >= 3: # from Cython 0.14
#    from setuptools.command.build_py import build_py_2to3 as build_py
#else:
#    from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

try:
    import numpy
except ImportError as e:
    print('**************************************************')
    print('* numpy was not installed in your system.  Please run')
    print('*     pip install numpy')
    print('* before installing pyscf.')
    print('**************************************************')
    raise e

topdir = os.path.abspath(os.path.join(__file__, '..'))

CLASSIFIERS = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: Apache Software License',
'Programming Language :: C',
'Programming Language :: Python',
'Programming Language :: Python :: 2.7',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
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
    with open(os.path.join(topdir, 'pyscf', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                return eval(line.strip().split(' = ')[1])
    raise ValueError("Version string not found")
VERSION = get_version()


if (sys.platform.startswith('linux') or
    sys.platform.startswith('cygwin') or
    sys.platform.startswith('gnukfreebsd')):
    ostype = 'linux'
    so_ext = '.so'
    LD_LIBRARY_PATH = 'LD_LIBRARY_PATH'
elif sys.platform.startswith('darwin'):
    ostype = 'mac'
    so_ext = '.dylib'
    LD_LIBRARY_PATH = 'DYLD_LIBRARY_PATH'
    from distutils.sysconfig import get_config_vars
    conf_vars = get_config_vars()
# setuptools/pip install by default generate "bundled" library.  Bundled
# library cannot be linked at compile time
# https://stackoverflow.com/questions/24519863/what-are-the-g-flags-to-build-a-true-so-mh-bundle-shared-library-on-mac-osx
# configs LDSHARED and CCSHARED and SO are hard coded in lib/pythonX.X/_sysconfigdata.py
# In some Python version, steuptools may correct these configs for OS X on the
# fly by _customize_compiler_for_shlib function or setup_shlib_compiler function
# in lib/pythonX.X/site-packages/setuptools/command/build_ext.py.
# The hacks below ensures that the OS X compiler does not generate bundle
# libraries.  Relevant code:
#    lib/pythonX.X/_sysconfigdata.py
#    lib/pythonX.X/distutils/command/build_ext.py
#    lib/pythonX.X/distutils/sysconfig.py,  get_config_vars()
#    lib/pythonX.X/distutils/ccompiler.py,  link_shared_object()
#    lib/pythonX.X/distutils/unixcompiler.py,  link()
    conf_vars['LDSHARED'] = conf_vars['LDSHARED'].replace('-bundle', '-dynamiclib')
    conf_vars['CCSHARED'] = " -dynamiclib"
    # On Mac OS, sysconfig.get_config_vars()["EXT_SUFFIX"] was set to
    # '.cpython-3*m-darwin.so'. numpy.ctypeslib module uses this parameter to
    # determine the extension of an external library. Python distutlis module
    # only generates the library with the extension '.so'.  It causes import
    # error at the runtime.  numpy.ctypeslib treats '.dylib' as the native
    # python extension.  Set 'EXT_SUFFIX' to '.dylib' can make distutlis
    # generate the libraries with extension '.dylib'.  It can be loaded by
    # numpy.ctypeslib
    if sys.version_info[0] >= 3:  # python3
        conf_vars['EXT_SUFFIX'] = '.dylib'
    else:
        conf_vars['SO'] = '.dylib'
elif sys.platform.startswith('win'):
    ostype = 'windows'
    so_ext = '.dll'
elif sys.platform.startswith('aix') or sys.platform.startswith('os400'):
    ostype = 'aix'
    so_ext = '.so'
    LD_LIBRARY_PATH = 'LIBPATH'
    if(os.environ.get('PYSCF_INC_DIR') is None):
        os.environ['PYSCF_INC_DIR'] = '/QOpenSys/pkgs:/QOpenSys/usr:/usr:/usr/local'
else:
    raise OSError('Unknown platform')
    ostype = None

#if 'CC' in os.environ:
#    compiler = os.environ['CC'].split()[0]
#else:
#    compiler = sysconfig.get_config_var("CC").split()[0]
#if 'gcc' in compiler or 'g++' in compiler:  # GNU compiler
#    so_ext = '.so'

#
# default include and library path
#
def check_version(version_to_test, version_min):
    return version_to_test.split('.') >= version_min.split('.')

# version : the lowest version
def search_lib_path(libname, extra_paths=None, version=None):
    paths = os.environ.get(LD_LIBRARY_PATH, '').split(os.pathsep)
    if 'PYSCF_INC_DIR' in os.environ:
        PYSCF_INC_DIR = os.environ['PYSCF_INC_DIR'].split(os.pathsep)
        for p in PYSCF_INC_DIR:
            paths = [p, os.path.join(p, 'lib'), os.path.join(p, '..', 'lib')] + paths
    if extra_paths is not None:
        paths += extra_paths

    len_libname = len(libname)
    for path in paths:
        full_libname = os.path.join(path, libname)
        if os.path.isfile(full_libname):
            if version is None or ostype == 'mac':
                return os.path.abspath(path)
            #elif ostype == 'mac':
            #    for f in os.listdir(path):
            #        f_name = f[:len_libname+1-len(so_ext)]
            #        f_version = f[len_libname+1-len(so_ext):-len(so_ext)]
            #        if (f_name == libname[:len_libname+1-len(so_ext)] and f_version and
            #            check_version(f_version, version)):
            #            return os.path.abspath(path)
            else:
                for f in os.listdir(path):
                    f_name = f[:len_libname]
                    f_version = f[len_libname+1:]
                    if (f_name == libname and f_version and
                        check_version(f_version, version)):
                        return os.path.abspath(path)

def search_inc_path(incname, extra_paths=None):
    paths = os.environ.get(LD_LIBRARY_PATH, '').split(os.pathsep)
    if 'PYSCF_INC_DIR' in os.environ:
        PYSCF_INC_DIR = os.environ['PYSCF_INC_DIR'].split(os.pathsep)
        for p in PYSCF_INC_DIR:
            paths = [p, os.path.join(p, 'include'), os.path.join(p, '..', 'include')] + paths
    if extra_paths is not None:
        paths += extra_paths
    for path in paths:
        full_incname = os.path.join(path, incname)
        if os.path.exists(full_incname):
            return os.path.abspath(path)

if 'LDFLAGS' in os.environ:
    blas_found = any(x in os.environ['LDFLAGS']
                     for x in ('blas', 'atlas', 'openblas', 'mkl', 'Accelerate'))
else:
    blas_found = False

blas_include = []
blas_lib_dir = []
blas_libraries = []
blas_extra_link_flags = []
blas_extra_compile_flags = []
if not blas_found:
    np_blas = numpy.__config__.get_info('blas_opt')
    blas_include = np_blas.get('include_dirs', [])
    blas_lib_dir = np_blas.get('library_dirs', [])
    blas_libraries = np_blas.get('libraries', [])
    blas_path_guess = [search_lib_path('lib'+x+so_ext, blas_lib_dir)
                       for x in blas_libraries]
    blas_extra_link_flags = np_blas.get('extra_link_args', [])
    blas_extra_compile_flags = np_blas.get('extra_compile_args', [])
    if ostype == 'mac':
        if blas_extra_link_flags:
            blas_found = True
    else:
        if None not in blas_path_guess:
            blas_found = True
            blas_lib_dir = list(set(blas_path_guess))

if not blas_found:  # for MKL
    mkl_path_guess = search_lib_path('libmkl_rt'+so_ext, blas_lib_dir)
    if mkl_path_guess is not None:
        blas_libraries = ['mkl_rt']
        blas_lib_dir = [mkl_path_guess]
        blas_found = True
        print("Using MKL library in %s" % mkl_path_guess)

if not blas_found:
    possible_blas = ('blas', 'atlas', 'openblas')
    for x in possible_blas:
        blas_path_guess = search_lib_path('libblas'+so_ext, blas_lib_dir)
        if blas_path_guess is not None:
            blas_libraries = [x]
            blas_lib_dir = [blas_path_guess]
            blas_found = True
            print("Using BLAS library %s in %s" % (x, blas_path_guess))
            break

if not blas_found:
    print("****************************************************************")
    print("*** WARNING: BLAS library not found.")
    print("* You can include the BLAS library in the global environment LDFLAGS, eg")
    print("*   export LDFLAGS='-L/path/to/blas/lib -lblas'")
    print("* or specify the BLAS library path in  PYSCF_INC_DIR")
    print("*   export PYSCF_INC_DIR=/path/to/blas/lib:/path/to/other/lib")
    print("****************************************************************")
    raise RuntimeError

distutils_lib_dir = 'lib.{platform}-{version[0]}.{version[1]}'.format(
    platform=sysconfig.get_platform(),
    version=sys.version_info)

pyscf_lib_dir = os.path.join(topdir, 'pyscf', 'lib')
build_lib_dir = os.path.join('build', distutils_lib_dir, 'pyscf', 'lib')
default_lib_dir = [build_lib_dir] + blas_lib_dir
default_include = ['.', 'build', pyscf_lib_dir] + blas_include

if not os.path.exists(os.path.join(topdir, 'build')):
    os.mkdir(os.path.join(topdir, 'build'))
with open(os.path.join(topdir, 'build', 'config.h'), 'w') as f:
    f.write('''
#if defined _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
#define WITH_RANGE_COULOMB
#define FINT int
''')

def make_ext(pkg_name, relpath, srcs, libraries=[], library_dirs=default_lib_dir,
             include_dirs=default_include, extra_compile_flags=[],
             extra_link_flags=[], **kwargs):
    if '/' in relpath:
        relpath = os.path.join(*relpath.split('/'))
    if (os.path.isfile(os.path.join(pyscf_lib_dir, 'build', 'CMakeCache.txt')) and
        os.path.isfile(os.path.join(pyscf_lib_dir, *pkg_name.split('.')) + so_ext)):
        return None
    else:
        if sys.platform.startswith('darwin'):
            soname = pkg_name.split('.')[-1]
            extra_link_flags = extra_link_flags + ['-install_name', '@loader_path/'+soname+so_ext]
            runtime_library_dirs = []
        elif sys.platform.startswith('aix') or sys.platform.startswith('os400'):
            extra_compile_flags = extra_compile_flags + ['-fopenmp']
            extra_link_flags = extra_link_flags + ['-lblas', '-lgomp', '-Wl,-brtl']
            runtime_library_dirs = ['$ORIGIN', '.']
        else:
            extra_compile_flags = extra_compile_flags + ['-fopenmp']
            extra_link_flags = extra_link_flags + ['-fopenmp']
            runtime_library_dirs = ['$ORIGIN', '.']
        srcs = make_src(relpath, srcs)
        return Extension(pkg_name, srcs,
                         libraries = libraries,
                         library_dirs = library_dirs,
                         include_dirs = include_dirs + [os.path.join(pyscf_lib_dir,relpath)],
                         extra_compile_args = extra_compile_flags,
                         extra_link_args = extra_link_flags,
# Be careful with the ld flag "-Wl,-R$ORIGIN" in the shell.
# When numpy.distutils is imported, the default CCompiler of distutils will be
# overwritten. Compilation is executed in shell and $ORIGIN will be converted to ''
                         runtime_library_dirs = runtime_library_dirs,
                         **kwargs)

def make_src(relpath, srcs):
    srcpath = os.path.join(pyscf_lib_dir, relpath)
    abs_srcs = []
    for src in srcs.split():
        if '/' in src:
            abs_srcs.append(os.path.relpath(os.path.join(srcpath, *src.split('/'))))
        else:
            abs_srcs.append(os.path.relpath(os.path.join(srcpath, src)))
    return abs_srcs

#
# Check libcint
#
extensions = []
if 1:
    libcint_lib_path = search_lib_path('libcint'+so_ext, [pyscf_lib_dir,
                                                          os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                          os.path.join(pyscf_lib_dir, 'deps', 'lib64')],
                                       version='3.0')
    libcint_inc_path = search_inc_path('cint.h', [pyscf_lib_dir,
                                                  os.path.join(pyscf_lib_dir, 'deps', 'include')])
    if libcint_lib_path and libcint_inc_path:
        print("****************************************************************")
        print("* libcint found in %s." % libcint_lib_path)
        print("****************************************************************")
        default_lib_dir += [libcint_lib_path]
        default_include += [libcint_inc_path]
    else:
        srcs = '''g3c2e.c breit.c fblas.c rys_roots.c g2e_coulerf.c misc.c
cint3c1e_a.c cint2c2e.c cint1e.c cint1e_a.c g2e.c cint_bas.c g1e.c
cart2sph.c cint2e_coulerf.c optimizer.c g2c2e.c c2f.c cint3c1e.c
g3c1e.c cint3c2e.c g4c1e.c cint2e.c autocode/intor4.c
autocode/int3c1e.c autocode/int3c2e.c autocode/dkb.c autocode/breit1.c
autocode/gaunt1.c autocode/grad1.c autocode/intor2.c autocode/intor3.c
autocode/hess.c autocode/intor1.c autocode/grad2.c'''
        if os.path.exists(os.path.join(pyscf_lib_dir, 'libcint')):
            extensions.append(
                make_ext('pyscf.lib.libcint', 'libcint/src', srcs, blas_libraries,
                         extra_compile_flags=blas_extra_compile_flags,
                         extra_link_flags=blas_extra_link_flags)
            )
            default_include.append(os.path.join(pyscf_lib_dir, 'libcint','src'))
        else:
            print("****************************************************************")
            print("*** WARNING: libcint library not found.")
            print("* You can download libcint library from http://github.com/sunqm/libcint")
            print("* May need to set PYSCF_INC_DIR if libcint library was not installed in the")
            print("* system standard install path (/usr, /usr/local, etc). Eg")
            print("*   export PYSCF_INC_DIR=/path/to/libcint:/path/to/other/lib")
            print("****************************************************************")
            raise RuntimeError

extensions += [
    make_ext('pyscf.lib.libnp_helper', 'np_helper',
             'condense.c npdot.c omp_reduce.c pack_tril.c transpose.c',
             blas_libraries,
             extra_compile_flags=blas_extra_compile_flags,
             extra_link_flags=blas_extra_link_flags),
    make_ext('pyscf.lib.libcgto', 'gto',
             '''fill_int2c.c fill_nr_3c.c fill_r_3c.c fill_int2e.c ft_ao.c
             grid_ao_drv.c fastexp.c deriv1.c deriv2.c nr_ecp.c nr_ecp_deriv.c
             autocode/auto_eval1.c ft_ao_deriv.c fill_r_4c.c''',
             ['cint', 'np_helper']),
    make_ext('pyscf.lib.libcvhf', 'vhf',
             '''fill_nr_s8.c nr_incore.c nr_direct.c optimizer.c nr_direct_dot.c
             time_rev.c r_direct_o1.c rkb_screen.c r_direct_dot.c
             rah_direct_dot.c rha_direct_dot.c hessian_screen.c''',
             ['cgto', 'np_helper', 'cint']),
    make_ext('pyscf.lib.libao2mo', 'ao2mo',
             'restore_eri.c nr_ao2mo.c nr_incore.c r_ao2mo.c',
             ['cvhf', 'cint', 'np_helper']),
    make_ext('pyscf.lib.libcc', 'cc',
             'ccsd_pack.c ccsd_grad.c ccsd_t.c uccsd_t.c',
             ['cvhf', 'ao2mo', 'np_helper']),
    make_ext('pyscf.lib.libfci', 'mcscf',
             '''fci_contract.c fci_contract_nosym.c fci_rdm.c fci_string.c
             fci_4pdm.c select_ci.c''',
             ['np_helper']),
    make_ext('pyscf.lib.libmcscf', 'mcscf', 'nevpt_contract.c',
             ['fci', 'cvhf', 'ao2mo']),
    make_ext('pyscf.lib.libri', 'ri', 'r_df_incore.c',
             ['cint', 'ao2mo', 'np_helper']),
    make_ext('pyscf.lib.libhci', 'hci', 'hci.c', ['np_helper']),
    make_ext('pyscf.lib.libpbc', 'pbc', 'ft_ao.c optimizer.c fill_ints.c grid_ao.c',
             ['cgto', 'cint']),
    make_ext('pyscf.lib.libmbd', os.path.join('extras', 'mbd'), 'dipole.c', []),
    make_ext('pyscf.lib.libdft', 'dft',
             '''CxLebedevGrid.c grid_basis.c nr_numint.c r_numint.c
             numint_uniform_grid.c''',
             ['cvhf', 'cgto', 'cint', 'np_helper']),
]

#
# Check libxc
#
DFT_AVAILABLE = 0
if 1:
    libxc_lib_path = search_lib_path('libxc'+so_ext, [pyscf_lib_dir,
                                                      os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                      os.path.join(pyscf_lib_dir, 'deps', 'lib64')],
                                     version='5')
    libxc_inc_path = search_inc_path('xc.h', [pyscf_lib_dir,
                                              os.path.join(pyscf_lib_dir, 'deps', 'include')])
    if libxc_lib_path and libxc_inc_path:
        print("****************************************************************")
        print("* libxc found in %s." % libxc_lib_path)
        print("****************************************************************")
        default_lib_dir += [libxc_lib_path]
        default_include += [libxc_inc_path]
        extensions += [
            make_ext('pyscf.lib.libxc_itrf', 'dft', 'libxc_itrf.c', ['xc']),
        ]
        DFT_AVAILABLE = 1
    else:
        print("****************************************************************")
        print("*** WARNING: libxc library not found.")
        print("* You can download libxc library from http://www.tddft.org/programs/libxc/down.php?file=4.3.4/libxc-4.3.4.tar.gz")
        print("* libxc library needs to be compiled with the flag --enable-shared")
        print("* May need to set PYSCF_INC_DIR if libxc library was not installed in the")
        print("* system standard install path (/usr, /usr/local, etc). Eg")
        print("*   export PYSCF_INC_DIR=/path/to/libxc:/path/to/other/lib")
        print("****************************************************************")

#
# Check xcfun
#
if 1:
    xcfun_lib_path = search_lib_path('libxcfun'+so_ext, [pyscf_lib_dir,
                                                         os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                         os.path.join(pyscf_lib_dir, 'deps', 'lib64')])
    xcfun_inc_path = search_inc_path('xcfun.h', [pyscf_lib_dir,
                                                 os.path.join(pyscf_lib_dir, 'deps', 'include')])
    if xcfun_lib_path and xcfun_inc_path:
        print("****************************************************************")
        print("* xcfun found in %s." % xcfun_lib_path)
        print("****************************************************************")
        default_lib_dir += [xcfun_lib_path]
        default_include += [xcfun_inc_path]
        extensions += [
            make_ext('pyscf.lib.libxcfun_itrf', 'dft', 'xcfun_itrf.c', ['xcfun']),
        ]
        DFT_AVAILABLE = 1

extensions = [x for x in extensions if x is not None]

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        if not DFT_AVAILABLE:
            print("****************************************************************")
            print("*** WARNING: DFT is not available.")
            print("****************************************************************")

# Python ABI updates since 3.5
# https://www.python.org/dev/peps/pep-3149/
class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        from distutils.sysconfig import get_config_var
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix

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
    packages=find_packages(exclude=['*dmrgscf*', '*fciqmcscf*', '*icmpspt*',
                                    '*shciscf*', '*xianci*', '*nao*',
                                    '*future*', '*test*', '*examples*',
                                    '*setup.py']),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtWithoutPlatformSuffix,
              'install': PostInstallCommand},
    install_requires=['numpy', 'scipy', 'h5py'],
    extras_require={
        'geomopt': ['pyberny>=0.6.2', 'geometric>=0.9.7.2'],
    },
    setup_requires = ['numpy'],
)

