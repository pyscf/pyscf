import os
import sys
import sysconfig
from setuptools import setup, find_packages, Extension
import numpy

if sys.version_info[0] >= 3: # from Cython 0.14
    from distutils.command.build_py import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py
from distutils.command.install import install

topdir = os.path.abspath(os.path.join(__file__, '..'))

CLASSIFIERS = [
'Development Status :: 5 - Production/Stable',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: BSD License',
'Programming Language :: C',
'Programming Language :: Python',
'Programming Language :: Python :: 2.7',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
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
def get_version():
    with open(os.path.join(topdir, 'pyscf', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                return eval(line.strip().split(' = ')[1])
    raise ValueError("Version string not found")
VERSION = get_version()


#
# default include and library path
#
np_blas = numpy.__config__.blas_opt_info
blas_include = []#np_blas['include_dirs']
blas_lib_dir = np_blas['library_dirs']
blas_library = np_blas['libraries']

distutils_lib_dir = 'lib.{platform}-{version[0]}.{version[1]}'.format(
    platform=sysconfig.get_platform(),
    version=sys.version_info)

if (sys.platform.startswith('linux') or
    sys.platform.startswith('gnukfreebsd')):
    so_ext = '.so'
elif sys.platform.startswith('darwin'):
    so_ext = '.dylib'
elif sys.platform.startswith('win'):
    so_ext = '.dll'
else:
    raise OSError('Unknown platform')

pyscf_lib_dir = os.path.join(topdir, 'pyscf', 'lib')
build_lib_dir = os.path.join('build', distutils_lib_dir, 'pyscf', 'lib')
default_lib_dir = [os.path.join(pyscf_lib_dir,'deps','lib'), build_lib_dir] + blas_lib_dir
default_include = ['.', 'build', pyscf_lib_dir,
                   os.path.join(pyscf_lib_dir,'deps','include')] + blas_include

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
             include_dirs=default_include, **kwargs):
    if '/' in relpath:
        relpath = os.path.join(*relpath.split('/'))
    if (os.path.isfile(os.path.join(pyscf_lib_dir, 'build', 'CMakeCache.txt')) and
        os.path.isfile(os.path.join(pyscf_lib_dir, *pkg_name.split('.')) + so_ext)):
        return None
    else:
        srcs = make_src(relpath, srcs)
        return Extension(pkg_name, srcs,
                         libraries = libraries,
                         library_dirs = library_dirs,
                         include_dirs = include_dirs + [os.path.join(pyscf_lib_dir,relpath)],
                         extra_compile_args = ['-fopenmp'],
                         extra_link_args = ['-fopenmp'],
# Be careful with the ld flag "-Wl,-R$ORIGIN" in the shell.
# When numpy.distutils is imported, the default CCompiler of distutils will be
# overwritten. Compilation is executed in shell and $ORIGIN will be converted to ''
                         runtime_library_dirs = ['$ORIGIN', '.'],
                         **kwargs)

def make_src(relpath, srcs):
    srcpath = os.path.join(pyscf_lib_dir, relpath)
    abs_srcs = []
    for src in srcs.split():
        if '/' in src:
            abs_srcs.append(os.path.join(srcpath, *src.split('/')))
        else:
            abs_srcs.append(os.path.join(srcpath, src))
    return abs_srcs


def search_lib_path(libname, extra_paths=None, version=None):
    paths = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    if 'PYSCF_INC_DIR' in os.environ:
        PYSCF_INC_DIR = os.environ['PYSCF_INC_DIR']
        paths = [PYSCF_INC_DIR,
                 os.path.join(PYSCF_INC_DIR, 'lib'),
                 os.path.join(PYSCF_INC_DIR, '..', 'lib'),
                ] + paths
    if extra_paths is not None:
        paths += extra_paths
    for path in paths:
        full_libname = os.path.join(path, libname)
        if version is not None:
            full_libname = full_libname + '.' + version
        if os.path.exists(full_libname):
            return os.path.abspath(path)

def search_inc_path(incname, extra_paths=None):
    paths = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    if 'PYSCF_INC_DIR' in os.environ:
        paths = [PYSCF_INC_DIR,
                 os.path.join(PYSCF_INC_DIR, 'include'),
                ] + paths
    if extra_paths is not None:
        paths += extra_paths
    for path in paths:
        inc_path = os.path.join(os.path.dirname(path), 'include')
        full_incname = os.path.join(path, incname)
        if os.path.exists(full_incname):
            return os.path.abspath(path)

#
# Check libcint
#
extensions = []
if 1:
    print pyscf_lib_dir
    libcint_lib_path = search_lib_path('libcint'+so_ext, [os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                          os.path.join(pyscf_lib_dir, 'deps', 'lib64')],
                                       version='3.0')
    libcint_inc_path = search_inc_path('cint.h', [os.path.join(pyscf_lib_dir, 'deps', 'include')])
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
autocode/hess.c autocode/intor1.c autocode/grad2.c '''
        if os.path.exists(os.path.join(pyscf_lib_dir, 'libcint')):
            extensions.append(
                make_ext('pyscf.lib.libcint', 'libcint/src', srcs, blas_library)
            )
        else:
            print("****************************************************************")
            print("*** WARNING: libcint library not found.")
            print("* You can download libcint library from http://github.com/sunqm/libcint")
            print("* May need to set PYSCF_INC_DIR if libcint library was not installed in the")
            print("* system standard install path (/usr, /usr/local, etc)")
            print("****************************************************************")
            raise RuntimeError

extensions += [
    make_ext('pyscf.lib.libnp_helper', 'np_helper',
             'condense.c npdot.c omp_reduce.c pack_tril.c transpose.c',
             blas_library),
    make_ext('pyscf.lib.libcgto', 'gto',
             '''fill_int2c.c fill_nr_3c.c fill_r_3c.c fill_int2e.c ft_ao.c
             grid_ao_drv.c fastexp.c deriv1.c deriv2.c nr_ecp.c autocode/auto_eval1.c''',
             ['cint', 'np_helper']),
    make_ext('pyscf.lib.libcvhf', 'vhf',
             '''fill_nr_s8.c nr_incore.c nr_direct.c optimizer.c nr_direct_dot.c
             time_rev.c r_direct_o1.c rkb_screen.c r_direct_dot.c
             rah_direct_dot.c rha_direct_dot.c''',
             ['cgto', 'np_helper', 'cint']),
    make_ext('pyscf.lib.libao2mo', 'ao2mo',
             'restore_eri.c nr_ao2mo.c nr_incore.c r_ao2mo.c',
             ['cvhf', 'cint', 'np_helper']),
    make_ext('pyscf.lib.libfci', 'mcscf',
             '''fci_contract.c fci_contract_nosym.c fci_rdm.c fci_string.c
             fci_4pdm.c select_ci.c''',
             ['np_helper']),
    make_ext('pyscf.lib.libmcscf', 'mcscf', 'nevpt_contract.c',
             ['fci', 'cvhf', 'ao2mo']),
    make_ext('pyscf.lib.libri', 'ri', 'r_df_incore.c',
             ['cint', 'ao2mo', 'np_helper']),
    make_ext('pyscf.lib.libhci', 'hci', 'hci.c', ['np_helper']),
    make_ext('pyscf.lib.libpbc', 'pbc', 'ft_ao.c fill_ints.c grid_ao.c', ['cgto', 'cint']),
    make_ext('pyscf.lib.libmbd', os.path.join('extras', 'mbd'), 'dipole.c', []),
    make_ext('pyscf.lib.libdft', 'dft',
             'CxLebedevGrid.c grid_basis.c nr_numint.c r_numint.c',
             ['cvhf', 'cgto', 'cint']),
]

#
# Check libxc
#
DFT_AVAILABLE = 0
if 1:
    libxc_lib_path = search_lib_path('libxc'+so_ext, [os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                      os.path.join(pyscf_lib_dir, 'deps', 'lib64')],
                                     version='4.0.0')
    libxc_inc_path = search_inc_path('xc.h', [os.path.join(pyscf_lib_dir, 'deps', 'include')])
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
        print("You can download libxc library from http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-3.0.0.tar.gz")
        print("libxc library needs to be compiled with the flag --enable-shared")
        print("May need to set PYSCF_INC_DIR if libxc library was not installed in the")
        print("system standard install path (/usr, /usr/local, etc)")
        print("****************************************************************")

#
# Check xcfun
#
if 1:
    xcfun_lib_path = search_lib_path('xcfun'+so_ext, [os.path.join(pyscf_lib_dir, 'deps', 'lib'),
                                                      os.path.join(pyscf_lib_dir, 'deps', 'lib64')])
    xcfun_inc_path = search_inc_path('xcfun.h', [os.path.join(pyscf_lib_dir, 'deps', 'include')])
    if xcfun_lib_path and xcfun_inc_path:
        print("****************************************************************")
        print("* xcfun found in %s." % xcfun_lib_path)
        print("****************************************************************")
        default_lib_dir += [xcfun_lib_path]
        default_include += [xcfun_inc_path]
        extensions += [
            make_ext('pyscf.lib.xcfun_itrf', 'dft', 'xcfun_itrf.c', ['xcfun']),
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
    package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat']}, # any package contains *.so *.dat files
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*dmrgscf*', '*fciqmcscf*', '*icmpspt*',
                                    '*shciscf*', '*xianci*', '*nao*',
                                    '*future*', '*test*', '*examples*',
                                    '*setup.py']),
    ext_modules=extensions,
    cmdclass={'build_py': build_py,
              'install': PostInstallCommand,
             },
    install_requires=['numpy', 'scipy', 'h5py'],
)

