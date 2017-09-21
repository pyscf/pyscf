import os
import sys
import sysconfig
from setuptools import setup, find_packages, Extension
import numpy

if sys.version_info[0] >= 3: # from Cython 0.14
    from distutils.command.build_py import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py

CLASSIFIERS = [
'Development Status :: 1.4 stable',
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


np_blas = numpy.__config__.blas_opt_info
blas_include = np_blas['include_dirs']
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

topdir = os.path.abspath(os.path.join(__file__, '..'))
pyscf_lib_dir = os.path.join(topdir, 'pyscf', 'lib')
build_lib_dir = os.path.join('build', distutils_lib_dir, 'pyscf', 'lib')
default_lib_dir = [os.path.join(pyscf_lib_dir,'deps','lib'), build_lib_dir] + blas_lib_dir
default_include = ['.', 'build', pyscf_lib_dir,
                   os.path.join(pyscf_lib_dir,'deps','include')] + blas_include

with open(os.path.join(topdir, 'build', 'config.h'), 'w') as f:
    f.write('''
#if defined _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
''')

def make_ext(pkg_name, relpath, srcs, libraries=[], library_dirs=default_lib_dir,
             include_dirs=default_include, **kwargs):
    if (os.path.isfile(os.path.join(pyscf_lib_dir, 'build', 'CMakeCache.txt')) and
        os.path.isfile('.'.join([os.path.join(pyscf_lib_dir, *pkg_name.split('.')), so_ext]))):
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

extensions = [
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
    make_ext('pyscf.lib.libxc_itrf', 'dft', 'libxc_itrf.c', ['xc']),
    make_ext('pyscf.lib.xcfun_itrf', 'dft', 'xcfun_itrf.c', ['xcfun']),
]
extensions = [x for x in extensions if x is not None]

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
                                    '*shciscf*', '*xianci*', '*nao*',
                                    '*future*', '*test*', '*examples*',
                                    '*setup.py']),
    ext_modules=extensions,
    cmdclass={'build_py': build_py}
)

