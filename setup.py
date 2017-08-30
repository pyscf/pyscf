import sys
import subprocess
from distutils.core import setup, Command
from distutils.command.install import install
from distutils.command.build import build
import os

if sys.version_info[0] >= 3: # from Cython 0.14
    from distutils.command.build_py import build_py_2to3 as build_py
else:
    from distutils.command.build_py import build_py


class PyscfBuild(build):

    def run(self):
        # run original build code
        build.run(self)

        build_path = os.path.abspath(self.build_temp)

        cmd_cmake = ["cmake", ".."]
        cmd_make = ["make"]

        if self.debug:
            cmd_cmake.append("-DCMAKE_BUILD_TYPE=Debug")
            cmd_make.append("VERBOSE=1")

        def compile():
            try:
                os.mkdir("lib/build")
                print("cd lib/build")
            except:
                print("cd lib/build")
            os.chdir("lib/build")
            print("cmake")
            subprocess.call(cmd_cmake)
            print("make")
            subprocess.call(cmd_make)

            os.chdir("../..")
            print("cd ../../")

        self.execute(compile, [], "compile pyscf")

        self.mkpath(self.build_lib)

class PyscfClean(Command):
    user_options = []
    description = "clean build and lib/build directories"

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        cmd = "rm -r build lib/build lib/*.so"
        subprocess.call(cmd, shell=True)

class PyscfInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options("build", ("build_scripts", "build_scripts"))

    def run(self):
        # run original install code
        install.run(self)

        self.copy_tree(self.build_lib, self.install_lib)


CLASSIFIERS = [
    'Development Status :: Development',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved',
    'Programming Language :: C',
    'Programming Language :: Fortran',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]

NAME             = 'pyscf'
MAINTAINER       = 'Qiming Sun, Peter Koval'
MAINTAINER_EMAIL = 'osirpt.sun@gmail.com, koval.peter@gmail.com'
DESCRIPTION      = """PySCF: Python-based Simulations of Chemistry Framework.
                      Including TDDFT calculation from mbpt_lcao project"""
#LONG_DESCRIPTION = ''
URL              = 'http://www.pyscf.org'
DOWNLOAD_URL     = 'https://github.com/kovalp/pyscf'
LICENSE          = 'BSD 2-clause "Simplified" License (BSD2)'
AUTHOR           = 'Qiming Sun, Peter Koval'
AUTHOR_EMAIL     = 'osirpt.sun@gmail.com, koval.peter@gmail.com'
PLATFORMS        = ['Linux', 'Mac OS-X', 'Unix']
VERSION          = '1.3.0'

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
    cmdclass={'build_py': build_py,
              'build': PyscfBuild,
              'install': PyscfInstall,
              'veryclean': PyscfClean},
)

#msg = subprocess.check_output(
#    'mkdir -p lib/build && cd lib/build && cmake .. && make install',
#    shell=True, stderr=subprocess.STDOUT)
