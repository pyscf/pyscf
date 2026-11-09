"""Record the ARM64 runtime and the unresolved DF4C regression input/output."""

import json
import os
from pathlib import Path
import platform
import sys

import numpy
import scipy
import scipy.linalg
from pyscf import df, gto, lib, scf
from pyscf.dft import libxc


def main():
    output = Path('arm64-diagnostics')
    output.mkdir(exist_ok=True)
    header = Path(gto.__file__).resolve().parents[1] / 'lib/deps/include/cint.h'
    maps = Path('/proc/self/maps')
    environment = dict(
        pid=os.getpid(), machine=platform.machine(), python=sys.version,
        numpy=numpy.__version__, scipy=scipy.__version__, libxc=libxc.__version__,
        libcint_header=[line for line in header.read_text().splitlines()
                        if line.startswith('#define CINT_VERSION')],
        loaded_libraries=sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                 if any(name in line for name in ('libcint', 'libxc.', 'libopenblas'))}),
    )
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps(environment), flush=True)
    numpy.show_config()
    scipy.show_config()

    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='cc-pvdz', verbose=0)
    arrays = {}
    original_eigh = scipy.linalg.eigh

    def record_eigh(matrix, *args, **kwargs):
        values, vectors = original_eigh(matrix, *args, **kwargs)
        index = len(arrays) // 3
        arrays[f'metric_{index}'] = matrix.copy()
        arrays[f'eigenvalues_{index}'] = values.copy()
        arrays[f'eigenvectors_{index}'] = vectors.copy()
        return values, vectors

    with lib.temporary_env(lib.param, LIGHT_SPEED=1):
        nao = mol.nao_nr() * 4
        numpy.random.seed(1)
        dm = numpy.random.random((2, nao, nao)) + numpy.random.random((2, nao, nao)) * 1j
        dm[0] += scf.dhf.time_reversal_matrix(mol, dm[0])
        dm[1] += scf.dhf.time_reversal_matrix(mol, dm[1])
        fit = df.DF4C(mol)
        with lib.temporary_env(scipy.linalg, eigh=record_eigh):
            vj, vk = fit.get_jk(dm, hermi=0, omega=0.9)
        exact_j, exact_k = scf.dhf.get_jk(mol, dm, hermi=0, omega=0.9)
    arrays.update(dm=dm, vj=vj, vk=vk, exact_j=exact_j, exact_k=exact_k)
    numpy.savez_compressed(output / 'df4c.npz', **arrays)
    fp_j, fp_k = lib.fp(vj), lib.fp(vk)
    result = dict(fp_j=[fp_j.real, fp_j.imag], fp_k=[fp_k.real, fp_k.imag],
                  reference_j_difference=abs(fp_j - (1364.9812117487595+215.73320482400422j)),
                  reference_k_difference=abs(fp_k - (159.036202745021+687.903428296142j)),
                  exact_j_max_error=float(abs(vj-exact_j).max()),
                  exact_k_max_error=float(abs(vk-exact_k).max()))
    (output / 'df4c.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
