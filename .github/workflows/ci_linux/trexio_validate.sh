#!/usr/bin/env bash
# Build trexio-validate and install the executable into /usr/local.
#
# trexio-validate (https://github.com/TREX-CoE/trexio-validate) recomputes the
# contents of a TREXIO file from the basis set stored in it and compares the
# result with the stored data, which is what
# pyscf/tools/test/test_trexio_validate.py uses to check the files written by
# pyscf.tools.trexio.
#
# It needs libcint, which build_pyscf.sh has already unpacked into
# pyscf/lib/deps, so run this after that script.  TREXIO is fetched and linked
# statically, because the pip `trexio` package ships no C headers.

set -e

src="${RUNNER_TEMP:-/tmp}/trexio-validate"
prefix="${TREXIO_VALIDATE_PREFIX:-/usr/local}"
deps="$PWD/pyscf/lib/deps"

if [ ! -f "$deps/lib/libcint.so" ]; then
    echo "libcint not found in $deps; run build_pyscf.sh first" >&2
    exit 1
fi

rm -rf "$src"
git clone --depth 1 https://github.com/TREX-CoE/trexio-validate.git "$src"

cmake -S "$src" -B "$src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTREXIO_VALIDATE_FETCH_TREXIO=ON \
    -DTREXIO_VALIDATE_BUILD_TESTS=OFF \
    -DTREXIO_VALIDATE_BUILD_EXAMPLES=OFF \
    -DLIBCINT_INCLUDE_DIR="$deps/include" \
    -DLIBCINT_LIBRARY="$deps/lib/libcint.so" \
    -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build "$src/build" -j4

if mkdir -p "$prefix" 2>/dev/null && [ -w "$prefix" ]; then
    cmake --install "$src/build"
else
    sudo cmake --install "$src/build"
fi

# A bundled TREXIO is linked statically into the executable, so only the
# program is installed; the test module falls back to it when the Python
# module is absent.
"$prefix/bin/trexio-validate" --version
