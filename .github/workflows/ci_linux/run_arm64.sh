#!/usr/bin/env bash

set -e

# Run the manylinux container natively; QEMU exceeded the CI time limit.
test "$(uname -m)" = aarch64
exec docker run --rm -v "${GITHUB_WORKSPACE:?}:/src/pyscf:rw" --workdir=/src/pyscf \
    -e RUNNER_OS=Linux -e PYSCF_CI_PROFILE=manylinux-aarch64 \
    -e PYSCF_CI_PYTHON_ABI="${PYSCF_CI_PYTHON_ABI:?}" \
    "$PYSCF_CI_ARM64_IMAGE" bash -ex ./.github/workflows/run_ci.sh "$@"
