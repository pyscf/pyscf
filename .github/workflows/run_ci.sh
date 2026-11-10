#!/usr/bin/env bash

set -e

if [ "$RUNNER_OS" == "Linux" ]; then
    os='linux'
elif [ "$RUNNER_OS" == "macOS" ]; then
    os='macos'
else
    echo "$RUNNER_OS not supported"
    exit 1
fi

if [ "$os" = "linux" ] && [ -n "${PYSCF_CI_ARM64_IMAGE:-}" ]; then
    exec bash ./.github/workflows/ci_linux/run_arm64.sh "$@"
fi

if [ "${PYSCF_CI_PROFILE:-}" = "manylinux-aarch64" ]; then
    export PATH="/opt/python/${PYSCF_CI_PYTHON_ABI:?}/bin:$PATH"
    ./.github/workflows/ci_linux/python_deps.sh
    ./.github/workflows/ci_linux/deps_yum.sh
else
    ./.github/workflows/ci_"$os"/deps_apt.sh
    ./.github/workflows/ci_"$os"/python_deps.sh
fi
./.github/workflows/ci_"$os"/build_pyscf.sh
./.github/workflows/run_tests.sh "$@"
