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

./.github/workflows/ci_"$os"/deps_apt.sh
./.github/workflows/ci_"$os"/python_deps.sh
./.github/workflows/ci_"$os"/build_pyscf.sh
./.github/workflows/run_tests.sh
