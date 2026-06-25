# cibuildwheel Evaluation For Windows Packaging

## Scope

This note evaluates whether the current Windows wheel release path should be migrated from the custom `build-wheel.ps1` + `verify-wheel.ps1` flow to `cibuildwheel`.

## Current State

The current Windows release path is now stable in these areas:

- PEP 517 wheel builds via `python -m build --wheel --no-isolation`
- explicit runtime staging inside the build flow
- targeted installed-wheel verification with:
  - artifact checks
  - import checks
  - smoke checks
  - packaging regression examples
  - diagnostic expected-failure checks
- a reproducible local environment description in `conda/windows/environment.yml`
- a matching GitHub Actions Windows runner path using:
  - `actions/setup-python`
  - `msys2/setup-msys2`
  - `build-wheel.ps1`
  - `verify-wheel.ps1`

## What cibuildwheel Would Improve

Potential benefits:

- one more standardized wheel entry point across platforms
- better consistency with the already existing macOS `cibuildwheel` jobs
- easier future matrix expansion across Python versions and architectures
- clearer wheel-oriented CI semantics than hand-written per-platform glue

## What Makes Migration Non-Trivial

The current Windows flow still depends on project-specific behavior that is not just a normal Python wheel build:

1. External runtime preparation is part of the build path
   - the flow stages external runtime files into `pyscf/lib/deps/win64/bin`
   - this is not just a plain `pip wheel` invocation

2. The build depends on MSYS2 UCRT64 toolchain setup
   - GCC and OpenBLAS come from `msys2/setup-msys2`
   - this environment must remain aligned with the wheel contents

3. The project uses custom CMake-driven native builds
   - the build is routed through setuptools build hooks into CMake
   - it is not yet a fully generic `cibuildwheel` drop-in case

4. Verification is an explicit post-build gate
   - `verify-wheel.ps1` is a real release-quality contract, not an optional smoke check
   - any migration still needs this verification layer

## Decision

Decision: `NO-GO for immediate migration`

Reason:

- the current Windows native-runner flow is now stable and verified
- migrating immediately would add moving parts without removing the custom runtime staging or verification requirements
- the current pain points were mostly correctness, reproducibility, and diagnosis; those have now been addressed directly

This is not a rejection of `cibuildwheel` in general. It is a sequencing decision.

## Recommended Future Entry Criteria

Revisit migration only when all of these are true:

1. the Windows native-runner path stays green for multiple release cycles
2. runtime staging remains stable and no longer changes frequently
3. the project wants broader Python-version or architecture expansion
4. maintaining separate Windows/macOS/Linux wheel workflows becomes a real maintenance burden

## Recommended Migration Shape

If migration is revisited later, use this order:

1. Keep `build-wheel.ps1` and `verify-wheel.ps1` as the authoritative build/verify contracts
2. Introduce `cibuildwheel` first as a wrapper around the existing Windows build assumptions, not as a rewrite
3. Preserve the post-build verification gate unchanged
4. Compare:
   - existing Windows runner time
   - new `cibuildwheel` runner time
   - failure clarity
   - maintenance complexity
5. Migrate only if the wrapper demonstrably reduces maintenance without weakening verification

## Bottom Line

The correct current strategy is:

- keep Windows on the native GitHub Actions runner path
- keep local Windows packaging aligned with that path
- treat `cibuildwheel` as a later optimization, not as unfinished mandatory work
