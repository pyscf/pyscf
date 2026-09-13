# PySCF Python coding style

Load this template when writing, modifying, or reviewing Python code in PySCF.
Customize as needed.

- Resolve routine implementation choices using nearby code and existing APIs.
  Ask when an unresolved scientific convention or requirement materially affects
  the result; do not silently invent units, normalization factors, or reference
  values.

## Implementation

- PySCF favors compact scientific code. Keep implementations concise while
  making algorithmic steps and numerical assumptions clear.
- Follow nearby code when conventions differ, unless changing the style is
  part of the task.
- Make the smallest complete change that solves the problem. Avoid unrelated
  cleanup, renaming, or reformatting.
- Do not substantially refactor, rewrite, or reorganize existing helper
  functions unless the task clearly requires it. If substantial refactoring is
  necessary, explain why the localized approach is insufficient.
- Reuse existing PySCF utilities before adding equivalent third-party helpers.
- Use PySCF's logger for diagnostic output rather than Python's standard
  `logging` module or `print` statements.
- Keep dependency direction consistent with the existing module structure.
  `data`, `gto`, `lib`, and `tools` are foundational modules. Reuse them where
  appropriate. Chek their existing imports before adding a dependency. Avoid
  introducing dependencies from foundational code into method-specific modules.
- Prefer module-scope imports. When circular imports arise, resolve them
  through appropriate dependency boundaries or localized imports rather than
  adding eager imports to package `__init__.py` files. Only expose new
  submodules through `__init__.py` when required by the public API.
- Prefer side-effect-free implementations. Do not rely on mutations of shared
  objects to make other objects behave correctly. When modifying an object,
  prefer creating a new instance unless in-place updates are
  performance-critical or required by the API. Use the returned object
  explicitly, even when the operation mutates in place.
- When introducing new attributes with immutable defaults, consider class-level
  defaults when consistent with the surrounding class. Extensions such as
  gpu4pyscf may reuse PySCF methods without running the corresponding PySCF
  `__init__` method to initialize attributes.

## Formatting and linting

- Follow the repository's formatting and lint configuration, which permits
  compact conventions. Do not mechanically apply generic PEP 8 formatting.
- Break long numerical calls and Boolean expressions at logical boundaries,
  matching the surrounding continuation-line indentation.

## Documentation

- Document non-obvious assumptions.
- For functions outside the public API, a short docstring explaining their
  purpose is sufficient.
- Update documentation when changing parameters, return values, supported
  shapes, symmetry, units, or public behavior.
