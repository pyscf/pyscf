# PySCF Python coding style

Load this template when writing, modifying, or reviewing Python code in PySCF,
including tests. Customize as needed.

Prioritize scientific correctness and intended API contracts. Keep implementation
and validation proportional to the change; apply the guidance below where relevant.

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
  cleanup, renaming, or reformatting. Refactor existing helpers substantially
  only when the task requires it, and explain why a localized change is
  insufficient.
- Reuse existing PySCF utilities when their intended contracts fit the operation.
  Do not infer support from implementation details or carefully tuned parameters
  that happen to produce the desired result. If broader reuse is needed,
  explicitly generalize the function's contract and implementation.
- Use PySCF's logger for diagnostic output rather than Python's standard
  `logging` module or `print` statements.
- Keep dependency direction consistent with the existing module structure.
  `data`, `gto`, `lib`, and `tools` are foundational modules. Reuse them where
  appropriate. Check their existing imports before adding a dependency. Avoid
  introducing dependencies from foundational code into method-specific modules.
- Prefer module-scope imports. When circular imports arise, resolve them
  through appropriate dependency boundaries or localized imports rather than
  adding eager imports to package `__init__.py` files. Only expose new
  submodules through `__init__.py` when required by the public API.
- Prefer side-effect-free implementations. Avoid unintended shared-state
  mutation. Do not rely on mutations of shared objects to make other objects
  behave correctly. When modifying an object, prefer creating a new instance
  unless in-place updates are performance-critical or required by the API.
- When introducing new attributes with immutable defaults, consider class-level
  defaults when consistent with the surrounding class. Extensions such as
  gpu4pyscf may reuse PySCF methods without running the corresponding PySCF
  `__init__` method to initialize attributes.
- Prefer validation at the appropriate boundary and trust established internal
  invariants. Avoid redundant checks and speculative handling of extreme corner
  cases unless relevant to the task or supported use cases. Retain checks needed
  to prevent concrete correctness or safety failures.
- Prefer a clear failure to a silent fallback. If an operation cannot meet its
  contract, let the error propagate or raise an informative exception.

## Formatting and linting

- Follow the repository's formatting and lint configuration, which permits
  compact conventions. Do not mechanically apply generic PEP 8 formatting.
- Break long numerical calls and Boolean expressions at logical boundaries,
  matching the surrounding continuation-line indentation.

## Test value and readability

- Test meaningful scientific results, supported behavior, and concrete regressions.
  Use independent expected results rather than duplicating implementation logic.
  Avoid tests added solely for coverage. Assert internal details or data layout
  only when they are part of an important behavioral contract.
- Use small, representative systems and minimal mocking. Mock a dependency only
  when needed to isolate meaningful behavior; keep that behavior real. Avoid
  artificial states created only to reach code. Reuse or extend existing tests
  instead of adding redundant cases.
- Prefer adding regression tests to existing test files. Create a new test file
  only when no existing file is suitable.
- Prefer direct setup and assertions with one focused behavior per test. Avoid
  nested loops mixing methods, options, or behaviors when they obscure the case;
  use separate, descriptive tests instead. Simple loops or parameterization are
  useful for closely related cases; accept repetition for clarity.
- Make the setup, behavior, and expected result clear without tracing helper
  layers. Use descriptive names and distinguish actual and expected results.
  Make assertion failures explain what went wrong and identify the failing case
  in loops or parameterized tests. Briefly explain non-obvious scientific setups,
  regressions, reference values, and tolerances.
- Match test complexity to behavioral risk. Introduce fixtures, factories, or
  helpers only when they materially simplify meaningful tests. Avoid building a
  test framework for a small change or exhaustive method/option combinations
  without a concrete reason.

## Documentation

- Treat functions and classes listed in a module's `__all__` as public APIs.
  Absence from `__all__` does not by itself make an API private.
- Give public APIs clear docstrings describing their purpose, inputs, outputs,
  and important parameters, attributes, and defaults. Explain shapes, units,
  and limitations where relevant.
- Keep internal docstrings brief, but explain non-obvious assumptions and contracts.
- Update documentation when changing parameters, return values, supported
  shapes, symmetry, units, or public behavior.

### Comments and docstrings

- Write in plain, natural English, using short, direct sentences.
- Prefer concrete explanations over abstract or formal language.
- Explain why code works a certain way, not what is already obvious.
- Document non-obvious assumptions, numerical conventions, and edge cases.
