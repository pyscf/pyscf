# PySCF PR review

Load this template when reviewing a PySCF contribution or checking a
contributor's own work before submitting or updating a PR. Customize as needed.

Prioritize scientific correctness and intended API contracts. Keep implementation
and validation proportional to the change; apply the checks below where relevant.

Review the proposed diff, PR description, and relevant surrounding code and
callers. Report concrete findings with locations, consequences, and suggested
corrections; distinguish required fixes from questions and optional improvements.
State any material review or validation gaps.

## References and data

- Check that relevant programs and literature are properly cited. Connect
  non-obvious algorithms and scientific conventions to their references in
  comments or docstrings where useful.
- Verify that cited references exist, links or identifiers point to the intended
  sources, and those sources support the associated claims.
- For constants, parameter tables, or databases, check that sources are cited
  and that license or redistribution concerns are addressed. Flag unclear
  permissions rather than assuming that a citation establishes permission.
- Document the units and any transformations needed to reproduce constants or
  table values.

## Examples and documentation

- Check that a new feature includes a runnable example demonstrating important
  flags, attributes, and options, with comments explaining what the settings
  do and why they are used.
- Treat functions and classes listed in a module's `__all__` as public APIs.
  Absence from `__all__` does not by itself make an API private.
- Check that public APIs have clear docstrings describing their purpose,
  inputs, outputs, and important parameters, attributes, and defaults. Explain
  shapes, units, and limitations where relevant.
- Explain intentional changes to public behavior and provide migration guidance
  when replacing an existing API.
- Flag comments and docstrings that are verbose, vague, or hard to read.
- Check that explanations are accurate, useful, and not redundant with the code.

## Reuse and duplication

- Search for existing implementations whose intended contracts fit the operation
  before suggesting reuse or accepting duplicate code.
- Reuse existing shared definitions of constants and parameter tables instead
  of adding duplicate definitions.
- When a separate implementation of similar functionality is necessary, check
  for a comment or docstring explaining why the existing implementation is
  insufficient and why the new one is preferred.
- Respect function contracts when reusing code. Judge reuse by what the
  function is designed to support. Flag calls that bypass the intended contract
  through input manipulation, compensating factors, or disabled terms, even if
  numerical tests pass.

## Implementation and behavior

- Check for side effects on existing functions or objects. Intentional mutations
  need clear documentation, comments, or warnings as appropriate. Prefer fixing
  accidental side effects to merely adding a comment.
- Determine whether scanners are affected. If so, inspect `reset` and verify
  that cached state is handled properly.
- Check initialization paths, particularly supplied intermediates that bypass
  an ordinary setup step.
- When rejecting an unsupported feature or option combination, provide an error
  message that clearly identifies the limitation.
- Remove debugging code accidentally left in the contribution, such as
  breakpoints and temporary diagnostic output.
- Tie requests for guards or recovery to concrete failure modes in supported
  use cases. Prefer validation at the appropriate boundary and trust established
  internal invariants. Avoid redundant checks and speculative handling of extreme
  corner cases unless relevant to the task or supported use cases. Retain checks
  needed to prevent concrete correctness or safety failures.
- Prefer a clear failure to a silent fallback. Check that operations that cannot
  meet their contract propagate errors or raise informative exceptions instead
  of silently substituting defaults, results, or alternative algorithms.
- For complex-valued orbitals or wavefunction coefficients, derive the required
  conjugation from the bra/ket definitions and tensor index conventions. Pay
  particular attention to formulas inherited from real-valued methods, where
  missing conjugation is invisible. Check that contractions, symmetries, and
  dtypes preserve the intended complex-valued expressions.
- Check for potential API breaks when removing or changing functions, method
  signatures, attributes, defaults, or return values.
  Lack of internal uses does not establish that an API is safe to change.
  Preserve compatibility for deprecated APIs until their planned removal, and
  flag intentional breaking changes that lack a deprecation or migration path.

## Performance

- Check for repeated expensive work and unnecessarily large intermediates.
- Require representative benchmarks when performance is the reason for added
  complexity or moving Python code into C.

## Human readability

- Check that a reader familiar with the method can understand the code.
  Suggest breaking complicated expressions into understandable steps where useful.
- Prefer direct setup and assertions with one focused behavior per test. Flag
  nested loops mixing methods, options, or behaviors when they obscure the case;
  suggest separate, descriptive tests instead. Simple loops or parameterization
  are useful for closely related cases; accept modest repetition for clarity.
- Check that setup, behavior, and expected results are clear without tracing
  helper layers. Names should describe the case, actual and expected results
  should be distinguishable, and assertion failures should explain what went
  wrong and identify the failing case. Check for brief explanations of non-obvious
  scientific setups, regressions, reference values, and tolerances.

## Tests and validation

- Check that claims in the PR description and examples accurately describe the
  implemented method and its capabilities.
- Check that tests protect important scientific results and supported behavior,
  and that a bug fix has a regression test exercising the failure. Expected
  results should be independent of the implementation. Flag assertions about
  internal details or data layout unless they are part of an important contract.
  Request missing tests for realistic failures, not branch coverage for its own sake.
- Prefer small, representative, deterministic systems and minimal mocking. Mocking
  should isolate a needed dependency while keeping the behavior under test real.
  Question artificial states created only to reach code. Suggest simplifying,
  consolidating, or removing redundant tests and tests with no meaningful assertion.
- Flag new test files when the tests would fit naturally in an existing file.
- Match test complexity to behavioral risk. Flag fixtures, factories, helper
  layers, or parameter matrices that do not materially simplify meaningful tests.
  Avoid exhaustive combinations of methods and options without a concrete reason.

## PR context

- Search for related PySCF GitHub issues and check that the PR explains and
  links the relevant relationships. If GitHub is unavailable, report that this
  check could not be completed.
- If an AI agent was used, check that the PR message discloses the assistance.
