# PySCF PR review

Load this template when reviewing a PySCF contribution or checking a
contributor's own work before submitting or updating a PR. Customize as needed.

Review the proposed diff, PR description, and relevant surrounding code and
callers. Apply the checks below where they concern the change. Report concrete
findings with locations, consequences, and suggested corrections; distinguish
required fixes from questions and optional improvements. State any material
review or validation gaps.

## References and data

- Check that relevant programs and literature are properly cited. Connect
  non-obvious algorithms and scientific conventions to their references in
  comments or docstrings where useful.
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

## Reuse and duplication

- Search the codebase for similar functionality. Determine whether an existing
  implementation can be reused or adapted instead of adding duplicate code.
- Reuse existing shared definitions of constants and parameter tables instead
  of adding duplicate definitions.
- When a separate implementation of similar functionality is necessary, check
  for a comment or docstring explaining why the existing implementation is
  insufficient and why the new one is preferred.

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
- Normal computation follows established assumptions. Unless targeting special
  systems, trivial checks and speculative handling of extreme corner cases can
  be skipped.

## Performance

- Check for repeated expensive work and unnecessarily large intermediates.
- Require representative benchmarks when performance is the reason for added
  complexity or moving Python code into C.

## Human readability

- Check that a human reader familiar with the method can understand the code.
- Avoid complicated long expressions. Break them into understandable
  algorithmic steps where appropriate.

## Scientific validation

- Check that claims in the PR description and examples accurately describe the
  implemented method and its capabilities.
- For a bug fix, check that a regression test exercises the failing behavior.
- Use small, deterministic systems and cover relevant supported branches.
  Avoid an exhaustive combination of methods and options without a reason.

## PR context

- Search for related PySCF GitHub issues and check that the PR explains and
  links the relevant relationships. If GitHub is unavailable, report that this
  check could not be completed.
- If an AI agent was used, check that the PR message discloses the assistance.
