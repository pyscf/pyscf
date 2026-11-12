**Summary**
One or two sentences: what changes and why.

**Motivation and context**
Link the related issue(s), e.g. `Fixes #1234`.

**How has this been tested?**
- New/changed tests: `python -m pytest pyscf/.../test_xxx.py`
- Commands and results (quote the relevant output):

**Notes**
Per the [AI policy](https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md), briefly describe any AI assistance: the tool or agent used and what it was used for (code generation, testing, debugging, drafting, none).

**Checklist**
- [ ] I have read the [CONTRIBUTING.md](https://github.com/pyscf/pyscf/blob/master/CONTRIBUTING.md)
- [ ] Changes and PR description checked against the [PR review template](https://github.com/pyscf/pyscf/blob/master/agent-recipes/skills/pyscf-pr-review.template.md)
- [ ] Bug fixes are backed by a regression test
- [ ] Test tolerances and reference values are not relaxed just to make tests pass
- [ ] New features include a runnable example and cited references for non-obvious algorithms or constants
- [ ] `python -m flake8 <touched files>` passes
- [ ] Docstrings added/updated for public API changes
- [ ] No unrelated reformatting or leftover debugging code
