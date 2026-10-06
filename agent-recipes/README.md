# Agent Recipes

These files help you guide an AI agent working on PySCF. Use and customize
them as needed.

- [AGENTS.md](agents/AGENTS.md) explains the repository layout and development
  environment.
- [Coding style](skills/pyscf-coding-style.template.md) guides how to write and
  review PySCF Python code.
- [PR review](skills/pyscf-pr-review.template.md) helps check a contribution
  before submitting or updating a pull request, or when reviewing a github pull
  request.

Ask your agent to read `agent-recipes/agents/AGENTS.md` and the relevant
template at the start of a task. For example:

```text
Read agent-recipes/agents/AGENTS.md and
agent-recipes/skills/pyscf-pr-review.template.md, then review my changes
against <base branch or commit> and the accompanying PR description.
```

If you use these often, you can add the guidance from `agents/AGENTS.md` to an
`AGENTS.md` in the root of your checkout. Keep any instructions already there.
Ask your agent to read the templates when needed, or install them using the
skill format your agent supports. It may not find these files on its own.
