# Contributing to DepthCast

Thanks for your interest in improving DepthCast. This guide explains how
to set up a development environment, the standards the codebase holds
itself to, and how to get a change merged.

## Code of conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By taking part you agree to uphold
it.

## Development setup

```bash
git clone https://github.com/Hilary-Henshaw/depthcast.git
cd depthcast
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

No dataset is required to develop or test DepthCast; the synthetic
generator (`depthcast.data.generate_lob_matrix`) supplies FI-2010-shaped
data on demand.

## Quality gates

Every change must pass the same checks CI runs. Run them locally before
opening a pull request:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # static types (strict)
pytest --cov            # tests with coverage
```

Conventions enforced across the project:

- **Type hints everywhere.** Public and private functions are fully
  annotated; `mypy` runs in strict mode.
- **Modern typing syntax.** Use `str | None`, `list[str]`, and
  `dict[str, int]` rather than the `typing` aliases.
- **Logging, not printing.** Library code logs through
  `depthcast.logging_utils.get_logger`; it never calls `print`.
- **Validated configuration.** New tunables belong in the Pydantic config
  models, not as bare constants threaded through call sites.
- **Line length 79.** Enforced by Ruff.
- **No em dashes** in comments or docs; use a hyphen or rewrite.

## Tests

- Add unit tests for new logic and keep total coverage at or above the
  current level (80% is the floor).
- Mark anything that trains a model with `@pytest.mark.integration` and
  keep it tiny (a few-channel model on a few hundred synthetic events) so
  the suite stays fast on a laptop CPU.
- Test names should read like documentation:
  `test_split_rejects_empty_split`, not `test_split_2`.

## Commit and pull-request flow

1. Create a topic branch off `main`.
2. Make focused commits with clear messages (imperative mood).
3. Ensure all quality gates pass.
4. Open a pull request describing the motivation, the approach, and any
   trade-offs. Link related issues.
5. A maintainer will review. Address feedback by pushing follow-up
   commits rather than force-pushing over the discussion.

## Reporting bugs and proposing features

Open a GitHub issue with a minimal reproduction (the synthetic generator
is ideal for this) or a clear description of the proposed behaviour and
its rationale. Security issues should follow
[SECURITY.md](SECURITY.md) instead of the public tracker.
