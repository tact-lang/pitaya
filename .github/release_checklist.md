# Release Checklist

Use this checklist to cut a clean release.

## Preflight

- [ ] Working tree clean; CI green on main/default branch
- [ ] Decide new version (semver): `vX.Y.Z`

## Versioning & Changelog

- [ ] Update version in `pyproject.toml` (`[project].version`)
- [ ] Update version in `src/__init__.py` (`__version__`)
- [ ] Update `CHANGELOG.md`
  - [ ] Move items from “Unreleased” under a new `X.Y.Z - YYYY-MM-DD` section
  - [ ] Add link references at bottom (compare previous tag → new tag)
  - [ ] Start a fresh “Unreleased” section

## Docs & Help

- [ ] README is accurate and links to docs with working relative links
- [ ] docs/ updated for any new flags/strategies/TUI changes
- [ ] CLI `--help` text is current (examples reflect latest behavior)

## Quality gate

- [ ] Lint & format
  - [ ] `ruff check .`
  - [ ] `black --check .` (or `black .` if you plan to format)
- [ ] Optional type check: `mypy` (if configured)
- [ ] Run smoke tests locally
  - [ ] `pitaya --version`
  - [ ] `pitaya "Create a HELLO.txt file with 'Hello from Pitaya' text in it and commit it"`
  - [ ] One advanced run (e.g., `--strategy best-of-n -S n=3`)

## Release & Publish

- [ ] Tag and push
  - [ ] `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] Create GitHub Release
  - [ ] Use notes from `CHANGELOG.md`
  - [ ] Attach highlights and breaking changes
- [ ] Build distribution
  - [ ] `python -m build` (requires `build`) or `hatch build`
- [ ] Verify artifacts in `dist/` install cleanly in a fresh venv
  - [ ] `pip install dist/*.whl` then `pitaya --version`
- [ ] Publish to PyPI (if applicable)
  - [ ] `twine upload dist/*` (or your release automation)

## Post‑release

- [ ] Verify `pip install pitaya` works on a clean machine
- [ ] Verify README/doc links render on GitHub
- [ ] Announce release notes in your preferred channels

Notes

- Docker images are not published from this repo by default; override images per run with `--docker-image` as needed.
- Keep `pyproject.toml` repository URLs up to date (we use `https://github.com/tact-lang/pitaya`).
