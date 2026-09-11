# Contributing

Contributions are welcome for review, but no license is currently granted. Opening a pull request does not change ownership or grant redistribution rights.

Before submitting a change:

1. Keep credentials, private paths, run data, databases, browser state, and generated artifacts out of the tree.
2. Run `python -m pytest` in a Python 3.11+ virtual environment.
3. Run `git diff --check` and inspect `git status`.
4. Document user-visible behavior and preserve the loopback-only, tool-free assistant boundary.