# Releasing M-Ana

1. Update `MAna/_version.py` and `CHANGELOG.txt`.
2. Run lint, tests, and any relevant optional integration tests.
3. Remove old artifacts and run `python -m build`.
4. Run `twine check dist/*` and inspect the wheel contents and metadata.
5. Commit the release, create a signed `vX.Y.Z` tag, and push the tag.
6. The release workflow creates a GitHub Release and attaches the wheel and
   source distribution.

The version in `MAna/_version.py` is the single source of truth. Do not add a
second literal version to packaging configuration.
