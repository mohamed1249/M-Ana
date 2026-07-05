# Versioning and compatibility

M-Ana uses semantic versioning:

- Patch releases fix compatible behavior and documentation.
- Minor releases add backward-compatible APIs and may add optional dependencies.
- Major releases may remove deprecated APIs or change public contracts.

The public API is the documented set of names exported from package namespaces.
Private names beginning with `_`, test helpers, and undocumented implementation
details are not compatibility promises.
