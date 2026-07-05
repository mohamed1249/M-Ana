# Security Policy

## Supported versions

Security fixes are provided for the latest M-Ana minor release. Users should
upgrade to the newest available release before reporting a problem.

## Reporting a vulnerability

Please use the repository's private GitHub security advisory form. Do not open a
public issue for an unpatched vulnerability and do not include live credentials,
private datasets, or production connection strings in a report.

Include the affected version, a minimal reproduction, the expected impact, and
any known mitigation. Acknowledgement should arrive within seven days; release
timing depends on severity and the complexity of a safe fix.

## Credential safety

M-Ana adapters accept credentials through arguments or environment variables.
The repository must never contain real API keys. Any credential found in an old
notebook or commit should be revoked rather than merely deleted from the latest
revision.
