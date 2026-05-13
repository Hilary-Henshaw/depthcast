# Security Policy

## Supported Versions

DepthCast is pre-1.0. Security fixes are applied to the latest released
minor version on the `main` branch.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a Vulnerability

Please report suspected vulnerabilities privately rather than opening a
public issue. Email **henshaw.hilary@gmail.com** with:

- a description of the issue and its impact,
- steps to reproduce (the synthetic data generator is ideal for a
  minimal reproduction), and
- any suggested remediation.

You can expect an acknowledgement within three business days and a
resolution plan once the report is triaged. We will credit reporters in
the release notes unless you ask us not to.

## Handling of Untrusted Inputs

DepthCast loads model checkpoints with PyTorch's `weights_only=True` path
and reconstructs the architecture from a JSON configuration, so loading a
checkpoint never executes arbitrary pickled code. Dataset files are
parsed as plain numeric matrices. Even so, only load checkpoints and data
from sources you trust.
