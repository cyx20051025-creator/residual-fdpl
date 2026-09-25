# Model Zoo

## Core Assets

The following assets are staged in a separate private repository. They can be
attached to a GitHub Release after redistribution permission is confirmed:

| Asset | Status | Planned storage |
|---|---|---|
| 3-RCAB seed 42 final weights | staged private | Release asset |
| 3-RCAB seed 43 final weights | staged private | Release asset |
| 3-RCAB seed 44 final weights | staged private | Release asset |
| Matched no-RFDPL final weights | staged private | Release asset |
| 1,024-pair SIDD validation bundle | staged private | Release asset |
| Selected paired-result JSON files | staged private | Private repo or Release asset |
| Full training histories and logs | pending | Release asset or external archive |

No checkpoint is committed to the repository.

## Naming

The public bundle will use the following pattern:

```text
residual-fdpl-v0.1.0-weights/
|-- weights/
|-- results/
`-- SHA256SUMS
```

## External Baselines

RIDNet, SwinIR, NAFNet, and other external model implementations must retain
their upstream attribution and license notices. Their pretrained weights are
not assumed to be redistributable.
