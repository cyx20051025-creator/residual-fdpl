# Model Zoo

## Core Assets

The following assets are staged in the companion full repository. The six
final checkpoints are approved for the public `v3.1-review` GitHub Release.
Large data and per-image result files remain separate:

| Asset | Status | Planned storage |
|---|---|---|
| 3-RCAB seed 42 final weights | approved | `v3.1-review` Release asset |
| 3-RCAB seed 43 final weights | approved | `v3.1-review` Release asset |
| 3-RCAB seed 44 final weights | approved | `v3.1-review` Release asset |
| Matched no-RFDPL final weights | approved | `v3.1-review` Release asset |
| 1,024-pair SIDD validation bundle | pending dataset terms | Later Release asset |
| Selected paired-result JSON files | private review reference | Full repository or later Release asset |
| Full training histories and logs | pending | Release asset or external archive |

No checkpoint is committed to ordinary Git history. Each Release asset is
listed in `SHA256SUMS` in the companion full repository.

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
