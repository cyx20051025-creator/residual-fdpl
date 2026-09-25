# Model Zoo

## Core Assets

The six final checkpoints are available in the public `v3.1-review.3` GitHub
Release. Large data and per-image result files remain separate:

| Asset | Status | Planned storage |
|---|---|---|
| 3-RCAB seed 42 final weights | available | `v3.1-review.3` Release asset |
| 3-RCAB seed 43 final weights | available | `v3.1-review.3` Release asset |
| 3-RCAB seed 44 final weights | available | `v3.1-review.3` Release asset |
| Matched no-RFDPL final weights | available | `v3.1-review.3` Release asset |
| 1,024-pair SIDD+ validation bundle | available under upstream research/education notice | `v3.1-review.3` Release asset |
| Selected paired-result JSON files | private review reference | Full repository or later Release asset |
| Full training histories and logs | pending | Release asset or external archive |

No checkpoint is committed to ordinary Git history. Each Release asset is
listed in `CHECKPOINT_SHA256SUMS`, distributed with the Release.

## Naming

The public bundle will use the following pattern:

```text
residual-fdpl-v0.1.0-weights/
|-- weights/
|-- results/
`-- CHECKPOINT_SHA256SUMS
```

## External Baselines

RIDNet, SwinIR, NAFNet, and other external model implementations must retain
their upstream attribution and license notices. Their pretrained weights are
not assumed to be redistributable.
