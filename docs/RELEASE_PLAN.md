# Release Plan

## Source Release

The first public GitHub release is source-only until the authors confirm code,
result, data, and weight redistribution terms.

Required before creating the public remote:

1. Keep the first public repository source-only. Paper `results/*.json` files
   are staged in the private asset repository until redistribution is approved.
2. Confirm attribution and licenses for RRDB, RCAB, VGG19, and any optional
   external baselines.
3. Add the final paper DOI and citation metadata after publication.
4. Run one clean checkout, package build, test, and smoke verification.
5. Supply the GitHub owner and create the public remote.

The author email is intentionally public in `pyproject.toml`.

The repository intentionally has no automatic GitHub Release publishing
workflow yet. Publishing before the decisions above would make irreversible
license and redistribution commitments.

The private full repository contains the six canonical checkpoints, selected
paper-result JSON files, their `SHA256SUMS`, and a 1,024-pair SIDD evaluation
bundle. The checkpoints are approved for the public `v3.1-review` Release. The
evaluation bundle remains local until dataset redistribution is confirmed.

## Optional Asset Release

If redistribution is approved, weights and large result bundles should be
uploaded as release assets, not committed to Git:

```text
residual-fdpl-<version>-weights/
|-- weights/
|-- results/
`-- SHA256SUMS
```

Each bundle must document its provenance, upstream license, checkpoint naming,
and SHA-256 manifest before publication.
