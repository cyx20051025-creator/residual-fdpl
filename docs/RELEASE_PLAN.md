# Release Plan

## Source Release

The first public GitHub release is source-only until the authors confirm code,
result, data, and weight redistribution terms.

Required before tagging:

1. Select and add a code license.
2. Keep the first public repository source-only. Paper `results/*.json` files
   are staged in the private asset repository until redistribution is approved.
3. Confirm attribution and licenses for RRDB, RCAB, VGG19, and any optional
   external baselines.
4. Add final `CITATION.cff` metadata after the paper identifiers are known.
5. Run one clean checkout, package build, test, and smoke verification.

The author email is intentionally public in `pyproject.toml`.

The repository intentionally has no automatic GitHub Release publishing
workflow yet. Publishing before the decisions above would make irreversible
license and redistribution commitments.

The private asset staging repository is prepared separately. It contains the
six canonical checkpoints, selected paper-result JSON files, their
`SHA256SUMS`, and a 1,024-pair SIDD evaluation bundle. It remains local until
the GitHub owner is supplied and a private remote can be created.

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
