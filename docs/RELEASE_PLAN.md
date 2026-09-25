# Release Plan

## Current Review Release

The public repository is available at:

<https://github.com/cyx20051025-creator/residual-fdpl>

The second-review revision is tagged `v3.1-review.3`. It contains the
canonical v3.1 pipeline, strict configurations, CPU tests, the Table 3
reproduction script, and public SHA-256 manifests for checkpoints and the
1,024-pair evaluation set.

The following assets are attached to the `v3.1-review.3` GitHub Release:

```text
rcab3_seed42_fdpl_final.pth
rcab3_seed42_nofdpl_final.pth
rcab3_seed43_fdpl_final.pth
rcab3_seed43_nofdpl_final.pth
rcab3_seed44_fdpl_final.pth
rcab3_seed44_nofdpl_final.pth
CHECKPOINT_SHA256SUMS
SIDD_QUICK_EVAL_SHA256SUMS
SIDD_QUICK_EVAL_ZIP_SHA256SUMS
sidd_quick_eval_256.zip
```

| Table 3 row | Protocol | Seed | Reported gain |
|---|---|---|---|
| 1 | sliding, 128-tile / 32-overlap / stride-96 | 42 | `+0.6475` dB |
| 2 | sliding, 128-tile / 32-overlap / stride-96 | 43 | `+0.5985` dB |
| 3 | sliding, 128-tile / 32-overlap / stride-96 | 44 | `+0.4101` dB |
| 4 | direct, 256 x 256 | 42 | `+0.5995` dB |
| 5 | direct, 256 x 256 | 43 | `+0.6017` dB |

The two assets for each row and their reported gains are mapped explicitly in
`README.md` and `docs/PAPER_ARTIFACTS.md`. Seed 44 has no direct row.

The optional SIDD+ evaluation archive is provided for open research and
educational reproduction under the upstream notice in
`docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`. It is not included in ordinary Git
history and is not relicensed under Apache-2.0. Per-image result JSON files
remain in the private review reference.

## Post-Acceptance Release

After acceptance:

1. Freeze the accepted source and paper artifact mapping.
2. Add the final paper DOI, journal metadata, and BibTeX information.
3. Preserve the SIDD+ evaluation notice with any permanent release and decide
   whether the per-image JSON files should be added.
4. Publish a permanent `v3.1.0` release and archive DOI.
5. Preserve the review tag and its assets for auditability.
