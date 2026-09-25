# Release Plan

## Current Review Release

The public repository is available at:

<https://github.com/cyx20051025-creator/residual-fdpl>

The second-review revision is tagged `v3.1-review.2`. It contains the
source-only canonical v3.1 pipeline, strict configurations, CPU tests, the
Table 3 reproduction script, and public SHA-256 manifests for checkpoints and
the 1,024-pair evaluation set.

The six final checkpoints are attached to the `v3.1-review.2` GitHub Release:

```text
rcab3_seed42_fdpl_final.pth
rcab3_seed42_nofdpl_final.pth
rcab3_seed43_fdpl_final.pth
rcab3_seed43_nofdpl_final.pth
rcab3_seed44_fdpl_final.pth
rcab3_seed44_nofdpl_final.pth
CHECKPOINT_SHA256SUMS
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

The SIDD+ evaluation images and per-image result JSON files remain outside the
public repository until redistribution terms are confirmed. The public data
manifest allows a locally obtained official copy to be verified exactly.

## Post-Acceptance Release

After acceptance:

1. Freeze the accepted source and paper artifact mapping.
2. Add the final paper DOI, journal metadata, and BibTeX information.
3. Confirm whether the SIDD evaluation bundle and per-image JSON files may be
   redistributed.
4. Publish a permanent `v3.1.0` release and archive DOI.
5. Preserve the review tag and its assets for auditability.
