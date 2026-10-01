# Clean-Clone Audit

Date: 2026-10-01

## Source Revision

- Commit under audit:
  `f2de176c6da1b832d493b28a40418c408311b433`
- Commit subject: `Include direct seed-44 Table 3 comparison`

The audit cloned this committed revision into a new directory and ran the
tests with only the cloned source tree:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
33 passed
```

## Table 3 Coverage

The cloned revision exposes six rows:

```text
(1, sliding, 42, 0.6475)
(2, sliding, 43, 0.5985)
(3, sliding, 44, 0.4101)
(4, direct, 42, 0.5995)
(5, direct, 43, 0.6017)
(6, direct, 44, 0.3955)
```

The direct seed-44 row uses the released pair:

- `rcab3_seed44_nofdpl_final.pth`
- `rcab3_seed44_fdpl_final.pth`

## Direct Seed-44 Reference Result

The completed 1,024-pair reference comparison is:

`paired_component_seed44_direct.json`

SHA-256:

`301c00e3c215a337e20df964b9117e85215a56c74f1c5b50c28ae5177428b751`

Reported PSNR delta:

`+0.3954527690075338 dB`

The reference result is private review evidence. The public repository
contains the checkpoint mapping and reproduction script required to perform
the same comparison.

