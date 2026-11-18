# Post-core replication protocol

Frozen: 2026-11-17, before the third-endpoint and alternate-split runs
started.

## Purpose

The two-seed primary sweep established a calibration-density recovery on
SIDD+. This post-core block tests whether the simplified 117-feature-only
selector reproduces the result on a third fine-tuned NAFNet checkpoint and
whether the curve is stable under a second split seed.

## Shared settings

- Population: SIDD+ 1,024 image-level units.
- Candidate family, risk thresholds, `0.02` cap, source candidate and
  bootstrap settings remain frozen.
- Ten leaves per budget.
- Budgets: 16, 64, 128, 256 and 512 calibration images.
- Ridge input: 117 image-statistics features plus intercept.

## Replication A: third endpoint

- Endpoint: fine-tuned NAFNet seed 64.
- Split seed: `20261127`, matching the primary two-seed sweep.
- Output: `selector_no_source_budget_20261117/seed64`.

## Replication B: alternate split

- Endpoints: fine-tuned NAFNet seeds 62 and 63.
- Split seed: `20261227`.
- Output: `selector_no_source_alt_split_20261227/seed62` and
  `.../seed63`.

## Decision criteria

The experiment block is strong enough to stop if:

1. The primary source-free two-seed sweep selects the same action as the
   original selector in essentially all splits.
2. The selector-minus-fixed trend remains monotone across the five budgets.
3. The 512-image selector remains positive against the fixed target in both
   primary endpoints.
4. The third endpoint follows the same qualitative density recovery.
5. The alternate split does not reverse the direction of the 256/512 result.

Failure of criteria 4 or 5 is reported as endpoint or split sensitivity; the
protocol will not be changed after observing outcomes. Repeated splits reuse
the same images and are not independent populations.
