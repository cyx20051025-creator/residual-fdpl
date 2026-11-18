# Nine-way source-free selector replay

Status: deterministic replay using already completed GPU records.

## Invariant

Once the source columns are absent from the target-Ridge design, the fitted
target-Ridge candidates depend on the endpoint, crops, targets and split, but
not on the source-predictor seed. Source-only and scaled-source candidates do
depend on that seed.

## Procedure

For each of the three fine-tuned NAFNet endpoints:

1. Load the actual source-free GPU record for source rule 56 and all ten
   splits.
2. Load the original selector record for source rules 56, 57 and 58.
3. Verify identical calibration and evaluation scene lists.
4. Keep source-only, scaled-source, fixed-target and no-correction summaries
   from the source-specific original record.
5. Replace only the capped and uncapped target-Ridge summaries with the
   source-free summaries for the same endpoint and split.
6. Reapply the frozen candidate-eligibility and action-selection function.
7. Select the recorded outer result for the resulting action.

## Validation gate

For source 56, the replay must match the actual source-free GPU output in
every split, including selected action and outer mean PSNR. The script stops
without writing the nine-way conclusion if any source-56 comparison fails.
The script also verifies that the no-correction and fixed-target summaries,
which do not depend on the source rule, are identical across sources 56/57/58
for every endpoint and split.

## Interpretation boundary

This replay closes the mismatch between the simplified Ridge map and the old
three-source-rule report. It is exact candidate-summary reuse, not nine new
GPU executions and not nine independent populations. All applications reuse
the same 29 scenes and ten shared partitions.
