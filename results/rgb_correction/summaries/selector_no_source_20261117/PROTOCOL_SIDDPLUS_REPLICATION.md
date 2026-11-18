# SIDD+ source-free selector replication

Protocol frozen: 2026-11-17, before running endpoint seed 63.

## Purpose

The source-free selector passes the three-seed SIDD-CC comparison. This addendum
tests whether removing the source columns preserves the selector on the SIDD+
boundary population for a second independently trained endpoint.

## Fixed factors

- Population: `sidd_valid_1024`.
- Endpoint: project fine-tuned NAFNet seed 63.
- Source predictor: seed 56.
- Ten image-level leaves with 16 calibration and 1008 evaluation images.
- Split seed: `20261127`.
- Candidate set, thresholds, projection and `0.02` cap are unchanged.

The seed-63 feature and source-offset files already exist and were not changed
using selector results.

## Only changed factor

The source columns are removed from the fitted capped and uncapped Ridge
candidates. Source-frozen, source-scaled and fixed-target candidates remain
unchanged.

## Decision rule

Compare the source-free selector with the original 16-image SIDD+ selector on
the same ten partitions. Report actions, selected metrics, positive-split rate
and the selector-minus-fixed-target gap. Do not reinterpret SIDD+ as
scene-level validation.
