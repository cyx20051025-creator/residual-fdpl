# Released-NAFNet calibration-density boundary

Frozen: 2026-11-17, after the 16-scene source-free run was launched and before
the 4- and 8-scene outputs were produced.

## Question

Is the conservative released-NAFNet result at the primary 16-scene budget a
sign that the endpoint cannot be corrected, or is it consistent with
insufficient calibration support?

## Fixed factors

- Population: SIDD-CC, 928 crops from 29 scenes.
- Endpoint: released `NAFNet-SIDD-width32.pth`.
- Source rule: seed 56.
- Ten fixed partitions with split seed `20261216`.
- Frozen candidates, risk thresholds, zero-luma projection and `0.02` cap.
- Ridge input: 117 image statistics plus intercept.

## Added conditions

- Four calibration scenes, leaving 25 outer evaluation scenes.
- Eight calibration scenes, leaving 21 outer evaluation scenes.
- Twelve calibration scenes, leaving 17 outer evaluation scenes.

The existing 16-scene result is retained as the primary comparison. The
smaller-budget conditions are exploratory boundary diagnostics, not new
operating points. No threshold or candidate will change after their outputs
are observed.

## Interpretation

- A monotone improvement from 4 to 8 to 12 to 16 scenes is consistent with a
  calibration-support boundary.
- A flat or reversed curve means that the weak result is better described as
  endpoint/population mismatch than as sample scarcity.
- In either case, repeated partitions reuse the same 29 scenes and do not
  constitute independent populations.
