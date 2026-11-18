# Source-free selector on released NAFNet

Protocol frozen before the new output.

## Question

Does the deployable 117-statistic Ridge input preserve the released-NAFNet
harm-control result previously obtained with the 117-plus-source Ridge map?

## Fixed factors

- Population: SIDD-CC, 928 crops from 29 scenes.
- Endpoint: released `NAFNet-SIDD-width32.pth`.
- Source rule: seed 56.
- Ten fixed 16/13-scene partitions, split seed `20261216`.
- Frozen candidates, risk thresholds, zero-luma projection and `0.02` cap.
- Source-frozen and source-scaled candidates remain available.

## Only changed factor

The capped and uncapped target-Ridge candidates use the 117 image statistics
plus intercept. The three source components are set to zero inside Ridge.

## Comparisons

- Source-free selected action and outer result.
- Existing original-input released-NAFNet result on the same partitions.
- Always-on fixed target, capped Ridge and uncapped Ridge.
- PSNR, SSIM, LPIPS and DISTS.

## Decision

No threshold or candidate may change after the outer result is observed.
The input simplification is retained only if it does not reverse the
released-NAFNet safety conclusion. Source rules 57 and 58 may subsequently be
replayed from cached source-specific summaries because the source-free Ridge
candidates are independent of the source seed; source 56 is the exact GPU
validation gate for that replay.
