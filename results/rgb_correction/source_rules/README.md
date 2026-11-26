# Fitted Source-Rule Artifacts

This directory contains the three fitted source-domain models used for source
seeds 56, 57, and 58 in the residual RGB-correction study. The artifacts are
derived numerical parameters produced by this project; they contain no source
images, target images, denoiser checkpoints, or third-party model weights.

Each `.pkl` file is a dictionary with:

- `schema_version` and `feature_schema`;
- the ordered 117-feature name list;
- five fitted ensemble members;
- the selected scale and aggregation rule;
- the source-domain selection record; and
- SHA-256 identifiers for the upstream source-feature and screen files.

Load an artifact with Python 3.11, NumPy, and scikit-learn 1.5.x:

```python
import pickle

with open("v3_common_median_rgb_seed56.pkl", "rb") as handle:
    source_rule = pickle.load(handle)
```

Only load pickle files whose SHA-256 digest matches `SHA256SUMS`. The exact
evaluation data, target feature caches, denoiser checkpoints, and metric
weights remain subject to their providers' terms and are not redistributed.
