# Results

This directory contains compact public summaries for the residual RGB
correction study under `rgb_correction/`. The six main TVC Table 3 comparisons
can still be regenerated from the released checkpoints and the 1,024-pair
evaluation bundle with
[`scripts/reproduce_table3.sh`](../scripts/reproduce_table3.sh).

The RGB-correction tree contains aggregate Markdown/CSV summaries and selected
split or sensitivity JSON files, not raw images or the complete per-image
analysis tree. See `rgb_correction/README.md` for its scope.

Do not commit checkpoints, datasets, HDF5 caches, full training histories, raw
logs, or third-party metric weights here.
