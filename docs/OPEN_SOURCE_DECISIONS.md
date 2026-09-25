# Open-Source Decisions

The first public review release is now published. The current decisions are:

1. Code license: Apache-2.0.
2. Public review revision: `v3.1-review`.
3. Public scope: canonical v3.1 source pipeline, configurations, CPU tests,
   Table 3 reproduction tooling, and public checksum manifests.
4. Checkpoints: the six final 3-RCAB weights are distributed as Release assets
   under Apache-2.0 to the extent permitted by applicable law.
5. External baselines: NAFNet, RIDNet, SwinIR, and other baseline weights are
   not redistributed. Upstream terms remain in `THIRD_PARTY_NOTICES.md`.
6. Author email: intentionally public in `pyproject.toml`.
7. Dataset images: the 1,024-pair SIDD+ validation archive is distributed
   only as a Release asset for open research and educational reproduction,
   with the complete upstream notice; derived HDF5 training caches remain
   private.
8. Exact per-image paper result JSON files: retained in the private asset
   repository for the review release.
9. Paper title, venue metadata, final DOI, and permanent archive DOI: added
   after acceptance.
