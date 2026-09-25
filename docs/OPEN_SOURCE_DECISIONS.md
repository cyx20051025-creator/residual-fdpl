# Open-Source Decisions

The first public review release is now published. The current decisions are:

1. Code license: Apache-2.0.
2. Public review revision: `v3.1-review.1`.
3. Public scope: canonical v3.1 source pipeline, configurations, CPU tests,
   Table 3 reproduction tooling, and public checksum manifests.
4. Checkpoints: six final 3-RCAB weights are distributed as Release assets.
5. External baselines: NAFNet, RIDNet, SwinIR, and other baseline weights are
   not redistributed. Upstream terms remain in `THIRD_PARTY_NOTICES.md`.
6. Author email: intentionally public in `pyproject.toml`.
7. Dataset images and derived HDF5 caches: not redistributed pending upstream
   terms; the exact evaluation file names and SHA-256 digests are public.
8. Exact per-image paper result JSON files: retained in the private asset
   repository for the review release.
9. Paper title, venue metadata, final DOI, and permanent archive DOI: added
   after acceptance.
