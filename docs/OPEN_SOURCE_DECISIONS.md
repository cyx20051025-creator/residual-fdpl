# Public Scope Decisions

The public release is now published. The current decisions are:

1. Code license: Apache-2.0.
2. Current public source revision: `v3.1-review-8`.
   The `v3.1-review` tag is retained as the asset Release.
3. Public scope: the v3.1 source pipeline, configurations, CPU tests,
   Table 3 reproduction tooling, and public checksum manifests.
4. Checkpoints: the six final 3-RCAB weights are distributed as Release assets
   under Apache-2.0 to the extent permitted by applicable law.
5. External baselines: NAFNet, RIDNet, SwinIR, and other baseline weights are
   not redistributed. Upstream terms remain in `THIRD_PARTY_NOTICES.md`.
6. Dataset images: the 1,024-pair SIDD+ validation archive is distributed
   only as a Release asset for open research and educational reproduction,
   with the complete upstream notice; derived HDF5 training caches remain
   private.
7. Historical per-image paper JSON files and full training histories are not
   distributed. The public Table 3 reproduction script regenerates the six
   comparison outputs from released assets.
8. Paper title, venue metadata, final DOI, and permanent archive DOI: added
   only when those bibliographic details become available.
