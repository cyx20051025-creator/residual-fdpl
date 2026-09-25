# Assets

This directory is reserved for the public download manifest. Checkpoint
download links point to the `v3.1-review.2` GitHub Release.

No dataset, checkpoint, HDF5 cache, or generated figure is bundled in the
public source repository. The six final checkpoints are distributed in the
review Release. Dataset images, quick-evaluation archives, and per-image
results remain outside normal Git history pending upstream terms.

The prepared private asset bundle contains:

```text
checkpoints/
|-- rcab3_seed42_fdpl_final.pth
|-- rcab3_seed42_nofdpl_final.pth
|-- rcab3_seed43_fdpl_final.pth
|-- rcab3_seed43_nofdpl_final.pth
|-- rcab3_seed44_fdpl_final.pth
`-- rcab3_seed44_nofdpl_final.pth

results/raw/
|-- selected paper, mechanism, and paired-significance JSON files
`-- table-to-source mapping in results/paper/README.md

release_bundles/
`-- sidd_quick_eval_256.zip
```

The quick-evaluation bundle contains 1,024 paired SIDD validation images and
expands to approximately 185 MB. A public
`docs/SIDD_QUICK_EVAL_SHA256SUMS` manifest records the exact expected files
without redistributing the images.
