# Third-Party Notices

The Apache-2.0 license in this repository applies to original project code and
the six author-created 3-RCAB checkpoint files distributed in the review
Release, to the extent permitted by applicable law. Dependencies, datasets,
external pretrained weights, and architectural references remain subject to
their own terms.

## Runtime Dependencies

| Component | Use | Upstream terms |
|---|---|---|
| PyTorch | Tensor operations, training, and inference | BSD-3-Clause |
| torchvision | VGG19 model and pretrained weights | BSD-3-Clause |
| NumPy | Numerical and array support | BSD-3-Clause |
| Pillow | PNG input and output | HPND |
| h5py | HDF5 data access | BSD-3-Clause |

The exact dependency versions used for validation are recorded in
`requirements/validated.txt`; `docs/REPRODUCIBILITY.md` summarizes the
environment.

## Architectural References

The implementation includes a compact residual-in-residual dense block (RRDB)
and residual channel attention block (RCAB). These components follow the
published ESRGAN and RCAN architectures. The repository does not redistribute
the original upstream projects or their pretrained weights. Users should cite
the corresponding papers when those components are used in derivative work.

The VGG perceptual loss uses the torchvision VGG19 implementation and its
available ImageNet-pretrained weights. VGG19 and the ImageNet dataset are not
redistributed by this repository.

## SIDD and Evaluation Data

The repository source tree and Git history do not distribute the complete
SIDD, Urban100, or derived HDF5 caches. The `v3.1-review` Release optionally
provides the 1,024-pair SIDD+ validation subset used for paper evaluation.
Its source and notice are recorded in `docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`.
The data are for open research and educational purposes, remain subject to the
upstream challenge terms, and are not relicensed under Apache-2.0.

## External Baselines

NAFNet, RIDNet, SwinIR, and other external baseline implementations are not
part of the first public source-only release. If they are added later, their
source attribution, upstream license text, and pretrained-weight terms must be
reviewed before redistribution.
