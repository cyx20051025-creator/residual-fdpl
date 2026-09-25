# SIDD+ Evaluation Data Notice

This notice applies to the optional 1,024-pair SIDD+ evaluation archive
distributed with the `v3.1-review.4` GitHub Release. It does not apply to the
project source code or the author-created checkpoints, which are licensed
separately under Apache-2.0.

## Upstream Source

The evaluation bundle is derived from the SIDD/SIDD+ validation data associated
with:

- SIDD project: <https://abdokamel.github.io/sidd/>
- NTIRE 2020 Real Image Denoising Challenge (SIDD+), Track 1:
  <https://competitions.codalab.org/competitions/22230>
- NTIRE 2020 Real Image Denoising Challenge (SIDD+), Track 2:
  <https://competitions.codalab.org/competitions/22231>

The SIDD project page states:

> The dataset and the associated code repositories are under the MIT License.

The data were also distributed through the NTIRE 2020 SIDD+ challenge. The
challenge terms are more specific about challenge data and are reproduced below
for the distributed evaluation subset.

## Upstream Challenge Terms

The following paragraph is reproduced from the NTIRE 2020 Real Image Denoising
Challenge terms and conditions:

> All data provided by NTIRE are freely available to the participants from the
> website of the challenge under license terms provided with the data. The data
> are available only for open research and educational purposes, within the
> scope of the challenge. NTIRE and the organizers make no warranties regarding
> the database, including but not limited to warranties of non-infringement or
> fitness for a particular purpose. The copyright of the images remains in
> property of their respective owners. By downloading and making use of the
> data, you accept full responsibility for using the data. You shall defend and
> indemnify NTIRE and the organizers, including their employees, Trustees,
> officers and agents, against any and all claims arising from your use of the
> data. You agree not to redistribute the data without this notice.

The terms also state that the training and validation data are made available
to challenge participants, while test ground truth is held by the organizers.

## Distributed Subset

The Release archive contains:

```text
siddplus_valid_noisy_srgb/
siddplus_valid_gt_srgb/
```

Each directory contains matching filenames `0000.PNG` through `1023.PNG`.
Every file in this preparation is a 256 x 256 RGB PNG. The archive contains
1,024 noisy/ground-truth pairs in total and is intended solely to reduce the
work required for research and educational reproduction of the paper.

This repository does not distribute the complete SIDD training data, raw/DNG
source files, test ground truth, or derived HDF5 caches. No augmentation was
applied when preparing this evaluation archive. The PNG files preserve the
evaluation files used by the released checkpoints; any crop, color conversion,
or resampling already present in the upstream data remains part of its
provenance.

## Redistribution Conditions

- Retain this notice and the upstream attribution with the archive.
- Do not relicense the images under Apache-2.0, MIT, or any other project
  license merely because they accompany this repository.
- Use the data only for open research and educational purposes within the
  scope described by the upstream terms.
- The copyright of the images remains with the respective owners.
- Cite the SIDD and NTIRE challenge publications when using the data.

## Citation

```bibtex
@inproceedings{Abdelhamed_2018_CVPR,
  author    = {Abdelhamed, Abdelrahman and Lin, Stephen and Brown, Michael S.},
  title     = {A High-Quality Denoising Dataset for Smartphone Cameras},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2018}
}

@inproceedings{Abdelhamed_2020_CVPRW,
  author    = {Abdelhamed, Abdelrahman and Afifi, Mahmoud and Timofte, Radu and Brown, Michael S. and others},
  title     = {NTIRE 2020 Challenge on Real Image Denoising: Dataset, Methods and Results},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2020}
}
```
