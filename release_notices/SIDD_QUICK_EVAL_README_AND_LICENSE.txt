SIDD+ 1,024-Pair Quick-Evaluation Archive
=========================================

Purpose
-------

This archive contains 1,024 noisy/ground-truth image pairs used to reproduce
the evaluation reported by the Residual FDPL paper. It is provided only to
reduce the work required for open research and educational reproduction.

Archive Layout
--------------

siddplus_valid_noisy_srgb/
  Matching filenames 0000.PNG through 1023.PNG

siddplus_valid_gt_srgb/
  Matching filenames 0000.PNG through 1023.PNG

Every image is a 256 x 256 RGB PNG. There are 2,048 image files forming 1,024
paired samples. No augmentation was applied when preparing this evaluation
archive.

Upstream Source
---------------

SIDD project:
  https://abdokamel.github.io/sidd/

NTIRE 2020 Real Image Denoising Challenge (SIDD+), Track 1:
  https://competitions.codalab.org/competitions/22230

NTIRE 2020 Real Image Denoising Challenge (SIDD+), Track 2:
  https://competitions.codalab.org/competitions/22231

The SIDD project page states:

  "The dataset and the associated code repositories are under the MIT
  License."

The data were also distributed through the NTIRE 2020 SIDD+ challenge. The
following challenge notice is retained for this distributed evaluation subset.

Upstream Challenge Notice
-------------------------

All data provided by NTIRE are freely available to the participants from the
website of the challenge under license terms provided with the data. The data
are available only for open research and educational purposes, within the
scope of the challenge. NTIRE and the organizers make no warranties regarding
the database, including but not limited to warranties of non-infringement or
fitness for a particular purpose. The copyright of the images remains in
property of their respective owners. By downloading and making use of the
data, you accept full responsibility for using the data. You shall defend and
indemnify NTIRE and the organizers, including their employees, Trustees,
officers and agents, against any and all claims arising from your use of the
data. You agree not to redistribute the data without this notice.

Redistribution and Use
----------------------

- Keep this notice with the archive when it is redistributed.
- Do not relicense the images under Apache-2.0, MIT, or a project license.
- Use the images only for open research and educational purposes.
- The image copyright remains with the respective owners.
- Cite SIDD and the NTIRE 2020 Real Image Denoising Challenge when using the
  images.

This archive does not contain the complete SIDD training data, raw/DNG source
files, test ground truth, or HDF5 caches. The files preserve the evaluation
PNGs used with the released checkpoints; any crop, color conversion, or
resampling already present in the upstream data remains part of its provenance.

Citation
--------

Abdelrahman Abdelhamed, Stephen Lin, and Michael S. Brown. "A High-Quality
Denoising Dataset for Smartphone Cameras." CVPR, 2018.

Abdelrahman Abdelhamed, Mahmoud Afifi, Radu Timofte, Michael S. Brown, et al.
"NTIRE 2020 Challenge on Real Image Denoising: Dataset, Methods and Results."
CVPR Workshops, 2020.
