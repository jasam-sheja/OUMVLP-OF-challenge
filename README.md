# OUMVLP-OF-challenge
This is the official starter kit for the [OUMVLP-OF challenge](https://of.iapr-tc4.org/).

![](https://miniodis-rproxy.lisn.upsaclay.fr/coda-v2-prod-public/logos/2025-02-27-1740631823/57db00e2427b/key_fig.png)


- [OUMVLP-OF-challenge](#oumvlp-of-challenge)
  - [Dataset Preparation](#dataset-preparation)
    - [Optical Flow Generation](#optical-flow-generation): Use this to generate your own optical flow data from RGB capture of subjects and their binary segmentation (silhouette).
    - [Test Dataset Preparation](#test-dataset-preparation): Use this to prepare the preprocessed dataset for the challenge.
  - [Optical Flow Data](#optical-flow-data): Use this to read the video files of optical flow videos.
  - [Submission](#submission)
    - [Submission Preparation](#submission-preparation): Use this to save the submission files (recommended).
    - [In Case of Manual Submission](#in-case-of-manual-submission): Use this to create the submission files manually.

## Dataset Preparation
To train your model you need to prepare a dataset with optical flow data. This competition **does not** provide any training samples. You can use any external dataset, such as CASIA-B and GREW (Gait-in-Wild), etc., to generate optical flow maps and train your models.

The following sections describe how to generate the optical flow data from your rgb and silhouette videos and how to prepare the test dataset.

### Optical Flow Generation
To generate your own training optical flow data from RGB capture of subjects and their binary segmentation (silhouette), you can use the following functions in `compute_OF_RAFT.py`:

```python
import compute_OF_RAFT as raft


raft.apply_video_files # to apply optical flow on video files
# Args:
#   - rgb_file: path to the RGB video file
#   - silhouette_file: path to the silhouette video file
#   - model: model to use for optical flow computation (default: raft)
# Returns:
#   - flow: optical flow images as numpy array

raft.apply_image_folders # to apply optical flow on image folders
# Args:
#   - rgb_folder: path to the RGB image folder
#   - silhouette_folder: path to the silhouette image folder
#   - model: model to use for optical flow computation (default: raft)
# Returns:
#   - flow: optical flow images as numpy array
```

To save the optical flow data, you can use the following function:

```python
import data_io
data_io.write_vid(flow, output_file)
```

To try the code on your own data, you run the following command:

```bash
python compute_OF_RAFT.py --video_file <path-to-rgb-video-file> --silhouette_file <path-to-sil-video-file> --output_file <path-to-output-optical-flow-video-file>
```
Please note that appying the above command is not recommended for large datasets. It is recommended to use the above functions in your own dataset preparation code.

**Note**: Please make sure that both the input rgbs and silhouettes are synchronized and have the same number of frames.

### Test Dataset Preparation
No training samples will be provided. The participants can use any external dataset, such as CASIA-B and GREW (Gait-in-Wild), etc., to generate optical flow maps and train their models.

After you get permission, you can download the dataset. Then, you can use the following code to extract the data.

```bash
git clone https://github.com/jasam-sheja/OUMVLP-OF-challenge.git
cd OUMVLP_OF-challenge
# move the downloaded zip files to the root directory
# Gallery: Use {password} sent to you via email
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-gallery.zip
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-gallery.zip
# Phase 1: Use {password} sent to you via email
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-probe-phase-1.zip
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-probe-phase-1.zip
unzip phase_1_meta.zip
# Phase 2: Use {password} sent to you via email
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-probe-phase-2.zip
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-probe-phase-2.zip
unzip phase_2_meta.zip
```

The dataset will have the following structure:
```
OUMVLP_OF_V1_IJCB2025OFcompetition-data-gallery
    | - data-gallery
        | - seq_0001.mp4        (Optical flow video in RGB format)
        | - seq_0002.mp4        (Optical flow video in RGB format)
        | - ...
    | - data-probe-phase-[1|2]
        | - seq_0001.mp4        (Optical flow video in RGB format)
        | - seq_0002.mp4        (Optical flow video in RGB format)
        | - ...
OUMVLP_OF_V2_IJCB2025OFcompetition-data-gallery
    | - data-gallery
        | - seq_0001.mp4        (Optical flow video in RGB format)
        | - seq_0002.mp4        (Optical flow video in RGB format)
        | - ...
    | - data-probe-phase-[1|2]
        | - seq_0001.mp4        (Optical flow video in RGB format)
        | - seq_0002.mp4        (Optical flow video in RGB format)
        | - ...
meta
    | - phase[1|2]_gallery_verification.json    (Distance matrix column index and its corresponding gallery file)
    | - probe[1|2]_probe_verification.json      (Distance matrix row index and its corresponding probe file)
    | - probe[1|2]_index_mapping.json           (Probe and gallery file to integer mapping)
```

## Optical Flow Data
To read the video files of optical flow maps, you can use this function `io.read_vid`. It'll return a numpy uint8 array of shape `FxHxWx3` where F is the number of frames.


## Submission

There are 4 files to submit:
1. ranking-v1.npz: The matches file for optical flow version 1.
2. ranking-v2.npz: The matches file for optical flow version 2.
3. dist-v1.npz: The distance file for optical flow version 1.
4. dist-v2.npz: The distance file for optical flow version 2.

When creating the submission zip file it should be so when extracted the files are in the root directory.

**Note**: For a valid submssion either both files of version 1, both files of version 2 should be submitted, or all four files should be submitted.

### Submission Preparation
To prepare your submission, we provide the following helper functions to save the files. 

General use of the helper functions:
```python
# unpack the meta files into meta directory before running this code
import data_io

# identification 
matches = {'probe1': ['gallery1', 'gallery2', ..., 'gallery14'],
           'probe2': ['gallery1', 'gallery2', ..., 'gallery14'], ...}
data_io.save_identification_submission(matches, 'path/to/submission', phase='phase1', version='v1')

# verification
distance_builder = VerificationSubmission(phase='phase1')
distance_builder.set_distance('probe1', 'gallery1', 0.5)
distance_builder.set_distance('probe1', 'gallery2', 0.6)
distance_builder.set_distance('probe2', 'gallery1', 0.7)
distance_builder.set_distance('probe2', 'gallery2', 0.8)
...
distance_builder.save('path/to/submission', version='v1')
```
**Note**: The submission files should be complete and the provided functions checks for the completeness of the submission.

Example of a dummy submission:
```python
import os
import random

import data_io # helper functions


phase = "phase1"  # phase1 or phase2
submission_folder = "submission"  # output directory
os.makedirs(submission_folder, exist_ok=False)
for version in ["v1", "v2"]:
    # identification submission
    gallery, gallery_file = zip(
        *data_io.get_identification_gallery_files(phase=phase, version=version)
    )
    matches = {} # probe matches
    for probe, probe_file in data_io.get_identification_probe_files(
        phase=phase, version=version
    ):
        matches[probe] = [random.choice(gallery) for _ in range(14)] # replace with your own model output
    data_io.save_identification_submission(
        matches, output_dir=submission_folder, phase=phase, version=version
    )
    # verification submission
    vsubmission = data_io.VerificationSubmission(phase=phase)
    for probe, _ in data_io.get_verification_probe_files(
        phase=phase, version=version
    ):
        for gallery, _ in data_io.get_verification_gallery_files(
            phase=phase, version=version
        ):
            vsubmission.set_distance(
                probe=probe,
                gallery=gallery,
                distance=random.uniform(0, 1), # replace with your own model output
            )
    vsubmission.save(
        output_dir=submission_folder,
        version=version,
    )
os.system(f"zip -j {submission_folder}.zip {submission_folder}/*")
# then submit submission.zip to the competition
```

### In Case of Manual Submission
If you decided to create the submission files manually, you must follow the following format:

#### **ranking-v1.npz** / **ranking-v2.npz**:
- Must be numpy compressed file with `ranking` as file label, i.e., `np.savez_compressed('ranking-v1.npz', ranking=mat)`.
- The matrix `mat` should be of shape `Nx14` where N is the number of probes in `meta/{phase}_index_mapping.json[probe]` and the columns are the top 14 gallery matches for each probe.
- The order of probe is in `meta/{phase}_index_mapping.json[probe]` and the order of gallery is in `meta/{phase}_index_mapping.json[gallery]`.

#### **dist-v1.npz** / **dist-v2.npz**:
- Must be numpy compressed file with `dist` as file label, i.e., `np.savez_compressed('dist-v1.npz', dist=mat)`.
- The matrix `mat` should be of shape `NxM` where N is the number of probes in `meta/{phase}_probe_verification.json` and M is the number of galleries in `meta/{phase}_gallery_verification.json`.
- The matrix should contain the distance between each probe-gallery pair. Small distance means high similarity.
- The order of probe is the same in `meta/{phase}_probe_verification.json` and the order of gallery is the same in `meta/{phase}_gallery_verification.json`.
