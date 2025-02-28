# OUMVLP-OF-challenge
This is the official starter kit for the [OUMVLP-OF challenge](https://of.iapr-tc4.org/).

## Test Dataset Preparation
No training samples will be provided. The participants can use any external dataset, such as CASIA-B and GREW (Gait-in-Wild), etc., to generate optical flow maps and train their models.

After you get permission, you can download the dataset. Then, you can use the following code to extract the data.

```bash
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-gallery.zip
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-gallery.zip
mkdir meta
# Phase 1
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-probe-phase-1.zip
mv OUMVLP_OF_V1_IJCB2025OFcompetition/data-probe-phase-1 OUMVLP_OF_V1_IJCB2025OFcompetition/data-probe
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-probe-phase-1.zip
mv OUMVLP_OF_V2_IJCB2025OFcompetition/data-probe-phase-1 OUMVLP_OF_V2_IJCB2025OFcompetition/data-probe
unzip phase_1_meta.zip
mv phase1_gallery_verification.json meta/gallery_verification.json
mv phase1_probe_verification.json meta/probe_verification.json
# Phase 2
unzip OUMVLP_OF_V1_IJCB2025OFcompetition-data-probe-phase-2.zip
mv OUMVLP_OF_V1_IJCB2025OFcompetition/data-probe-phase-2 OUMVLP_OF_V1_IJCB2025OFcompetition/data-probe
unzip OUMVLP_OF_V2_IJCB2025OFcompetition-data-probe-phase-2.zip
mv OUMVLP_OF_V2_IJCB2025OFcompetition/data-probe-phase-2 OUMVLP_OF_V2_IJCB2025OFcompetition/data-probe
unzip phase_2_meta.zip
mv phase2_gallery_verification.json meta/gallery_verification.json
mv phase2_probe_verification.json meta/probe_verification.json
```
The dataset will have the following structure:
```
OUMVLP_OF_V1_IJCB2025OFcompetition-data-gallery
    | - data-gallery
        | - seq_0001.mp4
        | - seq_0002.mp4
        | - ...
    | - data-probe
        | - seq_0001.mp4
        | - seq_0002.mp4
        | - ...
OUMVLP_OF_V2_IJCB2025OFcompetition-data-gallery
    | - data-gallery
        | - seq_0001.mp4
        | - seq_0002.mp4
        | - ...
    | - data-probe
        | - seq_0001.mp4
        | - seq_0002.mp4
        | - ...
meta
    | - gallery_verification.json
    | - probe_verification.json
```

## Optical Flow Data
To read the video files of optical flow maps, you can use this function `io.read_vid`. It'll return a numpy uint8 array of shape `FxHxWx3` where F is the number of frames.
