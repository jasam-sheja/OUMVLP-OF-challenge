"""
IO functions for reading and writing images and videos of optical flow maps (RGB images).

dependencies:
pip install imageio[ffmpeg] numpy

The author of this code snippet is Mohamad Ammar Alsherfawi Aljazaerly (https://github.com/jasam-sheja)
"""

import json
from pathlib import Path
from typing import Dict, List

import imageio.v3 as iio
import numpy as np


def read_folder(
    root: str | Path,
    exts: str | List[str] = None,
) -> np.ndarray:
    """
    Read a folder of images.

    Args:
        root: Path to the folder.
        exts: The file extension(s) to read. Default: ['.png', '.jpg'].

    Returns:
        A NumPy array with shape (#files, height, width, channels) and dtype uint8.

    Raises:
        FileNotFoundError: If no files are found.
    """
    root = Path(root)
    if exts is None:
        exts = [".png", ".jpg"]
    elif isinstance(exts, str):
        exts = [exts]
    files = []
    for exti in exts:
        files += list(root.glob(f"*{exti}"))
    files = sorted(files, key=lambda f: f.stem)
    if len(files) == 0:
        raise FileNotFoundError(f"no files found in {root} with {exts}")
    return np.stack([iio.imread(f) for f in files])


def write_vid(
    imgvol: np.ndarray,
    file: str | Path,
    output_params: List[str] = None,
    **kwargs,
) -> None:
    """
    Write a lossless video from an image volume.

    Args:
        imgvol: NumPy array of uint8 images with shape (#frames, height, width).
        file: The path to the output video file.
        output_params: List of additional ffmpeg output parameters.
        **kwargs: Additional parameters for imageio.v3.imwrite.
    """
    _output_params = ["-qp", "0", "-preset", "veryslow", "-pix_fmt", "bgr24"]
    if output_params is not None:
        _output_params += output_params
    return iio.imwrite(
        file, imgvol, codec="libx264rgb", fps=1, output_params=_output_params, **kwargs
    )


def read_vid(file: str | Path) -> np.ndarray:
    """
    Read a video file.

    Args:
        file: The path to the video file.

    Returns:
        A NumPy array with shape (#frames, height, width, channels) and dtype uint8.
    """
    return iio.imread(file)


def save_identification_submission(
    matches: Dict[str, List[str]],
    output_dir: Path | str,
    phase: str,  # phase1 phase2
    version: str,  # v1 v2
) -> None:
    """
    Save the identification submission file.
    """
    assert phase in ["phase1", "phase2"], f"Invalid phase: {phase}"
    output_dir = Path(output_dir)
    meta = Path(__file__).parent.joinpath("meta")
    indexing = meta.joinpath(f"{phase}_index_mapping.json")
    mapping = json.loads(indexing.read_text())
    assert len(matches) == len(
        mapping["probe"]
    ), f"Invalid number of matches for {phase}, expected {len(mapping['probe'])}, got {len(matches)}"
    output = np.full((len(mapping["probe"]), 14), -1, dtype=np.int32)
    for probe, gallery_matches in matches.items():
        if len(gallery_matches) != 14:
            raise ValueError(
                f"Invalid number of gallery matches for {probe}, expected 14, got {len(gallery_matches)}"
            )
        probe_idx = mapping["probe"][probe]
        for i, gallery in enumerate(gallery_matches):
            output[probe_idx, i] = mapping["gallery"][gallery]
    np.savez_compressed(output_dir.joinpath(f"ranking-{version}.npz"), ranking=output)


class VerificationSubmission:
    '''
    Verification submission builder.
    In order to create a verification submission without worrying about the order of the probe and gallery pairs,
    use this class to set the distance between a probe and gallery pairs.
    '''
    def __init__(self, phase: str):
        meta = Path(__file__).parent.joinpath("meta")
        self.probe_ver: List[str] = json.loads(meta.joinpath(f"{phase}_probe_verification.json").read_text())  # type: ignore
        self.gallery_ver: List[str] = json.loads(meta.joinpath(f"{phase}_gallery_verification.json").read_text())  # type: ignore
        self.dist = np.full(
            (len(self.probe_ver), len(self.gallery_ver)), float("nan"), dtype=np.float32
        )

    def set_distance(self, probe: str, gallery: str, distance: float) -> None:
        """
        Set the distance between a probe and gallery pair.
        Args:
            probe: The probe name as given in {phase}_probe_verification.json.
            gallery: The gallery name as given in {phase}_gallery_verification.json.
            distance: The distance between the probe and gallery images.
        """
        probe_idx = self.probe_ver.index(probe)
        gallery_idx = self.gallery_ver.index(gallery)
        self.dist[probe_idx, gallery_idx] = distance

    def save(self, output_dir: Path | str, version: str) -> None:
        """
        Save the verification submission file.
        """
        output_dir = Path(output_dir)
        if np.isnan(self.dist).any():
            raise ValueError("Invalid pairwise distance: NaN")
        np.savez_compressed(output_dir.joinpath(f"dist-{version}.npz"), dist=self.dist)


def save_verification_submission(
    dist: np.ndarray,
    output_dir: Path,
    phase: str,  # phase1 phase2
    version: str,  # v1 v2
) -> None:
    """
    Save the verification submission file.
    """
    assert phase in ["phase1", "phase2"], f"Invalid phase: {phase}"
    meta = Path(__file__).parent.joinpath("meta")
    probe_ver = json.loads(
        meta.joinpath(f"{phase}_probe_verification.json").read_text()
    )
    gallery_ver = json.loads(
        meta.joinpath(f"{phase}_gallery_verification.json").read_text()
    )
    assert dist.shape == (
        len(probe_ver),
        len(gallery_ver),
    ), f"Invalid shape for pairwise distance: {dist.shape}"
    if np.isnan(dist).any():
        raise ValueError("Invalid pairwise distance: NaN")
    np.savez_compressed(output_dir.joinpath(f"ver-{version}.npz"), dist=dist)
