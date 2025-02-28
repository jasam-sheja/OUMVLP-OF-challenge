"""
IO functions for reading and writing images and videos of optical flow maps (RGB images).

dependencies:
pip install imageio[ffmpeg] numpy

The author of this code snippet is Mohamad Ammar Alsherfawi Aljazaerly (https://github.com/jasam-sheja)
"""

from pathlib import Path
from typing import List

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
