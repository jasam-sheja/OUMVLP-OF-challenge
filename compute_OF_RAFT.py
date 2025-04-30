"""
# This code has been impelemented by:
#   Dr. Ammar Alsherfawi, Osaka University, Japan
#   Dr. Allam Shehata, Osaka University, Japan
#   Prof. Francisco M. Castro, University of Malaga, Spain.
# Email: allam@am.sanken.osaka-u.ac.jp; fcastro@uma.es

# Date: February 28th 2025

dependencies:
pytorch: 1.9.0+
pip install opencv-python cvbase imageio[ffmpeg]
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import cvbase
import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor, nn
from torchvision.models.optical_flow import raft_large

import data_io


def preprocess(batch: Tensor) -> Tensor:
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch


def read_video_frames(video_path, binary_path, w, h):

    frames_rgb = sorted(glob.glob(video_path + "/*.png"))

    frames_sil_names = [
        os.path.basename(os.path.normpath(XX)) for XX in frames_rgb
    ]  # get the bin sil seq synchronized
    frames_sil = [
        cv2.imread(binary_path + "/" + frame, 0) for frame in frames_sil_names
    ]  # read bin sil frames

    frames_rgb = [cv2.imread(frame) for frame in frames_rgb]
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_rgb]

    # preprocess sill
    frames_sil = [preprocess_sil(frame) for frame in frames_sil]
    frames_sil = [
        get_cropped_Sil_Loc(frame) for frame in frames_sil
    ]  # return cropped sills and bbox locations

    # get the bbx location
    location = [l[1] for l in frames_sil]
    location = np.stack(location, axis=0)
    # get cropped RGB sequence
    cropped_frames = [my_crop(frame, loc) for frame, loc in zip(frames_rgb, location)]

    # resize
    cropped_frames = [cv2.resize(frm, (w, h)) for frm in cropped_frames]

    cropped_frames = np.stack(cropped_frames, axis=0)
    tensor_imgs = torch.tensor(cropped_frames)

    return tensor_imgs
    # -------------------------------------------------------


def read_video_frames_full_res(video_path):
    frames_rgb = sorted(glob.glob(video_path + "/*.png"))
    frames_rgb = [cv2.imread(frame) for frame in frames_rgb]
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_rgb]

    frames_rgb = np.stack(frames_rgb, axis=0)
    tensor_imgs = torch.tensor(frames_rgb)

    return tensor_imgs


# --------------------------------------------
# --------------------------------------------
def preprocess_sil(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(img, [contours[k]], 0)
    return img


# --------------------------------------------
def get_ROI_location(binaryname):
    img = cv2.imread(binaryname, 0)
    img = preprocess_sil(img)
    h_top, h_bottom, w_left, w_right = compute_loc(img)

    return h_top, h_bottom, w_left, w_right


# -------------------------------------------------
def myWriteImg(filename, img):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    cv2.imwrite(filename, img)


# -------------------------------------------
def my_crop(img, location):
    w = np.size(img, 1)
    crop_h = int(location[3])
    crop_w = crop_h

    h_top = int(location[0])
    h_bottom = int(location[1])
    w_left = int(location[2] - np.ceil(crop_w / 2))
    w_right = int(w_left + crop_w)

    if np.size(img.shape) == 2:
        cropimg = np.zeros((crop_w, crop_h))
        if w_left < 0:
            print(img.shape, h_top, h_bottom, w_left, w_right, crop_w)
            cropimg[:, crop_w - w_right : crop_w] = img[h_top:h_bottom, 0:w_right]
        if w_right > w:
            cropimg[:, 0 : w - w_left] = img[h_top:h_bottom, w_left:w]
        if w_left >= 0 and w_right < w:
            cropimg = img[h_top:h_bottom, w_left:w_right]
    else:
        cropimg = np.zeros((crop_w, crop_h, 3))
        if w_left < 0:
            cropimg[:, crop_w - w_right : crop_w, :] = img[h_top:h_bottom, 0:w_right, :]
        if w_right > w:
            cropimg[:, 0 : w - w_left, :] = img[h_top:h_bottom, w_left:w, :]
        if w_left >= 0 and w_right < w:
            cropimg = img[h_top:h_bottom, w_left:w_right, :]
    return cropimg


# -------------------------------------------
def get_cropped_Sil_Loc(img):
    location = np.zeros((4))
    M = np.max(img, axis=1)
    F = np.where(M != 0)
    # top row
    location[0] = F[0][0]
    # bottom row
    location[1] = F[0][-1]
    # center col
    _, colindice = np.where(img != 0)
    location[2] = np.ceil(np.median(colindice))
    # height
    location[3] = np.abs(location[1] - location[0])

    # crop
    cropimg = my_crop(img, location)

    return cropimg, location


# ---------------------------------------
# get the location of the bbox: [height_top, height_bottom, width_left, width_right]
def compute_loc(img):
    # location = np.zeros((4))
    M = np.max(img, axis=1)
    F = np.where(M != 0)
    # top row
    h_top = int(F[0][0])
    # bottom row
    h_bottom = int((F[0][-1]))
    # height
    crop_h = int(np.abs(h_bottom - h_top))
    crop_w = crop_h
    # center col
    _, colindice = np.where(img != 0)
    center_w = np.ceil(np.median(colindice))

    w_left = int(center_w - np.ceil(crop_w / 2))
    w_right = int(w_left + crop_w)

    return h_top, h_bottom, w_left, w_right


def apply_video_files(
    rgb_file: str,
    silhouette_file: str,
    width: int = 256,
    height: int = 256,
    model: nn.Module = None,
) -> np.ndarray:
    """Generate optical flow for a video file and its corresponding silhouette file.
    Args:
        rgb_file (str): Path to the RGB video file.
        silhouette_file (str): Path to the silhouette video file.
    Returns:
        np.ndarray: Optical flow data.
    """
    rgb_file: Path = Path(rgb_file)
    silhouette_file: Path = Path(silhouette_file)

    # Check if the files exist
    if not rgb_file.exists():
        raise FileNotFoundError(f"RGB file not found: {rgb_file}")
    if not silhouette_file.exists():
        raise FileNotFoundError(f"Silhouette file not found: {silhouette_file}")

    # Read the RGB and silhouette files
    rgb = iio.imread(rgb_file)
    if not rgb.ndim == 4:
        raise ValueError(f"Invalid RGB file format: {rgb_file}")
    sil = iio.imread(silhouette_file)
    if not sil.ndim == 4:
        sil = sil[..., 0]  # Convert to 3D if it's 4D
    if not sil.ndim == 3:
        raise ValueError(f"Invalid silhouette file format: {silhouette_file}")

    return apply(rgb, sil, width, height, model)


def apply_image_folders(
    rgb_folder: str,
    silhouette_folder: str,
    width: int = 256,
    height: int = 256,
    model: nn.Module = None,
) -> np.ndarray:
    """Generate optical flow for a video folder and its corresponding silhouette folder.
    Args:
        rgb_file (str): Path to the RGB folder.
        silhouette_file (str): Path to the silhouette folder.
    Returns:
        np.ndarray: Optical flow data.
    """
    rgb_folder: Path = Path(rgb_folder)
    silhouette_folder: Path = Path(silhouette_folder)

    # Check if the files exist
    if not rgb_folder.exists():
        raise FileNotFoundError(f"RGB folder not found: {rgb_folder}")
    if not silhouette_folder.exists():
        raise FileNotFoundError(f"Silhouette folder not found: {silhouette_folder}")

    # Read the RGB and silhouette files
    rgb = data_io.read_folder(rgb_folder)
    if not rgb.ndim == 4:
        raise ValueError(f"Invalid RGB file format: {rgb_folder}")
    sil = data_io.read_folder(silhouette_folder)
    if sil.ndim == 4:
        sil = sil[..., 0]  # Convert to 3D if it's 4D
    if not sil.ndim == 3:
        raise ValueError(f"Invalid silhouette file format: {silhouette_folder}")

    return apply(rgb, sil, width, height, model)


@torch.inference_mode()
def apply(
    rgb: np.ndarray,
    sil: np.ndarray,
    width: int = 256,
    height: int = 256,
    model: nn.Module = None,
) -> np.ndarray:
    """"""
    """Generate optical flow for a sequence of RGB frames and their corresponding silhouettes.
    Args:
        rgb (np.ndarray): Array of RGB frames.
        sil (np.ndarray): Array of silhouette frames.
        width (int): Width of the output frames.
        height (int): Height of the output frames.
        model (nn.Module): Optical flow model.
    Returns:
        np.ndarray: Optical flow data.
    """
    if model is None:
        model = raft_large(pretrained=True, progress=False)
        if torch.cuda.is_available():
            model = model.cuda()
        model = model.eval()

    # Check the input dimensions
    if len(rgb) != len(sil):
        raise ValueError("RGB and silhouette sequences must have the same length.")
    if len(rgb) < 2:
        raise ValueError("At least two frames are required to compute optical flow.")
    if rgb.ndim != 4:
        raise ValueError("RGB sequence must be a 4D array, with shape (N, H, W, C).")
    if sil.ndim != 3:
        raise ValueError(
            "Silhouette sequence must be a 3D array, with shape (N, H, W)."
        )
    _, location = zip(*[get_cropped_Sil_Loc(frame) for frame in sil])

    rgb_crop = [my_crop(frame, loc) for frame, loc in zip(rgb, location)]
    rgb_crop_resized = [cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC) for frame in rgb_crop]
    rgb = np.stack(rgb_crop_resized, axis=0)
    del rgb_crop_resized, rgb_crop, sil, location

    # Process the RGB frames
    rgb: Tensor = torch.from_numpy(rgb)
    rgb = rgb.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
    rgb = preprocess(rgb)
    x, y = rgb[:-1], rgb[1:]
    x = torch.split(x, 2, dim=0)
    y = torch.split(y, 2, dim=0)

    # Compute optical flow
    flow = []
    for sp_ix in range(len(x)):
        flow.append(
            model(
                x[sp_ix].contiguous(),
                y[sp_ix].contiguous(),
            )[-1]
            .permute(0, 2, 3, 1)
            .to("cpu", non_blocking=True)
        )
    flow: Tensor = torch.cat(flow, dim=0)
    flow: np.ndarray = flow.cpu().numpy()

    # Convert flow to RGB
    flow_rgb = np.zeros((*flow.shape[:3], 3), dtype=np.uint8)
    for i in range(flow.shape[0]):
        flow_rgb[i] = (cvbase.flow2rgb(flow[i]) * 255).astype(np.uint8)
    return flow_rgb


# -------------------------------------------
# -------------------------------------------------
if __name__ == "__main__":

    # Prepare input
    # Input arguments
    parser = argparse.ArgumentParser(description="Build OF maps dataset")

    parser.add_argument(
        "--video_file",
        type=str,
        required=True,
        help="Path to  RGB sequences (images folder or video file)",
    )

    parser.add_argument(
        "--silhouette_file",
        type=str,
        required=True,
        help="Path to binary silhouettes (images folder or video file)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Full path for output files.",
    )

    args = parser.parse_args()
    print("This is a sample code to compute optical flow maps using RAFT.")
    print("Please make sure to have the required dependencies installed.")
    print("Use the functions apply_video_files or apply_image_folders to compute the optical flow.")
    print("Please make sure that both the input rgbs and silhouettes are synchronized")
    print("This script will produce a normalized optical flow images by cropping and resizing with output resolution HxWx3")
    # Read the input arguments
    video_file: str = args.video_file
    silhouette_file: str = args.silhouette_file
    output_file: str = args.output_file
    # Initialize some parameters...
    print('loading the model...')
    np.random.seed(0)
    model = raft_large(pretrained=True, progress=False).to("cuda")
    model = model.eval()

    if os.path.isdir(video_file) and os.path.isdir(silhouette_file):
        print("Computing optical flow for folders...")
        flow_rgb = apply_image_folders(video_file, silhouette_file)
    elif os.path.isfile(video_file) and os.path.isfile(silhouette_file):
        print("Computing optical flow for files...")
        flow_rgb = apply_video_files(video_file, silhouette_file)
    else:
        parser.error(
            "Either both video_file and silhouette_file should be directories or both should be files."
        )
    # Save the optical flow as a video
    print("Saving the optical flow as a video...")
    data_io.write_vid(flow_rgb, output_file)
