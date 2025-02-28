"""
# This code has been impelemented by Dr. Allam Shehata, Osaka University, Japan, and Prof. Francisco M. Castro, University of Malaga, Spain.
# Email: allam@am.sanken.osaka-u.ac.jp; fcastro@uma.es

# Date: February 28th 2025

dependencies:
pytorch: 1.9.0
pip install opencv-python cvbase 
"""

import cvbase
import numpy as np
import argparse
import os
import glob
import cv2
import torch
from torchvision.models.optical_flow import raft_large
import torchvision.transforms as T


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
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


# -------------------------------------------
# -------------------------------------------------
if __name__ == "__main__":

    # Prepare input
    # Input arguments
    parser = argparse.ArgumentParser(description="Build OF maps dataset")

    parser.add_argument(
        "--videodir",
        type=str,
        required=False,
        default="change_this_path/",
        help="Path to  RGB sequences",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="change_this_path/",
        help="Full path for output files.",
    )

    args = parser.parse_args()
    videodir = args.videodir
    outdir = args.outdir

    dataset_binary_path = "path to the corressponding binary silhouettes/"
    # Initialize some parameters...
    np.random.seed(0)
    model = raft_large(pretrained=True, progress=False).to("cuda")
    model = model.eval()
    id_folders = sorted(glob.glob(videodir + "/*/"))

    # The desired resolution of the generated OF
    wd = 256
    ht = 256
    # ==============Subjects loop======================
    for id in id_folders:
        view_folders = glob.glob(id + "RGB/*/")

        per_id = os.path.basename(os.path.normpath(id))
        # print('identity',per_id)

        check_exist = "path to the generated OF maps/" + per_id
        # -------- check the existence of of folder -----------
        if os.path.exists(check_exist):
            continue
        # -----------------------------------------------------
        for view in view_folders:
            # prepare the bin sills dir
            view_id = os.path.basename(os.path.normpath(view))
            binary_path = os.path.join(
                dataset_binary_path, "Silhouette_" + view_id, per_id
            )  # the dir name

            if os.path.exists(binary_path) and os.path.exists(view):
                # print('binary_path',binary_path)
                # print('RGB view',view)
                # read video frames and return cropped, aligned, and resized frames seq
                frame_list = read_video_frames(view, binary_path, wd, ht)
                # reshape
                frame_list = frame_list.permute(0, 3, 1, 2)

                if len(frame_list) > 0:
                    # preprocess the frame tensor
                    frame_list = preprocess(frame_list)
                    x = frame_list[:-1]
                    y = frame_list[1:]
                    x = torch.split(x, 2, dim=0)
                    y = torch.split(y, 2, dim=0)
                    # Compute OF.
                    flow = []
                    with torch.no_grad():
                        for sp_ix in range(len(x)):
                            flow.append(
                                model(
                                    x[sp_ix].cuda().contiguous(),
                                    y[sp_ix].cuda().contiguous(),
                                )[-1]
                                .permute(0, 2, 3, 1)
                                .cpu()
                                .numpy()
                            )

                    # Stack of maps
                    flow = np.concatenate(flow, axis=0)
                    outdir_ = os.path.join(
                        outdir, id.split("/")[-2], "OF", view.split("/")[-2]
                    )
                    os.makedirs(outdir_, exist_ok=True)
                    for im_ix in range(len(flow)):
                        img = cvbase.flow2rgb(flow[im_ix]) * 255

                        img = img.astype(np.uint8)

                        outpath = os.path.join(
                            outdir_, "{:03d}".format(im_ix + 1) + ".png"
                        )

                        cv2.imwrite(outpath, img)
