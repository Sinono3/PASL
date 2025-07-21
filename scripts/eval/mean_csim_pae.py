import math
from math import asin, atan2, cos

import einops
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from jaxtyping import Float
from tqdm import tqdm
from torch import Tensor

from eval.TDDFA_ONNX import TDDFA_ONNX
from eval.utils import EvalDataset, load_ir50
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

CRIT_LABEL_TO_IDX = {"ff": 0, "fs": 1, "fp": 2, "ss": 3, "sp": 4, "pp": 5}

device = torch.device("cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


# 加载3DDFA-V2模型配置
cfg = yaml.load(open("./configs/bfm/mb1_120x120.yml"), Loader=yaml.SafeLoader)

# 初始化FaceBoxes和TDDFA
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)


def P2sRt(P):
    """decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R):
    """compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
    """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


# 定义函数计算两张图像之间的旋转误差
def calculate_rotation(image: Float[torch.Tensor, "height width channel"]):
    image = image.cpu().numpy()
    # 使用FaceBoxes检测人脸
    boxes = face_boxes(image)

    n = len(boxes)
    if n == 0:
        print("No face detected, exit")
        return None

    # 使用3DDFA-V2进行3D姿势估计
    param_lst, roi_box_lst = tddfa(image, boxes)

    try:
        param_lst, roi_box_lst = tddfa(image, boxes)
        param = param_lst[0]
    except Exception:
        print("3D pose estimation failed, skipping")
        return None

    P1 = param[:12].reshape(3, -1).copy()  # camera matrix
    s, R1, t3d = P2sRt(P1)
    angle = matrix2angle(R1)
    yaw, pitch, roll = angle
    return yaw * (180 / math.pi)


def to_criterion_idx(fake_angle, real_angle):
    if abs(real_angle) <= 30 and abs(fake_angle) <= 30:
        return CRIT_LABEL_TO_IDX["ff"]
    elif abs(real_angle) >= 60 and abs(fake_angle) >= 60:
        return CRIT_LABEL_TO_IDX["pp"]
    elif (abs(real_angle) >= 30 and abs(real_angle) <= 60) and (
        abs(fake_angle) >= 30 and abs(fake_angle) <= 60
    ):
        return CRIT_LABEL_TO_IDX["ss"]
    elif (abs(real_angle) <= 30 and abs(fake_angle) >= 60) or (
        abs(fake_angle) <= 30 and abs(real_angle) >= 60
    ):
        return CRIT_LABEL_TO_IDX["fp"]
    elif (
        abs(real_angle) <= 30 and abs(fake_angle) >= 30 and abs(fake_angle) <= 60
    ) or (abs(fake_angle) <= 30 and abs(real_angle) >= 30 and abs(real_angle) <= 60):
        return CRIT_LABEL_TO_IDX["fs"]
    else:
        return CRIT_LABEL_TO_IDX["sp"]


# Computes face angles for each image and omits entries which have no detected faces.
def imgs_to_angle(
    batchsize: int,
    fake_img: Float[Tensor, "batch channel height width"],
    gt_img: Float[Tensor, "batch channel height width"],
    real_img: Float[Tensor, "batch channel height width"],
):
    # Reorder the dimensions
    # (because `calculate_rotation` expects images in this format)
    fake_img2 = einops.rearrange(fake_img, "b c h w -> b h w c")
    gt_img2 = einops.rearrange(gt_img, "b c h w -> b h w c")
    real_img2 = einops.rearrange(real_img, "b c h w -> b h w c")

    fake_angle = torch.empty(batchsize)
    gt_angle = torch.empty(batchsize)
    real_angle = torch.empty(batchsize)

    error_count = 0
    successful_idx = []
    for b in range(batchsize):
        fake_angle1 = calculate_rotation(fake_img2[b])
        gt_angle1 = calculate_rotation(gt_img2[b])
        real_angle1 = calculate_rotation(real_img2[b])

        if None not in [fake_angle1, gt_angle1, real_angle1]:
            fake_angle[b] = fake_angle1
            gt_angle[b] = gt_angle1
            real_angle[b] = real_angle1
            successful_idx.append(b)
        else:
            error_count += 1

    # Only use successful entries
    batchsize = len(successful_idx)
    fake_img = fake_img[successful_idx]
    gt_img = gt_img[successful_idx]
    real_img = real_img[successful_idx]
    fake_angle = fake_angle[successful_idx]
    gt_angle = gt_angle[successful_idx]
    real_angle = real_angle[successful_idx]
    return (
        batchsize,
        fake_img,
        gt_img,
        real_img,
        fake_angle,
        gt_angle,
        real_angle,
        error_count,
    )


def calculate_csim_for_all_tasks(fake_dir, gt_dir, real_dir):
    print("Loading PAE model...")
    # Order must be the same as in LABEL_TO_INDEX
    criteria = [
        load_ir50("ff", "./weights/pae/ff_backbone_ir_50_epoch_80.pth", device),
        load_ir50("fs", "./weights/pae/fs_backbone_ir_50_epoch_80.pth", device),
        load_ir50("fp", "./weights/pae/fp_backbone_ir_50_epoch_120.pth", device),
        load_ir50("ss", "./weights/pae/ss_backbone_ir_50_epoch_100.pth", device),
        load_ir50("sp", "./weights/pae/sp_backbone_ir_50_epoch_150.pth", device),
        load_ir50("pp", "./weights/pae/pp_backbone_ir_50_epoch_100.pth", device),
    ]
    for crit in criteria:
        crit.eval()

    dataset = EvalDataset(fake_dir, gt_dir, real_dir)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    ard_list = []

    resize = T.Resize((112, 112), T.InterpolationMode.BILINEAR, antialias=True)

    error_count_total = 0

    # 遍历两个文件夹中的图像并计算旋转误差
    for labels, fake_img, gt_img, real_img in tqdm(dataloader, desc="Processing"):
        batchsize = fake_img.shape[0]
        fake_img = fake_img.to(device, torch.float32)
        gt_img = gt_img.to(device, torch.float32)
        real_img = real_img.to(device, torch.float32)

        # Resize and normalize image to [0-1]
        fake_img = resize(fake_img) / 255.0
        gt_img = resize(gt_img) / 255.0
        real_img = resize(real_img) / 255.0

        (
            batchsize,
            fake_img,
            gt_img,
            real_img,
            fake_angle,
            gt_angle,
            real_angle,
            error_count,
        ) = imgs_to_angle(batchsize, fake_img, gt_img, real_img)
        error_count_total += error_count

        # Calculate ARD
        ard = torch.abs(fake_angle - gt_angle)
        ard_list.append(ard)

        samples_to_crit_idx = torch.tensor(
            [to_criterion_idx(fake_angle[i], real_angle[i]) for i in range(batchsize)]
        )
        crit_idx_to_samples = [
            torch.nonzero(samples_to_crit_idx == crit_idx, as_tuple=True)[0]
            for crit_idx in range(6)
        ]

        fake_embs, gt_embs, real_embs = [], [], []

        # Run each crit
        for crit_idx in range(6):
            crit = criteria[crit_idx]
            sample_idxs = crit_idx_to_samples[crit_idx]

            if len(sample_idxs) < 1:
                continue

            with torch.torch.no_grad():
                fake_embs.append(crit(fake_img[sample_idxs]))
                gt_embs.append(crit(gt_img[sample_idxs]))
                real_embs.append(crit(real_img[sample_idxs]))

        fake_embs = torch.cat(fake_embs)
        gt_embs = torch.cat(gt_embs)
        real_embs = torch.cat(real_embs)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        fake_gt_csim = cos(fake_embs, gt_embs)
        fake_real_csim = cos(fake_embs, real_embs)
        real_gt_csim = cos(real_embs, gt_embs)

        fake_gt_csim_list.append(fake_gt_csim)
        fake_real_csim_list.append(fake_real_csim)
        real_gt_csim_list.append(real_gt_csim)

    print("fake_gt_csim:", torch.cat(fake_gt_csim_list).mean().item())
    print("fake_real_csim:", torch.cat(fake_real_csim_list).mean().item())
    print("real_gt_csim:", torch.cat(real_gt_csim_list).mean().item())
    print("ard:", torch.cat(ard_list).mean().item())
    print("error count:", error_count_total)


if __name__ == "__main__":
    fake_image_folder = "./output/eval/mpie/250000"
    gt_image_folder = "./output/eval/mpie/250000ground_truth"
    real_image_folder = "./output/eval/mpie/250000real"

    calculate_csim_for_all_tasks(fake_image_folder, gt_image_folder, real_image_folder)
