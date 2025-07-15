import math
from math import asin, atan2, cos, sin

import einops
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from eval.TDDFA_ONNX import TDDFA_ONNX
from eval.utils import EvalDataset, load_ir50
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

device = torch.device("cpu")
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


# 加载3DDFA-V2模型配置
cfg = yaml.load(open("./configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)

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


def angle2matrix(angles):
    """get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    """
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)


# 定义函数计算两张图像之间的旋转误差
def calculate_rotation(image: torch.Tensor):
    image = image.cpu().numpy()
    # 使用FaceBoxes检测人脸
    boxes = face_boxes(image)

    n = len(boxes)
    if n == 0:
        print(f"No face detected, exit")
        return None

    # 使用3DDFA-V2进行3D姿势估计
    param_lst, roi_box_lst = tddfa(image, boxes)

    try:
        param_lst, roi_box_lst = tddfa(image, boxes)
        param = param_lst[0]
    except Exception as e:
        print(f"3D pose estimation failed, skipping")
        return None

    P1 = param[:12].reshape(3, -1).copy()  # camera matrix
    s, R1, t3d = P2sRt(P1)
    angle = matrix2angle(R1)
    yaw, pitch, roll = angle
    return yaw * (180 / math.pi)


def to_criterion_label(fake_angle, real_angle):
    if abs(real_angle) <= 30 and abs(fake_angle) <= 30:
        return "ff"
    elif abs(real_angle) >= 60 and abs(fake_angle) >= 60:
        return "pp"
    elif (abs(real_angle) >= 30 and abs(real_angle) <= 60) and (
        abs(fake_angle) >= 30 and abs(fake_angle) <= 60
    ):
        return "ss"
    elif (abs(real_angle) <= 30 and abs(fake_angle) >= 60) or (
        abs(fake_angle) <= 30 and abs(real_angle) >= 60
    ):
        return "fp"
    elif (
        abs(real_angle) <= 30 and abs(fake_angle) >= 30 and abs(fake_angle) <= 60
    ) or (abs(fake_angle) <= 30 and abs(real_angle) >= 30 and abs(real_angle) <= 60):
        return "fs"
    else:
        return "sp"


def calculate_csim_for_all_tasks(fake_dir, gt_dir, real_dir):
    print("Loading PAE model...")
    criterion = {
        "ff": load_ir50("ff", "./weights/POE/FF/Backbone_IR_50_Epoch_80.pth", device),
        "fs": load_ir50("fs", "./weights/POE/FS/Backbone_IR_50_Epoch_80.pth", device),
        "fp": load_ir50("fp", "./weights/POE/FP/Backbone_IR_50_Epoch_120.pth", device),
        "ss": load_ir50("ss", "./weights/POE/SS/Backbone_IR_50_Epoch_100.pth", device),
        "sp": load_ir50("sp", "./weights/POE/SP/Backbone_IR_50_Epoch_150.pth", device),
        "pp": load_ir50("pp", "./weights/POE/PP/Backbone_IR_50_Epoch_100.pth", device),
    }
    criterion["fs"].eval()
    criterion["fp"].eval()
    criterion["ss"].eval()
    criterion["sp"].eval()
    criterion["pp"].eval()
    criterion["ff"].eval()

    dataset = EvalDataset(fake_dir, gt_dir, real_dir)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    ard_list = []

    resize = T.Resize((112, 112), T.InterpolationMode.BILINEAR, antialias=True)

    err = 0
    # 遍历两个文件夹中的图像并计算旋转误差
    for label, fake_img, gt_img, real_img in tqdm(dataset, desc="Processing"):
        fake_img = fake_img.to(device, torch.float32)
        gt_img = gt_img.to(device, torch.float32)
        real_img = real_img.to(device, torch.float32)

        fake_img, gt_img, real_img = einops.rearrange(
            [fake_img, gt_img, real_img], "b c h w -> b h w c"
        )

        fake_angle = calculate_rotation(fake_img)
        gt_angle = calculate_rotation(gt_img)
        real_angle = calculate_rotation(real_img)

        if None in [fake_angle, gt_angle, real_angle]:
            err += 1
            continue

        fake_angle = torch.tensor([fake_angle])
        gt_angle = torch.tensor([gt_angle])
        ard = torch.abs(fake_angle - gt_angle)
        ard_list.append(ard)

        # Prepare image tensors for model
        fake_img, gt_img, real_img = einops.rearrange(
            [fake_img, gt_img, real_img], "b h w c -> b 1 c h w"
        )

        fake_img = resize(fake_img) / 255.0
        gt_img = resize(gt_img) / 255.0
        real_img = resize(real_img) / 255.0

        # print(fake_embs.shape)
        # print(real_embs.shape)
        # print(gt_embs.shape)

        label = to_criterion_label(fake_angle, real_angle)
        with torch.torch.no_grad():
            fake_embs = criterion[label](fake_img)
            gt_embs = criterion[label](gt_img)
            real_embs = criterion[label](real_img)

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
    print("error count:", err)


fake_image_folder = "./output/eval/mpie/250000"
gt_image_folder = "./output/eval/mpie/250000ground_truth"
real_image_folder = "./output/eval/mpie/250000real"

calculate_csim_for_all_tasks(fake_image_folder, gt_image_folder, real_image_folder)
