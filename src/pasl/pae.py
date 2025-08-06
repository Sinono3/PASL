from math import atan2, asin, cos
import math
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import yaml
from jaxtyping import Num, Float, Bool, Int64

# from lightning.fabric import Fabric
from torch import Tensor

# from pasl.eval.TDDFA_ONNX import TDDFA_ONNX
from pasl.ir50 import IR_50, load_ir50
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from tddfa.TDDFA_ONNX import TDDFA_ONNX


class PAE(nn.Module):
    IDX_TO_LABEL = ["ff", "fs", "fp", "ss", "sp", "pp"]
    LABEL_TO_IDX = {"ff": 0, "fs": 1, "fp": 2, "ss": 3, "sp": 4, "pp": 5}

    def __init__(self):
        super().__init__()
        # 初始化FaceBoxes和TDDFA
        self.face_boxes = FaceBoxes_ONNX()

        # 加载3DDFA-V2模型配置
        # TODO: hardcode this config or something.
        tddfa_cfg = dict(
            arch="mobilenet",  # MobileNet V1
            widen_factor=1.0,
            checkpoint_fp="external/tddfa/weights/mb1_120x120.onnx",
            bfm_fp="external/tddfa/configs/bfm_noneck_v3.pkl",  # or configs/bfm_noneck_v3_slim.pkl
            size=120,
            num_params=62,
        )
        self.tddfa = TDDFA_ONNX(**tddfa_cfg)

        # Order must be the same as in IDX_TO_LABEL
        self.models = nn.ModuleList(
            [
                IR_50([112, 112]),
                IR_50([112, 112]),
                IR_50([112, 112]),
                IR_50([112, 112]),
                IR_50([112, 112]),
                IR_50([112, 112]),
            ]
        )

    def load_from_paths(self, paths: dict[str, str]):
        for idx in range(6):
            id = PAE.IDX_TO_LABEL[idx]
            path = paths[id]
            self.models[idx] = load_ir50(id, path, "cpu")

    @staticmethod
    def _angles_to_model_idx(a, b):
        a = abs(a)
        b = abs(b)
        # Test for each face orientation
        a_f = a < 30
        a_p = a >= 60
        a_s = 30 <= a < 60
        b_f = b < 30
        b_p = b >= 60
        b_s = 30 <= b < 60

        if b_f and a_f:
            return PAE.LABEL_TO_IDX["ff"]
        if b_p and a_p:
            return PAE.LABEL_TO_IDX["pp"]
        elif b_s and a_s:
            return PAE.LABEL_TO_IDX["ss"]
        elif (b_f and a_p) or (a_f and b_p):
            return PAE.LABEL_TO_IDX["fp"]
        elif (b_f and a_s) or (a_f and b_s):
            return PAE.LABEL_TO_IDX["fs"]
        else:
            return PAE.LABEL_TO_IDX["sp"]

    @staticmethod
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

    def matrix2angle(self, R):
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
    def calculate_rotation(self, image: Float[torch.Tensor, "height width channel"]):
        image = image.cpu().numpy()
        # 使用FaceBoxes检测人脸
        boxes = self.face_boxes(image)

        n = len(boxes)
        if n == 0:
            print("No face detected, exit")
            return None

        # 使用3DDFA-V2进行3D姿势估计
        try:
            param_lst, roi_box_lst = self.tddfa(image, boxes)
            param = param_lst[0]
        except Exception:
            print("3D pose estimation failed, skipping")
            return None

        P1 = param[:12].reshape(3, -1).copy()  # camera matrix
        s, R1, t3d = PAE.P2sRt(P1)
        angle = self.matrix2angle(R1)
        yaw, pitch, roll = angle
        return yaw * (180 / math.pi)

    def img_to_angle(
        self, img: Float[Tensor, "b c h w"]
    ) -> tuple[Float[Tensor, " b"], Bool[Tensor, " b"]]:
        """
        Computes face angles for each image.

        Returns: angles of images, and a validity mask,
        informing which batches are invalid.
        """

        batchsize = img.shape[0]
        device = next(self.parameters()).device

        # Reorder the dimensions and *255
        # (because `calculate_rotation` expects images in this format)
        img = einops.rearrange(img, "b c h w -> b h w c") * 255.0

        angles = torch.empty(batchsize, device=device, dtype=torch.float32)
        # Generate a mask of which elements are valid,
        # because calculate_rotation can fail.
        valid_mask = torch.empty(batchsize, device=device, dtype=torch.bool)

        for b in range(batchsize):
            angle = self.calculate_rotation(img[b])

            if angle is not None:
                angles[b] = angle
                valid_mask[b] = True
            else:
                angles[b] = math.nan
                valid_mask[b] = False

        return (
            angles,
            valid_mask,
        )

    def angles_to_processing_list(
        self,
        angles: Num[Tensor, "b 2"],
        mask: Bool[Tensor, " b"],
        angle_to_model_idx_fn=None,
    ) -> list[Int64[Tensor, " batch_idx"]]:
        """
        Should contain two angles per batch,
        representing the two images to be compared.

        Returns a list of 6 tensors, each corresponding
        to an encoder model. Each tensor represents the
        indexes of the batches (referent to the angles tensor passed)
        that should be fed to *that* encoder.

        Kind of hard to explain clearly... but basically you give it pairs of
        angles of face images that will be compared against each other, and
        for each encoder it returns the indexes of what batches correspond to it.
        """
        assert angles.size(1) == 2, "angles dimension should be two"

        if angle_to_model_idx_fn is None:
            angle_to_model_idx_fn = PAE._angles_to_model_idx

        samples_to_idx = torch.tensor(
            [
                angle_to_model_idx_fn(batch[0].item(), batch[1].item())
                for batch in angles
            ]
        )
        idx_to_samples = [
            torch.nonzero(samples_to_idx == model_idx, as_tuple=True)[0]
            for model_idx in range(6)
        ]
        return idx_to_samples

    def forward(
        self,
        img: Float[Tensor, "b c h w"],
        model_idx_to_batches: list[Int64[Tensor, " batch_idx"]],
    ):
        batchsize = img.shape[0]
        device = next(self.parameters()).device
        embeds = torch.zeros(batchsize, 512, device=device, dtype=torch.float32)

        # Run each model
        for model_idx, batches in enumerate(model_idx_to_batches):
            if len(batches) < 1:
                continue

            with torch.no_grad():
                embeds[batches] = self.models[model_idx](img[batches])

        return embeds
