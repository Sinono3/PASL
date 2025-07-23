import os

import einops
import torch
import torch.linalg
import torch.nn.functional as F
from torch.utils import data
from torchvision import io, transforms
from torch import Tensor
from jaxtyping import Float

from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA
from deca.decalib.utils import util
from deca.decalib.utils.config import cfg as deca_cfg


def read_rgb(path, device):
    img = io.read_image(path)
    # uint8->float, [0-255]->[0-1]
    img = img.to(device, torch.float32) / 255.0
    return img


# Crop image by 20px on the right, pad 10px on the top and resize into original.
# NOTE: Why is this done?
def preprocess(img: Float[Tensor, "c h w"]) -> Float[Tensor, "c h w"]:
    c, h, w = img.shape
    # Shave 20 pixels off the right edge image
    cropped_img = img[:, :, :-20]
    # Pad top of image with 10 pixels of RGB (0, 0, 0)
    img = F.pad(cropped_img, (0, 0, 10, 0), mode="constant", value=0)
    # Resize back to original size
    img = F.interpolate(
        img.unsqueeze(0),  # need to add batch dim
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return img


class LMDataset(data.Dataset):
    deca: DECA
    face_detector: detectors.FAN

    def __init__(
        self,
        root_path,
        list_path,
        transform=None,
        device="cpu",
    ):
        self.device = device
        self.deca = DECA(config=deca_cfg, device=device)
        self.face_detector = detectors.FAN(device=device)
        self.transform = transform or (lambda x: x)
        self.root_path = root_path
        self.samples_src = []
        self.samples_ref = []
        self.samples_gt = []
        self.angles_src = []
        self.angles_ref = []

        with open(list_path) as F:
            for line_num, line in enumerate(F):
                try:
                    line = line.strip("\n")
                    line_split = line.split(" ")
                    path1 = line_split[0]
                    path2 = line_split[1]
                    path3 = line_split[2] if len(line_split) > 2 else path2  # gt
                    basename1 = os.path.basename(path1)
                    basename2 = os.path.basename(path2)
                    # The angle of the subject in the image is included in the basename.
                    angle1 = int(basename1.split("_")[2])
                    angle2 = int(basename2.split("_")[2])
                except Exception:
                    print(f"Error reading dataset list on line {line_num}")
                    continue

                self.samples_src.append(path1)
                self.samples_ref.append(path2)
                self.samples_gt.append(path3)
                self.angles_src.append(angle1)
                self.angles_ref.append(angle2)

    def __len__(self):
        return len(self.samples_src)

    def __getitem__(self, idx):
        p1, p2, p3 = self.samples_src[idx], self.samples_ref[idx], self.samples_gt[idx]
        src_ang, ref_ang = self.angles_src[idx], self.angles_ref[idx]

        src_path = os.path.join(self.root_path, p1)
        ref_path = os.path.join(self.root_path, p2)
        gt_path = os.path.join(self.root_path, p3)

        src = read_rgb(src_path, self.device)
        ref = read_rgb(ref_path, self.device)
        gt = read_rgb(gt_path, self.device)

        depth, lm = get_depth_render(
            self.deca, self.face_detector, src_path, ref_path, self.device
        )
        depth = depth.to(self.device)
        lm = lm.to(self.device)

        # NOTE: Why is this done?
        depth = preprocess(depth)
        lm = preprocess(lm)

        # apply transforms
        src_t = self.transform(src)
        ref_t = self.transform(ref)
        depth_t = self.transform(depth)
        lm_t = self.transform(lm)
        gt_t = self.transform(gt)

        src_ang = (torch.tensor(src_ang, device=self.device, dtype=torch.long),)
        ref_ang = (torch.tensor(ref_ang, device=self.device, dtype=torch.long),)

        return (
            src_t,
            ref_t,
            lm_t,
            depth_t,
            lm_t,
            gt_t,
            src_ang,
            ref_ang,
        )


def get_depth_render(
    deca: DECA, face_detector: detectors.FAN, src_path, ref_path, device
):
    td = datasets.TestData(
        [src_path, ref_path], face_detector, iscrop=True, sample_step=10
    )
    src = td[0]["image"].to(device)
    ref = td[1]["image"].to(device)

    # Matrix inverse is slow.
    # TODO: Since the transform we get here is a similarity
    # (https://nvision-user-guide.readthedocs.io/en/latest/geometric_transformations.html#similarity)
    # we could compute the inverse in a cheaper way.
    # But since it's 3x3, it won't matter that much.
    tform_inv_t = torch.linalg.inv(td[1]["tform"].to(device)).transpose(-2, -1)
    original = td[1]["original_image"].to(device)

    with torch.no_grad():
        src_embed = deca.encode(src.unsqueeze(0))
        ref_embed = deca.encode(ref.unsqueeze(0))

        new_embed = {}
        # the types of embeddings are
        # ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']
        # We take `shape`, `tex`, `light`, `detail` from the source
        new_embed["shape"] = src_embed["shape"]
        new_embed["tex"] = src_embed["tex"]
        new_embed["light"] = src_embed["light"]
        new_embed["detail"] = src_embed["detail"]
        # Use the other embeddings from the reference.
        new_embed["exp"] = ref_embed["exp"]
        new_embed["pose"] = ref_embed["pose"]
        new_embed["cam"] = ref_embed["cam"]
        new_embed["images"] = ref_embed["images"]

        opdict, visdict = deca.decode(
            new_embed,
            render_orig=True,
            original_image=original.unsqueeze(0),
            tform=tform_inv_t.unsqueeze(0),
        )

    depth: Float[torch.Tensor, "b c h w"] = deca.render.render_depth(
        opdict["trans_verts"]
    ).to(device)
    lm: Float[torch.Tensor, "b c h w"] = visdict["landmarks2d"].to(device)

    # Remove batch dimension
    depth = einops.rearrange(depth, "1 c h w -> c h w")
    lm = einops.rearrange(lm, "1 c h w -> c h w")

    # Repeat depth channel x3 to RGB channels
    depth = einops.repeat(depth, "c h w -> (3 c) h w")
    # Clamp into [0,1]
    depth = depth.clamp(0, 1)
    lm = lm.clamp(0.1)
    return depth, lm


def get_eval_loader_vgg(
    root_dir,
    list_path,
    img_size=256,
    batch_size=32,
    imagenet_normalize=True,
    num_workers=0,
    drop_last=False,
    device="cpu",
):
    print("Preparing DataLoader for the evaluation phase...")
    if imagenet_normalize:
        height, width = 299, 299
    else:
        height, width = img_size, img_size

    transform = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ]
    )

    ds = LMDataset(
        root_dir,
        list_path,
        transform=transform,
        device=device,
    )
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True,
        drop_last=drop_last,
    )
