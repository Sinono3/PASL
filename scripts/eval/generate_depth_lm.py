#!/usr/bin/env python
# Script used to generate (1) landmark images and (2) depth images for
# items in a dataset.
#
# This was originally done on the DataLoader, but it is more efficient to do
# it all at once.


import os
import pathlib
import shutil

import einops
import hydra
import torch
import torch.linalg
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from torch.utils import data
from torchvision.io import write_png
from tqdm import tqdm

from deca.decalib.datasets import datasets, detectors
from deca.decalib.deca import DECA
from deca.decalib.utils.config import cfg as deca_cfg


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


class Dataset(data.Dataset):
    def __init__(
        self,
        root_path,
        list_path,
    ):
        self.root_path = root_path
        self.samples_src = []
        self.samples_ref = []

        with open(list_path) as F:
            for line_num, line in enumerate(F):
                try:
                    line = line.strip("\n")
                    line_split = line.split(" ")
                    path1 = line_split[0]
                    path2 = line_split[1]
                except Exception:
                    print(f"Error reading dataset list on line {line_num}")
                    continue

                self.samples_src.append(path1)
                self.samples_ref.append(path2)

    def __len__(self):
        return len(self.samples_src)

    def __getitem__(self, idx):
        p1, p2 = self.samples_src[idx], self.samples_ref[idx]
        src_path = os.path.join(self.root_path, p1)
        ref_path = os.path.join(self.root_path, p2)
        return src_path, ref_path, p1, p2


# Generate batches of depth and landmark images
def render_depth_lm(
    deca: DECA,
    face_detector: detectors.FAN,
    src_path_list: list[str],
    ref_path_list: list[str],
    device: torch.device | str,
):
    assert len(src_path_list) == len(ref_path_list)
    batchsize = len(src_path_list)

    td = datasets.TestData(
        src_path_list + ref_path_list, face_detector, iscrop=True, sample_step=10
    )

    # Recover data from datasets.TestData
    src_td = {"image": []}
    ref_td = {"image": [], "tform": [], "original_image": []}
    for i in range(0, batchsize):
        src_td["image"].append(td[i]["image"])
    for i in range(batchsize, batchsize * 2):
        ref_td["image"].append(td[i]["image"])
        ref_td["tform"].append(td[i]["tform"])
        ref_td["original_image"].append(td[i]["original_image"])

    # Stack along batch dimension
    src_td["image"] = einops.pack(src_td["image"], "* c h w")[0].to(device)
    ref_td["image"] = einops.pack(ref_td["image"], "* c h w")[0].to(device)
    ref_td["tform"] = einops.pack(ref_td["tform"], "* rows cols")[0].to(device)
    ref_td["original_image"] = einops.pack(ref_td["original_image"], "* c h w")[0].to(
        device
    )

    ref_tform_inv_t = torch.linalg.inv(ref_td["tform"]).transpose(-2, -1)
    ref_original = ref_td["original_image"]

    with torch.no_grad():
        src_embeds = deca.encode(src_td["image"])
        ref_embeds = deca.encode(ref_td["image"])

        new_embeds = {}
        # the types of embeddings are
        # ['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail']
        # We take `shape`, `tex`, `light`, `detail` from the source
        new_embeds["shape"] = src_embeds["shape"]
        new_embeds["tex"] = src_embeds["tex"]
        new_embeds["light"] = src_embeds["light"]
        new_embeds["detail"] = src_embeds["detail"]
        # Use the other embeddings from the references.
        new_embeds["exp"] = ref_embeds["exp"]
        new_embeds["pose"] = ref_embeds["pose"]
        new_embeds["cam"] = ref_embeds["cam"]
        new_embeds["images"] = ref_embeds["images"]

        opdict, visdict = deca.decode(
            new_embeds,
            render_orig=True,
            original_image=ref_original,
            tform=ref_tform_inv_t,
        )

    depth: Float[torch.Tensor, "b c h w"] = deca.render.render_depth(
        opdict["trans_verts"]
    ).to(device)
    lm: Float[torch.Tensor, "b c h w"] = visdict["landmarks2d"].to(device)

    # Clamp into [0,1]
    depth = depth.clamp(0, 1)
    lm = lm.clamp(0, 1)
    return depth, lm


@torch.no_grad()
def generate_depth_lm(
    cfg,
    device,
):
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    face_detector = detectors.FAN(device=device)
    output_dir = pathlib.Path(cfg.output_dir) / "dataset" / cfg.output_label

    ds = Dataset(cfg.dataset.root_path, cfg.dataset.eval_list.path)
    loader = data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    depth_dir = output_dir / "depth"
    lm_dir = output_dir / "lm"
    shutil.rmtree(depth_dir, ignore_errors=True)
    shutil.rmtree(lm_dir, ignore_errors=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating depth and lm images of `{cfg.dataset.name}` (with `{cfg.dataset.eval_list.name}` list) in {output_dir}"
    )
    # # DEBUG: Only do one batch
    # import itertools
    # loader = itertools.islice(loader, 1)

    for src_path, ref_path, src_base, ref_base in tqdm(loader):
        depth, lm = render_depth_lm(deca, face_detector, src_path, ref_path, device)

        # Convert to appropiate image format
        depth = (depth.clamp(0, 1) * 255.0).to("cpu", torch.uint8)
        lm = (lm.clamp(0, 1) * 255.0).to("cpu", torch.uint8)

        # Save output
        minibatch = depth.shape[0]
        for b in range(minibatch):
            src_base1 = src_base[b].replace("/", "%")
            ref_base1 = ref_base[b].replace("/", "%")
            basename = f"{src_base1}_{ref_base1}.png"
            depth_path = depth_dir / basename
            lm_path = lm_dir / basename
            write_png(depth[b], str(depth_path))
            write_png(lm[b], str(lm_path))


@hydra.main(
    version_base=None,
    config_path="../../configs/",
    config_name="base_generate_depth_lm",
)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_depth_lm(cfg, device)


if __name__ == "__main__":
    main()
