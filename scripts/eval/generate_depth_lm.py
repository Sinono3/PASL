#!/usr/bin/env python
# Script used to generate (1) landmark images and (2) depth images for
# items in a dataset.
#
# This was originally done on the DataLoader, but it is more efficient to do
# it all at once.


import os
import shutil
from pathlib import Path

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

import pasl.render_deca as rd
from deca.decalib.datasets import detectors
from deca.decalib.deca import DECA
from deca.decalib.utils.config import cfg as deca_cfg
from pasl.utils import set_seed


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

        print(f"Processing sample list {list_path}")
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
        print(f"Finished processing sample list {list_path}")

    def __len__(self):
        return len(self.samples_src)

    def __getitem__(self, idx):
        p1, p2 = self.samples_src[idx], self.samples_ref[idx]
        src_path = os.path.join(self.root_path, p1)
        ref_path = os.path.join(self.root_path, p2)
        return src_path, ref_path, p1, p2


@torch.no_grad()
def generate_depth_lm(
    cfg,
    device,
):
    output_dir = Path(cfg.output_dir) / "dataset" / str(cfg.output_label)
    if output_dir.is_dir():
        print("Output directory already exists. Aborting.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = data.DataLoader(
        Dataset(cfg.dataset.root_path, cfg.list.path),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    depth_dir: Path = output_dir / "depth"
    lm_dir: Path = output_dir / "lm"
    depth_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DECA model")
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    face_detector = detectors.FAN(device=device)

    print(
        f"Generating depth and lm images of `{cfg.dataset.name}` (with list {cfg.list.name})\n"
        f"Saving to {output_dir}"
    )
    # # DEBUG: Only do N batches
    # import itertools
    # N = 2
    # loader = itertools.islice(loader, N)

    for i, (src_path, ref_path, src_base, ref_base) in enumerate(tqdm(loader)):
        depth, lm = rd.render_depth_lm_batch(
            deca, face_detector, src_path, ref_path, device
        )

        # Convert to appropiate image format
        depth = (depth.clamp(0, 1) * 255.0).to("cpu", torch.uint8)
        lm = (lm.clamp(0, 1) * 255.0).to("cpu", torch.uint8)

        # Save output
        minibatch = depth.size(0)
        for k in range(minibatch):
            # list sample index = base batch img index + sample index in batch
            sample_idx = i * cfg.batch_size + (k + 1)
            basename = f"{sample_idx:04}.png"
            depth_path = depth_dir / basename
            lm_path = lm_dir / basename
            write_png(depth[k], str(depth_path))
            write_png(lm[k], str(lm_path))

    print("Generation finished")


@hydra.main(
    version_base=None,
    config_path="../../configs/",
    config_name="base_generate_depth_lm",
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_depth_lm(cfg, device)


if __name__ == "__main__":
    main()
