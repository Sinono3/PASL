"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path

import einops
import hydra
import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from torchvision.io import write_png
from torchvision.utils import save_image
from tqdm import tqdm

from decalib.datasets import detectors
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from pasl.data import get_data_loader
import pasl.render
from pasl.core.model import PaslModel
from pasl.utils import set_seed


@torch.no_grad()
def generate_samples(
    nets,
    deca,
    face_detector,
    loader,
    cfg,
    device,
):
    output_dir: Path = (
        Path(cfg.globals.output_dir) / "eval" / str(cfg.generate.output_label)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating sample outputs with list {cfg.list.name}")
    print(f"Saving to {output_dir}")

    # # DEBUG: Only do N batches
    # import itertools
    # N = 10
    # loader = itertools.islice(loader, N)

    for i, minibatch in enumerate(tqdm(loader)):
        src, ref, gt, ang_src, ang_ref, src_path, ref_path, gt_path = minibatch
        src, ref, gt = src.to(device), ref.to(device), gt.to(device)
        ang_src, ang_ref = ang_src.to(device), ang_ref.to(device)

        depth, lm = pasl.render.depth_from_src_ref_paths(
            deca, face_detector, src_path, ref_path, device
        )
        depth = einops.repeat(depth, "b 1 h w -> b 3 h w")

        if cfg.model.masks:
            masks = depth
        else:
            masks = None

        for j in range(cfg.generate.num_outputs):
            with torch.no_grad():
                style = nets.style_encoder(src)
                output = nets.generator(depth, lm, style, masks=masks)

            # Convert to RGB byte image format to save
            output = (output.clamp(0, 1) * 255.0).to("cpu", torch.uint8)

            minibatch = src.size(0)
            for k in range(minibatch):
                # dataset sample index = base batch img index + sample index in batch
                sample_idx = i * cfg.generate.batch_size + (k + 1)
                # output index
                output_idx = j + 1
                basename = f"{sample_idx:04}_{output_idx:02}.png"
                write_png(output[k], str(output_dir / basename))

    print("Generation finished")


def generate_debug_grid(
    nets,
    deca,
    face_detector,
    loader,
    cfg,
    device,
):
    minibatch = next(iter(loader))
    src, ref, gt, ang_src, ang_ref, src_path, ref_path, gt_path = minibatch
    src, ref, gt = src.to(device), ref.to(device), gt.to(device)
    ang_src, ang_ref = ang_src.to(device), ang_ref.to(device)

    depth, lm = pasl.render.depth_from_src_ref_paths(
        deca, face_detector, src_path, ref_path, device
    )
    depth = einops.repeat(depth, "b 1 h w -> b 3 h w")

    with torch.no_grad():
        style = nets.style_encoder(src)
        output = nets.generator(
            depth, lm, style, masks=(depth if cfg.model.masks else None)
        )

    # # DEBUG: load images
    # original = []
    # print(ref_path)
    # for path in ref_path:
    #     original.append(read_image(path))
    # original = einops.pack(original, "* c h w")[0].to(device, torch.float32) / 255.0

    # Tile on grid
    images = [src, ref, lm, depth, output]
    grid = einops.rearrange(images, "img b c h w -> c (b h) (img w)")
    save_image(grid, "output/debug_grid.png")


@hydra.main(
    version_base=None, config_path="../../configs/", config_name="base_generate_samples"
)
def main(cfg: DictConfig):
    set_seed(cfg.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_data_loader(
        root_path=cfg.list.root_path,
        list_path=cfg.list.path,
        img_size=cfg.model.img_size,
        batch_size=cfg.generate.batch_size,
        num_workers=cfg.globals.num_workers,
        drop_last=True,
    )

    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    face_detector = detectors.FAN(device=device)
    solver = PaslModel(cfg).to(device)
    solver.load_from_path(cfg.model.nets_ema_path)

    # DEBUG: Outputs all the images for a single batch (src, ref, gt, depth, lm, output)
    # generate_debug_grid(solver.nets_ema, deca, face_detector, loader, cfg, device)
    generate_samples(solver.nets_ema, deca, face_detector, loader, cfg, device)


if __name__ == "__main__":
    main()
