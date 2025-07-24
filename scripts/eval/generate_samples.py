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

from deca.decalib.datasets import detectors
from deca.decalib.deca import DECA
from deca.decalib.utils.config import cfg as deca_cfg
from pasl.data import get_data_loader
from pasl.render_deca import render_depth_lm_batch
from pasl.solver_lm_perceptual import Solver
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
    output_dir: Path = Path(cfg.output_dir) / "eval" / str(cfg.output_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Generating sample outputs of `{cfg.dataset.name}` (with list {cfg.list.name})\n"
        f"Saving to {output_dir}"
    )

    # # DEBUG: Only do N batches
    # import itertools
    # N = 10
    # loader = itertools.islice(loader, N)

    for i, minibatch in enumerate(tqdm(loader)):
        src, ref, gt, ang_src, ang_ref, src_path, ref_path, gt_path = minibatch
        src, ref, gt = src.to(device), ref.to(device), gt.to(device)
        ang_src, ang_ref = ang_src.to(device), ang_ref.to(device)

        depth, lm = render_depth_lm_batch(
            deca, face_detector, src_path, ref_path, device
        )
        depth = einops.repeat(depth, "b 1 h w -> b 3 h w")

        if cfg.model.masks:
            masks = depth
        else:
            masks = None

        for j in range(cfg.num_outputs_per_domain):
            with torch.no_grad():
                style = nets.style_encoder(src)
                output = nets.generator(depth, lm, style, masks=masks)

            # Convert to RGB byte image format to save
            output = (output.clamp(0, 1) * 255.0).to("cpu", torch.uint8)

            minibatch = src.size(0)
            for k in range(minibatch):
                # dataset sample index = base batch img index + sample index in batch
                sample_idx = i * cfg.batch_size + (k + 1)
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

    # DEBUG: Generate our own depth and LM
    depth, lm = render_depth_lm_batch(deca, face_detector, src_path, ref_path, device)
    depth = einops.repeat(depth, "b 1 h w -> b 3 h w")

    with torch.no_grad():
        style = nets.style_encoder(src)
        output = nets.generator(
            depth, lm, style, masks=(depth if cfg.model.masks else None)
        )

    # Tile on grid
    images = [src, ref, depth, lm, output]
    grid = einops.rearrange(images, "img b c h w -> c (b h) (img w)")
    save_image(grid, f"debug_grid_{cfg.batch_size}.png")


@hydra.main(
    version_base=None, config_path="../../configs/", config_name="base_generate"
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = get_data_loader(
        root_path=cfg.dataset.root_path,
        list_path=cfg.list.path,
        img_size=cfg.model.img_size,
        batch_size=cfg.batch_size,
        drop_last=True,
    )

    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)
    face_detector = detectors.FAN(device=device)
    solver = Solver(cfg, device)
    solver.load_from_path(cfg.model.nets_ema_path)
    # DEBUG: Outputs all the images for a single batch (src, ref, gt, depth, lm, output)
    generate_debug_grid(solver.nets_ema, deca, face_detector, loader, cfg, device)
    # generate_samples(solver.nets_ema, deca, face_detector, loader, cfg, device)


if __name__ == "__main__":
    main()
