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
from omegaconf import DictConfig
from torchvision.io import write_png
from tqdm import tqdm

from pasl.data import get_data_loader
from pasl.solver_lm_perceptual import Solver
from pasl.utils import set_seed


@torch.no_grad()
def generate_samples(
    nets,
    cfg,
    device,
):
    output_dir: Path = Path(cfg.output_dir) / "eval" / str(cfg.output_label)
    if output_dir.is_dir():
        print("Output directory already exists. Aborting.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_lm_root_path = Path(cfg.output_dir) / "dataset" / f"{cfg.list.name}"
    loader = get_data_loader(
        root_path=cfg.dataset.root_path,
        depth_lm_root_path=depth_lm_root_path,
        list_path=cfg.list.path,
        img_size=cfg.model.img_size,
        batch_size=cfg.batch_size,
        drop_last=True,
    )

    print(
        f"Generating sample outputs of `{cfg.dataset.name}` (with list {cfg.list.name})\n"
        f"Saving to {output_dir}"
    )

    # # DEBUG: Only do N batches
    # import itertools
    # N = 10
    # loader = itertools.islice(loader, N)

    # new: src, ref, gt, depth, lm, ang_src, ang_ref
    # x_src: img, img2, img_lm, img_lm2, lm, gt, torch.tensor(name_angle),torch.tensor(name2_angle)
    for i, (src, ref, gt, depth, lm, ang_src, ang_ref) in enumerate(tqdm(loader)):
        src = src.to(device)
        ref = ref.to(device)
        gt = gt.to(device)
        depth = depth.to(device)
        lm = lm.to(device)
        ang_src = ang_src.to(device)
        ang_ref = ang_ref.to(device)

        depth224 = F.interpolate(
            depth,  # need to add batch dim
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        # # Broadcast depth to RGB
        # depth224 = einops.repeat(depth224, "1 h w -> 3 h w")

        if cfg.model.masks:
            masks = depth224
        else:
            masks = None

        for j in range(cfg.num_outputs_per_domain):
            with torch.no_grad():
                style = nets.style_encoder(src)
                output = nets.generator(depth, lm, style, masks=masks)

            # Convert to image format to save
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


@hydra.main(
    version_base=None, config_path="../../configs/", config_name="base_generate"
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    solver = Solver(cfg, device)
    solver.load_from_path(cfg.model.nets_ema_path)
    generate_samples(solver.nets_ema, cfg, device)


if __name__ == "__main__":
    main()
