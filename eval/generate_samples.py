"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import hydra
import torch
from omegaconf import DictConfig
from torch.backends import cudnn

from core.solver_lm_perceptual import Solver
from metrics.eval import generate_images


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)

    solver = Solver(cfg)
    solver.load_from_path(cfg.model.nets_ema_path)
    print(solver.nets_ema.keys())

    with torch.no_grad():
        generate_images(
            solver.nets_ema,
            cfg,
            mode="latent",
            list_path=cfg.dataset.eval_list,
            output_dir=cfg.output_dir,
            output_label=cfg.output_label,
        )


if __name__ == "__main__":
    main()
