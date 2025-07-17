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

from core.solver_lm_perceptual import Solver as SolverMpie

# from core.solver_lm_perceptual_vox1 import Solver as SolverVox1
from metrics.eval import generate_images


@hydra.main(version_base=None, config_path="../configs/eval/", config_name="mpie250000")
def main(cfg: DictConfig):
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)

    solver = SolverMpie(cfg)
    solver._load_checkpoint(cfg.resume_iter)
    with torch.no_grad():
        generate_images(
            solver.nets_ema,
            cfg,
            step=cfg.resume_iter,
            mode="latent",
            list_path="./datasets/mpie_cross_test_cvpr_full.txt",
            output_dir=cfg.eval_dir,
        )


if __name__ == "__main__":
    main()
