"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float
from tensorboardX import SummaryWriter
from torch import Tensor

import pasl.utils_lm as utils

from .model_lm_talking import build_model


class Solver(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.nets, self.nets_ema = build_model(cfg)
        self.writer = SummaryWriter(
            Path(cfg.globals.output_dir) / "log" / "test_reconstruction"
        )

        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + "_ema", module)

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ("ema" not in name) and ("fan" not in name):
                print("Initializing %s..." % name)
                network.apply(utils.he_init)

    # Loads nets_ema from a path
    def load_from_path(self, path):
        pickle = torch.load(path, map_location=self.device)
        self.nets_ema.generator.load_state_dict(pickle["generator"])
        self.nets_ema.style_encoder.load_state_dict(pickle["style_encoder"])

    @torch.no_grad()
    def sample(
        self,
        src: Float[torch.Tensor, "b c h w"],
        depth: Float[torch.Tensor, "b c h w"],
        lm: Float[torch.Tensor, "b c h w"],
    ):
        src = src.to(self.device)
        depth = depth.to(self.device)
        lm = lm.to(self.device)

        if self.cfg.model.masks:
            masks = lm
        else:
            masks = None

        x_fake = self.nets_ema.generator(depth, lm, src, masks=masks)
        return x_fake

    @torch.no_grad()
    def extract(self, src: Float[Tensor, "b c h w"]) -> Float[Tensor, "b embed"]:
        src = src.to(self.device)
        s_ref = self.nets_ema.style_encoder(src)
        return s_ref
