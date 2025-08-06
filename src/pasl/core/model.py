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

# from lightning.fabric import Fabric
from jaxtyping import Float
from torch import Tensor

import pasl.utils_lm as utils

from pasl.core.architecture import (
    Generator,
    StyleEncoder,
    Discriminator_img_pix,
    Discriminator_img2_pix,
    build_model,
)


class PaslModel(nn.Module):
    generator: Generator
    style_encoder: StyleEncoder
    discriminator: Discriminator_img_pix
    discriminator2: Discriminator_img2_pix

    def __init__(self, cfg):
        super().__init__()
        generator, style_encoder, discriminator, discriminator2 = build_model(cfg)
        self.generator = generator
        self.style_encoder = style_encoder
        self.discriminator = discriminator
        self.discriminator2 = discriminator2

    def init(self):
        self.apply(utils.he_init)

    def load_from_path(self, path, device=None):
        if device is None:
            device = next(self.parameters()).device

        self.load_state_dict(torch.load(path, map_location=device))

    @torch.no_grad()
    def sample(
        self,
        src_style: Float[torch.Tensor, "b embed"],
        depth: Float[torch.Tensor, "b c h w"],
        lm: Float[torch.Tensor, "b c h w"],
        masks=False,
    ):
        masks = lm if masks else None
        return self.generator(depth, lm, src_style, masks=masks)

    @torch.no_grad()
    def extract(self, src: Float[Tensor, "b c h w"]) -> Float[Tensor, "b embed"]:
        return self.style_encoder(src)
